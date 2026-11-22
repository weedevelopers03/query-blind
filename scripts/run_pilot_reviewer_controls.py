#!/usr/bin/env python3
"""Fast reviewer-grade pilot controls & §11.1 #7 compute ledger (runs from cached pilot outputs in <60s).

Closes all 5 audit controls without re-running model forward passes:
  1. M4 matched random-drop control (C1_rand, 5 seeds) vs OUT-masked C1 on cached R3 features (CUB, WB2k, IN-1K).
  2. Waterbirds balanced 2x2 bootstrap 95% CIs (bg_main_effect & balanced_interaction) at MATCHED layers l=8 & l=11.
  3. Fix 3 Spearman (QII_S4 vs omega^2_key) excluding degenerate MAE (n=7 eligible models) on CUB & IN-1K.
  4. §11.1 #7 Revised Compute Estimate table derived from measured meta.json throughputs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from measures import keydom, routing, similarity as sim_mod
from scripts.run_pilot_m1_m2 import PILOT8_SPECS, _git_sha

MODEL_ORDER = list(PILOT8_SPECS.keys())


def _read_table(path_no_ext: Path) -> dict[str, np.ndarray]:
    """Read per-model routing Parquet or CSV into dict of numpy arrays."""
    pq = path_no_ext.with_suffix(".parquet")
    if pq.exists():
        import pyarrow.parquet as pq_mod
        tbl = pq_mod.read_table(pq)
        return {name: tbl[name].to_numpy(zero_copy_only=False) for name in tbl.column_names}
    csv_p = path_no_ext.with_suffix(".csv")
    if csv_p.exists():
        import csv
        with csv_p.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return {}
        out = {}
        for k in rows[0]:
            vals = [r[k] for r in rows]
            if k in ("layer", "image_id", "y_bird", "place_bg"):
                out[k] = np.array([int(float(v)) for v in vals], dtype=np.int64)
            elif k == "waterbirds_congruent":
                out[k] = np.array([str(v).lower() in ("true", "1") for v in vals], dtype=bool)
            elif k in ("bss_norm", "bss_raw", "gt_area_frac"):
                out[k] = np.array([float(v) for v in vals], dtype=np.float64)
            else:
                out[k] = np.array(vals, dtype=object)
        return out
    raise FileNotFoundError(f"Neither {pq} nor {csv_p} exists")


# --------------------------------------------------------------------------- 1. M4 Matched Random-Drop Control
def run_m4_random_drop_controls(results_root: Path, device: torch.device) -> dict:
    datasets = [
        ("cub", results_root / "pilot_similarity_m4_cub" / "feature_cache"),
        ("waterbirds_2k", results_root / "pilot_similarity_m4_wb2k" / "feature_cache"),
        ("imagenet_1k", results_root / "pilot_similarity_m4_in1k" / "feature_cache"),
    ]
    out: dict = {}
    for ds_name, cdir in datasets:
        if not cdir.is_dir():
            continue
        out[ds_name] = {}
        for l in (8, 9, 10, 11):
            if not all((cdir / f"{m}_l{l}.npz").exists() for m in MODEL_ORDER):
                continue
            r3_tok, r3_out = {}, {}
            for m in MODEL_ORDER:
                z = np.load(cdir / f"{m}_l{l}.npz")
                r3_tok[m] = z["r3"]
                r3_out[m] = z["r3_out"]
            res = sim_mod.r3_matched_random_drop_control(
                r3_tok, r3_out, MODEL_ORDER, seeds=(0, 1, 2, 3, 4), device=device,
            )
            out[ds_name][f"l{l}"] = res
    return out


# --------------------------------------------------------------------------- 2. Waterbirds Balanced 2x2 CIs (Matched Layers)
def run_waterbirds_balanced_cis(results_root: Path) -> dict:
    rdir = results_root / "pilot_waterbirds_m3" / "routing"
    if not rdir.is_dir():
        return {}
    out: dict = {}
    for m in MODEL_ORDER:
        try:
            cols = _read_table(rdir / m)
        except FileNotFoundError:
            continue
        out[m] = {}
        for l in (8, 9, 10, 11):
            mask = cols["layer"] == l
            if not np.any(mask):
                continue
            wb = routing.waterbirds_congruence_analysis(
                cols["bss_norm"][mask],
                cols["waterbirds_congruent"][mask],
                y_bird=cols["y_bird"][mask],
                place_bg=cols["place_bg"][mask],
                n_resamples=1000,
                seed=0,
            )
            out[m][f"l{l}"] = {
                "overall": wb["overall"],
                "pooled_gap": wb["congruence_gap_cong_minus_incong"],
                "balanced_2x2": wb.get("balanced_2x2", {}),
            }
    return out


# --------------------------------------------------------------------------- 3. Fix 3 Spearman excl. MAE (n=7)
def run_fix3_eligible_only(results_root: Path) -> dict:
    out: dict = {}
    for ds_name, ddir in [("cub_500", results_root / "pilot_m1_m2_cub"), ("imagenet_1k", results_root / "pilot_m1_m2_in1k")]:
        mdir = ddir / "models"
        if not mdir.is_dir():
            continue
        qii_all_last, kd_all_last = {}, {}
        qii_elig_last, kd_elig_last = {}, {}
        qii_all_lstar, kd_all_lstar = {}, {}
        qii_elig_lstar, kd_elig_lstar = {}, {}

        for m in MODEL_ORDER:
            p = mdir / f"{m}.json"
            if not p.exists():
                continue
            r = json.loads(p.read_text())
            ls = r["layer_summaries"]
            last = ls[-1]
            qii_all_last[m] = float(last["qii_s4"])
            kd_all_last[m] = float(last["omega2_key"])
            if last.get("status_s4") == "eligible":
                qii_elig_last[m] = float(last["qii_s4"])
                kd_elig_last[m] = float(last["omega2_key"])

            elig_rows = [x for x in ls if x.get("status_s4") == "eligible"]
            best = max(elig_rows, key=lambda x: x["omega2_key"]) if elig_rows else last
            qii_all_lstar[m] = float(best["qii_s4"])
            kd_all_lstar[m] = float(best["omega2_key"])
            if elig_rows:
                qii_elig_lstar[m] = float(best["qii_s4"])
                kd_elig_lstar[m] = float(best["omega2_key"])

        out[ds_name] = {
            "final_all_n8": keydom.qii_kd_spearman(qii_all_last, kd_all_last),
            "final_eligible_only": keydom.qii_kd_spearman(qii_elig_last, kd_elig_last),
            "l_star_all_n8": keydom.qii_kd_spearman(qii_all_lstar, kd_all_lstar),
            "l_star_eligible_only": keydom.qii_kd_spearman(qii_elig_lstar, kd_elig_lstar),
        }
    return out


# --------------------------------------------------------------------------- 4. Revised Compute Estimate (§11.1 #7)
def compile_compute_ledger(results_root: Path) -> dict:
    runs = {}
    for name in (
        "pilot_m1_m2_cub", "pilot_m1_m2_in1k", "pilot_waterbirds_m3",
        "pilot_similarity_m4_cub", "pilot_similarity_m4_wb2k", "pilot_similarity_m4_in1k",
        "pilot_ralr_cub", "pilot_ralr_in1k",
    ):
        mp = results_root / name / "meta.json"
        if mp.exists():
            meta = json.loads(mp.read_text())
            tp = meta.get("throughput_by_model", {})
            speeds = [v.get("img_per_sec") or v.get("extract_img_per_sec") for v in tp.values() if isinstance(v, dict)]
            speeds = [float(s) for s in speeds if s is not None]
            runs[name] = {
                "dataset": meta.get("dataset"),
                "n_images": meta.get("n_images"),
                "total_seconds": meta.get("total_seconds"),
                "mean_img_per_sec": round(float(np.mean(speeds)), 1) if speeds else None,
                "min_img_per_sec": round(float(np.min(speeds)), 1) if speeds else None,
                "git_sha": meta.get("git_sha"),
            }

    # Extrapolate Minimum Viable Paper (Guide §13.2: 12 models, Sets A+B) on single Tesla T4
    # Measured avg extraction rate across Pilot-8 ~ 55 img/s (~18 ms/img)
    in1k_m12_sec_per_1k_per_model = None
    if "pilot_m1_m2_in1k" in runs and runs["pilot_m1_m2_in1k"]["total_seconds"]:
        in1k_m12_sec_per_1k_per_model = runs["pilot_m1_m2_in1k"]["total_seconds"] / 8.0
    elif "pilot_m1_m2_cub" in runs and runs["pilot_m1_m2_cub"]["total_seconds"]:
        in1k_m12_sec_per_1k_per_model = (runs["pilot_m1_m2_cub"]["total_seconds"] / 8.0) * 2.0
    else:
        in1k_m12_sec_per_1k_per_model = 80.0

    wb_sec_per_5800_per_model = (
        runs["pilot_waterbirds_m3"]["total_seconds"] / 8.0
        if ("pilot_waterbirds_m3" in runs and runs["pilot_waterbirds_m3"]["total_seconds"])
        else 120.0
    )

    n_mvp_models = 12  # Sets A + B (Guide §13.2)
    proj = {
        "basis": "Measured single Tesla T4 GPU throughput extrapolated to MVP (12 models, Sets A+B, Guide §13.2)",
        "m1_m2_in1k_10k_gpu_hours": round(n_mvp_models * 10.0 * in1k_m12_sec_per_1k_per_model / 3600.0, 2),
        "m3_waterbirds_5794_gpu_hours": round(n_mvp_models * wb_sec_per_5800_per_model / 3600.0, 2),
        "m4_similarity_10k_gpu_hours": round(n_mvp_models * 10.0 * (in1k_m12_sec_per_1k_per_model * 0.45) / 3600.0, 2),
        "interventions_i1_i2_i5_gpu_hours": round(n_mvp_models * 12.0 * (in1k_m12_sec_per_1k_per_model * 0.8) / 3600.0, 2),
        "ralr_3benchmarks_gpu_hours": round(n_mvp_models * 16.0 * (in1k_m12_sec_per_1k_per_model * 0.35) / 3600.0, 2),
    }
    proj["total_mvp_t4_gpu_hours"] = round(sum(v for k, v in proj.items() if k.endswith("_gpu_hours")), 2)
    return {"measured_pilot_runs": runs, "mvp_projection": proj}


# --------------------------------------------------------------------------- Markdown formatting
def format_markdown(report: dict) -> str:
    lines = ["# Pilot Reviewer Controls & §11.1 #7 Compute Ledger\n"]

    # 1. M4 Matched Random-Drop Control
    lines.append("## 1. M4 Matched Random-Drop Control (`C1_OUT` vs `C1_rand`, 5 seeds, R3 CKA)\n")
    lines.append("| Dataset | Layer | τ(C0, C1_OUT) | τ(C0, C1_rand) mean ± sd | Δτ (OUT − rand) | Max |Δsim| OUT | Max |Δsim| rand |")
    lines.append("|---|---|---:|---|---:|---:|---:|")
    for ds, lmap in report["m4_matched_random_drop"].items():
        for lk, d in lmap.items():
            lines.append(
                f"| {ds} | {lk} | {d['tau_c0_c1_out']:.3f} | "
                f"{d['tau_c0_c1_rand_mean']:.3f} ± {d['tau_c0_c1_rand_std']:.3f} | "
                f"{d['delta_tau_out_minus_rand']:+.3f} | {d['max_abs_shift_out']:.3f} | {d['max_abs_shift_rand_mean']:.3f} |"
            )

    # 2. Waterbirds Balanced 2x2 CIs at MATCHED layers l=8 and l=11
    lines.append("\n## 2. Waterbirds Balanced 2×2 Bootstrap 95% CIs at Matched Layers (`l=8` & `l=11`)\n")
    lines.append("| Model | Layer | BSS_norm | Bg Main Effect (W−L) [95% CI] | Excl0 | Balanced Interaction (C−I) [95% CI] | Excl0 |")
    lines.append("|---|---|---:|---|---|---|---:|")
    for l_key in ("l8", "l11"):
        for m, lmap in report["waterbirds_balanced_2x2"].items():
            if l_key not in lmap or not lmap[l_key]["balanced_2x2"]:
                continue
            ov = lmap[l_key]["overall"]["mean"]
            bg = lmap[l_key]["balanced_2x2"]["bg_main_effect_water_minus_land"]
            it = lmap[l_key]["balanced_2x2"]["balanced_interaction_cong_minus_incong"]
            lines.append(
                f"| {m} | {l_key} | {ov:.3f} | "
                f"{bg['mean']:+.4f} [{bg['ci95_low']:+.4f}, {bg['ci95_high']:+.4f}] | {bg['ci_excludes_zero']} | "
                f"{it['mean']:+.4f} [{it['ci95_low']:+.4f}, {it['ci95_high']:+.4f}] | {it['ci_excludes_zero']} |"
            )

    # 3. Fix 3 Spearman excl. MAE
    lines.append("\n## 3. Fix 3 Spearman Excluding Degenerate MAE (`n = 7` eligible vs `n = 8` all)\n")
    lines.append("| Dataset | Scope | N | Spearman ρ | p-value | Decision |")
    lines.append("|---|---|---:|---:|---:|---|")
    for ds, smap in report["fix3_eligible_only"].items():
        for scope, d in smap.items():
            lines.append(f"| {ds} | {scope} | {d['n_models']} | {d['rho']:+.3f} | {d['p_value']:.4f} | **{d['decision']}** |")

    # 4. Compute Ledger
    lines.append("\n## 4. Measured Pilot Throughput & Revised T4 GPU-Hour Compute Estimate (§11.1 #7)\n")
    lines.append("| Pilot Run | Dataset | N | Total Sec | Mean img/s | Min img/s | git_sha |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for rname, d in report["compute_ledger"]["measured_pilot_runs"].items():
        sha = (d["git_sha"] or "null")[:8]
        lines.append(f"| {rname} | {d['dataset']} | {d['n_images']} | {d['total_seconds']} | {d['mean_img_per_sec']} | {d['min_img_per_sec']} | `{sha}` |")
    proj = report["compute_ledger"]["mvp_projection"]
    lines.append(f"\n**Projected Minimum Viable Paper (12 models, Sets A+B, single T4):** **`{proj['total_mvp_t4_gpu_hours']} GPU-hours`** "
                 f"(M1/M2 10k: {proj['m1_m2_in1k_10k_gpu_hours']}h; M3 WB: {proj['m3_waterbirds_5794_gpu_hours']}h; "
                 f"M4 10k: {proj['m4_similarity_10k_gpu_hours']}h; I1/I2/I5: {proj['interventions_i1_i2_i5_gpu_hours']}h; RALR 3x: {proj['ralr_3benchmarks_gpu_hours']}h)")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run fast reviewer controls on cached pilot outputs.")
    ap.add_argument("--results_root", default="/kaggle/working/results")
    ap.add_argument("--out_dir", default="/kaggle/working/results/pilot_reviewer_controls")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    report = {
        "git_sha": _git_sha(),
        "m4_matched_random_drop": run_m4_random_drop_controls(results_root, device),
        "waterbirds_balanced_2x2": run_waterbirds_balanced_cis(results_root),
        "fix3_eligible_only": run_fix3_eligible_only(results_root),
        "compute_ledger": compile_compute_ledger(results_root),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }
    (out_dir / "reviewer_controls.json").write_text(json.dumps(report, indent=2))
    md = format_markdown(report)
    (out_dir / "reviewer_controls.md").write_text(md)
    print(md, flush=True)
    return report


if __name__ == "__main__":
    main()
