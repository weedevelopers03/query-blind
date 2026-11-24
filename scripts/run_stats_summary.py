#!/usr/bin/env python3
"""Run frozen OSF §3 & §6 confirmatory statistical suite across saved experiment directories.

Reads outputs produced by:
  - run_pilot_m1_m2.py (--m1_dir)
  - run_pilot_interventions_i1_i2.py (--i1_i2_dir)
  - run_pilot_waterbirds_m3.py (--m3_dir)
  - run_pilot_similarity_m4.py (--m4_dir) + optional run_pilot_reviewer_controls.py (--reviewer_controls_json)
  - run_pilot_ralr.py (--ralr_dirs name=path,name2=path2)

Outputs:
  - <out_dir>/confirmatory_stats.json
  - <out_dir>/summary_confirmatory_stats.md
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stats import confirmatory as C


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_registry(path: Path) -> dict[str, dict[str, Any]]:
    rows = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {r["id"]: r for r in rows}


def _pick_l_star_from_m1(model_json: dict[str, Any]) -> int:
    lsums = model_json["layer_summaries"]
    eligible = [r for r in lsums if float(r["entropy_norm"]) <= 0.90]
    pool = eligible if eligible else lsums
    best = max(pool, key=lambda r: float(r["omega2_key"]))
    return int(best["layer"])


def _load_h1_from_m1_dir(
    m1_dir: Path,
    *,
    ssl_id: str = "dino_vits16",
    sup_id: str = "deit3_small_patch16_224",
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any] | None:
    p_ssl = m1_dir / "models" / f"{ssl_id}.json"
    p_sup = m1_dir / "models" / f"{sup_id}.json"
    if not (p_ssl.exists() and p_sup.exists()):
        return None
    j_ssl = _load_json(p_ssl)
    j_sup = _load_json(p_sup)
    l_ssl = _pick_l_star_from_m1(j_ssl)
    l_sup = _pick_l_star_from_m1(j_sup)
    pi_ssl = j_ssl.get("per_image", {})
    pi_sup = j_sup.get("per_image", {})
    # Support keys "l11" or "11"
    k_ssl = f"l{l_ssl}" if f"l{l_ssl}" in pi_ssl else str(l_ssl)
    k_sup = f"l{l_sup}" if f"l{l_sup}" in pi_sup else str(l_sup)
    if k_ssl not in pi_ssl or k_sup not in pi_sup:
        return None
    q_ssl = np.asarray(pi_ssl[k_ssl]["qii_s4"], dtype=np.float64)
    q_sup = np.asarray(pi_sup[k_sup]["qii_s4"], dtype=np.float64)
    res = C.test_h1_qii_s4_pair(q_ssl, q_sup, ssl_name=ssl_id, sup_name=sup_id, n_boot=n_boot, seed=seed)
    res["l_star_ssl"] = l_ssl
    res["l_star_sup"] = l_sup
    return res


def _load_h2_and_i2_from_dir(i1_i2_dir: Path, *, seed: int = 0) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    models_dir = i1_i2_dir / "models"
    report_p = i1_i2_dir / "i1_i2_report.json"
    if models_dir.exists() and list(models_dir.glob("*.json")):
        model_dicts = [_load_json(mp) for mp in sorted(models_dir.glob("*.json"))]
    elif report_p.exists():
        rep_all = _load_json(report_p)
        model_dicts = [rep_all[k] for k in sorted(rep_all.keys()) if isinstance(rep_all[k], dict) and "model_id" in rep_all[k]]
    else:
        return None, None
    h2_records: list[dict[str, Any]] = []
    i2_rows: list[dict[str, Any]] = []

    for d in model_dicts:
        mid = d["model_id"]
        obj = d.get("objective")
        # H2: L_int=1, S4
        s4_l1 = [r for r in d.get("i1_results", []) if int(r["L_int"]) == 1 and r["surrogate"] == "S4"]
        if s4_l1:
            rec: dict[str, Any] = {
                "model_id": mid,
                "objective": obj,
                "delta_box_auc": float(s4_l1[0]["delta_box_auc"]),
                "box_auc_clean": float(s4_l1[0]["box_auc_clean"]),
            }
            npz_p = i1_i2_dir / "per_image" / f"{mid}.npz"
            if npz_p.exists():
                z = np.load(npz_p)
                if "i1_l1_s4_delta_box_auc" in z and "clean_box_auc" in z:
                    rec["delta_box_auc_per_image"] = z["i1_l1_s4_delta_box_auc"]
                    rec["box_auc_clean_per_image"] = z["clean_box_auc"]
            h2_records.append(rec)

        # H2b I2 leg: final layer (max layer, usually l=11), mean_v, equal-count FG vs BG_eq_FG
        contrasts = [r for r in d.get("i2_fg_bg_contrasts", []) if r.get("mode") == "mean_v"]
        if contrasts:
            c_last = max(contrasts, key=lambda r: int(r["layer"]))
            if c_last.get("fg_cos_drop") is not None and c_last.get("bg_eq_fg_cos_drop") is not None:
                i2_rows.append({
                    "model_id": mid,
                    "objective": obj,
                    "layer": int(c_last["layer"]),
                    "fg_cos_drop": float(c_last["fg_cos_drop"]),
                    "bg_eq_fg_cos_drop": float(c_last["bg_eq_fg_cos_drop"]),
                })

    h2_res = C.test_h2_i1_supervision_contrast(h2_records) if h2_records else None
    i2_res = C.test_h2b_i2_log_ratio_leg(i2_rows, seed=seed) if i2_rows else None
    if h2_res is not None:
        h2_res["per_model_delta_box_auc"] = {r["model_id"]: float(r["delta_box_auc"]) for r in h2_records}
    return h2_res, i2_res


def _read_m3_routing_table(path: Path) -> dict[str, np.ndarray]:
    if path.suffix == ".parquet":
        import pandas as pd  # optional dependency (available on Kaggle)
        df = pd.read_parquet(path, columns=["layer", "bss_norm", "y_bird", "place_bg", "area_tercile"])
        return {k: df[k].to_numpy() for k in ("layer", "bss_norm", "y_bird", "place_bg", "area_tercile")}
    # CSV fallback (zero external deps)
    cols: dict[str, list] = {k: [] for k in ("layer", "bss_norm", "y_bird", "place_bg", "area_tercile")}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cols["layer"].append(int(float(row["layer"])))
            cols["bss_norm"].append(float(row["bss_norm"]))
            cols["y_bird"].append(int(float(row["y_bird"])))
            cols["place_bg"].append(int(float(row["place_bg"])))
            cols["area_tercile"].append(int(float(row["area_tercile"])))
    return {
        "layer": np.asarray(cols["layer"], dtype=np.int64),
        "bss_norm": np.asarray(cols["bss_norm"], dtype=np.float64),
        "y_bird": np.asarray(cols["y_bird"], dtype=np.int64),
        "place_bg": np.asarray(cols["place_bg"], dtype=np.int64),
        "area_tercile": np.asarray(cols["area_tercile"], dtype=np.int64),
    }


def _load_h2b_m3_from_dir(
    m3_dir: Path,
    *,
    layer: int = 8,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any] | None:
    rdir = m3_dir / "routing"
    if not rdir.exists():
        return None
    files = list(sorted(rdir.glob("*.parquet"))) + list(sorted(rdir.glob("*.csv")))
    if not files:
        return None
    seen: dict[str, Path] = {}
    for fp in files:
        seen.setdefault(fp.stem, fp)

    bss_by_model: dict[str, np.ndarray] = {}
    y_bird = place_bg = area_tercile = None
    for mid, fp in sorted(seen.items()):
        tbl = _read_m3_routing_table(fp)
        mask = (tbl["layer"] == int(layer))
        if not np.any(mask):
            continue
        bss_by_model[mid] = tbl["bss_norm"][mask]
        if y_bird is None:
            y_bird = tbl["y_bird"][mask]
            place_bg = tbl["place_bg"][mask]
            area_tercile = tbl["area_tercile"][mask]
    if not bss_by_model or y_bird is None:
        return None
    res = C.test_h2b_m3_leg(bss_by_model, y_bird, place_bg, area_tercile, n_boot=n_boot, seed=seed)
    res["layer"] = int(layer)
    return res


def _load_h3_from_m4_dir(
    m4_dir: Path,
    *,
    reviewer_controls_json: Path | None = None,
    layer_keys: list[str] | None = None,
    levels: list[str] | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any] | None:
    rep_p = m4_dir / "similarity_m4_report.json"
    if not rep_p.exists():
        return None
    rep = _load_json(rep_p)
    ds_name = str(rep.get("dataset", "imagenet")).lower()
    layers_dict = rep.get("layers", {})
    eff_layers = layer_keys if layer_keys else list(layers_dict.keys())
    eff_levels = levels if levels else ["R2", "R3"]

    # Optional matched random-drop C1_rand lookup from reviewer_controls.json (R3 CKA)
    rc_by_layer: dict[str, float] = {}
    c1_rand_source = None
    c1_rand_verified_on_same_zoo = False
    m4_models = set(rep.get("models", []))
    if reviewer_controls_json is not None and reviewer_controls_json.exists():
        rc = _load_json(reviewer_controls_json)
        rc_models = set(rc.get("models", []))
        m4r = rc.get("m4_matched_random_drop", {})
        ds_key = "imagenet_1k" if "imagenet" in ds_name else ("waterbirds_2k" if "waterbirds" in ds_name else "cub")
        if isinstance(m4r, dict) and ds_key in m4r and isinstance(m4r[ds_key], dict):
            for lk, entry in m4r[ds_key].items():
                if isinstance(entry, dict) and "tau_c0_c1_rand_mean" in entry:
                    rc_by_layer[lk] = float(entry["tau_c0_c1_rand_mean"])
            if rc_by_layer:
                same_zoo = bool(m4_models and rc_models and m4_models == rc_models)
                c1_rand_verified_on_same_zoo = same_zoo
                tag = "SAME-ZOO VERIFIED" if same_zoo else f"MISMATCHED ZOO ({len(rc_models) or 8} vs {len(m4_models)})"
                c1_rand_source = f"{reviewer_controls_json.name}:{ds_key} (R3 CKA; {tag})"

    def _extract_samples(block: dict[str, Any], s_offset: int) -> np.ndarray:
        if "samples" in block and block["samples"]:
            return np.asarray(block["samples"], dtype=np.float64)
        mu = float(block["mean"])
        sd = max((float(block["ci95_high"]) - float(block["ci95_low"])) / (2.0 * 1.959964), 1e-6)
        rng = np.random.default_rng(seed + 101 + s_offset)
        return rng.normal(loc=mu, scale=sd, size=200)

    cell_results: list[dict[str, Any]] = []
    idx_counter = 0
    for lk in eff_layers:
        lr = layers_dict.get(lk)
        if lr is None:
            continue
        for lv in eff_levels:
            lv_obj = lr.get("levels", {}).get(lv)
            if lv_obj is None:
                continue
            d = lv_obj.get("metrics", {}).get("cka")
            if d is None or "diff_same_minus_noise" not in d or "diff_cross_minus_noise" not in d:
                continue
            ds = _extract_samples(d["diff_same_minus_noise"], idx_counter * 2)
            dc = _extract_samples(d["diff_cross_minus_noise"], idx_counter * 2 + 1)
            tau_rand = rc_by_layer.get(lk) if lv == "R3" else None
            cell_results.append(
                C.test_h3_m4_artifact_sensitivity(
                    ds, dc,
                    tau_c0_c1=float(d["tau_c0_c1_full"]),
                    tau_c0_c1_rand=tau_rand,
                    n_boot=n_boot,
                    seed=seed + idx_counter,
                    label=f"{lk}:{lv}:cka",
                )
            )
            idx_counter += 1

    if not cell_results:
        return None
    if len(cell_results) == 1:
        return cell_results[0]
    return C.test_h3_m4_multicell(
        cell_results,
        c1_rand_source=c1_rand_source,
        c1_rand_verified_on_same_zoo=c1_rand_verified_on_same_zoo,
    )


def _load_h5_from_ralr_dirs(ralr_specs: list[str]) -> dict[str, Any] | None:
    if not ralr_specs:
        return None
    benchmarks: dict[str, dict[str, dict[str, float]]] = {}
    for spec in ralr_specs:
        if "=" in spec:
            bname, pstr = spec.split("=", 1)
        else:
            pstr = spec
            bname = Path(pstr).name
        rep_p = Path(pstr) / "ralr_report.json"
        if not rep_p.exists():
            continue
        rep = _load_json(rep_p)
        mmap: dict[str, dict[str, float]] = {}
        for mid, mr in rep.get("models", {}).items():
            pd = mr.get("paired_diffs", {}).get("ralr_adaptive_minus_cls_l_star", {})
            if "box_auc" in pd:
                mmap[mid] = {
                    "delta_box_auc": float(pd["box_auc"]["mean_diff"]),
                    "delta_otsu": float(pd["corloc_otsu"]["mean_diff"]) if "corloc_otsu" in pd else float("nan"),
                }
        if mmap:
            benchmarks[bname] = mmap
    if not benchmarks:
        return None
    return C.test_h5_ralr_sign_test(benchmarks)


def format_markdown_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# Confirmatory Statistical Summary (OSF §3 Holm-Bonferroni Family & §6 Confound Regression)\n",
        "## 1. Five-Test Holm-Bonferroni Family (α = 0.05, two-sided)\n",
        "| ID | Raw p | Holm Rank | Step α | Holm p_adj | Reject Holm? | Key Statistic |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in payload["holm_family"]:
        pr = f"{row['p_raw']:.4f}" if row.get("p_raw") is not None and np.isfinite(row["p_raw"]) else "—"
        rk = str(row["holm_rank"]) if row.get("holm_rank") is not None else "—"
        ast = f"{row['alpha_step']:.4f}" if row.get("alpha_step") is not None else "—"
        pa = f"{row['p_adj']:.4f}" if row.get("p_adj") is not None else "—"
        rej = "**YES**" if row.get("reject_holm") else "no"
        stat = row.get("headline_stat", "—")
        lines.append(f"| **{row['id']}** | `{pr}` | {rk} | `{ast}` | `{pa}` | {rej} | {stat} |")

    if payload.get("details", {}).get("H3", {}).get("per_cell"):
        h3d = payload["details"]["H3"]
        lines.append(f"\n## 1B. H3 Multi-Cell M4 Breakdown ({h3d['n_cells_triggered']}/{h3d['n_cells_total']} CKA cells < τ_noise; Bonferroni-adjusted inside H3)\n")
        if h3d.get("c1_rand_source"):
            lines.append(f"*Note: `C1_rand` reference values loaded from `{h3d['c1_rand_source']}`.*\n")
        lines.append("| Cell | τ(C0,C1) | Δsame | Δcross | Cell p_raw | Below Noise? | C1_rand Gap |")
        lines.append("|---|---:|---:|---:|---:|---|---:|")
        for c in h3d["per_cell"]:
            rg = f"{c['rand_gap']:+.4f}" if c.get("rand_gap") is not None else "—"
            bn = "YES" if c.get("both_below_noise") else "no"
            lines.append(
                f"| `{c['cell']}` | {c['tau_c0_c1']:.3f} | {c['mean_delta_same']:+.3f} | "
                f"{c['mean_delta_cross']:+.3f} | `{c['p_raw']:.4f}` | {bn} | {rg} |"
            )

    if payload.get("confound_regression"):
        lines.append("\n## 2. §10.3 Descriptive Confound Regression (Underpowered; Descriptive Only)\n")
        for yname, cr in payload["confound_regression"].items():
            lines.append(f"### Outcome: `{yname}` (`{cr.get('status')}`)\n")
            if cr.get("n_complete", 0) >= 3:
                lines.append(
                    f"- `n_complete = {cr['n_complete']}`, `p_active = {cr['p_active_predictors']}`, "
                    f"ridge={cr['rank_deficient_ridge_applied']}"
                )
                lines.append(
                    f"- `R²(obj only) = {cr['r2_objective_only']:.4f}` (adj `{cr.get('r2_adj_objective_only')}`) | "
                    f"`R²(covs only) = {cr['r2_covariates_only']:.4f}` (adj `{cr.get('r2_adj_covariates_only')}`) | "
                    f"`R²(full) = {cr['r2_full']:.4f}` (adj `{cr.get('r2_adj_full')}`) | "
                    f"`ΔR²_adj(obj | covs) = {cr.get('delta_r2_adj_objective_given_covariates')}`"
                )
                if cr.get("models_with_verify_flag"):
                    lines.append(f"- Models carrying `verify: true` in registry: `{cr['models_with_verify_flag']}`")
                if cr.get("skipped_missing_covariates"):
                    lines.append(f"- Skipped (missing covariates): `{cr['skipped_missing_covariates']}`")
            else:
                lines.append(f"- Insufficient complete registry rows (`n={cr.get('n_complete', 0)}`).")
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Run OSF §3 & §6 confirmatory statistical suite.")
    ap.add_argument("--m1_dir", default=None)
    ap.add_argument("--i1_i2_dir", default=None)
    ap.add_argument("--m3_dir", default=None)
    ap.add_argument("--m3_layer", type=int, default=8)
    ap.add_argument("--m4_dir", default=None)
    ap.add_argument("--m4_layers", default="l8,l9,l10,l11", help="Comma-separated M4 layer keys (default: all 4 last-third layers).")
    ap.add_argument("--m4_levels", default="R2,R3", help="Comma-separated M4 levels (default: R2,R3).")
    ap.add_argument("--reviewer_controls_json", default=None)
    ap.add_argument("--ralr_dirs", default="", help="Comma-separated name=path entries for RALR benchmarks.")
    ap.add_argument("--registry", default=str(REPO_ROOT / "configs" / "checkpoints.yaml"))
    ap.add_argument("--out_dir", default="results/confirmatory_stats")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = _load_registry(Path(args.registry))

    tests_in: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    confound_regs: dict[str, Any] = {}

    # H1
    if args.m1_dir:
        h1 = _load_h1_from_m1_dir(Path(args.m1_dir), n_boot=args.n_boot, seed=args.seed)
        if h1 is not None:
            h1["headline_stat"] = f"ΔQII_S4={h1['mean_diff']:+.4f} [{h1['ci95'][0]:+.4f}, {h1['ci95'][1]:+.4f}]"
            tests_in.append(h1)
            details["H1"] = {k: v for k, v in h1.items() if k != "boot_samples"}

    # H2 + H2b-I2
    h2 = i2_leg = None
    if args.i1_i2_dir:
        h2, i2_leg = _load_h2_and_i2_from_dir(Path(args.i1_i2_dir), seed=args.seed)
        if h2 is not None:
            h2["headline_stat"] = (
                f"β={h2['beta_label_text_vs_ssl']:+.4f} ({h2['fit_mode']}, id={h2['identifiable']})"
            )
            tests_in.append(h2)
            details["H2"] = h2

    # H2b-M3 + conjunction
    m3_leg = None
    if args.m3_dir:
        m3_leg = _load_h2b_m3_from_dir(Path(args.m3_dir), layer=args.m3_layer, n_boot=args.n_boot, seed=args.seed)
    if m3_leg is not None and i2_leg is not None:
        h2b = C.test_h2b_conjunction(m3_leg, i2_leg)
        h2b["headline_stat"] = (
            f"p_M3={h2b['p_m3']:.4f} (terc={m3_leg['terciles_held']}/3), "
            f"p_I2={h2b['p_i2']:.4f} (ΔR={i2_leg['delta_R_ssl_minus_lt']:+.3f})"
        )
        tests_in.append(h2b)
        details["H2b"] = h2b
    elif m3_leg is not None:
        details["H2b_M3_only"] = {k: v for k, v in m3_leg.items() if k != "boot_samples"}
    elif i2_leg is not None:
        details["H2b_I2_only"] = {k: v for k, v in i2_leg.items() if k != "boot_samples"}

    # H3
    if args.m4_dir:
        rc_p = Path(args.reviewer_controls_json) if args.reviewer_controls_json else None
        m4_layers = [x.strip() for x in args.m4_layers.split(",") if x.strip()]
        m4_levels = [x.strip() for x in args.m4_levels.split(",") if x.strip()]
        h3 = _load_h3_from_m4_dir(
            Path(args.m4_dir),
            reviewer_controls_json=rc_p,
            layer_keys=m4_layers,
            levels=m4_levels,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        if h3 is not None:
            if "n_cells_triggered" in h3:
                rg_txt = f", R3 min rand_gap={h3['r3_min_rand_gap']:+.3f}" if h3.get("r3_min_rand_gap") is not None else ""
                h3["headline_stat"] = (
                    f"{h3['n_cells_triggered']}/{h3['n_cells_total']} cells < noise "
                    f"(strongest {h3['strongest_cell']}: Δsame={h3['strongest_delta_same']:+.3f}, "
                    f"Δcross={h3['strongest_delta_cross']:+.3f}{rg_txt})"
                )
            else:
                h3["headline_stat"] = (
                    f"Δsame={h3['mean_delta_same']:+.3f}, Δcross={h3['mean_delta_cross']:+.3f}, "
                    f"rand_gap={h3['rand_gap']}"
                )
            tests_in.append(h3)
            details["H3"] = {k: v for k, v in h3.items() if k != "boot_samples"}

    # H5
    ralr_specs = [s.strip() for s in args.ralr_dirs.split(",") if s.strip()]
    if ralr_specs:
        h5 = _load_h5_from_ralr_dirs(ralr_specs)
        if h5 is not None:
            b_bits = [
                f"{bn}:{bv['primary_box_auc_excl_mae']['wins']}/{bv['primary_box_auc_excl_mae']['n']}"
                for bn, bv in h5["benchmarks"].items()
            ]
            h5["headline_stat"] = " | ".join(b_bits)
            tests_in.append(h5)
            details["H5"] = h5

    holm_rows = C.holm_bonferroni(tests_in, alpha=0.05)

    # §10.3 Descriptive confound regression on available model-level outcomes
    if i2_leg is not None and i2_leg.get("per_model"):
        r_map = {r["model_id"]: float(r["R_m"]) for r in i2_leg["per_model"]}
        confound_regs["I2_log_ratio_R_m"] = C.fit_confound_regression_descriptive(r_map, registry)
    if h2 is not None and h2.get("per_model_delta_box_auc"):
        confound_regs["I1_S4_L1_delta_box_auc"] = C.fit_confound_regression_descriptive(
            h2["per_model_delta_box_auc"], registry
        )

    payload = {
        "holm_alpha": 0.05,
        "holm_family": holm_rows,
        "details": details,
        "confound_regression": confound_regs,
    }
    (out_dir / "confirmatory_stats.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = format_markdown_summary(payload)
    (out_dir / "summary_confirmatory_stats.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
