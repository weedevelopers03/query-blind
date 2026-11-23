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
    if not models_dir.exists():
        return None, None
    h2_records: list[dict[str, Any]] = []
    i2_rows: list[dict[str, Any]] = []

    for mp in sorted(models_dir.glob("*.json")):
        d = _load_json(mp)
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
    layer_key: str = "l11",
    level: str = "R2",
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any] | None:
    rep_p = m4_dir / "similarity_m4_report.json"
    if not rep_p.exists():
        return None
    rep = _load_json(rep_p)
    lr = rep.get("layers", {}).get(layer_key)
    if lr is None:
        return None
    d = lr["levels"][level]["metrics"]["cka"]
    tau_c0_c1 = float(d["tau_c0_c1_full"])

    # Prefer raw split-half arrays if present; else reconstruct normal approximation from reported mean & 95% CI
    def _extract_samples(block: dict[str, Any]) -> np.ndarray:
        if "samples" in block and block["samples"]:
            return np.asarray(block["samples"], dtype=np.float64)
        mu = float(block["mean"])
        sd = max((float(block["ci95_high"]) - float(block["ci95_low"])) / (2.0 * 1.959964), 1e-6)
        rng = np.random.default_rng(seed + 101)
        return rng.normal(loc=mu, scale=sd, size=200)

    ds = _extract_samples(d["diff_same_minus_noise"])
    dc = _extract_samples(d["diff_cross_minus_noise"])

    tau_rand = None
    if reviewer_controls_json is not None and reviewer_controls_json.exists():
        rc = _load_json(reviewer_controls_json)
        # Look for matched C1_rand entry in reviewer_controls.json
        c1r = rc.get("m4_c1_random_drop", {})
        if isinstance(c1r, dict):
            key = f"{layer_key}_{level}_cka"
            if key in c1r:
                tau_rand = float(c1r[key]["tau_c0_c1_rand_mean"])
            elif layer_key in c1r and level in c1r[layer_key]:
                tau_rand = float(c1r[layer_key][level]["tau_c0_c1_rand_mean"])

    return C.test_h3_m4_artifact_sensitivity(
        ds, dc,
        tau_c0_c1=tau_c0_c1,
        tau_c0_c1_rand=tau_rand,
        n_boot=n_boot,
        seed=seed,
        label=f"{layer_key}:{level}:cka",
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
                    f"- `R²(obj only) = {cr['r2_objective_only']:.4f}` | "
                    f"`R²(covs only) = {cr['r2_covariates_only']:.4f}` | "
                    f"`R²(full) = {cr['r2_full']:.4f}` | "
                    f"`ΔR²(obj | covs) = {cr['delta_r2_objective_given_covariates']:+.4f}`"
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
    ap.add_argument("--m4_layer", default="l11")
    ap.add_argument("--m4_level", default="R2")
    ap.add_argument("--reviewer_controls_json", default=None)
    ap.add_argument("--ralr_dirs", default="", help="Comma-separated name=path entries for RALR benchmarks.")
    ap.add_argument("--registry", default=str(REPO_ROOT / "configs" / "checkpoints.yaml"))
    ap.add_argument("--out_dir", default="results/confirmatory_stats")
    ap.add_argument("--n_boot", type=int, default=2000)
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
        h3 = _load_h3_from_m4_dir(
            Path(args.m4_dir),
            reviewer_controls_json=rc_p,
            layer_key=args.m4_layer,
            level=args.m4_level,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        if h3 is not None:
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
