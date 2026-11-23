"""Failing-if-broken unit & integration tests for OSF §3 & §6 confirmatory statistical suite."""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_stats_summary as stats_cli
from stats import confirmatory as C


def test_bootstrap_p_and_holm_stepdown_exact():
    # Exact OSF §3 formula: (2 * min(#{<=0}, #{>=0}) + 1) / (B + 1)
    b_all_pos = np.ones(1999, dtype=np.float64)
    assert abs(C.bootstrap_two_sided_p(b_all_pos, 0.0) - (1.0 / 2000.0)) < 1e-12

    b_sym = np.array([-1.0, -0.5, 0.5, 1.0])
    assert abs(C.bootstrap_two_sided_p(b_sym, 0.0) - 1.0) < 1e-12

    # 5-test Holm step-down: Step 1 (0.005 < 0.01) REJECTS; Step 2 (0.013 > 0.0125) FAILS;
    # Step 3 (0.015 < 0.0167) MUST NOT reject because Step 2 stopped the chain!
    tests = [
        {"id": "H1", "p_raw": 0.40},
        {"id": "H2", "p_raw": 0.013},
        {"id": "H2b", "p_raw": 0.015},
        {"id": "H3", "p_raw": 0.005, "boot_samples": np.linspace(-0.4, -0.1, 2000)},
        {"id": "H5", "p_raw": 0.08},
    ]
    res = {r["id"]: r for r in C.holm_bonferroni(tests, alpha=0.05)}
    assert res["H3"]["holm_rank"] == 1 and res["H3"]["reject_holm"] is True
    assert abs(res["H3"]["alpha_step"] - 0.01) < 1e-12
    assert abs(res["H3"]["p_adj"] - 0.025) < 1e-12
    assert res["H3"]["ci_holm"] is not None and abs(res["H3"]["ci_holm_level"] - 0.99) < 1e-5

    assert res["H2"]["holm_rank"] == 2 and res["H2"]["reject_holm"] is False
    assert abs(res["H2"]["p_adj"] - 0.052) < 1e-12
    # Critical step-down gate: H2b raw p=0.015 < 0.0167, MUST STILL BE FALSE after H2 fails!
    assert res["H2b"]["holm_rank"] == 3 and res["H2b"]["reject_holm"] is False
    # Monotone adjusted p: p_adj(H2b) = max(0.052, 3 * 0.015) = 0.052
    assert abs(res["H2b"]["p_adj"] - 0.052) < 1e-12


def test_crossed_reml_lmm_matches_dense_reml_and_recovers_effects():
    # 1) Exact algebraic verification against brute-force dense (MN x MN) covariance REML
    rng = np.random.default_rng(7)
    M, N = 4, 15
    m_idx = np.repeat(np.arange(M), N)
    i_idx = np.tile(np.arange(N), M)
    n = M * N
    u_m = np.array([-0.03, 0.02, -0.01, 0.04])
    v_i = rng.normal(0.0, 0.05, size=N)
    x_grp = np.repeat([0.0, 0.0, 1.0, 1.0], N)
    x_cov = rng.normal(0.0, 1.0, size=n)
    X = np.column_stack([np.ones(n), x_grp, x_cov])
    y = 0.01 + 0.06 * x_grp - 0.08 * x_cov + u_m[m_idx] + v_i[i_idx] + rng.normal(0.0, 0.02, size=n)

    fit = C.fit_crossed_reml_lmm(y, X, m_idx, i_idx, col_names=["b0", "grp", "cov"])

    # Compute dense -2*REML directly at the returned (sigma2_model, sigma2_image, sigma2_e)
    Zm = np.eye(M)[m_idx]
    Zi = np.eye(N)[i_idx]
    tau_m = fit["sigma2_model"] / fit["sigma2_e"]
    tau_i = fit["sigma2_image"] / fit["sigma2_e"]
    V = np.eye(n) + tau_m * (Zm @ Zm.T) + tau_i * (Zi @ Zi.T)
    Vinv = np.linalg.inv(V)
    XtVX = X.T @ Vinv @ X
    beta_dense = np.linalg.solve(XtVX, X.T @ Vinv @ y)
    r = y - X @ beta_dense
    s2_dense = float((r @ Vinv @ r) / (n - 3))
    cov_dense = s2_dense * np.linalg.inv(XtVX)
    _, ld_V = np.linalg.slogdet(V)
    _, ld_X = np.linalg.slogdet(XtVX)
    neg2_dense = ld_V + ld_X + (n - 3) * math.log(s2_dense) + (n - 3) * (1.0 + math.log(2.0 * math.pi))

    assert np.allclose(list(fit["beta"].values()), beta_dense, atol=1e-8)
    assert np.allclose(fit["cov_beta"], cov_dense, atol=1e-8)
    assert abs(fit["neg2_reml"] - neg2_dense) < 1e-8

    # 2) Larger 11-model (7 SSL + 4 label+text) + MAE excluded H2 test
    rng2 = np.random.default_rng(123)
    N2 = 120
    img_eff = rng2.normal(0.0, 0.04, size=N2)
    recs = []
    for mid in sorted(C.SSL_MODELS):
        cl = rng2.uniform(0.6, 0.9, size=N2)
        dy = 0.0 + (-0.05) * (cl - 0.75) + rng2.normal(0.0, 0.005) + img_eff + rng2.normal(0.0, 0.02, size=N2)
        recs.append({"model_id": mid, "delta_box_auc": float(dy.mean()), "box_auc_clean": float(cl.mean()),
                     "delta_box_auc_per_image": dy, "box_auc_clean_per_image": cl})
    for mid in sorted(C.LABEL_TEXT_MODELS):
        cl = rng2.uniform(0.6, 0.9, size=N2)
        dy = 0.06 + (-0.05) * (cl - 0.75) + rng2.normal(0.0, 0.005) + img_eff + rng2.normal(0.0, 0.02, size=N2)
        recs.append({"model_id": mid, "delta_box_auc": float(dy.mean()), "box_auc_clean": float(cl.mean()),
                     "delta_box_auc_per_image": dy, "box_auc_clean_per_image": cl})
    # Include MAE with extreme garbage value — MUST be excluded!
    recs.append({"model_id": "mae_vitb16", "delta_box_auc": -99.0, "box_auc_clean": 0.5,
                 "delta_box_auc_per_image": np.full(N2, -99.0), "box_auc_clean_per_image": np.full(N2, 0.5)})

    h2 = C.test_h2_i1_supervision_contrast(recs)
    assert h2["excluded_models"] == ["mae_vitb16"]
    assert h2["identifiable"] is True
    assert h2["n_ssl"] == 7 and h2["n_label_text"] == 4
    assert h2["fit_mode"] == "per_image_crossed_reml_lmm"
    assert abs(h2["beta_label_text_vs_ssl"] - 0.06) < 0.015
    assert h2["p_raw"] < 0.001

    # Single-model group MUST flag identifiable=False
    h2_single = C.test_h2_i1_supervision_contrast(recs[:3] + [recs[7]])
    assert h2_single["identifiable"] is False


def test_h2b_scale_invariance_log_ratio_and_iut_conjunction():
    # Scale invariance: multiplying both FG and BG_eq by 10x leaves R_m invariant (up to eps)
    r1 = C.compute_i2_log_ratio(0.20, 0.02)
    r2 = C.compute_i2_log_ratio(0.02, 0.002)
    assert abs(r1 - r2) < 1e-3, f"Log ratio not scale-invariant: {r1} vs {r2}"

    # 11-model I2 leg where SSL has high ratio (10x) at wildly varying absolute scales,
    # and label+text has lower ratio (3x)
    rows = []
    scales = [0.40, 0.30, 0.25, 0.15, 0.08, 0.04, 0.02]
    for mid, sc in zip(sorted(C.SSL_MODELS), scales):
        rows.append({"model_id": mid, "fg_cos_drop": sc, "bg_eq_fg_cos_drop": sc / 12.0})
    for mid, sc in zip(sorted(C.LABEL_TEXT_MODELS), [0.35, 0.20, 0.10, 0.05]):
        rows.append({"model_id": mid, "fg_cos_drop": sc, "bg_eq_fg_cos_drop": sc / 3.0})
    rows.append({"model_id": "mae_vitb16", "fg_cos_drop": 0.001, "bg_eq_fg_cos_drop": 0.9})

    i2_res = C.test_h2b_i2_log_ratio_leg(rows, seed=0)
    assert i2_res["excluded_models"] == ["mae_vitb16"]
    assert i2_res["perm_mode"] == "exact_330"  # C(11, 7) = 330
    assert i2_res["predicted_direction_ok"] is True
    assert i2_res["p_i2"] < 0.01  # 1 / 330 = 0.00303 (or 2/330 if symmetric)

    # M3 leg synthetic Waterbirds 2x2 where label+text has larger Water-Land main effect across all 3 terciles
    rng = np.random.default_rng(0)
    N = 600
    y_bird = np.tile([0, 0, 1, 1], N // 4)
    place_bg = np.tile([0, 1, 0, 1], N // 4)
    terc = np.repeat([0, 1, 2], N // 3)
    bss_map = {}
    for mid in sorted(C.SSL_MODELS):
        bss_map[mid] = 0.4 + 0.05 * place_bg + rng.normal(0.0, 0.05, size=N)
    for mid in sorted(C.LABEL_TEXT_MODELS):
        bss_map[mid] = 0.4 + 0.25 * place_bg + rng.normal(0.0, 0.05, size=N)
    m3_res = C.test_h2b_m3_leg(bss_map, y_bird, place_bg, terc, n_boot=500, seed=0)
    assert m3_res["predicted_direction_ok"] is True
    assert m3_res["terciles_held"] == 3
    assert m3_res["p_m3"] < 0.01

    # Conjunction passes when BOTH legs pass
    conj_pass = C.test_h2b_conjunction(m3_res, i2_res)
    assert conj_pass["p_raw"] == max(m3_res["p_m3"], i2_res["p_i2"])
    assert conj_pass["p_raw"] < 0.01

    # Conjunction FAILS (p_raw = 1.0) if I2 leg is null/wrong direction even when M3 passes strongly!
    i2_fail = {**i2_res, "predicted_direction_ok": False, "p_i2": 0.80}
    conj_fail = C.test_h2b_conjunction(m3_res, i2_fail)
    assert conj_fail["p_raw"] == 1.0


def test_h3_m4_and_h5_exact_sign_test_rules():
    # H3: both same & cross < 0 + rand_gap >= 0.10
    rng = np.random.default_rng(5)
    h3 = C.test_h3_m4_artifact_sensitivity(
        rng.normal(-0.25, 0.04, size=50),
        rng.normal(-0.22, 0.04, size=50),
        tau_c0_c1=0.65,
        tau_c0_c1_rand=0.92,
        n_boot=500,
        seed=0,
    )
    assert h3["both_below_noise"] is True
    assert h3["rand_gap_ok"] is True
    assert h3["p_raw"] < 0.01

    # H5 exact OSF §3 binomial p-values check:
    # 11/11 -> 0.0009765625; 10/11 -> 0.01171875; 11/12 -> 0.00634765625
    models_11 = sorted(C.SSL_MODELS | C.LABEL_TEXT_MODELS)
    b1 = {m: {"delta_box_auc": 0.02, "delta_otsu": 0.05} for m in models_11}
    b1["mae_vitb16"] = {"delta_box_auc": -0.01, "delta_otsu": -0.02}  # 11/12 sens, 11/11 prim
    b2 = {m: {"delta_box_auc": (0.02 if i < 10 else -0.01), "delta_otsu": 0.05} for i, m in enumerate(models_11)}
    b2["mae_vitb16"] = {"delta_box_auc": 0.01, "delta_otsu": 0.05}

    h5 = C.test_h5_ralr_sign_test({"cub": b1, "imagenet": b2})
    assert abs(h5["benchmarks"]["cub"]["primary_box_auc_excl_mae"]["p_two_sided"] - 0.0009765625) < 1e-9
    assert abs(h5["benchmarks"]["cub"]["sensitivity_box_auc_all"]["p_two_sided"] - 0.00634765625) < 1e-9
    assert abs(h5["benchmarks"]["imagenet"]["primary_box_auc_excl_mae"]["p_two_sided"] - 0.01171875) < 1e-9
    # 2nd-smallest primary p across 2 benchmarks is 0.01171875 (passes Step 2 <= 0.0125, fails Step 1 <= 0.01)
    assert abs(h5["p_raw"] - 0.01171875) < 1e-9

    # Secondary Otsu 11/11 MUST NEVER rescue a failed primary box-AUC (5/11 -> p=1.0)
    b_fail_auc = {m: {"delta_box_auc": (0.02 if i < 5 else -0.02), "delta_otsu": 0.20} for i, m in enumerate(models_11)}
    h5_fail = C.test_h5_ralr_sign_test({"cub": b_fail_auc, "imagenet": b_fail_auc})
    assert h5_fail["benchmarks"]["cub"]["secondary_otsu_excl_mae"]["wins"] == 11
    assert h5_fail["p_raw"] > 0.50


def test_confound_regression_and_cli_smoke(tmp_path: Path):
    # Synthetic end-to-end directories for CLI smoke test
    m1_dir = tmp_path / "m1"
    (m1_dir / "models").mkdir(parents=True)
    for mid, val in [("dino_vits16", 0.95), ("deit3_small_patch16_224", 0.94)]:
        (m1_dir / "models" / f"{mid}.json").write_text(json.dumps({
            "model_id": mid,
            "layer_summaries": [{"layer": 11, "entropy_norm": 0.70, "omega2_key": 0.50}],
            "per_image": {"l11": {"qii_s4": np.full(30, val).tolist()}},
        }))

    i12_dir = tmp_path / "i12"
    (i12_dir / "models").mkdir(parents=True)
    (i12_dir / "per_image").mkdir(parents=True)
    rng = np.random.default_rng(1)
    for mid in ["dino_vits16", "ibot_vits16", "deit3_small_patch16_224", "openclip_vitb16", "mae_vitb16"]:
        is_lt = mid in C.LABEL_TEXT_MODELS
        dy = rng.normal(0.05 if is_lt else 0.0, 0.01, size=20).astype(np.float32)
        cl = rng.uniform(0.6, 0.9, size=20).astype(np.float32)
        np.savez_compressed(i12_dir / "per_image" / f"{mid}.npz", clean_box_auc=cl, i1_l1_s4_delta_box_auc=dy)
        (i12_dir / "models" / f"{mid}.json").write_text(json.dumps({
            "model_id": mid,
            "i1_results": [{"L_int": 1, "surrogate": "S4", "delta_box_auc": float(dy.mean()), "box_auc_clean": float(cl.mean())}],
            "i2_fg_bg_contrasts": [{"layer": 11, "mode": "mean_v", "fg_cos_drop": 0.20, "bg_eq_fg_cos_drop": 0.06 if is_lt else 0.01}],
        }))

    m3_dir = tmp_path / "m3"
    (m3_dir / "routing").mkdir(parents=True)
    for mid in ["dino_vits16", "ibot_vits16", "deit3_small_patch16_224", "openclip_vitb16"]:
        is_lt = mid in C.LABEL_TEXT_MODELS
        with (m3_dir / "routing" / f"{mid}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["layer", "bss_norm", "y_bird", "place_bg", "area_tercile"])
            for i in range(60):
                yb = (i // 2) % 2
                pb = i % 2
                tc = i % 3
                val = 0.4 + (0.20 if is_lt else 0.04) * pb + float(rng.normal(0, 0.02))
                w.writerow([8, val, yb, pb, tc])

    m4_dir = tmp_path / "m4"
    m4_dir.mkdir(parents=True)
    (m4_dir / "similarity_m4_report.json").write_text(json.dumps({
        "layers": {"l11": {"levels": {"R2": {"metrics": {"cka": {
            "tau_c0_c1_full": 0.60,
            "diff_same_minus_noise": {"mean": -0.25, "ci95_low": -0.35, "ci95_high": -0.15},
            "diff_cross_minus_noise": {"mean": -0.22, "ci95_low": -0.32, "ci95_high": -0.12},
        }}}}}}
    }))

    ralr_dir = tmp_path / "ralr_cub"
    ralr_dir.mkdir(parents=True)
    (ralr_dir / "ralr_report.json").write_text(json.dumps({
        "models": {
            m: {"paired_diffs": {"ralr_adaptive_minus_cls_l_star": {
                "box_auc": {"mean_diff": 0.03}, "corloc_otsu": {"mean_diff": 0.05}
            }}}
            for m in ["dino_vits16", "ibot_vits16", "deit3_small_patch16_224", "openclip_vitb16", "mae_vitb16"]
        }
    }))

    out_dir = tmp_path / "stats_out"
    stats_cli.main([
        "--m1_dir", str(m1_dir),
        "--i1_i2_dir", str(i12_dir),
        "--m3_dir", str(m3_dir),
        "--m4_dir", str(m4_dir),
        "--ralr_dirs", f"cub={ralr_dir}",
        "--out_dir", str(out_dir),
        "--n_boot", "200",
    ])
    assert (out_dir / "confirmatory_stats.json").exists()
    assert (out_dir / "summary_confirmatory_stats.md").exists()
    out_j = json.loads((out_dir / "confirmatory_stats.json").read_text())
    assert len(out_j["holm_family"]) == 5
    assert out_j["confound_regression"]["I2_log_ratio_R_m"]["status"] == "DESCRIPTIVE_ONLY_UNDERPOWERED"
