"""Frozen OSF §3 & §6 confirmatory statistical tests for Query-Blind Localization Audit.

Implements verbatim:
- Two-sided bootstrap p-value p_boot = (2 * min(#{theta* <= 0}, #{theta* >= 0}) + 1) / (B + 1)
- Step-down Holm-Bonferroni family correction (m = 5 primary tests, alpha = 0.05) with Holm-adjusted CIs
- H1: Matched S/16 paired image-level bootstrap on head-weighted QII_S4 at l*
- H2: Exact crossed random-effects REML LMM
      Delta_box_AUC_{m,i} ~ I(label+text) + box_auc_clean_{m,i} + (1 | model_id) + (1 | image_id)
      excluding mae_vitb16 (reconstruction), with automatic single-model identifiability flag
- H2b: Intersection-union conjunction (p_H2b = max(p_M3, p_I2)) combining:
      * M3 leg: Waterbirds balanced 2x2 Bg Main Effect (Water - Land) vs max size-matched SSL across area terciles
      * I2 leg: scale-invariant log ratio R_m = log((FG + eps) / (BG_eq + eps)) between-family contrast
- H3: M4 artifact-controlled similarity Delta_tau = tau(C0, C1) - tau_noise + C1_rand gap >= 0.10
- H5: Exact two-sided binomial sign test on primary box-AUC (n=11 excl. MAE primary + n=12 sensitivity)
- §10.3: Descriptive multi-covariate confound regression from configs/checkpoints.yaml
"""
from __future__ import annotations

import itertools
import math
import re
from typing import Any

import numpy as np
from scipy import optimize, stats

# --------------------------------------------------------------------------- Model family maps (OSF §1 & §5)
SSL_MODELS = frozenset({
    "dino_vits16",
    "ibot_vits16",
    "mocov3_vitb16",
    "dinov2_vitb14",
    "dinov2_vitb14_reg",
    "dinov2_vits14",
    "dinov2_vits14_reg",
})
LABEL_TEXT_MODELS = frozenset({
    "deit3_small_patch16_224",
    "deit3_small_patch16_224_in21k",
    "openclip_vitb16",
    "metaclip_vitb16",
})
EXCLUDED_RECON_MODELS = frozenset({"mae_vitb16"})

SSL_OBJECTIVES = frozenset({"self_distillation", "contrastive"})
LABEL_TEXT_OBJECTIVES = frozenset({"label_supervised", "image_text_contrastive"})
RECON_OBJECTIVES = frozenset({"reconstruction"})

MODEL_SIZE_BUCKET = {
    "dino_vits16": "S",
    "ibot_vits16": "S",
    "deit3_small_patch16_224": "S",
    "deit3_small_patch16_224_in21k": "S",
    "dinov2_vits14": "S",
    "dinov2_vits14_reg": "S",
    "mocov3_vitb16": "B",
    "mae_vitb16": "B",
    "dinov2_vitb14": "B",
    "dinov2_vitb14_reg": "B",
    "openclip_vitb16": "B",
    "metaclip_vitb16": "B",
}


def classify_supervision_group(model_id: str, objective: str | None = None) -> str:
    """Map model_id / objective to binary OSF §3 contrast group: 'SSL', 'label+text', or 'reconstruction'."""
    if model_id in EXCLUDED_RECON_MODELS or (objective in RECON_OBJECTIVES):
        return "reconstruction"
    if model_id in SSL_MODELS or (objective in SSL_OBJECTIVES):
        return "SSL"
    if model_id in LABEL_TEXT_MODELS or (objective in LABEL_TEXT_OBJECTIVES):
        return "label+text"
    raise ValueError(f"Cannot classify supervision group for model_id={model_id!r}, objective={objective!r}")


# --------------------------------------------------------------------------- Bootstrap & Holm-Bonferroni (OSF §3)
def bootstrap_two_sided_p(boot_samples: np.ndarray, null_val: float = 0.0) -> float:
    """OSF §3 exact two-sided bootstrap p-value: (2 * min(#{theta* <= null}, #{theta* >= null}) + 1) / (B + 1)."""
    b = np.asarray(boot_samples, dtype=np.float64)
    b = b[np.isfinite(b)]
    B = int(b.size)
    if B == 0:
        return float("nan")
    n_le = int(np.sum(b <= null_val))
    n_ge = int(np.sum(b >= null_val))
    return float(min(1.0, (2 * min(n_le, n_ge) + 1) / (B + 1)))


def bootstrap_ci_at_alpha(boot_samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile confidence interval at level 1 - alpha."""
    b = np.asarray(boot_samples, dtype=np.float64)
    b = b[np.isfinite(b)]
    if b.size == 0:
        return float("nan"), float("nan")
    lo = float(np.quantile(b, alpha / 2.0))
    hi = float(np.quantile(b, 1.0 - alpha / 2.0))
    return lo, hi


def paired_bootstrap_test(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Paired image-level bootstrap test of mean(a - b) vs 0 with exact OSF §3 p_boot."""
    a_arr = np.asarray(a, dtype=np.float64).reshape(-1)
    b_arr = np.asarray(b, dtype=np.float64).reshape(-1)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"Paired arrays must have identical shape, got {a_arr.shape} vs {b_arr.shape}")
    ok = np.isfinite(a_arr) & np.isfinite(b_arr)
    diff = a_arr[ok] - b_arr[ok]
    n = int(diff.size)
    if n == 0:
        return {
            "n": 0,
            "mean_diff": float("nan"),
            "p_boot": float("nan"),
            "ci95": [float("nan"), float("nan")],
            "boot_samples": np.array([], dtype=np.float64),
        }
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    boot_means = diff[idx].mean(axis=1)
    mean_diff = float(diff.mean())
    p_boot = bootstrap_two_sided_p(boot_means, null_val=0.0)
    lo, hi = bootstrap_ci_at_alpha(boot_means, alpha=alpha)
    return {
        "n": n,
        "mean_diff": mean_diff,
        "p_boot": p_boot,
        "ci95": [lo, hi],
        "boot_samples": boot_means,
    }


def holm_bonferroni(tests: list[dict[str, Any]], *, alpha: float = 0.05) -> list[dict[str, Any]]:
    """Step-down Holm-Bonferroni procedure for primary hypothesis family (OSF §3).

    Each item in `tests` must contain `id` and `p_raw` (float or None/NaN).
    Optional `boot_samples` (np.ndarray) triggers computation of the exact Holm step-level CI
    at confidence level 1 - alpha_step.
    """
    out: list[dict[str, Any]] = []
    valid_indices = [
        i for i, t in enumerate(tests)
        if t.get("p_raw") is not None and np.isfinite(float(t["p_raw"]))
    ]
    m = len(valid_indices)
    sorted_valid = sorted(valid_indices, key=lambda i: float(tests[i]["p_raw"]))

    step_meta: dict[int, dict[str, Any]] = {}
    running_adj = 0.0
    stopped = False
    for rank0, idx in enumerate(sorted_valid):
        p_raw = float(tests[idx]["p_raw"])
        denom = m - rank0
        alpha_step = float(alpha) / float(denom)
        p_adj = min(1.0, max(running_adj, p_raw * denom))
        running_adj = p_adj
        if not stopped and p_raw < alpha_step:
            reject = True
        else:
            reject = False
            stopped = True
        ci_holm = None
        if tests[idx].get("boot_samples") is not None:
            bs = np.asarray(tests[idx]["boot_samples"], dtype=np.float64)
            if bs.size > 0:
                lo, hi = bootstrap_ci_at_alpha(bs, alpha=alpha_step)
                ci_holm = [lo, hi]
        step_meta[idx] = {
            "holm_rank": rank0 + 1,
            "holm_m": m,
            "alpha_step": alpha_step,
            "p_adj": p_adj,
            "reject_holm": reject,
            "ci_holm_level": round(1.0 - alpha_step, 5),
            "ci_holm": ci_holm,
        }

    for i, t in enumerate(tests):
        rec = {k: v for k, v in t.items() if k != "boot_samples"}
        if i in step_meta:
            rec.update(step_meta[i])
        else:
            rec.update({
                "holm_rank": None,
                "holm_m": m,
                "alpha_step": None,
                "p_adj": None,
                "reject_holm": False,
                "ci_holm_level": None,
                "ci_holm": None,
            })
        out.append(rec)
    return out


# --------------------------------------------------------------------------- H1: Matched S/16 QII_S4
def test_h1_qii_s4_pair(
    qii_ssl: np.ndarray,
    qii_sup: np.ndarray,
    *,
    ssl_name: str = "dino_vits16",
    sup_name: str = "deit3_small_patch16_224",
    min_effect: float = 0.02,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """OSF §3 H1: paired image-level bootstrap of QII_S4(SSL) - QII_S4(label) at unlabelled l*."""
    res = paired_bootstrap_test(qii_ssl, qii_sup, n_boot=n_boot, seed=seed)
    meets_min = bool(np.isfinite(res["mean_diff"]) and abs(res["mean_diff"]) >= min_effect and res["mean_diff"] > 0)
    return {
        "id": "H1",
        "pair": f"{ssl_name} - {sup_name}",
        "n_images": res["n"],
        "mean_ssl": float(np.nanmean(qii_ssl)),
        "mean_sup": float(np.nanmean(qii_sup)),
        "mean_diff": res["mean_diff"],
        "ci95": res["ci95"],
        "p_raw": res["p_boot"],
        "min_effect_threshold": min_effect,
        "meets_min_effect_and_direction": meets_min,
        "boot_samples": res["boot_samples"],
    }


# --------------------------------------------------------------------------- H2: Crossed REML LMM
def fit_crossed_reml_lmm(
    y: np.ndarray,
    X: np.ndarray,
    model_idx: np.ndarray,
    image_idx: np.ndarray,
    *,
    col_names: list[str] | None = None,
) -> dict[str, Any]:
    """Exact REML solver for crossed random-intercept LMM:
        y = X beta + u_{model} + v_{image} + e
    Uses block-diagonal Sherman-Morrison over images + tiny M x M Woodbury over models.
    Exact for both balanced and unbalanced image x model panels, O(n * p + M^3) per eval.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    X = np.asarray(X, dtype=np.float64)
    m_raw = np.asarray(model_idx, dtype=np.int64).reshape(-1)
    i_raw = np.asarray(image_idx, dtype=np.int64).reshape(-1)

    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X, m_raw, i_raw = y[ok], X[ok], m_raw[ok], i_raw[ok]
    n, p = X.shape
    if n <= p + 2:
        raise ValueError(f"Insufficient observations (n={n}, p={p}) for REML LMM.")

    _, m_codes = np.unique(m_raw, return_inverse=True)
    _, i_codes = np.unique(i_raw, return_inverse=True)
    M = int(m_codes.max() + 1)
    N = int(i_codes.max() + 1)

    # Pre-build sparse incidence helpers
    img_counts = np.bincount(i_codes, minlength=N).astype(np.float64)
    # Indicator matrix Z_m (n x M) - M is tiny (<= 12)
    Z_m = np.zeros((n, M), dtype=np.float64)
    Z_m[np.arange(n), m_codes] = 1.0

    def _apply_A_inv(W: np.ndarray, tau_i: float) -> np.ndarray:
        """Apply A^{-1} = (I + tau_i Z_i Z_i^T)^{-1} to matrix W (n x k) in O(n k)."""
        if tau_i <= 0.0:
            return W
        scale = tau_i / (1.0 + img_counts * tau_i)  # (N,)
        # Sum rows of W per image
        sums = np.zeros((N, W.shape[1]), dtype=np.float64)
        np.add.at(sums, i_codes, W)
        return W - (scale[i_codes, None] * sums[i_codes])

    def _eval_reml(tau_m: float, tau_i: float) -> tuple[float, np.ndarray, np.ndarray, float]:
        """Evaluate -2 * log_reml, beta_hat, cov_beta, sigma2_e at variance ratios (tau_m, tau_i)."""
        tau_m = max(float(tau_m), 0.0)
        tau_i = max(float(tau_i), 0.0)
        logdet_A = float(np.sum(np.log1p(img_counts * tau_i)))

        # Compute A^{-1} [X, y] and A^{-1} Z_m
        Xy = np.column_stack([X, y])
        Ainv_Xy = _apply_A_inv(Xy, tau_i)

        if tau_m > 1e-12:
            Ainv_Zm = _apply_A_inv(Z_m, tau_i)
            Zt_Ainv_Z = Z_m.T @ Ainv_Zm
            C = Zt_Ainv_Z + (1.0 / tau_m) * np.eye(M, dtype=np.float64)
            C = 0.5 * (C + C.T)
            sign_c, logdet_c = np.linalg.slogdet(C)
            if sign_c <= 0:
                return 1e30, np.zeros(p), np.full((p, p), np.nan), float("nan")
            logdet_V = logdet_A + M * math.log(tau_m) + float(logdet_c)
            Zt_Ainv_Xy = Z_m.T @ Ainv_Xy
            Cinv_Zt_Ainv_Xy = np.linalg.solve(C, Zt_Ainv_Xy)
            Vinv_Xy = Ainv_Xy - Ainv_Zm @ Cinv_Zt_Ainv_Xy
        else:
            logdet_V = logdet_A
            Vinv_Xy = Ainv_Xy

        Vinv_X = Vinv_Xy[:, :p]
        Vinv_y = Vinv_Xy[:, p]
        Xt_Vinv_X = X.T @ Vinv_X
        Xt_Vinv_X = 0.5 * (Xt_Vinv_X + Xt_Vinv_X.T)
        sign_x, logdet_x = np.linalg.slogdet(Xt_Vinv_X)
        if sign_x <= 0:
            return 1e30, np.zeros(p), np.full((p, p), np.nan), float("nan")

        beta = np.linalg.solve(Xt_Vinv_X, X.T @ Vinv_y)
        r = y - X @ beta
        Vinv_r = Vinv_y - Vinv_X @ beta
        rss = float(r @ Vinv_r)
        if rss <= 1e-18:
            rss = 1e-18
        dof = float(n - p)
        sigma2_e = rss / dof
        neg2_reml = logdet_V + float(logdet_x) + dof * math.log(sigma2_e) + dof * (1.0 + math.log(2.0 * math.pi))
        cov_beta = sigma2_e * np.linalg.inv(Xt_Vinv_X)
        return float(neg2_reml), beta, cov_beta, float(sigma2_e)

    # Optimize in bounded log1p / softplus or non-negative box (tau_m, tau_i >= 0) with multi-start
    best_val = float("inf")
    best_params = (0.0, 0.0)
    starts = [(0.0, 0.0), (0.05, 0.2), (0.5, 1.0), (2.0, 0.1), (0.01, 2.0)]
    for s_m, s_i in starts:
        v0, _, _, _ = _eval_reml(s_m, s_i)
        if v0 < best_val:
            best_val, best_params = v0, (s_m, s_i)

    res = optimize.minimize(
        lambda th: _eval_reml(math.exp(th[0]) - 1e-8 if th[0] > -18 else 0.0,
                              math.exp(th[1]) - 1e-8 if th[1] > -18 else 0.0)[0],
        x0=np.array([math.log(best_params[0] + 1e-3), math.log(best_params[1] + 1e-3)]),
        method="L-BFGS-B",
        bounds=[(-20.0, 12.0), (-20.0, 12.0)],
    )
    cand_m = math.exp(res.x[0]) if res.x[0] > -18 else 0.0
    cand_i = math.exp(res.x[1]) if res.x[1] > -18 else 0.0
    # Also check exact boundary combinations (tau_m=0, tau_i=opt), (tau_m=opt, tau_i=0), (0, 0)
    candidates = [(cand_m, cand_i), (0.0, cand_i), (cand_m, 0.0), (0.0, 0.0)]
    for cm, ci in candidates:
        v, _, _, _ = _eval_reml(cm, ci)
        if v < best_val:
            best_val, best_params = v, (cm, ci)

    tau_m, tau_i = best_params
    neg2_reml, beta, cov_beta, sigma2_e = _eval_reml(tau_m, tau_i)
    se = np.sqrt(np.maximum(np.diag(cov_beta), 1e-18))
    names = col_names if col_names is not None else [f"x{k}" for k in range(p)]
    return {
        "n_obs": n,
        "n_models": M,
        "n_images": N,
        "names": names,
        "beta": {nm: float(b) for nm, b in zip(names, beta)},
        "se": {nm: float(s) for nm, s in zip(names, se)},
        "cov_beta": cov_beta.tolist(),
        "sigma2_e": float(sigma2_e),
        "sigma2_model": float(tau_m * sigma2_e),
        "sigma2_image": float(tau_i * sigma2_e),
        "neg2_reml": float(neg2_reml),
    }


def test_h2_i1_supervision_contrast(
    model_records: list[dict[str, Any]],
    *,
    min_effect: float = 0.02,
) -> dict[str, Any]:
    """OSF §3 H2: I1-S4 L_int=1 Δbox-AUC contrast between SSL and label+text (MAE excluded).

    Each record in `model_records` has:
      - `model_id`: str
      - `objective`: str (optional)
      - `delta_box_auc`: float (model mean)
      - `box_auc_clean`: float (model mean)
      - Optional per-image arrays for exact crossed REML LMM:
          `delta_box_auc_per_image`: np.ndarray (N,)
          `box_auc_clean_per_image`: np.ndarray (N,)
    """
    kept = []
    excluded = []
    for rec in model_records:
        mid = rec["model_id"]
        grp = classify_supervision_group(mid, rec.get("objective"))
        if grp == "reconstruction":
            excluded.append(mid)
            continue
        kept.append({**rec, "group": grp, "is_label_text": 1.0 if grp == "label+text" else 0.0})

    n_ssl = sum(1 for r in kept if r["group"] == "SSL")
    n_lt = sum(1 for r in kept if r["group"] == "label+text")
    identifiable = (n_ssl >= 2 and n_lt >= 2)
    M = len(kept)

    if n_ssl == 0 or n_lt == 0:
        return {
            "id": "H2",
            "n_ssl": n_ssl,
            "n_label_text": n_lt,
            "excluded_models": excluded,
            "identifiable": False,
            "fit_mode": "insufficient_groups",
            "beta_label_text_vs_ssl": float("nan"),
            "se": float("nan"),
            "t_stat": float("nan"),
            "df": 0,
            "p_raw": float("nan"),
            "ci95": [float("nan"), float("nan")],
            "min_effect_threshold": min_effect,
            "meets_min_effect": False,
        }

    has_per_image = all(
        r.get("delta_box_auc_per_image") is not None and r.get("box_auc_clean_per_image") is not None
        for r in kept
    )

    if has_per_image:
        ys, xs, mids, iids = [], [], [], []
        for m_i, r in enumerate(kept):
            dy = np.asarray(r["delta_box_auc_per_image"], dtype=np.float64).reshape(-1)
            cl = np.asarray(r["box_auc_clean_per_image"], dtype=np.float64).reshape(-1)
            N = dy.size
            ys.append(dy)
            xs.append(np.column_stack([np.ones(N), np.full(N, r["is_label_text"]), cl]))
            mids.append(np.full(N, m_i, dtype=np.int64))
            iids.append(np.arange(N, dtype=np.int64))
        y_all = np.concatenate(ys)
        X_all = np.vstack(xs)
        m_all = np.concatenate(mids)
        i_all = np.concatenate(iids)
        # Center clean covariate so intercept is interpretable
        X_all[:, 2] = X_all[:, 2] - np.nanmean(X_all[:, 2])

        lmm = fit_crossed_reml_lmm(
            y_all, X_all, m_all, i_all,
            col_names=["intercept", "I_label_text", "box_auc_clean_c"],
        )
        beta = lmm["beta"]["I_label_text"]
        se = lmm["se"]["I_label_text"]
        # Conservative cluster-level df = M - 2 (model-level binary factor)
        df = max(M - 2, 1)
        t_stat = beta / max(se, 1e-12)
        p_raw = float(2.0 * stats.t.sf(abs(t_stat), df=df))
        t_crit = float(stats.t.ppf(0.975, df=df))
        ci95 = [float(beta - t_crit * se), float(beta + t_crit * se)]
        extra = {
            "fit_mode": "per_image_crossed_reml_lmm",
            "sigma2_model": lmm["sigma2_model"],
            "sigma2_image": lmm["sigma2_image"],
            "sigma2_e": lmm["sigma2_e"],
            "beta_clean_auc": lmm["beta"]["box_auc_clean_c"],
            "n_obs": lmm["n_obs"],
            "n_images": lmm["n_images"],
        }
    else:
        # Model-level OLS fallback when only summary JSONs are available
        y_m = np.array([float(r["delta_box_auc"]) for r in kept], dtype=np.float64)
        g_m = np.array([r["is_label_text"] for r in kept], dtype=np.float64)
        c_m = np.array([float(r["box_auc_clean"]) for r in kept], dtype=np.float64)
        c_m = c_m - np.mean(c_m)
        if M >= 4:
            X_m = np.column_stack([np.ones(M), g_m, c_m])
        else:
            X_m = np.column_stack([np.ones(M), g_m])
        p_cols = X_m.shape[1]
        df = max(M - p_cols, 1)
        XtX_inv = np.linalg.pinv(X_m.T @ X_m)
        b_hat = XtX_inv @ (X_m.T @ y_m)
        resid = y_m - X_m @ b_hat
        s2 = float((resid @ resid) / df)
        se = float(math.sqrt(max(s2 * XtX_inv[1, 1], 1e-18)))
        beta = float(b_hat[1])
        t_stat = beta / max(se, 1e-12)
        p_raw = float(2.0 * stats.t.sf(abs(t_stat), df=df))
        t_crit = float(stats.t.ppf(0.975, df=df))
        ci95 = [float(beta - t_crit * se), float(beta + t_crit * se)]
        extra = {
            "fit_mode": "model_summary_ols",
            "beta_clean_auc": float(b_hat[2]) if p_cols == 3 else None,
        }

    mean_ssl = float(np.mean([float(r["delta_box_auc"]) for r in kept if r["group"] == "SSL"]))
    mean_lt = float(np.mean([float(r["delta_box_auc"]) for r in kept if r["group"] == "label+text"]))
    return {
        "id": "H2",
        "n_models_included": M,
        "n_ssl": n_ssl,
        "n_label_text": n_lt,
        "excluded_models": excluded,
        "identifiable": identifiable,
        "identifiability_note": (
            "OK (>=2 checkpoints in both SSL and label+text)"
            if identifiable
            else "WARNING: <2 checkpoints in at least one group; family fixed effect and (1|model_id) are not separable."
        ),
        "mean_ssl": mean_ssl,
        "mean_label_text": mean_lt,
        "beta_label_text_vs_ssl": float(beta),
        "se": float(se),
        "t_stat": float(t_stat),
        "df": int(df),
        "p_raw": float(p_raw),
        "ci95": ci95,
        "min_effect_threshold": min_effect,
        "meets_min_effect": bool(np.isfinite(beta) and abs(beta) >= min_effect),
        **extra,
    }


# --------------------------------------------------------------------------- H2b: M3 + I2 Log-Ratio Conjunction
def _balanced_bg_main_effect(
    bss: np.ndarray,
    y_bird: np.ndarray,
    place_bg: np.ndarray,
) -> float:
    """Balanced 2x2 Waterbirds Background Main Effect:
       0.5 * [(mean(Water|Landbird) - mean(Land|Landbird)) + (mean(Water|Waterbird) - mean(Land|Waterbird))].
    """
    m00 = bss[(y_bird == 0) & (place_bg == 0)]
    m01 = bss[(y_bird == 0) & (place_bg == 1)]
    m10 = bss[(y_bird == 1) & (place_bg == 0)]
    m11 = bss[(y_bird == 1) & (place_bg == 1)]
    if min(m00.size, m01.size, m10.size, m11.size) == 0:
        return float("nan")
    return float(0.5 * ((np.nanmean(m01) - np.nanmean(m00)) + (np.nanmean(m11) - np.nanmean(m10))))


def test_h2b_m3_leg(
    bss_by_model: dict[str, np.ndarray],
    y_bird: np.ndarray,
    place_bg: np.ndarray,
    area_tercile: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
    objectives: dict[str, str] | None = None,
) -> dict[str, Any]:
    """OSF §3 H2b M3 leg: balanced 2x2 Bg Main Effect (Water - Land) for label+text vs max size-matched SSL.

    All arrays in `bss_by_model` are aligned (N,) over the exact same Waterbirds test images at a fixed layer.
    """
    y_bird = np.asarray(y_bird, dtype=np.int64).reshape(-1)
    place_bg = np.asarray(place_bg, dtype=np.int64).reshape(-1)
    area_tercile = np.asarray(area_tercile, dtype=np.int64).reshape(-1)
    N = int(y_bird.size)
    objectives = objectives or {}

    ssl_by_size: dict[str, list[str]] = {"S": [], "B": []}
    lt_models: list[str] = []
    excluded: list[str] = []

    for mid in bss_by_model:
        grp = classify_supervision_group(mid, objectives.get(mid))
        if grp == "reconstruction":
            excluded.append(mid)
            continue
        sz = MODEL_SIZE_BUCKET.get(mid, "B")
        if grp == "SSL":
            ssl_by_size.setdefault(sz, []).append(mid)
        else:
            lt_models.append(mid)

    # Point estimates per model overall and per area tercile
    bg_eff_overall: dict[str, float] = {}
    bg_eff_terciles: dict[str, dict[int, float]] = {}
    for mid, arr in bss_by_model.items():
        a = np.asarray(arr, dtype=np.float64).reshape(-1)
        bg_eff_overall[mid] = _balanced_bg_main_effect(a, y_bird, place_bg)
        bg_eff_terciles[mid] = {}
        for t in (0, 1, 2):
            mask = (area_tercile == t)
            bg_eff_terciles[mid][t] = _balanced_bg_main_effect(a[mask], y_bird[mask], place_bg[mask])

    # Pair each label+text model with max size-matched SSL (fallback to all SSL if size bucket empty)
    all_ssl = ssl_by_size["S"] + ssl_by_size["B"]
    if not lt_models or not all_ssl:
        return {
            "id": "H2b_M3",
            "p_m3": float("nan"),
            "mean_diff_vs_max_ssl": float("nan"),
            "terciles_held": 0,
            "excluded_models": excluded,
            "boot_samples": np.array([], dtype=np.float64),
        }

    pairs = []
    for lt in lt_models:
        sz = MODEL_SIZE_BUCKET.get(lt, "B")
        pool = ssl_by_size.get(sz) or all_ssl
        max_ssl = max(pool, key=lambda m: bg_eff_overall[m])
        pairs.append((lt, max_ssl, sz, pool))

    # Pre-group image indices by the 4 (y_bird, place_bg) cells for fast exact stratified-or-iid bootstrap
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, N, size=(int(n_boot), N))

    def _boot_bg_eff_matrix(arr: np.ndarray) -> np.ndarray:
        """Vectorized balanced 2x2 Bg Main Effect across all B bootstrap resamples."""
        a = np.asarray(arr, dtype=np.float64).reshape(-1)
        # Cell indicator masks on original images
        c00 = (y_bird == 0) & (place_bg == 0)
        c01 = (y_bird == 0) & (place_bg == 1)
        c10 = (y_bird == 1) & (place_bg == 0)
        c11 = (y_bird == 1) & (place_bg == 1)
        out = np.empty(int(n_boot), dtype=np.float64)
        # Process in chunks to keep RAM tiny (<50 MB even at N=5794, B=2000)
        chunk = 200
        for s in range(0, int(n_boot), chunk):
            e = min(s + chunk, int(n_boot))
            bi = boot_idx[s:e]  # (C, N)
            vals = a[bi]
            def _cmean(cmask: np.ndarray) -> np.ndarray:
                m = cmask[bi]
                cnt = np.maximum(m.sum(axis=1), 1)
                return (vals * m).sum(axis=1) / cnt
            out[s:e] = 0.5 * ((_cmean(c01) - _cmean(c00)) + (_cmean(c11) - _cmean(c10)))
        return out

    boot_cache: dict[str, np.ndarray] = {}
    needed_models = sorted({m for lt, mx, _, pool in pairs for m in ([lt, mx] + pool)})
    for mid in needed_models:
        boot_cache[mid] = _boot_bg_eff_matrix(bss_by_model[mid])

    # Average contrast across all label+text models vs their size-matched max SSL baseline
    pair_details = []
    boot_diffs_all = []
    for lt, max_ssl, sz, pool in pairs:
        ssl_min = min(bg_eff_overall[m] for m in pool)
        ssl_max = bg_eff_overall[max_ssl]
        diff_pt = bg_eff_overall[lt] - ssl_max
        # Conservative bootstrap: compare against max_m in size-matched SSL pool per resample
        pool_boot_max = np.max(np.vstack([boot_cache[m] for m in pool]), axis=0)
        b_diff = boot_cache[lt] - pool_boot_max
        boot_diffs_all.append(b_diff)
        t_wins = sum(
            1 for t in (0, 1, 2)
            if bg_eff_terciles[lt][t] > max(bg_eff_terciles[m][t] for m in pool)
        )
        lo, hi = bootstrap_ci_at_alpha(b_diff, alpha=0.05)
        pair_details.append({
            "label_text_model": lt,
            "size_bucket": sz,
            "max_size_matched_ssl": max_ssl,
            "bg_effect_lt": round(bg_eff_overall[lt], 4),
            "bg_effect_ssl_range": [round(ssl_min, 4), round(ssl_max, 4)],
            "ratio_vs_max_ssl": round(bg_eff_overall[lt] / max(ssl_max, 1e-6), 2),
            "diff_vs_max_ssl": round(diff_pt, 4),
            "diff_ci95": [round(lo, 4), round(hi, 4)],
            "p_boot_pair": bootstrap_two_sided_p(b_diff, 0.0),
            "terciles_held": int(t_wins),
        })

    agg_boot = np.mean(np.vstack(boot_diffs_all), axis=0)
    mean_diff = float(np.mean([d["diff_vs_max_ssl"] for d in pair_details]))
    p_m3 = bootstrap_two_sided_p(agg_boot, 0.0)
    lo, hi = bootstrap_ci_at_alpha(agg_boot, alpha=0.05)
    # Tercile rule: count how many terciles the mean label+text > max SSL holds (or min across pairs)
    terciles_held_agg = sum(
        1 for t in (0, 1, 2)
        if np.mean([bg_eff_terciles[lt][t] - max(bg_eff_terciles[m][t] for m in pool) for lt, _, _, pool in pairs]) > 0
    )
    return {
        "id": "H2b_M3",
        "n_images": N,
        "excluded_models": excluded,
        "mean_diff_vs_max_ssl": mean_diff,
        "ci95": [lo, hi],
        "p_m3": p_m3,
        "predicted_direction_ok": bool(mean_diff > 0),
        "terciles_held": int(terciles_held_agg),
        "tercile_rule_ok": bool(terciles_held_agg >= 2),
        "pair_details": pair_details,
        "boot_samples": agg_boot,
    }


def compute_i2_log_ratio(fg_cos_drop: float, bg_eq_cos_drop: float, *, eps: float = 1e-6) -> float:
    """OSF §3 scale-invariant I2 log ratio R_m = log((FG + eps) / (BG_eq + eps))."""
    return float(math.log((max(float(fg_cos_drop), 0.0) + eps) / (max(float(bg_eq_cos_drop), 0.0) + eps)))


def test_h2b_i2_log_ratio_leg(
    model_rows: list[dict[str, Any]],
    *,
    eps: float = 1e-6,
    n_perm_mc: int = 10000,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """OSF §3 H2b I2 leg: between-family contrast of scale-invariant log ratio R_m (MAE excluded).

    Predicted direction: mean_SSL(R_m) - mean_{label+text}(R_m) > 0.
    Uses exact enumeration over all C(M, n_SSL) permutations when C(M, n_SSL) <= 20,000.
    """
    ssl_vals, lt_vals = [], []
    per_model = []
    excluded = []

    for row in model_rows:
        mid = row["model_id"]
        grp = classify_supervision_group(mid, row.get("objective"))
        if grp == "reconstruction":
            excluded.append(mid)
            continue
        rm = compute_i2_log_ratio(row["fg_cos_drop"], row["bg_eq_fg_cos_drop"], eps=eps)
        per_model.append({
            "model_id": mid,
            "group": grp,
            "fg_cos_drop": float(row["fg_cos_drop"]),
            "bg_eq_fg_cos_drop": float(row["bg_eq_fg_cos_drop"]),
            "R_m": round(rm, 4),
        })
        if grp == "SSL":
            ssl_vals.append(rm)
        else:
            lt_vals.append(rm)

    n_ssl = len(ssl_vals)
    n_lt = len(lt_vals)
    if n_ssl == 0 or n_lt == 0:
        return {
            "id": "H2b_I2",
            "n_ssl": n_ssl,
            "n_label_text": n_lt,
            "excluded_models": excluded,
            "delta_R_ssl_minus_lt": float("nan"),
            "p_i2": float("nan"),
            "predicted_direction_ok": False,
            "per_model": per_model,
        }

    arr_ssl = np.array(ssl_vals, dtype=np.float64)
    arr_lt = np.array(lt_vals, dtype=np.float64)
    obs_diff = float(arr_ssl.mean() - arr_lt.mean())
    pooled = np.concatenate([arr_ssl, arr_lt])
    M = pooled.size
    total_sum = float(pooled.sum())

    # Exact permutation test when combinatorially small (e.g. C(11, 7) = 330, C(7, 5) = 21!)
    n_comb = math.comb(M, n_ssl)
    if n_comb <= 20000:
        perm_diffs = np.empty(n_comb, dtype=np.float64)
        for k, idx_tuple in enumerate(itertools.combinations(range(M), n_ssl)):
            s_sum = float(pooled[list(idx_tuple)].sum())
            perm_diffs[k] = (s_sum / n_ssl) - ((total_sum - s_sum) / n_lt)
        p_perm = float(np.mean(np.abs(perm_diffs) >= abs(obs_diff) - 1e-12))
        perm_mode = f"exact_{n_comb}"
    else:
        rng = np.random.default_rng(seed)
        perm_diffs = np.empty(n_perm_mc, dtype=np.float64)
        for k in range(n_perm_mc):
            perm = rng.permutation(pooled)
            perm_diffs[k] = perm[:n_ssl].mean() - perm[n_ssl:].mean()
        p_perm = float((int(np.sum(np.abs(perm_diffs) >= abs(obs_diff) - 1e-12)) + 1) / (n_perm_mc + 1))
        perm_mode = f"mc_{n_perm_mc}"

    # Group-stratified bootstrap CI across checkpoints
    rng_b = np.random.default_rng(seed + 17)
    b_ssl = arr_ssl[rng_b.integers(0, n_ssl, size=(n_boot, n_ssl))].mean(axis=1)
    b_lt = arr_lt[rng_b.integers(0, n_lt, size=(n_boot, n_lt))].mean(axis=1)
    b_diff = b_ssl - b_lt
    ci95 = list(bootstrap_ci_at_alpha(b_diff, alpha=0.05))

    return {
        "id": "H2b_I2",
        "n_ssl": n_ssl,
        "n_label_text": n_lt,
        "excluded_models": excluded,
        "mean_R_ssl": float(arr_ssl.mean()),
        "mean_R_label_text": float(arr_lt.mean()),
        "delta_R_ssl_minus_lt": obs_diff,
        "ci95": ci95,
        "p_i2": p_perm,
        "perm_mode": perm_mode,
        "predicted_direction_ok": bool(obs_diff > 0),
        "per_model": per_model,
        "boot_samples": b_diff,
    }


def test_h2b_conjunction(m3_res: dict[str, Any], i2_res: dict[str, Any]) -> dict[str, Any]:
    """OSF §3 H2b intersection-union test (IUT) conjunction: p_H2b = max(p_M3, p_I2)."""
    p_m3 = float(m3_res.get("p_m3", float("nan")))
    p_i2 = float(i2_res.get("p_i2", float("nan")))
    if np.isfinite(p_m3) and np.isfinite(p_i2):
        # If either leg goes in the OPPOSITE direction, set one-sided-guarded IUT p to 1.0
        dir_m3 = bool(m3_res.get("predicted_direction_ok", False))
        dir_i2 = bool(i2_res.get("predicted_direction_ok", False))
        terc_ok = bool(m3_res.get("tercile_rule_ok", True))
        if dir_m3 and dir_i2 and terc_ok:
            p_comb = max(p_m3, p_i2)
        else:
            p_comb = max(p_m3 if (dir_m3 and terc_ok) else 1.0, p_i2 if dir_i2 else 1.0)
    else:
        p_comb = float("nan")
    return {
        "id": "H2b",
        "p_m3": p_m3,
        "p_i2": p_i2,
        "p_raw": p_comb,
        "m3_direction_ok": bool(m3_res.get("predicted_direction_ok", False)),
        "m3_tercile_rule_ok": bool(m3_res.get("tercile_rule_ok", False)),
        "i2_direction_ok": bool(i2_res.get("predicted_direction_ok", False)),
        "m3_leg": {k: v for k, v in m3_res.items() if k != "boot_samples"},
        "i2_leg": {k: v for k, v in i2_res.items() if k != "boot_samples"},
    }


# --------------------------------------------------------------------------- H3: M4 Artifact Sensitivity
def test_h3_m4_artifact_sensitivity(
    delta_same_samples: np.ndarray,
    delta_cross_samples: np.ndarray,
    *,
    tau_c0_c1: float,
    tau_c0_c1_rand: float | None = None,
    min_rand_gap: float = 0.10,
    n_boot: int = 2000,
    seed: int = 0,
    label: str = "l11:R2:cka",
) -> dict[str, Any]:
    """OSF §3 H3: Delta_tau = tau(C0, C1) - tau_noise < 0 on both same-half and cross-half + C1_rand gap >= 0.10."""
    ds = np.asarray(delta_same_samples, dtype=np.float64).reshape(-1)
    dc = np.asarray(delta_cross_samples, dtype=np.float64).reshape(-1)
    ds = ds[np.isfinite(ds)]
    dc = dc[np.isfinite(dc)]
    if ds.size == 0 or dc.size == 0:
        return {"id": "H3", "cell": label, "p_raw": float("nan"), "rand_gap_ok": False}

    rng = np.random.default_rng(seed)
    bs_same = ds[rng.integers(0, ds.size, size=(n_boot, ds.size))].mean(axis=1)
    bs_cross = dc[rng.integers(0, dc.size, size=(n_boot, dc.size))].mean(axis=1)
    p_same = bootstrap_two_sided_p(bs_same, 0.0)
    p_cross = bootstrap_two_sided_p(bs_cross, 0.0)
    mean_same = float(ds.mean())
    mean_cross = float(dc.mean())

    # Conjunction: BOTH same-half and cross-half must be < 0
    dir_ok = (mean_same < 0.0 and mean_cross < 0.0)
    p_comb = max(p_same, p_cross) if dir_ok else 1.0
    rand_gap = (float(tau_c0_c1_rand) - float(tau_c0_c1)) if tau_c0_c1_rand is not None else None
    rand_gap_ok = bool(rand_gap is not None and rand_gap >= min_rand_gap)

    # Pass the more conservative bootstrap distribution (larger p) for Holm step CI
    dom_boot = bs_same if p_same >= p_cross else bs_cross
    return {
        "id": "H3",
        "cell": label,
        "tau_c0_c1": float(tau_c0_c1),
        "tau_c0_c1_rand": float(tau_c0_c1_rand) if tau_c0_c1_rand is not None else None,
        "rand_gap": round(rand_gap, 4) if rand_gap is not None else None,
        "rand_gap_ok": rand_gap_ok,
        "mean_delta_same": mean_same,
        "ci95_same": list(bootstrap_ci_at_alpha(bs_same, 0.05)),
        "p_same": p_same,
        "mean_delta_cross": mean_cross,
        "ci95_cross": list(bootstrap_ci_at_alpha(bs_cross, 0.05)),
        "p_cross": p_cross,
        "both_below_noise": dir_ok,
        "p_raw": p_comb,
        "boot_samples": dom_boot,
    }


def test_h3_m4_multicell(
    cell_results: list[dict[str, Any]],
    *,
    c1_rand_source: str | None = None,
    c1_rand_verified_on_same_zoo: bool = False,
) -> dict[str, Any]:
    """OSF §3 H3 multi-cell aggregation over all matched (l8..l11) x (R2, R3) CKA cells.

    Applies conservative within-H3 Bonferroni adjustment across all K evaluated cells:
        p_raw(H3) = min(1.0, K * min_c p_cell(c))
    so H3 pays the full K-cell multiplicity penalty BEFORE entering the 5-test Holm family.
    Also tracks whether all K cells trigger vs any cell (transparently reporting both), and
    whether the registered C1_rand gap >= 0.10 condition was verified on the exact same model zoo.
    """
    valid = [c for c in cell_results if c.get("p_raw") is not None and np.isfinite(float(c["p_raw"]))]
    K = len(valid)
    if K == 0:
        return {
            "id": "H3",
            "n_cells_total": 0,
            "n_cells_triggered": 0,
            "triggered_cells": [],
            "p_raw": float("nan"),
            "per_cell": [],
        }

    triggered = [
        c["cell"] for c in valid
        if bool(c.get("both_below_noise")) and float(c["p_raw"]) < (0.05 / K)
    ]
    best = min(valid, key=lambda c: (float(c["p_raw"]), float(c.get("mean_delta_same", 0.0))))
    p_bonf = float(min(1.0, K * float(best["p_raw"])))
    rand_ok_cells = [c["cell"] for c in valid if c.get("rand_gap_ok")]
    r3_gaps = [c["rand_gap"] for c in valid if ":R3:" in c["cell"] and c.get("rand_gap") is not None]
    r3_min_gap = round(float(min(r3_gaps)), 4) if r3_gaps else None
    all_registered_conditions_met = bool(
        len(triggered) > 0 and c1_rand_verified_on_same_zoo and (r3_min_gap is not None and r3_min_gap >= 0.10)
    )

    per_cell_clean = [{k: v for k, v in c.items() if k != "boot_samples"} for c in valid]
    out: dict[str, Any] = {
        "id": "H3",
        "aggregation": f"any_cell_bonferroni_across_{K}_cells",
        "n_cells_total": K,
        "n_cells_triggered": len(triggered),
        "all_cells_triggered": len(triggered) == K,
        "triggered_cells": triggered,
        "strongest_cell": best["cell"],
        "strongest_delta_same": round(float(best["mean_delta_same"]), 4),
        "strongest_delta_cross": round(float(best["mean_delta_cross"]), 4),
        "strongest_tau_c0_c1": round(float(best["tau_c0_c1"]), 4),
        "min_cell_p_raw": float(best["p_raw"]),
        "p_raw": p_bonf,
        "rand_gap_ok_cells": rand_ok_cells,
        "r3_min_rand_gap": r3_min_gap,
        "c1_rand_source": c1_rand_source,
        "c1_rand_verified_on_same_zoo": bool(c1_rand_verified_on_same_zoo),
        "all_registered_conditions_met": all_registered_conditions_met,
        "per_cell": per_cell_clean,
    }
    if best.get("boot_samples") is not None:
        out["boot_samples"] = best["boot_samples"]
    return out


# --------------------------------------------------------------------------- H5: RALR Exact Two-Sided Sign Test
def test_h5_ralr_sign_test(
    benchmarks: dict[str, dict[str, dict[str, float]]],
    *,
    exclude_models: frozenset[str] = EXCLUDED_RECON_MODELS,
) -> dict[str, Any]:
    """OSF §3 H5: exact two-sided binomial sign test on primary box-AUC across backbones.

    `benchmarks` maps benchmark_name -> {model_id -> {"delta_box_auc": float, "delta_otsu": float (optional)}}.
    Primary denominator excludes MAE (`n=11` on zoo12, `n=7` on pilot8). Sensitivity includes all models.
    Secondary Otsu is reported separately and NEVER rescues a failed primary box-AUC test.
    """
    bench_results = {}
    primary_p_list = []

    for bname, model_map in benchmarks.items():
        all_ids = sorted(model_map.keys())
        prim_ids = [m for m in all_ids if m not in exclude_models]

        def _sign_block(ids: list[str], key: str) -> dict[str, Any]:
            vals = [float(model_map[m][key]) for m in ids if key in model_map[m] and np.isfinite(model_map[m][key])]
            n = len(vals)
            wins = sum(1 for v in vals if v > 0.0)
            p_two = float(stats.binomtest(wins, n, 0.5, alternative="two-sided").pvalue) if n > 0 else float("nan")
            return {
                "n": n,
                "wins": wins,
                "losses_or_ties": n - wins,
                "mean_delta": float(np.mean(vals)) if n > 0 else float("nan"),
                "p_two_sided": p_two,
            }

        prim_auc = _sign_block(prim_ids, "delta_box_auc")
        sens_auc = _sign_block(all_ids, "delta_box_auc")
        sec_otsu_prim = _sign_block(prim_ids, "delta_otsu")
        sec_otsu_all = _sign_block(all_ids, "delta_otsu")

        bench_results[bname] = {
            "primary_box_auc_excl_mae": prim_auc,
            "sensitivity_box_auc_all": sens_auc,
            "secondary_otsu_excl_mae": sec_otsu_prim,
            "secondary_otsu_all": sec_otsu_all,
        }
        if np.isfinite(prim_auc["p_two_sided"]):
            primary_p_list.append((bname, prim_auc["p_two_sided"]))

    # Gate G3 rule: >=2 of 3 benchmarks must clear Holm on primary box-AUC.
    # For the single Holm-family entry representing H5 (Gate G3 >=2 benchmarks), the exact
    # order-statistic p-value that >=2 benchmarks clear threshold alpha_step is the 2nd smallest p-value!
    sorted_p = sorted(p for _, p in primary_p_list)
    if len(sorted_p) >= 2:
        p_gate_g3 = sorted_p[1]  # clearing Holm at p_(2) < alpha_step <=> >= 2 benchmarks clear alpha_step!
    elif len(sorted_p) == 1:
        p_gate_g3 = sorted_p[0]
    else:
        p_gate_g3 = float("nan")

    return {
        "id": "H5",
        "n_benchmarks": len(benchmarks),
        "benchmarks": bench_results,
        "p_raw": p_gate_g3,
        "p_rule_note": "2nd-smallest primary box-AUC sign-test p across benchmarks (p_(2) < alpha_step iff >=2 benchmarks clear Holm; secondary Otsu cannot rescue box-AUC).",
    }


# --------------------------------------------------------------------------- §10.3 Descriptive Confound Regression
def _parse_patch_size(arch_str: str) -> float:
    m = re.search(r"/(\d+)", str(arch_str))
    return float(m.group(1)) if m else float("nan")


def fit_confound_regression_descriptive(
    outcomes_by_model: dict[str, float],
    registry_rows: dict[str, dict[str, Any]],
    *,
    layer_frac_by_model: dict[str, float] | None = None,
    resolution_by_model: dict[str, float] | None = None,
    ridge_alpha: float = 1e-2,
) -> dict[str, Any]:
    """OSF §6 (§10.3) strictly DESCRIPTIVE multi-covariate confound regression:
        Y ~ objective_family + log10(params) + layer_frac + patch_size
            + log10(pretrain_images_m) + log10(epochs) + has_registers + resolution

    With 8-12 models, formal inference is underpowered; this reports standardized coefficients,
    R^2 (objective-only vs covariates-only vs full), and flags any missing/unverified registry fields.
    """
    layer_frac_by_model = layer_frac_by_model or {}
    resolution_by_model = resolution_by_model or {}

    complete_rows = []
    skipped_missing = []

    for mid, y_val in sorted(outcomes_by_model.items()):
        if not np.isfinite(y_val):
            continue
        reg = registry_rows.get(mid)
        if reg is None:
            skipped_missing.append({"model_id": mid, "reason": "not in registry"})
            continue
        params_m = reg.get("params_m")
        pretrain_m = reg.get("pretrain_images_m")
        epochs = reg.get("epochs")
        regs = reg.get("registers")
        patch_sz = _parse_patch_size(reg.get("arch", ""))
        res = resolution_by_model.get(mid, reg.get("native_res"))
        lf = layer_frac_by_model.get(mid, 1.0)

        missing_fields = []
        if not isinstance(params_m, (int, float)) or params_m <= 0:
            missing_fields.append("params_m")
        if not isinstance(pretrain_m, (int, float)) or pretrain_m <= 0:
            missing_fields.append("pretrain_images_m")
        if not isinstance(epochs, (int, float)) or epochs <= 0:
            missing_fields.append("epochs")
        if not isinstance(regs, int) or regs < 0:
            missing_fields.append("registers")
        if not np.isfinite(patch_sz):
            missing_fields.append("patch_size")
        if not isinstance(res, (int, float)) or res <= 0:
            missing_fields.append("resolution")

        if missing_fields:
            skipped_missing.append({"model_id": mid, "missing": missing_fields})
            continue

        grp = classify_supervision_group(mid, reg.get("objective"))
        complete_rows.append({
            "model_id": mid,
            "y": float(y_val),
            "is_label_text": 1.0 if grp == "label+text" else 0.0,
            "is_recon": 1.0 if grp == "reconstruction" else 0.0,
            "log10_params": math.log10(float(params_m)),
            "layer_frac": float(lf),
            "patch_size": float(patch_sz),
            "log10_pretrain_images_m": math.log10(float(pretrain_m)),
            "log10_epochs": math.log10(float(epochs)),
            "has_registers": 1.0 if regs > 0 else 0.0,
            "resolution": float(res),
            "verify_flag": bool(reg.get("verify", False)),
        })

    n = len(complete_rows)
    if n < 3:
        return {
            "status": "DESCRIPTIVE_ONLY_INSUFFICIENT_COMPLETE_ROWS",
            "n_complete": n,
            "skipped_missing_covariates": skipped_missing,
        }

    y = np.array([r["y"] for r in complete_rows], dtype=np.float64)
    y_var = float(np.sum((y - y.mean()) ** 2))

    cov_cols = [
        "is_label_text",
        "is_recon",
        "log10_params",
        "layer_frac",
        "patch_size",
        "log10_pretrain_images_m",
        "log10_epochs",
        "has_registers",
        "resolution",
    ]
    # Drop zero-variance columns (e.g., is_recon if MAE excluded, or layer_frac if all l=11)
    active_cols = [c for c in cov_cols if float(np.std([r[c] for r in complete_rows])) > 1e-12]
    X_raw = np.column_stack([[r[c] for r in complete_rows] for c in active_cols])
    X_std = (X_raw - X_raw.mean(axis=0, keepdims=True)) / X_raw.std(axis=0, keepdims=True)
    y_c = y - y.mean()

    def _fit_r2(Xm: np.ndarray, alpha_r: float) -> tuple[np.ndarray, float]:
        if Xm.shape[1] == 0 or y_var < 1e-18:
            return np.zeros(Xm.shape[1]), 0.0
        if alpha_r <= 0.0:
            # SVD minimum-norm OLS: numerically exact even under exact predictor collinearity
            # (e.g. patch_size=14 <=> native_res=518 across DINOv2 models).
            b, _, _, _ = np.linalg.lstsq(Xm, y_c, rcond=None)
        else:
            A = Xm.T @ Xm + alpha_r * np.eye(Xm.shape[1])
            b = np.linalg.solve(A, Xm.T @ y_c)
        pred = Xm @ b
        r2 = 1.0 - float(np.sum((y_c - pred) ** 2)) / y_var
        return b, r2

    mat_rank = int(np.linalg.matrix_rank(X_std))
    rank_def = (n <= len(active_cols) + 1) or (mat_rank < len(active_cols))
    # For pure descriptive variance decomposition (R²), always use exact SVD OLS when n > mat_rank + 1
    # so R²(full) >= max(R²(obj), R²(covs)) holds mathematically; apply ridge only when n <= mat_rank + 1.
    eff_alpha = float(ridge_alpha) if n <= mat_rank + 1 else 0.0
    beta_full, r2_full = _fit_r2(X_std, eff_alpha)

    obj_idx = [i for i, c in enumerate(active_cols) if c in ("is_label_text", "is_recon")]
    non_obj_idx = [i for i, c in enumerate(active_cols) if c not in ("is_label_text", "is_recon")]
    _, r2_obj_only = _fit_r2(X_std[:, obj_idx], 0.0) if obj_idx else (np.array([]), 0.0)
    _, r2_cov_only = _fit_r2(X_std[:, non_obj_idx], eff_alpha) if non_obj_idx else (np.array([]), 0.0)

    def _adj_r2(r2_val: float, k_rank: int) -> float | None:
        dof = n - k_rank - 1
        if dof <= 0:
            return None
        return round(float(1.0 - (1.0 - r2_val) * (n - 1) / dof), 4)

    rank_obj = int(np.linalg.matrix_rank(X_std[:, obj_idx])) if obj_idx else 0
    rank_cov = int(np.linalg.matrix_rank(X_std[:, non_obj_idx])) if non_obj_idx else 0
    r2_adj_obj = _adj_r2(r2_obj_only, rank_obj)
    r2_adj_cov = _adj_r2(r2_cov_only, rank_cov)
    r2_adj_full = _adj_r2(r2_full, mat_rank)

    return {
        "status": "DESCRIPTIVE_ONLY_UNDERPOWERED",
        "n_complete": n,
        "p_active_predictors": len(active_cols),
        "design_matrix_rank": mat_rank,
        "collinear_predictors_detected": mat_rank < len(active_cols),
        "rank_deficient_ridge_applied": eff_alpha > 0.0,
        "ridge_alpha": eff_alpha,
        "r2_objective_only": round(r2_obj_only, 4),
        "r2_covariates_only": round(r2_cov_only, 4),
        "r2_full": round(r2_full, 4),
        "delta_r2_objective_given_covariates": round(r2_full - r2_cov_only, 4),
        "r2_adj_objective_only": r2_adj_obj,
        "r2_adj_covariates_only": r2_adj_cov,
        "r2_adj_full": r2_adj_full,
        "delta_r2_adj_objective_given_covariates": (
            round(r2_adj_full - r2_adj_cov, 4) if (r2_adj_full is not None and r2_adj_cov is not None) else None
        ),
        "standardized_coefs": {c: round(float(b), 4) for c, b in zip(active_cols, beta_full)},
        "included_models": [r["model_id"] for r in complete_rows],
        "models_with_verify_flag": [r["model_id"] for r in complete_rows if r["verify_flag"]],
        "skipped_missing_covariates": skipped_missing,
    }
