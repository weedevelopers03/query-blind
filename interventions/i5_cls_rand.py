"""Intervention I5: Controlled [CLS] State Randomization & Across-Model Prediction Test (Guide §6)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr, spearmanr

DEFAULT_I5_LR_SWEEP = (8, 9, 10, 11)
DEFAULT_I5_MODES = ("cov_matched", "cross_img", "iso_norm")
DEFAULT_I5_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_CLS_PCA_K = 16


@dataclass(frozen=True)
class ClsStatePCA:
    mean: np.ndarray        # (D,) float64
    components: np.ndarray  # (D, k) float64 orthonormal columns
    stds: np.ndarray        # (k,) float64 singular-value stds along each PC


def fit_cls_state_pca(z_cls: np.ndarray, k: int = DEFAULT_CLS_PCA_K) -> ClsStatePCA:
    """Fit empirical mean + top-k PCA covariance basis to (I, D) [CLS] hidden states (Guide §6)."""
    X = np.asarray(z_cls, dtype=np.float64)
    I, D = X.shape
    mu = X.mean(axis=0)
    Xc = X - mu[None, :]
    k_eff = max(1, min(int(k), I - 1 if I > 1 else 1, D))
    if I <= 1:
        comps = np.eye(D, k_eff, dtype=np.float64)
        stds = np.zeros(k_eff, dtype=np.float64)
        return ClsStatePCA(mean=mu, components=comps, stds=stds)

    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:k_eff].T.copy()                       # (D, k_eff)
    stds = (S[:k_eff] / np.sqrt(max(I - 1, 1))).copy()  # (k_eff,)
    return ClsStatePCA(mean=mu, components=comps, stds=stds)


def sample_cls_state_override(
    mode: str,
    z_cls_clean: np.ndarray,  # (I, D)
    pca: ClsStatePCA,
    seed: int = 0,
) -> np.ndarray:
    """Generate (I, D) float32 controlled [CLS] state overrides matched to each image's exact ||z_cls||_2.

    Modes (Guide §6):
      - cov_matched: Empirical mean + top-k covariance Gaussian (S4-analog in hidden state space), norm-matched.
      - cross_img:   Real [CLS] hidden state from a random different image (derangement j != i, zero shift).
      - iso_norm:    Isotropic Gaussian N(0, I_D) rescaled to each image's exact ||z_cls||_2 (off-manifold control).
    """
    X = np.asarray(z_cls_clean, dtype=np.float64)
    I, D = X.shape
    norms = np.linalg.norm(X, axis=-1, keepdims=True)  # (I, 1)
    rng = np.random.default_rng(int(seed))

    if mode == "cov_matched":
        k_dim = int(pca.stds.shape[0])
        eps = rng.standard_normal((I, k_dim)) * pca.stds[None, :]  # (I, k)
        raw = pca.mean[None, :] + eps @ pca.components.T           # (I, D)
        r_norm = np.maximum(np.linalg.norm(raw, axis=-1, keepdims=True), 1e-12)
        return (raw / r_norm * norms).astype(np.float32)

    if mode == "cross_img":
        if I < 2:
            return X.astype(np.float32).copy()
        shifts = rng.integers(1, I, size=I)
        j_idx = (np.arange(I) + shifts) % I
        return X[j_idx].astype(np.float32).copy()

    if mode == "iso_norm":
        g = rng.standard_normal((I, D))
        g_norm = np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-12)
        return (g / g_norm * norms).astype(np.float32)

    raise ValueError(f"Unknown I5 randomization mode {mode!r}; expected one of {DEFAULT_I5_MODES}")


def bootstrap_across_model_correlation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float | int | list[float | None]]:
    """Across-model Spearman rho and Pearson r with bootstrap 95% CIs (Guide §6 I5 prediction test)."""
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[ok], yv[ok]
    n = int(xv.size)
    if n < 3 or np.std(xv) < 1e-12 or np.std(yv) < 1e-12:
        return {
            "n_models": n,
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
            "spearman_ci95": [None, None],
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "pearson_ci95": [None, None],
        }

    sp = spearmanr(xv, yv)
    pr = pearsonr(xv, yv)
    rng = np.random.default_rng(int(seed))
    sp_b, pr_b = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, yb = xv[idx], yv[idx]
        if np.std(xb) < 1e-12 or np.std(yb) < 1e-12:
            continue
        sp_b.append(float(spearmanr(xb, yb).statistic))
        pr_b.append(float(pearsonr(xb, yb).statistic))

    sp_ci = [round(float(v), 4) for v in np.percentile(sp_b, [2.5, 97.5])] if len(sp_b) >= 20 else [None, None]
    pr_ci = [round(float(v), 4) for v in np.percentile(pr_b, [2.5, 97.5])] if len(pr_b) >= 20 else [None, None]
    return {
        "n_models": n,
        "spearman_rho": round(float(sp.statistic), 4),
        "spearman_p": round(float(sp.pvalue), 4),
        "spearman_ci95": sp_ci,
        "pearson_r": round(float(pr.statistic), 4),
        "pearson_p": round(float(pr.pvalue), 4),
        "pearson_ci95": pr_ci,
    }
