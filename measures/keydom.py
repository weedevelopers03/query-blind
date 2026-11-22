"""Measure M2 — Key Dominance (Methodology Guide §3, Fix 3, §12).

Two-way additive logit model (§3.1):
    z(q, i) = q . k_i * scale = mu + alpha_q + beta_i + gamma_qi + eps
    SS_query = N * sum_q (alpha_bar_q)^2
    SS_key   = |Q| * sum_i (beta_bar_i)^2
    SS_resid = SS_total - SS_query - SS_key
    MS_resid = SS_resid / ((|Q| - 1) * (N - 1))

Mandatory corrections (§3.2):
  1. Use unbiased omega^2_key (NOT biased eta^2):
       omega^2_key = max(0, (SS_key - (N - 1) * MS_resid) / (SS_total + MS_resid))
  2. Fix |Q| = 64 across all models:
       1 global query + 31 random patch queries (same image) + 32 global queries (other images).
       (For MAP-head models where global query is fixed across images: 1 probe + 63 patch queries.)

Key-effect saliency map (§3.3-3.4):
  * beta_bar_i reshaped to (G, G) per head -> stored as keymap.npy (I, L', H, G, G) in fp16.
  * Head-weighted sum using w_h (global update contribution norm), then min-max normalized per image.
  * Evaluate threshold-free box-AUC of bilinearly upsampled key-effect map alongside omega^2_key.

Fix 3 check (§3.6):
  * Across-model Spearman correlation between head-weighted QII(S4) and head-weighted omega^2_key.
    rho < 0.8 -> keep both; rho >= 0.9 -> keep key dominance as primary and demote QII to appendix.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from ._common import atomic_save_npy, box_auc, upsample_map_bilinear

NUM_QUERIES: int = 64
N_SAME_IMG_PATCH: int = 31
N_OTHER_IMG_GLOBAL: int = 32


def build_query_set_64(
    q_g: np.ndarray | torch.Tensor,
    q_patches: np.ndarray | torch.Tensor,
    q_global_pool: np.ndarray | torch.Tensor | None,
    image_idx: int = 0,
    seed: int = 0,
    is_map_head: bool = False,
) -> np.ndarray:
    """Build the fixed |Q|=64 query set for one image (Guide §3.2 & §12).

    Standard [CLS] models:
      1 true global query + 31 random patch queries (same image) + 32 global queries (other images).
    MAP-head models (fixed image-independent global query):
      1 fixed probe query + 63 random patch queries (same image), avoiding 32 duplicate rows.

    Returns:
        Q: (H, 64, d_h) float32 array.
    """
    qg = q_g.detach().cpu().numpy() if isinstance(q_g, torch.Tensor) else np.asarray(q_g)
    qp = q_patches.detach().cpu().numpy() if isinstance(q_patches, torch.Tensor) else np.asarray(q_patches)
    if qg.ndim != 2 or qp.ndim != 3:
        raise ValueError(f"Expected q_g (H, d_h) and q_patches (H, N, d_h); got {qg.shape}, {qp.shape}")
    H, N, d_h = qp.shape
    if qg.shape != (H, d_h):
        raise ValueError(f"q_g shape {qg.shape} incompatible with q_patches {qp.shape}")

    rng = np.random.default_rng([int(seed), int(image_idx), 64])

    if is_map_head:
        n_pat = NUM_QUERIES - 1
        replace = N < n_pat
        if replace:
            warnings.warn(f"MAP-head image has N={N} < {n_pat} patches; sampling with replacement.")
        p_idx = rng.choice(N, size=n_pat, replace=replace)
        Q = np.concatenate([qg[:, None, :], qp[:, p_idx, :]], axis=1)
        assert Q.shape == (H, NUM_QUERIES, d_h)
        return Q.astype(np.float32)

    if q_global_pool is None:
        raise ValueError("q_global_pool is required to draw the 32 cross-image global queries (§3.2).")
    pool = q_global_pool.detach().cpu().numpy() if isinstance(q_global_pool, torch.Tensor) else np.asarray(q_global_pool)
    if pool.ndim != 3 or pool.shape[1:] != (H, d_h):
        raise ValueError(f"Expected q_global_pool (I, {H}, {d_h}), got {pool.shape}")
    I = pool.shape[0]
    if I < 2:
        raise ValueError("q_global_pool must contain at least 2 images to sample other-image queries.")

    if N < N_SAME_IMG_PATCH:
        raise ValueError(f"Image has only N={N} patches; need at least {N_SAME_IMG_PATCH} for |Q|=64.")
    p_idx = rng.choice(N, size=N_SAME_IMG_PATCH, replace=False)

    others = np.delete(np.arange(I), image_idx)
    replace_other = len(others) < N_OTHER_IMG_GLOBAL
    if replace_other:
        warnings.warn(f"Only {len(others)} other images in pool (< {N_OTHER_IMG_GLOBAL}); sampling with replacement.")
    o_idx = rng.choice(others, size=N_OTHER_IMG_GLOBAL, replace=replace_other)

    # pool[o_idx] is (32, H, d_h) -> transpose to (H, 32, d_h)
    q_other = np.transpose(pool[o_idx], (1, 0, 2))
    Q = np.concatenate([qg[:, None, :], qp[:, p_idx, :], q_other], axis=1)
    assert Q.shape == (H, NUM_QUERIES, d_h)
    return Q.astype(np.float32)


def two_way_anova_key_dominance(logits: np.ndarray | torch.Tensor) -> dict:
    """Two-way additive ANOVA on attention logits (..., Q, N) in float64 (Guide §§3.1-3.2).

    Returns:
        dict with omega2_key (...,), eta2_key (...,), beta_key (..., N),
        ss_total, ss_query, ss_key, ss_resid, ms_resid.
    """
    z = logits.detach().cpu().numpy() if isinstance(logits, torch.Tensor) else np.asarray(logits)
    z = z.astype(np.float64)
    if z.ndim < 2:
        raise ValueError(f"Expected logits with at least 2 dims (..., Q, N); got {z.shape}")
    n_q, n_k = z.shape[-2], z.shape[-1]
    if n_q < 2 or n_k < 2:
        raise ValueError(f"Need Q >= 2 and N >= 2 for two-way ANOVA; got Q={n_q}, N={n_k}")

    mu = z.mean(axis=(-2, -1), keepdims=True)
    alpha_bar = z.mean(axis=-1, keepdims=True) - mu   # (..., Q, 1)
    beta_bar = z.mean(axis=-2, keepdims=True) - mu    # (..., 1, N)

    ss_total = np.sum((z - mu) ** 2, axis=(-2, -1))
    ss_query = float(n_k) * np.sum(alpha_bar ** 2, axis=(-2, -1))
    ss_key = float(n_q) * np.sum(beta_bar ** 2, axis=(-2, -1))
    ss_resid = np.maximum(0.0, ss_total - ss_query - ss_key)

    df_resid = float((n_q - 1) * (n_k - 1))
    ms_resid = ss_resid / df_resid

    denom_omega = ss_total + ms_resid
    with np.errstate(divide="ignore", invalid="ignore"):
        omega2 = np.where(denom_omega > 0.0, (ss_key - float(n_k - 1) * ms_resid) / denom_omega, 0.0)
        eta2 = np.where(ss_total > 0.0, ss_key / ss_total, 0.0)
    omega2 = np.clip(omega2, 0.0, 1.0)
    eta2 = np.clip(eta2, 0.0, 1.0)

    return {
        "omega2_key": omega2,
        "eta2_key": eta2,
        "beta_key": beta_bar.squeeze(-2),  # (..., N)
        "ss_total": ss_total,
        "ss_query": ss_query,
        "ss_key": ss_key,
        "ss_resid": ss_resid,
        "ms_resid": ms_resid,
    }


def aggregate_keymap_heads(keymap_per_head: np.ndarray, head_weights: np.ndarray | None = None) -> np.ndarray:
    """Head-weighted sum of beta_bar_i maps, min-max normalized per image to [0, 1] (Guide §3.4)."""
    km = np.asarray(keymap_per_head, dtype=np.float64)  # (H, G_h, G_w)
    H = km.shape[0]
    if head_weights is None:
        w = np.full(H, 1.0 / H, dtype=np.float64)
    else:
        w = np.asarray(head_weights, dtype=np.float64)
        w_sum = w.sum()
        w = w / w_sum if w_sum > 0 else np.full(H, 1.0 / H, dtype=np.float64)

    agg = np.tensordot(w, km, axes=(0, 0))  # (G_h, G_w)
    lo, hi = float(agg.min()), float(agg.max())
    if hi > lo:
        agg = (agg - lo) / (hi - lo)
    else:
        agg = np.zeros_like(agg)
    return agg.astype(np.float32)


def compute_key_dominance_image(
    Q: np.ndarray | torch.Tensor,
    k_patch: np.ndarray | torch.Tensor,
    scale: float,
    grid: tuple[int, int],
    *,
    head_weights: np.ndarray | torch.Tensor | None = None,
    gt_mask: np.ndarray | None = None,
    allow_custom_q: bool = False,
) -> dict:
    """Compute Measure M2 (omega^2_key + key-effect map + box-AUC) for one image at one block.

    Args:
        Q:              (H, 64, d_h) fixed query set from build_query_set_64().
        k_patch:        (H, N, d_h) patch keys ONLY.
        scale:          attention scale factor from adapter.
        grid:           (G_h, G_w) patch grid with G_h * G_w == N.
        head_weights:   optional (H,) contribution weights w_h from compute_head_weights().
        gt_mask:        optional (H_img, W_img) binary mask for keymap box-AUC (§3.3).
        allow_custom_q: set True only in toy unit tests with |Q| != 64.
    """
    Q_t = Q.detach().cpu().float() if isinstance(Q, torch.Tensor) else torch.as_tensor(Q, dtype=torch.float32)
    k_t = k_patch.detach().cpu().float() if isinstance(k_patch, torch.Tensor) else torch.as_tensor(k_patch, dtype=torch.float32)

    if Q_t.ndim != 3 or k_t.ndim != 3:
        raise ValueError(f"Expected Q (H, |Q|, d_h) and k_patch (H, N, d_h); got {tuple(Q_t.shape)}, {tuple(k_t.shape)}")
    H, n_q, d_h = Q_t.shape
    H_k, N, d_k = k_t.shape
    if H != H_k or d_h != d_k:
        raise ValueError(f"Shape mismatch: Q {tuple(Q_t.shape)} vs k_patch {tuple(k_t.shape)}")
    if not allow_custom_q and n_q != NUM_QUERIES:
        raise ValueError(f"Guide §3.2/§12 requires fixed |Q|={NUM_QUERIES}; got {n_q} (pass allow_custom_q=True only for tests)")
    if grid[0] * grid[1] != N:
        raise ValueError(f"grid {grid} product != N ({N})")

    if head_weights is not None:
        w_h = (head_weights.detach().cpu().numpy() if isinstance(head_weights, torch.Tensor) else np.asarray(head_weights)).astype(np.float64)
        w_sum = w_h.sum()
        w_h = w_h / w_sum if w_sum > 0 else np.full(H, 1.0 / H, dtype=np.float64)
    else:
        w_h = np.full(H, 1.0 / H, dtype=np.float64)

    # Attention logits in fp32: (H, |Q|, N)
    logits = (Q_t @ k_t.transpose(-2, -1)) * float(scale)
    anova = two_way_anova_key_dominance(logits)

    omega2_per_head = anova["omega2_key"]  # (H,)
    eta2_per_head = anova["eta2_key"]      # (H,)
    omega2_weighted = float(np.sum(w_h * omega2_per_head))
    omega2_unweighted = float(np.mean(omega2_per_head))

    keymap = anova["beta_key"].reshape(H, grid[0], grid[1]).astype(np.float32)  # (H, G_h, G_w)
    keymap_agg = aggregate_keymap_heads(keymap, w_h)                             # (G_h, G_w), [0, 1]

    if gt_mask is not None:
        target_hw = gt_mask.shape[:2]
        km_up = upsample_map_bilinear(keymap, target_hw)
        auc_per_head = np.array([box_auc(km_up[h], gt_mask) for h in range(H)], dtype=np.float64)
        auc_agg = box_auc(upsample_map_bilinear(keymap_agg, target_hw), gt_mask)
    else:
        auc_per_head = np.full(H, np.nan, dtype=np.float64)
        auc_agg = float("nan")

    return {
        "omega2_per_head": omega2_per_head,
        "eta2_per_head": eta2_per_head,
        "omega2_weighted": omega2_weighted,
        "omega2_unweighted": omega2_unweighted,
        "keymap": keymap,
        "keymap_agg": keymap_agg,
        "box_auc_per_head": auc_per_head,
        "box_auc_agg": auc_agg,
        "head_weights": w_h,
    }


# --------------------------------------------------------------------------- Sanity checks (§3.5)
def spatial_shuffle_check(
    Q: np.ndarray | torch.Tensor,
    k_patch: np.ndarray | torch.Tensor,
    scale: float,
    grid: tuple[int, int],
    *,
    head_weights: np.ndarray | None = None,
    gt_mask: np.ndarray | None = None,
    seed: int = 0,
    tol: float = 1e-6,
    allow_custom_q: bool = False,
) -> dict:
    """Guide §3.5: shuffling keys spatially must leave omega^2_key invariant while box-AUC collapses."""
    k_np = k_patch.detach().cpu().numpy() if isinstance(k_patch, torch.Tensor) else np.asarray(k_patch)
    perm = np.random.default_rng(seed).permutation(k_np.shape[1])
    k_shuf = k_np[:, perm, :]

    orig = compute_key_dominance_image(Q, k_np, scale, grid, head_weights=head_weights, gt_mask=gt_mask, allow_custom_q=allow_custom_q)
    shuf = compute_key_dominance_image(Q, k_shuf, scale, grid, head_weights=head_weights, gt_mask=gt_mask, allow_custom_q=allow_custom_q)
    max_diff = float(np.max(np.abs(orig["omega2_per_head"] - shuf["omega2_per_head"])))
    return {
        "ok": max_diff < tol,
        "max_omega2_diff": max_diff,
        "box_auc_orig": orig["box_auc_agg"],
        "box_auc_shuffled": shuf["box_auc_agg"],
    }


def permutation_null_check(logits: np.ndarray | torch.Tensor, seed: int = 0) -> dict:
    """Guide §3.5: permute (q, i) cell assignments -> omega^2_key should be ~0 (unbiased)."""
    z = logits.detach().cpu().numpy() if isinstance(logits, torch.Tensor) else np.asarray(logits)
    z = z.astype(np.float64).copy()
    rng = np.random.default_rng(seed)
    flat = z.reshape(-1, z.shape[-2] * z.shape[-1])
    for row in flat:
        rng.shuffle(row)
    z_perm = flat.reshape(z.shape)
    res = two_way_anova_key_dominance(z_perm)
    return {
        "omega2_mean": float(np.mean(res["omega2_key"])),
        "omega2_max": float(np.max(res["omega2_key"])),
        "eta2_mean": float(np.mean(res["eta2_key"])),
    }


# --------------------------------------------------------------------------- Fix 3 gate check (§3.6)
def qii_kd_spearman(
    qii_s4_by_model: dict[str, float] | np.ndarray,
    omega2_by_model: dict[str, float] | np.ndarray,
) -> dict:
    """Guide Fix 3 / §3.6: across-model Spearman correlation between QII(S4) and omega^2_key.

    Decision rule (pre-registered at G1):
      rho < 0.8        -> 'keep_both' (show scatter of dissociating models)
      0.8 <= rho < 0.9 -> 'borderline' (inspect scatter before deciding)
      rho >= 0.9       -> 'demote_qii' (keep key dominance as primary, move QII to appendix)
    """
    if isinstance(qii_s4_by_model, dict) and isinstance(omega2_by_model, dict):
        keys = sorted(set(qii_s4_by_model) & set(omega2_by_model))
        x = np.array([qii_s4_by_model[k] for k in keys], dtype=np.float64)
        y = np.array([omega2_by_model[k] for k in keys], dtype=np.float64)
    else:
        keys = None
        x = np.asarray(qii_s4_by_model, dtype=np.float64)
        y = np.asarray(omega2_by_model, dtype=np.float64)

    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3:
        return {"rho": float("nan"), "p_value": float("nan"), "n_models": int(len(x)), "decision": "insufficient_models"}

    res = spearmanr(x, y)
    rho = float(res.statistic)
    pval = float(res.pvalue)
    if np.isnan(rho):
        decision = "undefined"
    elif rho < 0.8:
        decision = "keep_both"
    elif rho < 0.9:
        decision = "borderline"
    else:
        decision = "demote_qii"
    return {
        "rho": rho,
        "p_value": pval,
        "n_models": int(len(x)),
        "models": [k for k, ok in zip(keys, valid) if ok] if keys is not None else None,
        "decision": decision,
    }


def save_keymaps(keymaps: np.ndarray, path: str | Path) -> Path:
    """Store keymap.npy (I, L', H, G, G) in fp16 atomically (Guide §3.3 & §12)."""
    arr = np.asarray(keymaps, dtype=np.float16)
    if arr.ndim != 5:
        raise ValueError(f"Expected keymaps shape (I, L', H, G, G), got {arr.shape}")
    return atomic_save_npy(arr, path)
