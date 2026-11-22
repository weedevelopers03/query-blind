"""Measure M3 — Routing Decomposition of the Global Update (Methodology Guide §4, §12).

Norm-based attribution (Kobayashi et al., EMNLP 2020; Guide §4.1):
    Delta_g = gamma * (sum_h sum_i a_i^(h) W_O^(h) v_i^(h) + b_O)
    u_i     = gamma * sum_h a_i^(h) W_O^(h) v_i^(h)        # sum over heads BEFORE taking norm!
    c_i     = ||u_i||_2

Five token groups (§4.2):
    FG   : patch tokens with >= 50% overlap with GT mask/box (excluding OUT)
    BG   : patch tokens with <  50% overlap with GT mask/box (excluding OUT)
    OUT  : patch tokens whose L2 norm at block l exceeds median + 5 * MAD (per image, per block);
           OUT overrides FG/BG (primary ordering). Alternative spatial-retained ordering also reported.
    REG  : register tokens, where present
    SELF : global [CLS] token (plus any DeiT distillation token if present)

Headline statistics (§4.3):
    BSS_raw  = (sum_{i in BG} c_i + sum_{i in OUT} c_i) / sum_{i in patches U SELF} c_i
    BSS_norm = mean_{i in BG U OUT}(c_i) / mean_{i in FG}(c_i)   [PRIMARY for H2; 1.0 = no preference]
    RR       = ||x_g^(l)||_2 / ||Delta_g^(l)||_2                 [Lappe & Giese decoupling ratio]
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import torch

M3_SCHEMA_COLUMNS: tuple[str, ...] = (
    "model_id",
    "objective",
    "set",
    "layer",
    "layer_frac",
    "image_id",
    "bss_raw",
    "bss_norm",
    "bss_norm_alt",
    "share_fg",
    "share_bg",
    "share_out",
    "share_reg",
    "share_self",
    "share_fg_alt",
    "share_bg_alt",
    "residual_ratio",
    "n_out_tokens",
    "gt_area_frac",
    "area_tercile",
    "waterbirds_congruent",
    "y_bird",
    "place_bg",
    "resolution",
    "grid",
)


@torch.no_grad()
def token_contributions(
    attn_row_full: torch.Tensor,
    v_full: torch.Tensor,
    W_O: torch.Tensor,
    gamma: torch.Tensor | None = None,
    return_vectors: bool = False,
):
    """Guide §4.1 Kobayashi norm-based token attribution c_i = ||gamma * sum_h a_i^(h) W_O^(h) v_i^(h)||_2.

    CRITICAL (§4.1 & §12):
      1. Sums over heads BEFORE taking the L2 norm (taking per-head norms and summing overstates c_i).
      2. Applies LayerScale gamma when present before taking norms.

    Args:
        attn_row_full: (..., H, T) global-token attention weights over ALL T tokens.
        v_full:        (..., H, T, d_h) value vectors over ALL T tokens.
        W_O:           (D, H * d_h) output projection weight matrix.
        gamma:         optional (D,) LayerScale vector.
    Returns:
        c: (..., T) float32 contribution norms (and optionally u: (..., T, D) vectors).
    """
    a = attn_row_full.float()
    v = v_full.float()
    # Weighted values per head & token: (..., H, T, d_h) -> (..., T, H * d_h)
    av = (a.unsqueeze(-1) * v).transpose(-3, -2).flatten(-2)
    # Matmul with W_O^T automatically computes sum_h a_i^(h) W_O^(h) v_i^(h) BEFORE norm
    u = av @ W_O.float().to(av.device).T  # (..., T, D)
    if gamma is not None:
        u = u * gamma.float().to(av.device)
    c = u.norm(dim=-1)  # (..., T)
    if return_vectors:
        return c, u
    return c


def detect_outlier_tokens(
    token_norms: np.ndarray | torch.Tensor,
    patch_idx: list[int] | np.ndarray,
    k_mad: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Guide §4.2: patch tokens whose L2 norm at block l exceeds median + 5 * MAD (per image, per block).

    Args:
        token_norms: (..., T) token L2 norms at block l.
        patch_idx:   indices of the N patch tokens within T.
    Returns:
        out_mask_patches: (..., N) bool mask over patch tokens.
        n_out:            (...,) int count of outlier patch tokens.
    """
    r = token_norms.detach().cpu().numpy() if isinstance(token_norms, torch.Tensor) else np.asarray(token_norms)
    r_p = r[..., patch_idx].astype(np.float64)  # (..., N)
    med = np.median(r_p, axis=-1, keepdims=True)
    mad = np.median(np.abs(r_p - med), axis=-1, keepdims=True)
    thresh = med + float(k_mad) * mad
    # If mad == 0 (all patch norms identical), no token is an outlier
    out_mask = np.where(mad > 0.0, r_p > thresh, False)
    return out_mask, out_mask.sum(axis=-1).astype(np.int64)


def routing_decomposition_batch(
    c_all: np.ndarray | torch.Tensor,
    token_norms_all: np.ndarray | torch.Tensor,
    layout: Mapping,
    fg_patch_masks: np.ndarray,
    x_g_in_norm: np.ndarray | torch.Tensor,
    delta_g_norm: np.ndarray | torch.Tensor,
    *,
    k_mad: float = 5.0,
) -> dict[str, np.ndarray]:
    """Compute Guide §4.2-4.3 routing shares, BSS_raw, BSS_norm (primary & alt), and RR for a batch of I images.

    Args:
        c_all:           (I, T) per-token contribution norms from token_contributions().
        token_norms_all: (I, T) per-token residual-stream L2 norms at block l.
        layout:          adapter.token_layout() dict (cls_idx, register_idx, patch_idx, dist_idx).
        fg_patch_masks:  (I, G, G) or (I, N) binary foreground patch mask (>= 50% overlap, §8.3).
        x_g_in_norm:     (I,) ||x_g^(l)||_2 entering block l.
        delta_g_norm:    (I,) ||Delta_g^(l)||_2 attention-branch update (with LayerScale & bias).
    """
    c = (c_all.detach().cpu().numpy() if isinstance(c_all, torch.Tensor) else np.asarray(c_all)).astype(np.float64)
    norms = (token_norms_all.detach().cpu().numpy() if isinstance(token_norms_all, torch.Tensor) else np.asarray(token_norms_all)).astype(np.float64)
    xg_n = (x_g_in_norm.detach().cpu().numpy() if isinstance(x_g_in_norm, torch.Tensor) else np.asarray(x_g_in_norm)).astype(np.float64)
    dg_n = (delta_g_norm.detach().cpu().numpy() if isinstance(delta_g_norm, torch.Tensor) else np.asarray(delta_g_norm)).astype(np.float64)

    I, T = c.shape
    patch_idx = list(layout["patch_idx"])
    reg_idx = list(layout.get("register_idx", []))
    cls_idx = layout.get("cls_idx")
    dist_idx = list(layout.get("dist_idx", []))
    self_idx = ([cls_idx] if cls_idx is not None else []) + dist_idx

    N = len(patch_idx)
    fg_spatial = (np.asarray(fg_patch_masks).reshape(I, N) > 0)
    bg_spatial = ~fg_spatial

    out_mask, n_out = detect_outlier_tokens(norms, patch_idx, k_mad=k_mad)  # (I, N)
    # Primary ordering (§4.2): OUT overrides FG and BG
    fg_prim = fg_spatial & (~out_mask)
    bg_prim = bg_spatial & (~out_mask)
    bg_or_out = bg_prim | out_mask

    c_patches = c[:, patch_idx]  # (I, N)
    c_total = c.sum(axis=-1)     # (I,)
    c_self = c[:, self_idx].sum(axis=-1) if self_idx else np.zeros(I, dtype=np.float64)
    c_reg = c[:, reg_idx].sum(axis=-1) if reg_idx else np.zeros(I, dtype=np.float64)

    c_fg = np.where(fg_prim, c_patches, 0.0).sum(axis=-1)
    c_bg = np.where(bg_prim, c_patches, 0.0).sum(axis=-1)
    c_out = np.where(out_mask, c_patches, 0.0).sum(axis=-1)
    c_fg_alt = np.where(fg_spatial, c_patches, 0.0).sum(axis=-1)
    c_bg_alt = np.where(bg_spatial, c_patches, 0.0).sum(axis=-1)

    safe_tot = np.where(c_total > 0.0, c_total, np.nan)
    share_fg = c_fg / safe_tot
    share_bg = c_bg / safe_tot
    share_out = c_out / safe_tot
    share_reg = c_reg / safe_tot
    share_self = c_self / safe_tot
    share_fg_alt = c_fg_alt / safe_tot
    share_bg_alt = c_bg_alt / safe_tot

    # Guide §4.3: BSS_raw = (sum_{BG} c_i + sum_{OUT} c_i) / sum_{patches U SELF} c_i
    denom_raw = c_patches.sum(axis=-1) + c_self
    bss_raw = (c_bg + c_out) / np.where(denom_raw > 0.0, denom_raw, np.nan)

    # Guide §4.3 PRIMARY: BSS_norm = mean_{BG U OUT}(c_i) / mean_{FG}(c_i)
    n_fg = fg_prim.sum(axis=-1)
    n_bg_out = bg_or_out.sum(axis=-1)
    mean_fg = np.where(n_fg > 0, c_fg / np.maximum(n_fg, 1), np.nan)
    mean_bg_out = np.where(n_bg_out > 0, (c_bg + c_out) / np.maximum(n_bg_out, 1), np.nan)
    bss_norm = mean_bg_out / np.where(mean_fg > 0.0, mean_fg, np.nan)

    # Appendix alternative ordering (outliers kept in spatial FG/BG group)
    n_fg_alt = fg_spatial.sum(axis=-1)
    n_bg_alt = bg_spatial.sum(axis=-1)
    mean_fg_alt = np.where(n_fg_alt > 0, c_fg_alt / np.maximum(n_fg_alt, 1), np.nan)
    mean_bg_alt = np.where(n_bg_alt > 0, c_bg_alt / np.maximum(n_bg_alt, 1), np.nan)
    bss_norm_alt = mean_bg_alt / np.where(mean_fg_alt > 0.0, mean_fg_alt, np.nan)

    # Guide §4.3: Residual Ratio RR = ||x_g^(l)|| / ||Delta_g^(l)||
    residual_ratio = xg_n / np.where(dg_n > 0.0, dg_n, np.nan)
    gt_area_frac = fg_spatial.mean(axis=-1)

    return {
        "bss_raw": bss_raw,
        "bss_norm": bss_norm,
        "bss_norm_alt": bss_norm_alt,
        "share_fg": share_fg,
        "share_bg": share_bg,
        "share_out": share_out,
        "share_reg": share_reg,
        "share_self": share_self,
        "share_fg_alt": share_fg_alt,
        "share_bg_alt": share_bg_alt,
        "residual_ratio": residual_ratio,
        "n_out_tokens": n_out,
        "gt_area_frac": gt_area_frac,
    }


def assign_area_terciles(gt_area_frac: np.ndarray) -> np.ndarray:
    """Guide §4.4: stratify images into object-area terciles ('small', 'medium', 'large')."""
    a = np.asarray(gt_area_frac, dtype=np.float64)
    q1, q2 = np.nanpercentile(a, [100.0 / 3.0, 200.0 / 3.0])
    out = np.full(a.shape, "medium", dtype=object)
    out[a <= q1] = "small"
    out[a > q2] = "large"
    return out


def _boot_mean_ci(arr: np.ndarray, n_resamples: int = 1000, seed: int = 0) -> dict:
    v = np.asarray(arr, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    mean = float(v.mean())
    if v.size < 2:
        return {"n": int(v.size), "mean": mean, "ci95_low": mean, "ci95_high": mean}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_resamples, v.size))
    lo, hi = np.percentile(v[idx].mean(axis=1), [2.5, 97.5])
    return {"n": int(v.size), "mean": mean, "ci95_low": float(lo), "ci95_high": float(hi)}


def _boot_unpaired_diff_ci(a: np.ndarray, b: np.ndarray, n_resamples: int = 1000, seed: int = 0) -> dict:
    va = np.asarray(a, dtype=np.float64); va = va[np.isfinite(va)]
    vb = np.asarray(b, dtype=np.float64); vb = vb[np.isfinite(vb)]
    if va.size == 0 or vb.size == 0:
        return {"mean_diff": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "ci_excludes_zero": False}
    diff = float(va.mean() - vb.mean())
    if va.size < 2 or vb.size < 2:
        return {"mean_diff": diff, "ci95_low": diff, "ci95_high": diff, "ci_excludes_zero": False}
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, va.size, size=(n_resamples, va.size))
    ib = rng.integers(0, vb.size, size=(n_resamples, vb.size))
    diffs = va[ia].mean(axis=1) - vb[ib].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "mean_diff": diff,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "ci_excludes_zero": bool((lo > 0.0) or (hi < 0.0)),
    }


def waterbirds_congruence_analysis(
    bss_norm: np.ndarray,
    congruent: np.ndarray,
    area_terciles: np.ndarray | None = None,
    y_bird: np.ndarray | None = None,
    place_bg: np.ndarray | None = None,
    n_resamples: int = 1000,
    seed: int = 0,
) -> dict:
    """Guide §4.5 Waterbirds stress test: BSS_norm on background-congruent vs incongruent images."""
    vals = np.asarray(bss_norm, dtype=np.float64)
    cong = np.asarray(congruent, dtype=bool)
    res = {
        "overall": _boot_mean_ci(vals, n_resamples=n_resamples, seed=seed),
        "congruent": _boot_mean_ci(vals[cong], n_resamples=n_resamples, seed=seed),
        "incongruent": _boot_mean_ci(vals[~cong], n_resamples=n_resamples, seed=seed),
        "congruence_gap_cong_minus_incong": _boot_unpaired_diff_ci(vals[cong], vals[~cong], n_resamples=n_resamples, seed=seed),
    }
    if area_terciles is not None:
        terc = np.asarray(area_terciles)
        res["by_tercile"] = {}
        for t in ("small", "medium", "large"):
            m = terc == t
            res["by_tercile"][t] = {
                "congruent": _boot_mean_ci(vals[m & cong], n_resamples=n_resamples, seed=seed),
                "incongruent": _boot_mean_ci(vals[m & (~cong)], n_resamples=n_resamples, seed=seed),
                "gap": _boot_unpaired_diff_ci(vals[m & cong], vals[m & (~cong)], n_resamples=n_resamples, seed=seed),
            }
    if y_bird is not None and place_bg is not None:
        yb = np.asarray(y_bird)
        pb = np.asarray(place_bg)
        groups = {
            "landbird_on_land": (yb == 0) & (pb == 0),
            "landbird_on_water": (yb == 0) & (pb == 1),
            "waterbird_on_land": (yb == 1) & (pb == 0),
            "waterbird_on_water": (yb == 1) & (pb == 1),
        }
        res["by_group"] = {g: _boot_mean_ci(vals[m], n_resamples=n_resamples, seed=seed) for g, m in groups.items()}
    return res
