"""Intervention I1 (Query Replacement) & I2 (Token-Group Knockout) builders and read-outs (Guide §6)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from eval import metrics as M
from measures import qii
from measures._common import box_auc_torch

DEFAULT_L_INT_SWEEP = (1, 2, 3, 4)  # Covers {1, 2, 4, L/4} for L=12 pilot models (Guide §6)
DEFAULT_I1_SURROGATES = ("S1", "S2", "S3", "S4")
DEFAULT_I2_GROUPS = ("FG", "BG", "OUT", "REG")
DEFAULT_SEEDS = (0, 1, 2, 3, 4)


# --------------------------------------------------------------------------- I1 Surrogate Query Generator
def sample_surrogate_queries_batch(
    family: str,
    q_g_all: np.ndarray,  # (I, H, d_h)
    pca: qii.QueryPCA,
    seed: int = 0,
) -> np.ndarray:
    """Generate (I, H, d_h) float32 surrogate queries for family in {S1, S2, S3, S4} (exact M1 definitions).

    - S1: Isotropic Gaussian matched to each image's true ||q_g||_2 per head.
    - S2: Empirical mean query mu across images (deterministic, image-independent).
    - S3: Cross-image [CLS] query from a random different image (derangement, no self-match).
    - S4: PCA-matched Gaussian in empirical query subspace matched to true ||q_g||_2.
    """
    I, H, d_h = q_g_all.shape
    rng = np.random.default_rng(seed)

    if family == "S2":
        mu = pca.mean.astype(np.float32)  # (H, d_h)
        return np.broadcast_to(mu[None, :, :], (I, H, d_h)).copy()

    if family == "S1":
        norms = np.linalg.norm(q_g_all.astype(np.float64), axis=-1, keepdims=True)  # (I, H, 1)
        g = rng.standard_normal((I, H, d_h))
        g_norm = np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-12)
        return (g / g_norm * norms).astype(np.float32)

    if family == "S3":
        if I < 2:
            return q_g_all.astype(np.float32).copy()
        # Random derangement (j != i for every image)
        shifts = rng.integers(1, I, size=I)
        j_idx = (np.arange(I) + shifts) % I
        return q_g_all[j_idx].astype(np.float32).copy()

    if family == "S4":
        norms = np.linalg.norm(q_g_all.astype(np.float64), axis=-1, keepdims=True)  # (I, H, 1)
        out = np.empty((I, H, d_h), dtype=np.float32)
        for h in range(H):
            k_dim = int(pca.stds.shape[1])
            z = rng.standard_normal((I, k_dim)) * pca.stds[h][None, :]  # (I, k)
            # pca.components has shape (H, d_h, k_dim)
            raw = pca.mean[h][None, :] + z @ pca.components[h].T        # (I, d_h)
            r_norm = np.maximum(np.linalg.norm(raw, axis=-1, keepdims=True), 1e-12)
            out[:, h, :] = (raw / r_norm * norms[:, h, :]).astype(np.float32)
        return out

    raise ValueError(f"Unknown surrogate family {family!r}; expected one of S1..S4")


# --------------------------------------------------------------------------- I2 Token-Group Masks & Matched Random Control
def build_token_group_masks(
    fg_patch_masks: np.ndarray,   # (I, N) bool
    out_patch_masks: np.ndarray,  # (I, N) bool (median + 5*MAD)
    layout: dict,
) -> dict[str, np.ndarray]:
    """Build (I, T) boolean knockout masks for FG, BG, OUT, REG (Guide §4.2 & §6).

    Matches M3's primary token-group partition:
      - OUT : high-norm outlier patches (takes precedence over FG/BG)
      - FG  : foreground patches excluding OUT
      - BG  : background patches excluding OUT
      - REG : register tokens (all False if model has no registers)
    """
    fg_p = np.asarray(fg_patch_masks, dtype=bool)
    out_p = np.asarray(out_patch_masks, dtype=bool)
    I, N = fg_p.shape
    patch_idx = np.asarray(layout["patch_idx"], dtype=np.int64)
    reg_idx = np.asarray(layout.get("register_idx", []), dtype=np.int64)
    T = int(layout["n_tokens"]) if "n_tokens" in layout else int(max(patch_idx.max(), layout["cls_idx"]) + 1)

    masks = {g: np.zeros((I, T), dtype=bool) for g in DEFAULT_I2_GROUPS}
    masks["OUT"][:, patch_idx] = out_p
    masks["FG"][:, patch_idx] = fg_p & (~out_p)
    masks["BG"][:, patch_idx] = (~fg_p) & (~out_p)
    if reg_idx.size > 0:
        masks["REG"][:, reg_idx] = True
    return masks


def make_matched_random_patch_mask(
    group_mask: np.ndarray,  # (I, T) bool
    layout: dict,
    seed: int = 0,
    eligible_mask: np.ndarray | None = None,  # Optional (I, T) bool restricting pool (e.g. non-FG for BG)
) -> np.ndarray:
    """Guide §6: for each image i, randomly select the exact same count m_i = |group_i| from eligible patch_idx.

    Never selects [CLS] or register tokens. When `eligible_mask` is supplied (e.g. non-FG patches for BG
    knockout), samples exclusively from `patch_idx ∩ eligible_mask[i]`.
    """
    gm = np.asarray(group_mask, dtype=bool)
    I, T = gm.shape
    patch_idx = np.asarray(layout["patch_idx"], dtype=np.int64)
    N = patch_idx.size
    elig = np.asarray(eligible_mask, dtype=bool) if eligible_mask is not None else None
    rng = np.random.default_rng(seed)
    rand_mask = np.zeros((I, T), dtype=bool)
    counts = gm.sum(axis=1)
    for i in range(I):
        m = int(counts[i])
        if m <= 0:
            continue
        pool = patch_idx[elig[i, patch_idx]] if elig is not None else patch_idx
        if pool.size == 0:
            continue
        if m >= pool.size:
            rand_mask[i, pool] = True
        else:
            chosen = rng.choice(pool, size=m, replace=False)
            rand_mask[i, chosen] = True
    return rand_mask


# --------------------------------------------------------------------------- Functional & Map Read-Outs
def cosine_similarity_batch(z_clean: np.ndarray, z_int: np.ndarray) -> np.ndarray:
    """Per-image cosine similarity between clean and intervened global embeddings (I,)."""
    a = np.asarray(z_clean, dtype=np.float64)
    b = np.asarray(z_int, dtype=np.float64)
    na = np.maximum(np.linalg.norm(a, axis=-1), 1e-12)
    nb = np.maximum(np.linalg.norm(b, axis=-1), 1e-12)
    return np.sum(a * b, axis=-1) / (na * nb)


def kl_divergence_logits(logits_clean: np.ndarray, logits_int: np.ndarray) -> np.ndarray:
    """Per-image KL(p_clean || p_intervened) in nats (fp64, Guide §6)."""
    lc = torch.as_tensor(logits_clean, dtype=torch.float64)
    li = torch.as_tensor(logits_int, dtype=torch.float64)
    log_pc = F.log_softmax(lc, dim=-1)
    log_pi = F.log_softmax(li, dim=-1)
    pc = log_pc.exp()
    kl = (pc * (log_pc - log_pi)).sum(dim=-1)
    return kl.clamp_min(0.0).numpy()


def top1_agreement(logits_clean: np.ndarray, logits_int: np.ndarray) -> np.ndarray:
    """Per-image 1.0 if argmax(logits_int) == argmax(logits_clean), else 0.0."""
    return (np.argmax(logits_clean, axis=-1) == np.argmax(logits_int, axis=-1)).astype(np.float64)


def knn_accuracy_cosine(
    feats_query: np.ndarray,
    labels: np.ndarray,
    *,
    feats_ref: np.ndarray | None = None,
    labels_ref: np.ndarray | None = None,
    exclude_self: bool | None = None,
    k: int = 20,
    device: torch.device | str | None = None,
) -> tuple[float, np.ndarray]:
    """Cosine-weighted k-NN classification accuracy (k=20, Guide §6).

    Supports two modes:
      1. External gallery (`feats_ref` and `labels_ref` provided, `exclude_self=False` by default):
         evaluates each query against the full training gallery (e.g. CUB-200-2011 5,994-image train split).
      2. Leave-one-out (`labels_ref is None`, `exclude_self=True` by default):
         evaluates each query against `feats_ref` (or itself) excluding self-index `i`.
    """
    dev = torch.device(device) if device is not None else torch.device("cpu")
    y_q = np.asarray(labels, dtype=np.int64)
    y_r = np.asarray(labels_ref, dtype=np.int64) if labels_ref is not None else y_q
    n_q = y_q.shape[0]
    n_r = y_r.shape[0]
    excl = (labels_ref is None) if exclude_self is None else bool(exclude_self)
    if n_q == 0 or n_r == 0 or (excl and n_r < 2):
        return float("nan"), np.full(n_q, np.nan)
    k_eff = min(k, n_r - 1 if excl else n_r)

    Q = torch.as_tensor(feats_query, dtype=torch.float32, device=dev)
    Q = Q / torch.clamp(torch.linalg.norm(Q, dim=-1, keepdim=True), min=1e-12)
    R = Q if feats_ref is None else torch.as_tensor(feats_ref, dtype=torch.float32, device=dev)
    if feats_ref is not None:
        R = R / torch.clamp(torch.linalg.norm(R, dim=-1, keepdim=True), min=1e-12)

    sims = Q @ R.T  # (n_q, n_r)
    if excl and n_q == n_r:
        sims.fill_diagonal_(-float("inf"))
    top_vals, top_idx = torch.topk(sims, k=k_eff, dim=-1, largest=True, sorted=False)
    top_idx_np = top_idx.cpu().numpy()
    top_w = torch.softmax(top_vals / 0.07, dim=-1).cpu().numpy()  # standard DINO cosine-weighted kNN

    correct = np.zeros(n_q, dtype=np.float64)
    for i in range(n_q):
        nbr_labels = y_r[top_idx_np[i]]
        uniq, inv = np.unique(nbr_labels, return_inverse=True)
        scores = np.bincount(inv, weights=top_w[i])
        pred = uniq[int(np.argmax(scores))]
        correct[i] = 1.0 if pred == y_q[i] else 0.0
    return float(correct.mean()), correct


def _roc_auc_1d_exact(scores: np.ndarray, labels_bool: np.ndarray) -> float:
    """Tie-aware exact Mann-Whitney ROC-AUC for 1D scores and boolean labels."""
    y = np.asarray(labels_bool, dtype=bool)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    s = np.asarray(scores, dtype=np.float64)
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]
    # Average ranks for ties (1-based)
    n = s_sorted.size
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and s_sorted[j] == s_sorted[i]:
            j += 1
        avg_r = 0.5 * ((i + 1) + j)
        ranks[i:j] = avg_r
        i = j
    sum_pos_ranks = float(ranks[y_sorted].sum())
    return (sum_pos_ranks - 0.5 * n_pos * (n_pos + 1)) / float(n_pos * n_neg)


def eval_patch_box_auc_unablated(
    patch_maps: np.ndarray,           # (I, N) float32
    fg_patch_masks: np.ndarray,       # (I, N) bool
    ablated_patch_mask: np.ndarray | None = None,  # (I, N) bool True = ablated/excluded
) -> np.ndarray:
    """Exact ROC-AUC scored strictly over NON-ABLATED patch tokens (prevents mechanical zero-ablation artifacts).

    Returns (I,) float64 array with NaN for images where non-ablated tokens have 0 FG or 0 BG patches.
    """
    pm = np.asarray(patch_maps, dtype=np.float64)
    fg = np.asarray(fg_patch_masks, dtype=bool)
    I, N = pm.shape
    keep = np.ones((I, N), dtype=bool) if ablated_patch_mask is None else ~np.asarray(ablated_patch_mask, dtype=bool)
    out = np.full(I, np.nan, dtype=np.float64)
    for i in range(I):
        k_i = keep[i]
        if not k_i.any():
            continue
        out[i] = _roc_auc_1d_exact(pm[i, k_i], fg[i, k_i])
    return out


@torch.no_grad()
def eval_maps_box_auc_and_otsu(
    patch_maps: np.ndarray,           # (I, N) float32
    grid: tuple[int, int],
    target_size: int,
    gt_masks_np: np.ndarray,          # (I, S, S) uint8/bool
    gt_boxes_list: Sequence[Sequence[Sequence[float]]] | None = None,
    device: torch.device | str | None = None,
    compute_otsu: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Bilinearly upsample (I, N) patch maps and compute per-image box-AUC (GPU) and optional Otsu CorLoc."""
    dev = torch.device(device) if device is not None else torch.device("cpu")
    I = patch_maps.shape[0]
    arr = torch.as_tensor(patch_maps, dtype=torch.float32, device=dev).reshape(I, 1, grid[0], grid[1])
    up = F.interpolate(arr, size=(target_size, target_size), mode="bilinear", align_corners=False).squeeze(1)
    # Min-max normalize per image
    flat = up.reshape(I, -1)
    lo = flat.min(dim=-1).values.view(I, 1, 1)
    hi = flat.max(dim=-1).values.view(I, 1, 1)
    up_norm = torch.where(hi > lo, (up - lo) / (hi - lo).clamp_min(1e-12), torch.zeros_like(up))

    m_t = torch.as_tensor(gt_masks_np, dtype=torch.bool, device=dev)
    aucs = box_auc_torch(up_norm, m_t).cpu().numpy().astype(np.float64)

    otsu_hits = None
    if compute_otsu and gt_boxes_list is not None:
        up_np = up_norm.cpu().numpy()
        boxes = [M.otsu_box(up_np[i])[0] for i in range(I)]
        _, otsu_hits = M.corloc(boxes, gt_boxes_list, iou_thresh=0.5)
    return aucs, otsu_hits
