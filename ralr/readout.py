"""RALR — Routing-Aware Localization Readout (Methodology Guide §7.1-§7.4).

Zero-parameter, label-free readout and its mandatory baselines:
- Offline calibration on unlabelled X_cal (zero labels, zero boxes):
    1. Reject degenerate uniform layers with mean normalized entropy E(l) > 0.9.
    2. Select l* = argmax_{l eligible} KD(l) (key dominance omega^2_key).
    3. Branch decision with frozen global threshold tau_q = 0.5:
         branch = "HIGH" if QII_S4(l*) >= tau_q else "LOW".
- Per-image readout (Steps 6-9):
    * HIGH branch: head-weighted query-free key-effect map beta_i(l*, h) (from M2).
    * LOW branch : Step-7 shortcut token detection
                     shortcut = {i: norm_i > median + 5*MAD}
                                U {i: c_i > q90(c) AND keyaff_i < q30(keyaff)}
                   where keyaff_i = cosine(k_i, centroid of largest k-means cluster, k=2, 10 restarts, seed=0),
                   zero [CLS] attention to shortcut tokens, renormalize per head, head-weighted sum.
    * Bilinear upsample (G, G) -> (target_size, target_size) and min-max normalize to [0, 1].
- Mandatory baselines & ablations (Guide §7.2 & §7.4):
    * cls_last          : raw [CLS] attention at last block, head-averaged (DINO default)
    * cls_l_star        : raw [CLS] attention at l*, head-averaged (isolates layer selection alone)
    * cls_l_star_w      : head-weighted [CLS] attention at l*
    * ralr_high         : always-HIGH / key-effect map without shortcut removal
    * ralr_low          : always-LOW / shortcut-knockout [CLS] attention at l*
    * ralr_high_cleaned : key-effect map with Step-7 shortcut suppression
    * ralr_adaptive     : frozen tau_q = 0.5 branch selection
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F

from eval.metrics import minmax_normalize_map
from measures.routing import detect_outlier_tokens

FROZEN_TAU_Q = 0.5
ENTROPY_CEILING = 0.90


# --------------------------------------------------------------------------- Offline calibration (§7.1 Steps 1-5)
def select_layer_and_branch(
    layer_stats: Mapping[int, Mapping[str, float]],
    *,
    tau_q: float = FROZEN_TAU_Q,
    entropy_ceiling: float = ENTROPY_CEILING,
) -> dict:
    """Guide §7.1 Steps 2-4: choose l* and branch purely from unlabelled calibration statistics.

    Args:
        layer_stats: {layer_idx: {"kd": float, "qii_s4": float, "entropy": float}}
        tau_q:       frozen global QII_S4 threshold (default 0.5, Guide §7.2)
    """
    if not layer_stats:
        raise ValueError("layer_stats cannot be empty")
    layers_sorted = sorted(layer_stats.keys())
    eligible = [l for l in layers_sorted if float(layer_stats[l]["entropy"]) <= entropy_ceiling]
    all_degenerate = len(eligible) == 0
    pool = layers_sorted if all_degenerate else eligible

    l_star = max(pool, key=lambda l: float(layer_stats[l]["kd"]))
    qii_val = float(layer_stats[l_star]["qii_s4"])
    branch = "HIGH" if qii_val >= float(tau_q) else "LOW"
    return {
        "l_star": int(l_star),
        "branch": branch,
        "tau_q": float(tau_q),
        "entropy_ceiling": float(entropy_ceiling),
        "eligible_layers": [int(l) for l in eligible],
        "all_layers_degenerate": bool(all_degenerate),
        "kd_at_l_star": float(layer_stats[l_star]["kd"]),
        "qii_s4_at_l_star": qii_val,
        "entropy_at_l_star": float(layer_stats[l_star]["entropy"]),
    }


# --------------------------------------------------------------------------- Step 7: Key clustering & shortcuts (§7.3)
def kmeans_cosine_keys(
    k_concat: np.ndarray,
    *,
    k: int = 2,
    n_restarts: int = 10,
    max_iter: int = 30,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Guide §7.3: L2-normalize concatenated patch keys, run k-means (k=2, 10 restarts, seed 0).

    Args:
        k_concat: (N, D_k) float array of patch keys concatenated across heads at l*.
    Returns:
        labels:      (N,) int cluster assignment in {0, ..., k-1}
        centroids:   (k, D_k) L2-normalized cluster centroids
        dom_cluster: index of largest cluster by patch count
        inertia:     best within-cluster squared Euclidean distance
    """
    X = np.asarray(k_concat, dtype=np.float64)
    N, D = X.shape
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.maximum(norms, 1e-12)

    k_eff = min(k, N)
    rng = np.random.default_rng(seed)
    best_inertia = float("inf")
    best_labels = np.zeros(N, dtype=np.int64)
    best_centroids = Xn[:k_eff].copy()

    for _ in range(n_restarts):
        # k-means++ deterministic seeding for stable convergence
        c_idx = [int(rng.integers(0, N))]
        for _c in range(1, k_eff):
            dists = np.min(
                np.stack([np.sum((Xn - Xn[idx]) ** 2, axis=1) for idx in c_idx], axis=0),
                axis=0,
            )
            tot = float(dists.sum())
            if tot <= 1e-12:
                next_i = int(rng.integers(0, N))
            else:
                probs = dists / tot
                next_i = int(rng.choice(N, p=probs))
            c_idx.append(next_i)
        C = Xn[c_idx].copy()

        labels = np.zeros(N, dtype=np.int64)
        for _it in range(max_iter):
            # Euclidean distance on unit sphere: ||x - c||^2 = 2 - 2 x.c (when C normalized) or exact sq dist
            d2 = np.sum((Xn[:, None, :] - C[None, :, :]) ** 2, axis=-1)  # (N, k_eff)
            new_labels = np.argmin(d2, axis=1)
            if np.array_equal(new_labels, labels) and _it > 0:
                break
            labels = new_labels
            for c in range(k_eff):
                m = labels == c
                if np.any(m):
                    C[c] = Xn[m].mean(axis=0)
                else:
                    C[c] = Xn[int(rng.integers(0, N))]

        d2 = np.sum((Xn[:, None, :] - C[None, :, :]) ** 2, axis=-1)
        inertia = float(d2[np.arange(N), labels].sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centroids = C.copy()

    # L2-normalize centroids for cosine similarity keyaff_i = cosine(k_i, centroid_dom)
    c_norms = np.linalg.norm(best_centroids, axis=1, keepdims=True)
    best_centroids = best_centroids / np.maximum(c_norms, 1e-12)
    counts = np.bincount(best_labels, minlength=k_eff)
    dom_cluster = int(np.argmax(counts))
    return best_labels, best_centroids, dom_cluster, best_inertia


def detect_shortcut_tokens(
    patch_norms: np.ndarray,
    patch_contrib_c: np.ndarray,
    k_concat: np.ndarray,
    *,
    cls_patch_attn: np.ndarray | None = None,
    k_mad: float = 5.0,
    k_clusters: int = 2,
    n_restarts: int = 10,
    seed: int = 0,
    dominant_rule: str = "largest",
) -> dict:
    """Guide §7.1 Step 7 & §7.3 shortcut token detector for one image.

    shortcut = {i: norm_i > median + 5*MAD} U {i: c_i > q90(c) AND keyaff_i < q30(keyaff)}

    Args:
        patch_norms:     (N,) L2 norms of patch tokens at l*
        patch_contrib_c: (N,) Kobayashi token contribution norms c_i for patches at l*
        k_concat:        (N, H * d_h) concatenated patch keys at l*
        cls_patch_attn:  optional (N,) mean [CLS] attention over patches for §7.3 sanity check
        dominant_rule:   "largest" (default §7.3) or "lower_attn" (fallback if largest != background)
    """
    p_norms = np.asarray(patch_norms, dtype=np.float64)
    c = np.asarray(patch_contrib_c, dtype=np.float64)
    N = p_norms.shape[0]

    # Part 1: High-norm outlier tokens (exact M3 rule)
    out_norm_mask, _ = detect_outlier_tokens(p_norms, np.arange(N), k_mad=k_mad)

    # Part 2: Key clustering & affinity to dominant cluster
    labels, centroids, largest_c, _ = kmeans_cosine_keys(
        k_concat, k=k_clusters, n_restarts=n_restarts, seed=seed
    )

    largest_is_lower_attn = None
    dom_c = largest_c
    if cls_patch_attn is not None:
        attn_1d = np.asarray(cls_patch_attn, dtype=np.float64)
        cluster_means = [
            float(attn_1d[labels == ci].mean()) if np.any(labels == ci) else float("inf")
            for ci in range(centroids.shape[0])
        ]
        min_attn_c = int(np.argmin(cluster_means))
        largest_is_lower_attn = bool(largest_c == min_attn_c)
        if dominant_rule == "lower_attn":
            dom_c = min_attn_c

    X = np.asarray(k_concat, dtype=np.float64)
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    keyaff = Xn @ centroids[dom_c]  # (N,) cosine similarity in [-1, 1]

    q90_c = float(np.quantile(c, 0.90))
    q30_aff = float(np.quantile(keyaff, 0.30))
    out_short_mask = (c > q90_c) & (keyaff < q30_aff)

    shortcut_mask = np.logical_or(out_norm_mask, out_short_mask)
    return {
        "shortcut_mask": shortcut_mask,
        "out_norm_mask": out_norm_mask,
        "out_shortcut_mask": out_short_mask,
        "keyaff": keyaff,
        "cluster_labels": labels,
        "dominant_cluster": int(dom_c),
        "largest_is_lower_attn": largest_is_lower_attn,
        "n_shortcut": int(shortcut_mask.sum()),
        "n_out_norm": int(out_norm_mask.sum()),
        "n_out_keyaff": int(out_short_mask.sum()),
    }


# --------------------------------------------------------------------------- Upsampling & full readout (§7.1 Steps 6-8)
def upsample_patch_map(patch_map_1d: np.ndarray, grid: tuple[int, int], target_size: int) -> np.ndarray:
    """Reshape (N,) patch map to (G_h, G_w), bilinear upsample to (target_size, target_size), min-max normalize."""
    arr = np.asarray(patch_map_1d, dtype=np.float32).reshape(1, 1, grid[0], grid[1])
    up = F.interpolate(torch.from_numpy(arr), size=(target_size, target_size), mode="bilinear", align_corners=False)
    return minmax_normalize_map(up[0, 0].numpy())


def compute_ralr_maps_single_image(
    *,
    cls_attn_last: np.ndarray,     # (H, N) [CLS] patch attention at final layer L-1
    cls_attn_lstar: np.ndarray,    # (H, N) [CLS] patch attention at l*
    head_weights_lstar: np.ndarray,  # (H,) w_h at l*
    key_effect_lstar: np.ndarray,  # (H, N) key-effect map beta_i(l*, h) from M2
    patch_norms_lstar: np.ndarray, # (N,) ||x_out_i||_2 at l*
    patch_contrib_lstar: np.ndarray, # (N,) Kobayashi c_i at l*
    k_patch_lstar: np.ndarray,     # (H, N, d_h) patch keys at l*
    branch: str,
    grid: tuple[int, int],
    target_size: int,
    k_clusters: int = 2,
    dominant_rule: str = "largest",
    seed: int = 0,
) -> tuple[dict[str, np.ndarray], dict]:
    """Compute RALR readout + all §7.2/§7.4 baselines & ablations for one image.

    Returns:
        maps: dict mapping method name -> (target_size, target_size) float64 normalized map in [0, 1]
        diag: Step-7 diagnostics (n_shortcut, n_out_norm, n_out_keyaff, largest_is_lower_attn)
    """
    H, N = cls_attn_lstar.shape
    w_h = np.asarray(head_weights_lstar, dtype=np.float64)
    w_sum = float(w_h.sum())
    w_norm = w_h / w_sum if w_sum > 1e-12 else np.full(H, 1.0 / H, dtype=np.float64)

    # Baseline 1 (§7.4): raw [CLS] attention at last block, head-averaged
    raw_cls_last = np.asarray(cls_attn_last, dtype=np.float64).mean(axis=0)

    # Baseline 2 (§7.4, CRITICAL): raw [CLS] attention at l*, head-averaged
    raw_cls_lstar = np.asarray(cls_attn_lstar, dtype=np.float64).mean(axis=0)

    # Ablation: head-weighted [CLS] attention at l*
    raw_cls_lstar_w = (np.asarray(cls_attn_lstar, dtype=np.float64) * w_norm[:, None]).sum(axis=0)

    # Always-HIGH / Key-effect map without shortcut removal (§7.1 Step 6 HIGH & §7.4)
    raw_ralr_high = (np.asarray(key_effect_lstar, dtype=np.float64) * w_norm[:, None]).sum(axis=0)

    # Step 7 shortcut detection
    k_concat = np.transpose(np.asarray(k_patch_lstar, dtype=np.float64), (1, 0, 2)).reshape(N, -1)
    sdiag = detect_shortcut_tokens(
        patch_norms_lstar,
        patch_contrib_lstar,
        k_concat,
        cls_patch_attn=raw_cls_lstar,
        k_clusters=k_clusters,
        dominant_rule=dominant_rule,
        seed=seed,
    )
    smask = sdiag["shortcut_mask"]  # (N,) bool

    # Always-LOW (§7.1 Step 6 LOW): zero [CLS] attention to shortcut tokens, renormalize per head, head-weight
    attn_low = np.asarray(cls_attn_lstar, dtype=np.float64).copy()
    if not np.all(smask):
        attn_low[:, smask] = 0.0
        row_s = attn_low.sum(axis=1, keepdims=True)
        attn_low = np.where(row_s > 1e-12, attn_low / np.maximum(row_s, 1e-12), np.asarray(cls_attn_lstar, dtype=np.float64))
    raw_ralr_low = (attn_low * w_norm[:, None]).sum(axis=0)

    # Bonus ablation: key-effect map with Step-7 shortcut suppression (clamps shortcut patches to non-shortcut min)
    raw_ralr_high_cleaned = raw_ralr_high.copy()
    if np.any(smask) and not np.all(smask):
        raw_ralr_high_cleaned[smask] = float(raw_ralr_high[~smask].min())

    raw_adaptive = raw_ralr_high if branch == "HIGH" else raw_ralr_low

    maps = {
        "cls_last": upsample_patch_map(raw_cls_last, grid, target_size),
        "cls_l_star": upsample_patch_map(raw_cls_lstar, grid, target_size),
        "cls_l_star_w": upsample_patch_map(raw_cls_lstar_w, grid, target_size),
        "ralr_high": upsample_patch_map(raw_ralr_high, grid, target_size),
        "ralr_low": upsample_patch_map(raw_ralr_low, grid, target_size),
        "ralr_high_cleaned": upsample_patch_map(raw_ralr_high_cleaned, grid, target_size),
        "ralr_adaptive": upsample_patch_map(raw_adaptive, grid, target_size),
    }
    diag = {
        "n_shortcut": sdiag["n_shortcut"],
        "n_out_norm": sdiag["n_out_norm"],
        "n_out_keyaff": sdiag["n_out_keyaff"],
        "largest_is_lower_attn": sdiag["largest_is_lower_attn"],
    }
    return maps, diag
