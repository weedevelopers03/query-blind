"""Measure M1 — Query-Invariance Index (QII) (Methodology Guide §2, Fix 2, §12).

Definition (§2.1):
    For image x, block l, head h:
      1. Extract true global query q_g and patch keys k_i (patches ONLY; exclude [CLS]/registers/dist).
      2. a_true = softmax_i(q_g . k_i * scale) in fp32 over the N patch positions.
      3. For each surrogate query q~ from surrogate family S in {S1, S2, S3, S4},
         a_surr = softmax_i(q~ . k_i * scale) in fp32 over the N patch positions.
      4. QII(l, h, S) = 1 - mean_{q~} JSD(a_true || a_surr) / log(2),
         where JSD uses natural log (matches scipy.spatial.distance.jensenshannon(..., base=np.e)**2).

Mandatory controls (§2.3):
  * Shuffled-key null (QII must be invariant under spatial permutation of k_i).
  * Normalized entropy H(a_true) / log(N) — catches Fix 2 "degenerate invariance" (uniform attention).
  * Map quality box-AUC of bilinearly upsampled a_true against ground-truth mask.
  * Per-head QII AND head-contribution-weighted aggregate using w_h (§2.3).
  * NEVER average attention distributions across heads before computing JSD (§2.4).
  * MAP-head models (fixed learned probe query): S2 (cross-image global query) is undefined and excluded (§1.1, §2.2).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from ._common import (
    LOG2,
    box_auc,
    compute_head_weights,
    get_output_proj_and_layerscale,
    jsd_nats,
    normalized_entropy,
    patch_attention,
    upsample_map_bilinear,
)

SURROGATES: tuple[str, ...] = ("S1", "S2", "S3", "S4")
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_PCA_K: int = 32
ENTROPY_ELIGIBILITY_THRESHOLD: float = 0.9  # Guide §7.1: E(l) <= 0.9 rejects degenerate-uniform layers

QII_SCHEMA_COLUMNS: tuple[str, ...] = (
    "model_id",
    "objective",
    "set",
    "layer",
    "layer_frac",
    "head",
    "image_id",
    "surrogate",
    "seed",
    "jsd",
    "qii",
    "entropy_norm",
    "box_auc",
    "head_weight",
    "resolution",
    "grid",
)


# --------------------------------------------------------------------------- PCA fit for S4 (§2.2)
@dataclass
class QueryPCA:
    """Per-head empirical PCA fit of global queries at one (model, layer) for surrogate S4."""

    mean: np.ndarray          # (H, d_h) float64
    components: np.ndarray    # (H, d_h, k_max) orthonormal columns per head, float64
    stds: np.ndarray          # (H, k_max) singular-value stds per head (0 for unused slots), float64
    k_eff: np.ndarray         # (H,) int, active rank <= k per head
    n_samples: int
    k_requested: int

    @property
    def is_degenerate(self) -> np.ndarray:
        """True per head when empirical query variance is effectively zero (e.g. MAP-head fixed probe)."""
        return self.k_eff == 0

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            mean=self.mean,
            components=self.components,
            stds=self.stds,
            k_eff=self.k_eff,
            n_samples=np.int64(self.n_samples),
            k_requested=np.int64(self.k_requested),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "QueryPCA":
        d = np.load(Path(path))
        return cls(
            mean=d["mean"],
            components=d["components"],
            stds=d["stds"],
            k_eff=d["k_eff"],
            n_samples=int(d["n_samples"]),
            k_requested=int(d["k_requested"]),
        )


def fit_query_pca(q_global_pool: np.ndarray | torch.Tensor, k: int = DEFAULT_PCA_K, min_std: float = 1e-7) -> QueryPCA:
    """Fit per-head top-k PCA on empirical global queries (I, H, d_h) for S4 (Guide §2.2)."""
    X = q_global_pool.detach().cpu().numpy() if isinstance(q_global_pool, torch.Tensor) else np.asarray(q_global_pool)
    if X.ndim != 3:
        raise ValueError(f"Expected q_global_pool shape (I, H, d_h), got {X.shape}")
    I, H, d_h = X.shape
    X64 = X.astype(np.float64)
    k_max = min(int(k), d_h, max(I - 1, 1))

    mean = X64.mean(axis=0)  # (H, d_h)
    components = np.zeros((H, d_h, k_max), dtype=np.float64)
    stds = np.zeros((H, k_max), dtype=np.float64)
    k_eff = np.zeros(H, dtype=np.int64)

    if I >= 2:
        denom = np.sqrt(float(I - 1))
        for h in range(H):
            Xc = X64[:, h, :] - mean[h]
            _, s, vt = np.linalg.svd(Xc, full_matrices=False)
            sigma = s / denom
            active = int(np.sum(sigma[:k_max] > min_std))
            k_eff[h] = active
            if active > 0:
                components[h, :, :active] = vt[:active].T
                stds[h, :active] = sigma[:active]

    return QueryPCA(
        mean=mean,
        components=components,
        stds=stds,
        k_eff=k_eff,
        n_samples=I,
        k_requested=int(k),
    )


def _haar_orthogonal(k: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform (Haar-distributed) random orthogonal matrix R in O(k) (Mezzadri 2007)."""
    A = rng.standard_normal((k, k))
    Q, R = np.linalg.qr(A)
    diag = np.diag(R)
    signs = np.where(diag >= 0.0, 1.0, -1.0)
    return Q * signs[None, :]


# --------------------------------------------------------------------------- Surrogate families (§2.2)
def _seed_rng(seed: int, image_idx: int, salt: int) -> np.random.Generator:
    """Deterministic, non-colliding RNG per (seed, image_idx, surrogate_salt)."""
    return np.random.default_rng([int(seed), int(image_idx), int(salt)])


def make_surrogate_s1(q_g: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """S1: Isotropic Gaussian vector rescaled per head to ||q_g||_2."""
    q64 = np.asarray(q_g, dtype=np.float64)  # (H, d_h)
    g = rng.standard_normal(q64.shape)
    g_norm = np.linalg.norm(g, axis=-1, keepdims=True)
    q_norm = np.linalg.norm(q64, axis=-1, keepdims=True)
    g = np.where(g_norm > 0, g / g_norm, 0.0) * q_norm
    return g.astype(np.float32)


def make_surrogate_s2(
    image_idx: int,
    q_global_pool: np.ndarray,
    seed: int,
    n_seeds: int = 5,
    is_map_head: bool = False,
) -> np.ndarray | None:
    """S2: Global query from a different image (j != image_idx), same model & layer.

    Explicitly returns None for MAP-head models (§1.1, §2.2: no per-image query to swap).
    Draws distinct other-image indices across seeds 0..n_seeds-1 when I - 1 >= n_seeds.
    """
    if is_map_head:
        return None
    I = q_global_pool.shape[0]
    if I < 2:
        return None
    # Draw an ordered list of other-image indices deterministically per image_idx, then index by seed
    rng = np.random.default_rng([int(image_idx), 202])
    others = np.delete(np.arange(I), image_idx)
    if len(others) >= max(n_seeds, seed + 1):
        chosen = rng.choice(others, size=max(n_seeds, seed + 1), replace=False)
        j = int(chosen[seed])
    else:
        rng_s = _seed_rng(seed, image_idx, 202)
        j = int(rng_s.choice(others))
    return np.asarray(q_global_pool[j], dtype=np.float32)


def make_surrogate_s3(
    q_patches: np.ndarray,
    image_idx: int,
    seed: int,
    n_seeds: int = 5,
) -> np.ndarray | None:
    """S3: Randomly chosen patch query from the same image (H, N, d_h) -> (H, d_h)."""
    if q_patches is None:
        return None
    N = q_patches.shape[1]
    if N < 1:
        return None
    rng = np.random.default_rng([int(image_idx), 303])
    if N >= max(n_seeds, seed + 1):
        chosen = rng.choice(N, size=max(n_seeds, seed + 1), replace=False)
        p_idx = int(chosen[seed])
    else:
        p_idx = int(_seed_rng(seed, image_idx, 303).integers(0, N))
    return np.asarray(q_patches[:, p_idx, :], dtype=np.float32)


def make_surrogate_s4(
    q_g: np.ndarray,
    pca: QueryPCA | None,
    rng: np.random.Generator,
    is_map_head: bool = False,
) -> np.ndarray | None:
    """S4 (headline): Random Haar rotation of q_g within the whitened top-k PCA subspace,
    matched in both empirical covariance and exact per-head norm ||q_g||_2 (§2.2).

    Returns None for MAP-head models or degenerate zero-variance query pools.
    """
    if is_map_head or pca is None or bool(np.all(pca.is_degenerate)):
        return None
    q64 = np.asarray(q_g, dtype=np.float64)  # (H, d_h)
    H, d_h = q64.shape
    out = np.empty_like(q64)

    for h in range(H):
        ke = int(pca.k_eff[h])
        q_norm = np.linalg.norm(q64[h])
        if ke == 0 or q_norm == 0.0:
            out[h] = np.nan
            continue
        mu = pca.mean[h]
        U = pca.components[h, :, :ke]   # (d_h, ke)
        sig = pca.stds[h, :ke]          # (ke,)

        q_c = q64[h] - mu
        proj = U.T @ q_c                # (ke,)
        q_perp = q_c - U @ proj         # orthogonal complement outside top-k

        # Rotate in whitened coordinates so empirical covariance U diag(sig^2) U^T is preserved
        z = proj / sig
        R = _haar_orthogonal(ke, rng)
        z_rot = R @ z
        q_rot = mu + U @ (sig * z_rot) + q_perp

        # Match exact per-head norm ||q_g||_2 (§2.2: "matched norm and matched covariance")
        rot_norm = np.linalg.norm(q_rot)
        out[h] = q_rot * (q_norm / rot_norm) if rot_norm > 0 else q_rot

    return out.astype(np.float32)


# --------------------------------------------------------------------------- Core M1 computation
def aggregate_qii_heads(
    values_per_head: np.ndarray,
    head_weights: np.ndarray | None = None,
) -> tuple[float, float]:
    """Return (contribution_weighted_mean, unweighted_mean) across heads (§2.3)."""
    vals = np.asarray(values_per_head, dtype=np.float64)
    unweighted = float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else float("nan")
    if head_weights is None:
        return unweighted, unweighted
    w = np.asarray(head_weights, dtype=np.float64)
    valid = ~np.isnan(vals)
    if not np.any(valid):
        return float("nan"), float("nan")
    w_valid = w[valid]
    w_sum = w_valid.sum()
    weighted = float((w_valid * vals[valid]).sum() / w_sum) if w_sum > 0 else unweighted
    return weighted, unweighted


def classify_qii_outcome(
    qii_s4: float,
    entropy_norm: float,
    entropy_thresh: float = ENTROPY_ELIGIBILITY_THRESHOLD,
) -> str:
    """Guide Fix 2 / §2.5 / §7.1: separate degenerate uniform invariance from genuine invariance."""
    if np.isnan(qii_s4) or np.isnan(entropy_norm):
        return "undefined"
    if entropy_norm > entropy_thresh:
        return "degenerate_invariance"
    return "eligible"


def compute_qii_image(
    q_g: np.ndarray | torch.Tensor,
    k_patch: np.ndarray | torch.Tensor,
    scale: float,
    grid: tuple[int, int],
    *,
    q_global_pool: np.ndarray | torch.Tensor | None = None,
    image_idx: int = 0,
    q_patches: np.ndarray | torch.Tensor | None = None,
    pca: QueryPCA | None = None,
    head_weights: np.ndarray | torch.Tensor | None = None,
    gt_mask: np.ndarray | None = None,
    surrogates: Sequence[str] = SURROGATES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    is_map_head: bool = False,
    meta: Mapping | None = None,
) -> dict:
    """Compute Measure M1 (QII) + all §2.3 mandatory controls for a single image at one block.

    Args:
        q_g:           (H, d_h) true global query for this image.
        k_patch:       (H, N, d_h) patch keys ONLY (CLS/registers/dist already excluded).
        scale:         attention scale factor from adapter (e.g. 1 / sqrt(d_h)).
        grid:          (G_h, G_w) patch grid shape with G_h * G_w == N.
        q_global_pool: optional (I, H, d_h) global queries across dataset (required for S2).
        image_idx:     index of this image inside q_global_pool (excluded from S2 draws).
        q_patches:     optional (H, N, d_h) patch queries from the same image (required for S3).
        pca:           optional QueryPCA fit for this (model, layer) (required for S4).
        head_weights:  optional (H,) contribution weights w_h from compute_head_weights().
        gt_mask:       optional (H_img, W_img) binary foreground mask for threshold-free box-AUC.
        surrogates:    surrogate families to evaluate (default S1..S4, always reported separately).
        seeds:         random seeds per surrogate family (default 0..4).
        is_map_head:   True for MAP-head models (fixed learned probe -> S2/S4 excluded).
        meta:          optional dict with model_id, objective, set, layer, n_blocks, image_id, resolution
                       for emitting §2.6 long-format schema rows.

    Returns:
        dict with:
          - a_true:           (H, N) fp32 patch-only attention probabilities
          - entropy_norm:     (H,) normalized entropy H(a_true) / log(N)
          - entropy_weighted: float, head-weighted normalized entropy
          - box_auc_per_head: (H,) ROC-AUC per head (NaN if gt_mask is None)
          - box_auc_weighted: float, ROC-AUC of head-weighted attention map (NaN if gt_mask is None)
          - head_weights:     (H,) w_h (uniform if not supplied)
          - per_surrogate:    {S: {"jsd": (n_seeds, H), "qii": (n_seeds, H),
                                   "qii_per_head": (H,), "qii_weighted": float,
                                   "qii_unweighted": float, "excluded": bool}}
          - status_s4:        "eligible" | "degenerate_invariance" | "undefined"
          - records:          list of §2.6 long-format dicts (when meta is provided)
    """
    q_t = q_g.detach().cpu().float() if isinstance(q_g, torch.Tensor) else torch.as_tensor(q_g, dtype=torch.float32)
    k_t = k_patch.detach().cpu().float() if isinstance(k_patch, torch.Tensor) else torch.as_tensor(k_patch, dtype=torch.float32)

    if q_t.ndim != 2 or k_t.ndim != 3:
        raise ValueError(f"Expected q_g (H, d_h) and k_patch (H, N, d_h); got {tuple(q_t.shape)}, {tuple(k_t.shape)}")
    H, N, d_h = k_t.shape
    if q_t.shape != (H, d_h):
        raise ValueError(f"q_g shape {tuple(q_t.shape)} incompatible with k_patch {tuple(k_t.shape)}")
    if grid[0] * grid[1] != N:
        raise ValueError(f"grid {grid} product ({grid[0] * grid[1]}) != N ({N})")

    q_np = q_t.numpy()
    pool_np = None
    if q_global_pool is not None:
        pool_np = q_global_pool.detach().cpu().numpy() if isinstance(q_global_pool, torch.Tensor) else np.asarray(q_global_pool)
    qpat_np = None
    if q_patches is not None:
        qpat_np = q_patches.detach().cpu().numpy() if isinstance(q_patches, torch.Tensor) else np.asarray(q_patches)

    if head_weights is not None:
        w_h = (head_weights.detach().cpu().numpy() if isinstance(head_weights, torch.Tensor) else np.asarray(head_weights)).astype(np.float64)
        w_sum = w_h.sum()
        w_h = w_h / w_sum if w_sum > 0 else np.full(H, 1.0 / H, dtype=np.float64)
    else:
        w_h = np.full(H, 1.0 / H, dtype=np.float64)

    # True patch-only attention in fp32 (§2.1)
    a_true = patch_attention(q_t, k_t, scale).numpy()  # (H, N)
    ent_per_head = normalized_entropy(a_true, axis=-1)  # (H,)
    ent_weighted, _ = aggregate_qii_heads(ent_per_head, w_h)

    # Threshold-free map quality: box-AUC of bilinearly upsampled true attention (§2.3)
    if gt_mask is not None:
        target_hw = gt_mask.shape[:2]
        a_grids = a_true.reshape(H, grid[0], grid[1])
        a_up = upsample_map_bilinear(a_grids, target_hw)  # (H, H_img, W_img)
        auc_per_head = np.array([box_auc(a_up[h], gt_mask) for h in range(H)], dtype=np.float64)
        agg_grid = np.tensordot(w_h, a_grids, axes=(0, 0))
        auc_weighted = box_auc(upsample_map_bilinear(agg_grid, target_hw), gt_mask)
    else:
        auc_per_head = np.full(H, np.nan, dtype=np.float64)
        auc_weighted = float("nan")

    n_seeds = len(seeds)
    per_surrogate: dict[str, dict] = {}
    records: list[dict] = []

    layer_val = int(meta["layer"]) if meta and "layer" in meta else -1
    n_blocks = int(meta["n_blocks"]) if meta and "n_blocks" in meta else 1
    # Guide conventions: blocks l = 1..L, so 0-based layer l maps to layer_frac = (l + 1) / L
    layer_frac = float(layer_val + 1) / float(n_blocks) if meta and "layer" in meta else float("nan")
    grid_str = f"{grid[0]}x{grid[1]}"

    for S in surrogates:
        jsd_arr = np.full((n_seeds, H), np.nan, dtype=np.float64)
        qii_arr = np.full((n_seeds, H), np.nan, dtype=np.float64)
        excluded = False

        for s_pos, seed in enumerate(seeds):
            if S == "S1":
                q_s = make_surrogate_s1(q_np, _seed_rng(seed, image_idx, 101))
            elif S == "S2":
                q_s = make_surrogate_s2(image_idx, pool_np, seed, n_seeds=n_seeds, is_map_head=is_map_head) if pool_np is not None else None
            elif S == "S3":
                q_s = make_surrogate_s3(qpat_np, image_idx, seed, n_seeds=n_seeds)
            elif S == "S4":
                q_s = make_surrogate_s4(q_np, pca, _seed_rng(seed, image_idx, 404), is_map_head=is_map_head)
            else:
                raise ValueError(f"Unknown surrogate family {S!r}; expected one of {SURROGATES}")

            if q_s is None:
                excluded = True
                continue

            # Guard partial NaN heads (e.g. a single zero-variance head in S4)
            valid_heads = ~np.any(np.isnan(q_s), axis=-1)
            if not np.any(valid_heads):
                excluded = True
                continue

            q_s_safe = np.where(np.isnan(q_s), 0.0, q_s)
            a_surr = patch_attention(torch.from_numpy(q_s_safe), k_t, scale).numpy()  # (H, N)
            # Compute JSD strictly PER HEAD (Guide §2.4: never average across heads before JSD)
            jsd_h = jsd_nats(a_true, a_surr, axis=-1)  # (H,)
            qii_h = 1.0 - jsd_h / LOG2
            jsd_h = np.where(valid_heads, jsd_h, np.nan)
            qii_h = np.where(valid_heads, qii_h, np.nan)

            jsd_arr[s_pos] = jsd_h
            qii_arr[s_pos] = qii_h

            if meta is not None:
                for h in range(H):
                    records.append(
                        {
                            "model_id": meta.get("model_id"),
                            "objective": meta.get("objective"),
                            "set": meta.get("set"),
                            "layer": layer_val,
                            "layer_frac": layer_frac,
                            "head": h,
                            "image_id": meta.get("image_id", image_idx),
                            "surrogate": S,
                            "seed": int(seed),
                            "jsd": float(jsd_h[h]),
                            "qii": float(qii_h[h]),
                            "entropy_norm": float(ent_per_head[h]),
                            "box_auc": float(auc_per_head[h]),
                            "head_weight": float(w_h[h]),
                            "resolution": meta.get("resolution"),
                            "grid": grid_str,
                        }
                    )

        if excluded and np.all(np.isnan(qii_arr)):
            qii_per_head = np.full(H, np.nan, dtype=np.float64)
            qii_w, qii_u = float("nan"), float("nan")
        else:
            qii_per_head = np.nanmean(qii_arr, axis=0)
            qii_w, qii_u = aggregate_qii_heads(qii_per_head, w_h)

        per_surrogate[S] = {
            "jsd": jsd_arr,
            "qii": qii_arr,
            "qii_per_head": qii_per_head,
            "qii_weighted": qii_w,
            "qii_unweighted": qii_u,
            "excluded": excluded and bool(np.all(np.isnan(qii_arr))),
        }

    s4_val = per_surrogate["S4"]["qii_weighted"] if "S4" in per_surrogate else float("nan")
    return {
        "a_true": a_true,
        "entropy_norm": ent_per_head,
        "entropy_weighted": ent_weighted,
        "box_auc_per_head": auc_per_head,
        "box_auc_weighted": auc_weighted,
        "head_weights": w_h,
        "per_surrogate": per_surrogate,
        "status_s4": classify_qii_outcome(s4_val, ent_weighted),
        "records": records,
    }


def shuffled_key_null_check(
    q_g: np.ndarray | torch.Tensor,
    k_patch: np.ndarray | torch.Tensor,
    scale: float,
    grid: tuple[int, int],
    *,
    seed: int = 0,
    tol: float = 1e-6,
    **kwargs,
) -> tuple[bool, float]:
    """Guide §2.3 shuffled-key null sanity check: spatially permuting k_i must leave QII invariant."""
    k_np = k_patch.detach().cpu().numpy() if isinstance(k_patch, torch.Tensor) else np.asarray(k_patch)
    N = k_np.shape[1]
    perm = np.random.default_rng(seed).permutation(N)
    k_shuf = k_np[:, perm, :]

    res_orig = compute_qii_image(q_g, k_np, scale, grid, **kwargs)
    res_shuf = compute_qii_image(q_g, k_shuf, scale, grid, **kwargs)

    max_diff = 0.0
    for S, d_orig in res_orig["per_surrogate"].items():
        if d_orig["excluded"]:
            continue
        diff = np.nanmax(np.abs(d_orig["qii"] - res_shuf["per_surrogate"][S]["qii"]))
        if np.isfinite(diff):
            max_diff = max(max_diff, float(diff))
    return max_diff < tol, max_diff


@torch.no_grad()
def extract_m1_inputs_from_adapter(adapter, model, x: torch.Tensor, l: int) -> dict:
    """Live helper: extract patch-sliced q_g, k_patch, q_patches, head_weights, scale, grid from an adapter."""
    layout = adapter.token_layout(model, x) if adapter.family == "timm_vit" else adapter.token_layout(model)
    patch_idx = layout["patch_idx"]
    is_map_head = layout["cls_idx"] is None

    if is_map_head:
        q_g = adapter.get_global_query(model, x, l)
        k_full, v_full = adapter.get_keys_values(model, x, l)
        attn_full = adapter.attention_probs(model, x, l)  # (B, H, T)
        scale = float(model.attn_pool.scale)
        q_patches = None
    else:
        q_full, k_full, v_full = adapter.get_qkv(model, x, l)
        cls_idx = layout["cls_idx"]
        q_g = q_full[:, :, cls_idx, :]
        q_patches = q_full[:, :, patch_idx, :]
        scale = float(adapter.attn_scale(model, l)) if hasattr(adapter, "attn_scale") else float(q_full.shape[-1] ** -0.5)
        attn_full = adapter.attention_probs(model, x, l)[:, :, cls_idx, :]  # (B, H, T)

    W_O, _, gamma = get_output_proj_and_layerscale(adapter, model, l)
    w_h = compute_head_weights(attn_full, v_full, W_O, gamma=gamma)
    k_patch = k_full[:, :, patch_idx, :]
    return {
        "q_g": q_g,
        "k_patch": k_patch,
        "q_patches": q_patches,
        "head_weights": w_h,
        "scale": scale,
        "grid": layout["grid"],
        "is_map_head": is_map_head,
    }
