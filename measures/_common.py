"""Shared fp32/fp64 primitives for Measures M1-M3 (Methodology Guide §§2-4, §12).

Enforced invariants:
  * Softmax is always computed in fp32 over patch tokens only (never including [CLS]/registers/dist).
  * JSD is computed in natural log (base e, bounded in [0, log 2]), matching
    scipy.spatial.distance.jensenshannon(p, q, base=np.e)**2 (squared, NOT distance).
  * Map upsampling uses bilinear interpolation (never nearest).
  * Head weights w_h include W_O and LayerScale gamma when present (§2.3, §4.1, §12).
"""
from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata

LOG2 = float(np.log(2.0))


def patch_attention(q: torch.Tensor, k_patch: torch.Tensor, scale: float) -> torch.Tensor:
    """Compute softmax(q @ k_patch^T * scale) in fp32 over the patch dimension only.

    Args:
        q:       (..., H, d_h) or (..., H, Q, d_h)
        k_patch: (..., H, N, d_h) — patch keys ONLY (CLS/registers/dist already excluded).
        scale:   attention scale factor from adapter (typically 1 / sqrt(d_h)).
    Returns:
        Attention probabilities in float32 summing to 1 along the patch axis:
        (..., H, N) if q had shape (..., H, d_h), else (..., H, Q, N).
    """
    squeeze_q = q.ndim == k_patch.ndim - 1
    q_f = q.float().unsqueeze(-2) if squeeze_q else q.float()
    k_f = k_patch.float()
    logits = (q_f @ k_f.transpose(-2, -1)) * float(scale)
    probs = logits.softmax(dim=-1)
    return probs.squeeze(-2) if squeeze_q else probs


def jsd_nats(p: np.ndarray, q: np.ndarray, axis: int = -1) -> np.ndarray:
    """Jensen-Shannon divergence in nats (base e, range [0, log 2]).

    Numerically identical to scipy.spatial.distance.jensenshannon(p, q, base=np.e)**2,
    vectorized over leading dimensions in float64.
    """
    p64 = np.asarray(p, dtype=np.float64)
    q64 = np.asarray(q, dtype=np.float64)
    # Guard against tiny numerical drift in inputs while verifying valid distributions
    p_sum = p64.sum(axis=axis, keepdims=True)
    q_sum = q64.sum(axis=axis, keepdims=True)
    p64 = p64 / np.where(p_sum > 0, p_sum, 1.0)
    q64 = q64 / np.where(q_sum > 0, q_sum, 1.0)

    m = 0.5 * (p64 + q64)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_p = np.where(p64 > 0.0, p64 * np.log(p64 / m), 0.0).sum(axis=axis)
        kl_q = np.where(q64 > 0.0, q64 * np.log(q64 / m), 0.0).sum(axis=axis)
    jsd = 0.5 * (kl_p + kl_q)
    return np.clip(jsd, 0.0, LOG2)


def normalized_entropy(a: np.ndarray, axis: int = -1) -> np.ndarray:
    """Normalized Shannon entropy H(a) / log(N) in [0, 1] (Guide Fix 2 & §2.3)."""
    a64 = np.asarray(a, dtype=np.float64)
    n = a64.shape[axis]
    if n <= 1:
        return np.zeros(a64.shape[:axis] + a64.shape[axis + 1 :], dtype=np.float64)
    a_sum = a64.sum(axis=axis, keepdims=True)
    a64 = a64 / np.where(a_sum > 0, a_sum, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.where(a64 > 0.0, a64 * np.log(a64), 0.0).sum(axis=axis)
    return np.clip(ent / np.log(float(n)), 0.0, 1.0)


def upsample_map_bilinear(saliency_grid: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Bilinearly upsample a (G, G) or (..., G, G) saliency grid to target_hw=(H, W)."""
    arr = np.asarray(saliency_grid, dtype=np.float32)
    orig_shape = arr.shape
    if arr.ndim == 2:
        t = torch.from_numpy(arr)[None, None]
        out = F.interpolate(t, size=target_hw, mode="bilinear", align_corners=False)[0, 0]
        return out.numpy()
    gh, gw = orig_shape[-2:]
    flat = arr.reshape(-1, 1, gh, gw)
    out = F.interpolate(torch.from_numpy(flat), size=target_hw, mode="bilinear", align_corners=False)
    return out.numpy().reshape(*orig_shape[:-2], target_hw[0], target_hw[1])


def box_auc(saliency_map: np.ndarray, gt_mask: np.ndarray) -> float:
    """Exact threshold-free ROC-AUC of pixel saliency scores against a binary FG mask.

    Uses Mann-Whitney U with average tie ranks (identical to sklearn.metrics.roc_auc_score).
    Returns NaN if gt_mask is all-0 or all-1.
    """
    scores = np.asarray(saliency_map, dtype=np.float64).ravel()
    labels = (np.asarray(gt_mask) > 0).ravel()
    if scores.shape != labels.shape:
        raise ValueError(f"shape mismatch: saliency {saliency_map.shape} vs gt_mask {gt_mask.shape}")
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    u_stat = ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u_stat / (n_pos * n_neg))


@torch.no_grad()
def box_auc_torch(saliency_maps: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
    """Vectorized GPU/CPU exact tie-aware ROC-AUC matching box_auc() / scipy rankdata('average').

    Args:
        saliency_maps: (..., H, W) float tensor on CPU or CUDA.
        gt_masks:      (..., H, W) broadcastable binary tensor (same device or moved automatically).
    Returns:
        (...,) float64 tensor of ROC-AUC values (NaN where mask is all-0 or all-1).
    """
    orig_shape = saliency_maps.shape[:-2]
    dev = saliency_maps.device
    P = saliency_maps.shape[-2] * saliency_maps.shape[-1]
    s = saliency_maps.reshape(-1, P).float()
    m = (gt_masks.to(dev) > 0).expand_as(saliency_maps).reshape(-1, P)

    vals, idx = torch.sort(s, dim=-1)
    lbl = m.gather(-1, idx)
    pos = torch.arange(P, device=dev, dtype=torch.int64).unsqueeze(0).expand_as(vals)

    # Exact average rank for tied values: 0.5 * (first_idx + last_idx) + 1
    is_start = torch.ones_like(lbl, dtype=torch.bool)
    is_start[:, 1:] = vals[:, 1:] != vals[:, :-1]
    start_pos = torch.where(is_start, pos, torch.zeros_like(pos)).cummax(dim=-1).values

    is_end = torch.ones_like(lbl, dtype=torch.bool)
    is_end[:, :-1] = vals[:, :-1] != vals[:, 1:]
    rev_pos = torch.where(is_end, pos, torch.full_like(pos, P - 1)).flip(-1)
    # cummin from the right gives exact last index of each tie group
    end_pos = rev_pos.cummin(dim=-1).values.flip(-1)

    avg_rank = 0.5 * (start_pos.to(torch.float64) + end_pos.to(torch.float64)) + 1.0
    n_pos = lbl.sum(dim=-1).to(torch.float64)
    n_neg = float(P) - n_pos
    rank_sum = (avg_rank * lbl.to(torch.float64)).sum(dim=-1)
    u = rank_sum - n_pos * (n_pos + 1.0) / 2.0
    denom = n_pos * n_neg
    auc = torch.where(denom > 0.0, u / denom, torch.full_like(denom, float("nan")))
    return auc.reshape(orig_shape)


@torch.no_grad()
def jsd_nats_torch(p: torch.Tensor, q: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Exact fp64 Jensen-Shannon divergence in nats on CPU or CUDA (matches jsd_nats to <1e-12)."""
    p64 = p.to(torch.float64)
    q64 = q.to(torch.float64)
    p_sum = p64.sum(dim=dim, keepdim=True)
    q_sum = q64.sum(dim=dim, keepdim=True)
    p64 = p64 / torch.where(p_sum > 0, p_sum, torch.ones_like(p_sum))
    q64 = q64 / torch.where(q_sum > 0, q_sum, torch.ones_like(q_sum))

    m = 0.5 * (p64 + q64)
    kl_p = torch.where(p64 > 0.0, p64 * torch.log(p64 / m.clamp_min(1e-300)), torch.zeros_like(p64)).sum(dim=dim)
    kl_q = torch.where(q64 > 0.0, q64 * torch.log(q64 / m.clamp_min(1e-300)), torch.zeros_like(q64)).sum(dim=dim)
    return (0.5 * (kl_p + kl_q)).clamp(0.0, LOG2)


@torch.no_grad()
def normalized_entropy_torch(a: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Exact fp64 normalized entropy H(a) / log(N) on CPU or CUDA."""
    a64 = a.to(torch.float64)
    n = a64.shape[dim]
    if n <= 1:
        return torch.zeros(a64.shape[:dim] + a64.shape[dim + 1 :], dtype=torch.float64, device=a64.device)
    a_sum = a64.sum(dim=dim, keepdim=True)
    a64 = a64 / torch.where(a_sum > 0, a_sum, torch.ones_like(a_sum))
    ent = -torch.where(a64 > 0.0, a64 * torch.log(a64.clamp_min(1e-300)), torch.zeros_like(a64)).sum(dim=dim)
    return (ent / float(np.log(float(n)))).clamp(0.0, 1.0)


def boxes_to_mask(boxes_xyxy: Sequence[Sequence[float]], hw: tuple[int, int]) -> np.ndarray:
    """Rasterize 0-based [x1, y1, x2, y2) boxes into a binary (H, W) mask."""
    h, w = hw
    mask = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes_xyxy:
        ix1 = max(0, min(w, int(np.floor(x1))))
        iy1 = max(0, min(h, int(np.floor(y1))))
        ix2 = max(0, min(w, int(np.ceil(x2))))
        iy2 = max(0, min(h, int(np.ceil(y2))))
        if ix2 > ix1 and iy2 > iy1:
            mask[iy1:iy2, ix1:ix2] = 1
    return mask


def get_output_proj_and_layerscale(
    adapter, model, l: int
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Extract (W_O, b_O, gamma) in fp32 for block l across all 4 adapter families.

    W_O has shape (D_out, H * d_h), b_O has shape (D_out,) or None, gamma has shape (D_out,) or None.
    """
    fam = getattr(adapter, "family", "")
    if fam == "timm_vit":
        blk = model.blocks[l]
        if not isinstance(blk.attn.norm, torch.nn.Identity):
            raise NotImplementedError(f"non-Identity attn.norm ({type(blk.attn.norm)}) not supported for head split")
        W_O = blk.attn.proj.weight.detach().float()
        b_O = blk.attn.proj.bias.detach().float() if blk.attn.proj.bias is not None else None
        gamma = blk.ls1.gamma.detach().float() if hasattr(blk.ls1, "gamma") else None
        return W_O, b_O, gamma
    if fam == "open_clip_vit":
        vt = model.visual if hasattr(model, "visual") else model
        blk = vt.transformer.resblocks[l]
        W_O = blk.attn.out_proj.weight.detach().float()
        b_O = blk.attn.out_proj.bias.detach().float() if blk.attn.out_proj.bias is not None else None
        gamma = blk.ls_1.gamma.detach().float() if hasattr(blk.ls_1, "gamma") else None
        return W_O, b_O, gamma
    if fam == "map_head":
        ap = model.attn_pool
        W_O = ap.proj.weight.detach().float()
        b_O = ap.proj.bias.detach().float() if ap.proj.bias is not None else None
        return W_O, b_O, None
    if fam == "dinov3_hf":
        layer = model.model.layer[l]
        W_O = layer.attention.o_proj.weight.detach().float()
        b_O = layer.attention.o_proj.bias.detach().float() if layer.attention.o_proj.bias is not None else None
        ls = getattr(layer, "layer_scale1", None)
        gamma = None
        if ls is not None:
            for attr in ("lambda1", "gamma"):
                if hasattr(ls, attr):
                    gamma = getattr(ls, attr).detach().float()
                    break
        return W_O, b_O, gamma
    raise ValueError(f"Unsupported adapter family for W_O / LayerScale extraction: {fam!r}")


@torch.no_grad()
def compute_head_weights(
    attn_row_full: torch.Tensor,
    v_full: torch.Tensor,
    W_O: torch.Tensor,
    gamma: torch.Tensor | None = None,
    return_vectors: bool = False,
):
    """Guide §2.3 & §3.4 head contribution weights w_h = ||gamma * W_O^(h) (sum_i a_i^h v_i^h)||_2 / sum_h ||.||_2.

    Args:
        attn_row_full: (..., H, T) global-token attention weights over ALL T tokens.
        v_full:        (..., H, T, d_h) value vectors over ALL T tokens.
        W_O:           (D, H * d_h) output projection weight matrix.
        gamma:         optional (D,) LayerScale vector.
    Returns:
        weights (..., H) summing to 1 along head dim (and optionally head_vecs (..., H, D)).
    """
    a = attn_row_full.float()
    v = v_full.float()
    H, d_h = v.shape[-3], v.shape[-1]
    # Per-head value aggregate: (..., H, d_h)
    ctx = torch.einsum("...ht,...htd->...hd", a, v)
    # Reshape W_O from (D, H*d_h) -> (H, D, d_h)
    W_h = W_O.float().to(ctx.device).reshape(W_O.shape[0], H, d_h).permute(1, 0, 2)
    head_vecs = torch.einsum("...hd,hDd->...hD", ctx, W_h)
    if gamma is not None:
        head_vecs = head_vecs * gamma.float().to(ctx.device)
    norms = head_vecs.norm(dim=-1)  # (..., H)
    denom = norms.sum(dim=-1, keepdim=True)
    weights = torch.where(denom > 0, norms / denom, torch.full_like(norms, 1.0 / H))
    if return_vectors:
        return weights, head_vecs
    return weights


def atomic_save_npy(array: np.ndarray, path: str | Path) -> Path:
    """Write a .npy file atomically via temp file + rename (Guide §12: no half-written shards)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp.npy")
    os.close(fd)
    try:
        np.save(tmp_name, array)
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise
    return path


def write_records(records: Iterable[Mapping], path: str | Path, columns: Sequence[str] | None = None) -> Path:
    """Write long-format records to Parquet (if pandas/pyarrow available) or CSV/JSONL atomically."""
    rows = list(records)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        import pandas as pd  # raises cleanly if parquet requested without pandas

        df = pd.DataFrame(rows, columns=list(columns) if columns else None)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp.parquet")
        os.close(fd)
        try:
            df.to_parquet(tmp_name, index=False)
            os.replace(tmp_name, path)
        except Exception:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
            raise
        return path

    if suffix == ".jsonl":
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp.jsonl")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            os.replace(tmp_name, path)
        except Exception:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
            raise
        return path

    # Default: CSV
    fieldnames = list(columns) if columns else (list(rows[0].keys()) if rows else [])
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp.csv")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise
    return path
