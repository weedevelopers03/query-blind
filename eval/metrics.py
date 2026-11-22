"""Canonical localization & WSOL metrics (Methodology Guide §9).

Single source of truth for all localization metrics in the repo. Never reimplement inline.
Implements:
- Threshold-free audit metrics (Guide §9.1, tunes_on_boxes=False):
    * box_auc       : exact tie-aware pixel-level ROC AUC against binary GT mask
    * pxap          : exact tie-aware pixel-level Average Precision against binary GT mask
    * pointing_game : 1.0 if argmax pixel lies inside GT mask, else 0.0
- Box geometry & multi-instance matching (Guide §8.3 & §9.1):
    * bbox_iou, max_iou_any_box, box_hit
- Parameter-free & calibration-fixed box extractors (Guide §7.1 Step 9 & §9.2, tunes_on_boxes=False):
    * largest_cc_box_from_mask
    * otsu_threshold, otsu_box
    * calibrate_pseudo_box_threshold (Murtaza et al. WACV 2025 style: single scalar tau* fit on
      unlabelled calibration maps against Otsu pseudo-boxes; zero GT box leakage)
    * fixed_threshold_box
- Benchmark WSOL metrics:
    * corloc        : fraction of images with max_g IoU(pred, g) >= delta (default 0.5)
    * maxboxacc_v2  : Choe et al. CVPR 2020 swept-threshold multi-instance accuracy over
                      deltas=(0.3, 0.5, 0.7) (explicitly marked tunes_on_boxes=True per Guide §9.2 #3)
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.ndimage import label as cc_label
from scipy.stats import rankdata

DEFAULT_MAXBOXACC_DELTAS = (0.3, 0.5, 0.7)
DEFAULT_SWEEP_THRESHOLDS = tuple(np.linspace(0.05, 0.95, 19).tolist())


# --------------------------------------------------------------------------- Normalization & upsampling helpers
def minmax_normalize_map(smap: np.ndarray) -> np.ndarray:
    """Min-max normalize a 2D saliency map to [0, 1] in float64."""
    arr = np.asarray(smap, dtype=np.float64)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo <= 1e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - lo) / (hi - lo)


# --------------------------------------------------------------------------- Threshold-free metrics (§9.1)
def box_auc(saliency_map: np.ndarray, gt_mask: np.ndarray) -> float:
    """Exact tie-aware ROC AUC of a 2D saliency map against a binary GT mask (threshold-free)."""
    scores = np.asarray(saliency_map, dtype=np.float64).ravel()
    labels = np.asarray(gt_mask, dtype=bool).ravel()
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    pos_rank_sum = float(ranks[labels].sum())
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def pxap(saliency_map: np.ndarray, gt_mask: np.ndarray) -> float:
    """Exact tie-safe pixel-level Average Precision (stepwise PR area, matching sklearn average_precision_score)."""
    scores = np.asarray(saliency_map, dtype=np.float64).ravel()
    labels = np.asarray(gt_mask, dtype=bool).ravel()
    n_pos = int(labels.sum())
    if n_pos == 0 or n_pos == labels.size:
        return float("nan") if n_pos == 0 else 1.0

    # Sort descending; collapse ties so thresholded operating points match sklearn exactly
    order = np.argsort(-scores, kind="mergesort")
    s_sorted = scores[order]
    y_sorted = labels[order].astype(np.float64)

    distinct_end = np.where(np.diff(s_sorted) != 0)[0]
    threshold_idx = np.r_[distinct_end, y_sorted.size - 1]

    tps = np.cumsum(y_sorted)[threshold_idx]
    fps = (threshold_idx + 1).astype(np.float64) - tps
    prec = tps / np.maximum(tps + fps, 1.0)
    rec = tps / float(n_pos)

    rec_prev = np.r_[0.0, rec[:-1]]
    ap = float(np.sum((rec - rec_prev) * prec))
    return float(np.clip(ap, 0.0, 1.0))


def pointing_game(saliency_map: np.ndarray, gt_mask: np.ndarray) -> float:
    """Pointing game: 1.0 if argmax pixel lies inside gt_mask, else 0.0 (threshold-free)."""
    scores = np.asarray(saliency_map, dtype=np.float64).ravel()
    labels = np.asarray(gt_mask, dtype=bool).ravel()
    if not labels.any():
        return float("nan")
    return 1.0 if bool(labels[int(np.argmax(scores))]) else 0.0


# --------------------------------------------------------------------------- Bounding-box geometry
def bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Intersection-over-Union of two [x1, y1, x2, y2] boxes (float pixel coordinates)."""
    ax1, ay1, ax2, ay2 = (float(v) for v in box_a)
    bx1, by1, bx2, by2 = (float(v) for v in box_b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def max_iou_any_box(pred_box: Sequence[float], gt_boxes: Sequence[Sequence[float]]) -> float:
    """Multi-instance aware max IoU against all ground-truth boxes for an image (Guide §8.3)."""
    if len(gt_boxes) == 0:
        return float("nan")
    return float(max(bbox_iou(pred_box, g) for g in gt_boxes))


def box_hit(pred_box: Sequence[float], gt_boxes: Sequence[Sequence[float]], iou_thresh: float = 0.5) -> float:
    """1.0 if max IoU with any GT box >= iou_thresh, else 0.0."""
    m = max_iou_any_box(pred_box, gt_boxes)
    if not np.isfinite(m):
        return float("nan")
    return 1.0 if m >= iou_thresh else 0.0


# --------------------------------------------------------------------------- Mask -> largest-CC tight box
def largest_cc_box_from_mask(mask: np.ndarray) -> tuple[float, float, float, float]:
    """Tight [x1, y1, x2, y2] box of the largest 8-connected foreground component (Guide §7.1 Step 9).

    Uses pixel-extent convention [x_min, y_min, x_max + 1, y_max + 1] matching box_to_pixel_mask in preprocess.py.
    Falls back to full image box if mask is empty.
    """
    m = np.asarray(mask, dtype=bool)
    H, W = m.shape
    if not m.any():
        return (0.0, 0.0, float(W), float(H))
    structure = np.ones((3, 3), dtype=np.int32)
    labeled, n_comp = cc_label(m, structure=structure)
    if n_comp == 0:
        return (0.0, 0.0, float(W), float(H))
    counts = np.bincount(labeled.ravel())
    counts[0] = 0  # exclude background
    best_cc = int(np.argmax(counts))
    ys, xs = np.where(labeled == best_cc)
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def otsu_threshold(values: np.ndarray, n_bins: int = 256) -> float:
    """Standard parameter-free Otsu threshold in pure NumPy."""
    v = np.asarray(values, dtype=np.float64).ravel()
    vmin, vmax = float(v.min()), float(v.max())
    if vmax - vmin <= 1e-12:
        return vmax
    hist, bin_edges = np.histogram(v, bins=n_bins, range=(vmin, vmax))
    hist = hist.astype(np.float64)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    w1 = np.cumsum(hist)
    w2 = np.cumsum(hist[::-1])[::-1]
    mu1 = np.cumsum(hist * bin_centers) / np.maximum(w1, 1e-12)
    mu2 = (np.cumsum((hist * bin_centers)[::-1]) / np.maximum(w2[::-1], 1e-12))[::-1]
    between = w1[:-1] * w2[1:] * (mu1[:-1] - mu2[1:]) ** 2
    idx = int(np.argmax(between))
    return float(bin_centers[idx])


def otsu_box(saliency_map: np.ndarray) -> tuple[tuple[float, float, float, float], np.ndarray, float]:
    """Guide §7.1 Step 9: Otsu(S) -> largest connected component -> tight box (tunes_on_boxes=False)."""
    sn = minmax_normalize_map(saliency_map)
    thr = otsu_threshold(sn)
    mask = sn >= thr
    box = largest_cc_box_from_mask(mask)
    return box, mask, float(thr)


def fixed_threshold_box(saliency_map: np.ndarray, threshold: float) -> tuple[float, float, float, float]:
    """Extract tight largest-CC box at a fixed threshold tau in [0, 1] on a min-max normalized map."""
    sn = minmax_normalize_map(saliency_map)
    return largest_cc_box_from_mask(sn >= float(threshold))


def _strided_cc_box(normed_map: np.ndarray, threshold: float, max_cc_side: int = 56) -> tuple[float, float, float, float]:
    """Fast tight largest-CC box using integer subsampling when map side >= 2 * max_cc_side."""
    H, W = normed_map.shape
    stride = max(1, min(H, W) // max_cc_side)
    if stride <= 1:
        return largest_cc_box_from_mask(normed_map >= threshold)
    sub = normed_map[::stride, ::stride]
    sx1, sy1, sx2, sy2 = largest_cc_box_from_mask(sub >= threshold)
    return (
        min(float(W), sx1 * stride),
        min(float(H), sy1 * stride),
        min(float(W), sx2 * stride),
        min(float(H), sy2 * stride),
    )


def calibrate_pseudo_box_threshold(
    cal_maps: Sequence[np.ndarray],
    pseudo_boxes: Sequence[Sequence[float]] | None = None,
    candidate_thresholds: Sequence[float] = DEFAULT_SWEEP_THRESHOLDS,
    iou_thresh: float = 0.5,
) -> float:
    """Murtaza et al. (WACV 2025) style realistic threshold calibration (tunes_on_boxes=False).

    Selects a single scalar binarization threshold tau* on unlabelled calibration maps against
    noisy pseudo-boxes (if pseudo_boxes is None, uses per-image Otsu largest-CC boxes as self-generated
    label-free pseudo-boxes; zero GT boxes are ever touched).
    """
    if len(cal_maps) == 0:
        return 0.5
    normed = [minmax_normalize_map(m) for m in cal_maps]
    if pseudo_boxes is None:
        pboxes = [otsu_box(m)[0] for m in normed]
    else:
        pboxes = [tuple(b) for b in pseudo_boxes]

    best_tau = 0.5
    best_score = -1.0
    for tau in candidate_thresholds:
        t_val = float(tau)
        hits = []
        for m, pb in zip(normed, pboxes):
            box = _strided_cc_box(m, t_val)
            hits.append(1.0 if bbox_iou(box, pb) >= iou_thresh else 0.0)
        sc = float(np.mean(hits))
        if sc > best_score:
            best_score = sc
            best_tau = t_val
    return best_tau


# --------------------------------------------------------------------------- Dataset-level WSOL metrics
def corloc(
    pred_boxes: Sequence[Sequence[float]],
    gt_boxes_list: Sequence[Sequence[Sequence[float]]],
    iou_thresh: float = 0.5,
) -> tuple[float, np.ndarray]:
    """CorLoc (multi-instance aware): fraction of images with max IoU >= iou_thresh."""
    hits = np.array(
        [box_hit(pb, gbs, iou_thresh=iou_thresh) for pb, gbs in zip(pred_boxes, gt_boxes_list)],
        dtype=np.float64,
    )
    return float(np.nanmean(hits)), hits


def maxboxacc_v2(
    saliency_maps: Sequence[np.ndarray],
    gt_boxes_list: Sequence[Sequence[Sequence[float]]],
    thresholds: Sequence[float] = DEFAULT_SWEEP_THRESHOLDS,
    deltas: Sequence[float] = DEFAULT_MAXBOXACC_DELTAS,
) -> dict:
    """MaxBoxAccV2 (Choe et al., CVPR 2020): mean over delta in {0.3, 0.5, 0.7} of best-threshold box accuracy.

    NOTE (Guide §9.2 #3): tunes_on_boxes = True (threshold is swept on evaluation GT boxes). Always label clearly!
    """
    n = len(saliency_maps)
    if n == 0:
        return {"maxboxacc_v2": float("nan"), "by_delta": {}, "tunes_on_boxes": True}

    normed = [minmax_normalize_map(m) for m in saliency_maps]
    # Compute max IoU per (threshold, image) once so sweeping all 3 deltas is fast
    iou_grid = np.empty((len(thresholds), n), dtype=np.float64)
    for ti, tau in enumerate(thresholds):
        t_val = float(tau)
        for i in range(n):
            pbox = _strided_cc_box(normed[i], t_val)
            iou_grid[ti, i] = max_iou_any_box(pbox, gt_boxes_list[i])

    by_delta = {}
    best_accs = []
    for d in deltas:
        accs = np.nanmean(iou_grid >= float(d), axis=1)
        best_idx = int(np.argmax(accs))
        best_acc = float(accs[best_idx])
        best_accs.append(best_acc)
        by_delta[f"{d:.1f}"] = {
            "acc": best_acc,
            "best_tau": float(thresholds[best_idx]),
            "hits": (iou_grid[best_idx] >= float(d)).astype(np.float64),
        }

    return {
        "maxboxacc_v2": float(np.mean(best_accs)),
        "by_delta": by_delta,
        "tunes_on_boxes": True,
    }
