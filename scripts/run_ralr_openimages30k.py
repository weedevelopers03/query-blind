#!/usr/bin/env python3
"""Action A1 (§7, Gate G3): Fast GPU-vectorized RALR + PxAP + box-AUC + Otsu evaluation on OpenImages30k (5,000 masks).

Implements Guide §§7.1, 8.1, 9.1–9.2 & Supervisor Review (4) Action A1 + §4.4:
  - Unlabelled calibration (l*, branch, QII_S4, omega^2_key, H_norm) on X_cal = 1,000 unlabelled images (seed=0).
  - Evaluates BOTH full test set (4,999 images) AND strictly disjoint non-calibration split (3,999 images).
  - Exact GPU-vectorized box-AUC and tie-safe PxAP (verified < 1e-5 vs eval/metrics.py::pxap).
  - Reports Otsu CorLoc@0.5 alongside mean Otsu foreground fraction and degenerate rate (outside [0.02, 0.80], §4.4).
  - Supports --resume per model so Colab runs are 100% crash-safe.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import binomtest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.base import attn_probs
from eval import metrics as M
from eval.preprocess import build_image_transform, transform_mask_resize_center_crop
from extract.cache import last_third_blocks
from measures import keydom, qii, routing
from measures._common import (
    box_auc_torch,
    compute_head_weights,
    get_output_proj_and_layerscale,
    patch_attention,
)
from ralr import readout as R
from scripts.run_pilot_m1_m2 import (
    _git_sha,
    bootstrap_ci,
    load_pilot_model,
    paired_bootstrap_diff_ci,
    resolve_model_ids,
    score_layer_gpu,
)
from scripts.run_pilot_ralr import extract_batch_ralr

METHODS_OI = (
    "cls_last",
    "cls_l_star",
    "cls_l_star_w",
    "cls_l_star_clip98",
    "cls_l_star_no_out",
    "lost_l_star",
    "ralr_adaptive",
)


def pxap_torch_batch(saliency_maps: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
    """Exact tie-safe pixel-level Average Precision on GPU/CPU (matches eval/metrics.py::pxap to <1e-5).

    Args:
        saliency_maps: (B, H, W) float tensor
        gt_masks:      (B, H, W) bool/float tensor
    Returns:
        (B,) float64 tensor of per-image PxAP (nan where gt_mask has 0 foreground pixels).
    """
    B = saliency_maps.shape[0]
    scores = saliency_maps.reshape(B, -1).to(torch.float64)
    labels = (gt_masks.reshape(B, -1) > 0.5).to(torch.float64)
    out = torch.full((B,), float("nan"), dtype=torch.float64, device=scores.device)

    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    s_sorted = torch.gather(scores, 1, order)
    y_sorted = torch.gather(labels, 1, order)
    n_pos = labels.sum(dim=1)  # (B,)
    P = scores.shape[1]

    for b in range(B):
        np_b = float(n_pos[b].item())
        if np_b <= 0.0:
            continue
        if np_b >= P:
            out[b] = 1.0
            continue
        sb = s_sorted[b]
        yb = y_sorted[b]
        diff = sb[1:] != sb[:-1]
        distinct_end = torch.nonzero(diff, as_tuple=False).squeeze(-1)
        last_idx = torch.tensor([P - 1], device=scores.device, dtype=torch.long)
        thr_idx = torch.cat([distinct_end, last_idx], dim=0)
        tps = torch.cumsum(yb, dim=0)[thr_idx]
        prec = tps / (thr_idx.to(torch.float64) + 1.0)
        rec = tps / np_b
        rec_prev = torch.cat([torch.zeros(1, device=scores.device, dtype=torch.float64), rec[:-1]], dim=0)
        out[b] = torch.clamp(torch.sum((rec - rec_prev) * prec), 0.0, 1.0)
    return out


def _mask_to_tight_box(mask_bool: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.nonzero(mask_bool)
    if ys.size == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def load_openimages_pairs(img_dir: Path, mask_dir: Path, limit: int | None = None, seed: int = 0) -> list[dict]:
    stem_to_img = {p.stem: p for p in sorted(img_dir.glob("*")) if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    stem_to_msk = {p.stem: p for p in sorted(mask_dir.glob("*.png")) if p.is_file()}
    matched = sorted(s for s in stem_to_img if s in stem_to_msk)
    if not matched:
        raise FileNotFoundError(f"No matching image/mask stems in {img_dir} and {mask_dir}")
    if limit is not None and limit < len(matched):
        rng = np.random.default_rng(int(seed))
        idx = np.sort(rng.choice(len(matched), size=int(limit), replace=False))
        matched = [matched[int(i)] for i in idx]
    return [{"id": s, "image_path": stem_to_img[s], "mask_path": stem_to_msk[s]} for s in matched]


def _upsample_norm_batch(grids: torch.Tensor, target_size: int) -> torch.Tensor:
    """Bilinearly upsample (B, gh, gw) -> (B, S, S) and min-max normalize per image."""
    up = F.interpolate(grids.unsqueeze(1).float(), size=(target_size, target_size), mode="bilinear", align_corners=False).squeeze(1)
    B = up.shape[0]
    flat = up.reshape(B, -1)
    mn = flat.min(dim=1, keepdim=True).values
    mx = flat.max(dim=1, keepdim=True).values
    span = torch.clamp(mx - mn, min=1e-12)
    return ((flat - mn) / span).reshape(B, target_size, target_size)


def _lost_continuous_batch(k_patch: torch.Tensor) -> torch.Tensor:
    """Vectorized continuous LOST seed-expansion map on (B, H, N, d_h) -> (B, N)."""
    B, H, N, d_h = k_patch.shape
    kf = k_patch.transpose(1, 2).reshape(B, N, H * d_h).float()
    kf = F.normalize(kf, dim=-1, eps=1e-8)
    A = torch.bmm(kf, kf.transpose(1, 2))  # (B, N, N)
    pos = (A > 0).float()
    deg = pos.sum(dim=-1)  # (B, N)
    # Seed = lowest-degree patch; S = seed + patches positively correlated with seed
    seed_idx = deg.argmin(dim=-1)  # (B,)
    row_seed = A[torch.arange(B, device=A.device), seed_idx]  # (B, N)
    in_S = (row_seed > 0).float()
    in_S[torch.arange(B, device=A.device), seed_idx] = 1.0
    score = torch.bmm(A, in_S.unsqueeze(-1)).squeeze(-1)  # (B, N)
    return torch.clamp(score, min=0.0)


@torch.no_grad()
def run_one_model_openimages(
    model_id: str,
    items: list[dict],
    gt_cache: dict[int, tuple],
    device: torch.device,
    out_dir: Path,
    weights_cache: Path,
    *,
    batch_size: int = 64,
    cal_n: int = 1000,
    tau_q: float = R.FROZEN_TAU_Q,
    seed: int = 0,
    smoke_random: bool = False,
) -> dict:
    t0 = time.perf_counter()
    adapter, model, mean, std, target_size = load_pilot_model(
        model_id, device, weights_cache, smoke_random=smoke_random,
    )
    tf = build_image_transform(target_size, mean, std)
    n_img = len(items)

    # Precompute transformed GT masks & tight boxes once per resolution (224 or 196)
    if target_size not in gt_cache:
        masks, boxes = [], []
        for it in items:
            with Image.open(it["mask_path"]) as mp:
                m = transform_mask_resize_center_crop(mp, target_size).astype(bool)
            masks.append(m)
            boxes.append([_mask_to_tight_box(m)])
        gt_cache[target_size] = (np.stack(masks, axis=0), boxes)
    gt_masks_np, gt_boxes_list = gt_cache[target_size]

    # Deterministic X_cal (1,000 unlabelled images, seed=0) vs disjoint eval split
    rng_cal = np.random.default_rng(int(seed))
    cal_k = min(int(cal_n), n_img)
    cal_idx = np.sort(rng_cal.choice(n_img, size=cal_k, replace=False))
    cal_mask_bool = np.zeros(n_img, dtype=bool)
    cal_mask_bool[cal_idx] = True
    heldout_idx = np.where(~cal_mask_bool)[0]

    layers = last_third_blocks(adapter.n_blocks(model))
    last_l = layers[-1]

    # Pass 1: Extract on X_cal (1,000 unlabelled images) to select l* & branch (zero GT used!)
    t_cal = time.perf_counter()
    cal_buf = {l: {k: [] for k in ("q_g", "k_patch", "q_patches", "cls_attn_patch", "head_weights")} for l in layers}
    scales, grids = {}, {}
    for st in range(0, len(cal_idx), batch_size):
        batch_ids = cal_idx[st : st + batch_size]
        xb = torch.stack([tf(Image.open(items[i]["image_path"]).convert("RGB")) for i in batch_ids], dim=0).to(device)
        ex = extract_batch_ralr(adapter, model, xb, layers)
        for l in layers:
            for k in cal_buf[l]:
                cal_buf[l][k].append(ex[l][k])
            scales[l] = ex[l]["scale"]
            grids[l] = ex[l]["grid"]

    layer_stats = {}
    dummy_masks = np.ones((len(cal_idx), target_size, target_size), dtype=np.uint8)
    dummy_masks[:, 0, 0] = 0
    for l in layers:
        qg = torch.cat(cal_buf[l]["q_g"], 0).numpy()
        kp = torch.cat(cal_buf[l]["k_patch"], 0).numpy()
        qp = torch.cat(cal_buf[l]["q_patches"], 0).numpy()
        wh = torch.cat(cal_buf[l]["head_weights"], 0).numpy()
        pca = qii.fit_query_pca(qg, k=min(qii.DEFAULT_PCA_K, qg.shape[-1], max(1, len(cal_idx) - 1)))
        sc = score_layer_gpu(
            qg, kp, qp, wh, dummy_masks, scales[l], grids[l], pca,
            device=device, seeds=(seed,),
        )
        layer_stats[l] = {
            "kd": float(np.nanmean(sc["omega2_weighted"])),
            "qii_s4": float(np.nanmean(sc["qii_weighted_by_s"]["S4"])),
            "entropy": float(np.nanmean(sc["ent_weighted"])),
        }
    del cal_buf
    cal_sel = R.select_layer_and_branch(layer_stats, tau_q=tau_q)
    l_star = int(cal_sel["l_star"])
    branch = str(cal_sel["branch"])
    cal_sec = time.perf_counter() - t_cal
    print(f"[{model_id}] calibrated on {len(cal_idx)} unlabelled imgs in {cal_sec:.1f}s -> l*={l_star}, branch={branch}", flush=True)

    # Pass 2: Extract ONLY {l_star, last_l} across all n_img images & score directly in GPU batches!
    extract_layers = sorted({l_star, last_l})
    gh, gw = grids[l_star]
    t_ext = time.perf_counter()

    buf = {l: {k: [] for k in ("q_g", "k_patch", "q_patches", "cls_attn_patch", "head_weights", "patch_norms")} for l in extract_layers}
    for st in range(0, n_img, batch_size):
        en = min(n_img, st + batch_size)
        xb = torch.stack([tf(Image.open(items[i]["image_path"]).convert("RGB")) for i in range(st, en)], dim=0).to(device)
        ex = extract_batch_ralr(adapter, model, xb, extract_layers)
        for l in extract_layers:
            for k in buf[l]:
                t_val = ex[l][k]
                buf[l][k].append(t_val.half() if k in ("k_patch", "q_patches") else t_val)
    extract_sec = time.perf_counter() - t_ext

    qg_ls = torch.cat(buf[l_star]["q_g"], 0).float()
    kp_ls = torch.cat(buf[l_star]["k_patch"], 0)  # fp16 on CPU
    qp_ls = torch.cat(buf[l_star]["q_patches"], 0)
    wh_ls = torch.cat(buf[l_star]["head_weights"], 0).float()
    ca_ls = torch.cat(buf[l_star]["cls_attn_patch"], 0).float()
    pn_ls = torch.cat(buf[l_star]["patch_norms"], 0).float()
    ca_last = torch.cat(buf[last_l]["cls_attn_patch"], 0).float()
    del buf

    # Compute M2 key-effect maps beta_i(l*, h) in low-RAM GPU chunks (exact match to compute_key_effect_maps_gpu)
    qg_np = qg_ls.numpy()
    I_n, H_n, N_p, d_h = kp_ls.shape
    betas_ls = torch.empty((I_n, H_n, N_p), dtype=torch.float32)
    beta_chunk = 128
    for s in range(0, I_n, beta_chunk):
        e = min(I_n, s + beta_chunk)
        qp_c = qp_ls[s:e].float().numpy()
        Q_c = np.empty((e - s, H_n, keydom.NUM_QUERIES, d_h), dtype=np.float32)
        for bi, gi in enumerate(range(s, e)):
            Q_c[bi] = keydom.build_query_set_64(qg_np[gi], qp_c[bi], qg_np, image_idx=gi, seed=seed + gi)
        Qt = torch.as_tensor(Q_c, dtype=torch.float32, device=device)
        Kt = kp_ls[s:e].to(device=device, dtype=torch.float32)
        logits = (torch.einsum("bhqd,bhnd->bhqn", Qt, Kt) * float(scales[l_star])).to(torch.float64)
        mu = logits.mean(dim=(-2, -1), keepdim=True)
        betas_ls[s:e] = (logits.mean(dim=-2, keepdim=True) - mu).squeeze(-2).float().cpu()
    del qp_ls, qg_ls, qg_np
    gc.collect()

    # Score all methods in fast GPU batches
    per_img_auc = {m: np.empty(n_img, dtype=np.float64) for m in METHODS_OI}
    per_img_pxap = {m: np.empty(n_img, dtype=np.float64) for m in METHODS_OI}
    per_img_pt = {m: np.empty(n_img, dtype=np.float64) for m in METHODS_OI}
    per_img_otsu_hit = {m: np.empty(n_img, dtype=np.float64) for m in METHODS_OI}
    per_img_otsu_fg = {m: np.empty(n_img, dtype=np.float64) for m in METHODS_OI}

    score_bs = 128
    for st in range(0, n_img, score_bs):
        en = min(n_img, st + score_bs)
        gt_b = torch.from_numpy(gt_masks_np[st:en]).to(device)

        # Construct (B, gh, gw) raw grids on GPU
        m_last = ca_last[st:en].mean(dim=1).to(device)                                        # (B, N)
        m_ls = ca_ls[st:en].mean(dim=1).to(device)                                            # (B, N)
        m_lsw = (ca_ls[st:en] * wh_ls[st:en].unsqueeze(-1)).sum(dim=1).to(device)             # (B, N)
        # clip98: percentile clip top 2% per image on cls_l_star
        q98 = torch.quantile(m_ls, 0.98, dim=1, keepdim=True)
        m_clip98 = torch.minimum(m_ls, q98)
        # no_out: zero OUT patches (> median + 5*MAD) on cls_l_star
        pn_b = pn_ls[st:en]
        med = pn_b.median(dim=1, keepdim=True).values
        mad = (pn_b - med).abs().median(dim=1, keepdim=True).values
        out_b = (pn_b > (med + 5.0 * mad)).to(device)
        m_noout = m_ls.masked_fill(out_b, 0.0)
        # lost_l_star continuous
        m_lost = _lost_continuous_batch(kp_ls[st:en].to(device).float())
        # ralr_adaptive (HIGH branch for all models since QII_S4 >= 0.50; head-weighted key-effect map)
        if branch == "HIGH":
            m_ralr = (betas_ls[st:en].to(device) * wh_ls[st:en].to(device).unsqueeze(-1)).sum(dim=1)
        else:
            m_ralr = m_noout

        raw_dict = {
            "cls_last": m_last,
            "cls_l_star": m_ls,
            "cls_l_star_w": m_lsw,
            "cls_l_star_clip98": m_clip98,
            "cls_l_star_no_out": m_noout,
            "lost_l_star": m_lost,
            "ralr_adaptive": m_ralr,
        }
        for meth, grid_flat in raw_dict.items():
            maps_t = _upsample_norm_batch(grid_flat.reshape(-1, gh, gw), target_size)
            per_img_auc[meth][st:en] = box_auc_torch(maps_t, gt_b.float()).cpu().numpy()
            per_img_pxap[meth][st:en] = pxap_torch_batch(maps_t, gt_b).cpu().numpy()
            # Pointing game on GPU
            B_cur = maps_t.shape[0]
            argm = maps_t.reshape(B_cur, -1).argmax(dim=1)
            gt_flat = gt_b.reshape(B_cur, -1)
            per_img_pt[meth][st:en] = gt_flat[torch.arange(B_cur, device=device), argm].float().cpu().numpy()
            # Otsu CorLoc@0.5 + fg_frac on CPU
            maps_np = maps_t.cpu().numpy()
            for bi in range(B_cur):
                obox, omask, _ = M.otsu_box(maps_np[bi])
                per_img_otsu_fg[meth][st + bi] = float(omask.mean())
                per_img_otsu_hit[meth][st + bi] = 1.0 if M.max_iou_any_box(obox, gt_boxes_list[st + bi]) >= 0.5 else 0.0

    del kp_ls, betas_ls, wh_ls, ca_ls, pn_ls, ca_last, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    def _summarize(idx_sel: np.ndarray | None = None) -> dict:
        ms_out = {}
        for meth in METHODS_OI:
            a = per_img_auc[meth] if idx_sel is None else per_img_auc[meth][idx_sel]
            p = per_img_pxap[meth] if idx_sel is None else per_img_pxap[meth][idx_sel]
            pt = per_img_pt[meth] if idx_sel is None else per_img_pt[meth][idx_sel]
            oh = per_img_otsu_hit[meth] if idx_sel is None else per_img_otsu_hit[meth][idx_sel]
            of = per_img_otsu_fg[meth] if idx_sel is None else per_img_otsu_fg[meth][idx_sel]
            am, alo, ahi = bootstrap_ci(a, seed=seed)
            pm, plo, phi = bootstrap_ci(p, seed=seed)
            ptm, ptlo, pthi = bootstrap_ci(pt, seed=seed)
            om, olo, ohi = bootstrap_ci(oh, seed=seed)
            degen_rate = float(np.mean((of < 0.02) | (of > 0.80)))
            ms_out[meth] = {
                "box_auc": {"mean": round(am, 4), "ci95": [round(alo, 4), round(ahi, 4)], "tunes_on_boxes": False},
                "pxap": {"mean": round(pm, 4), "ci95": [round(plo, 4), round(phi, 4)], "tunes_on_boxes": False},
                "pointing": {"mean": round(ptm, 4), "ci95": [round(ptlo, 4), round(pthi, 4)], "tunes_on_boxes": False},
                "corloc_otsu": {
                    "mean": round(om, 4), "ci95": [round(olo, 4), round(ohi, 4)],
                    "mean_fg_frac": round(float(of.mean()), 4),
                    "degen_frac_outside_002_080": round(degen_rate, 4),
                    "tunes_on_boxes": False,
                },
            }
        return ms_out

    def _paired(idx_sel: np.ndarray | None = None) -> dict:
        out_p = {}
        for base_m in ("cls_l_star", "cls_l_star_w", "cls_last", "cls_l_star_clip98", "cls_l_star_no_out", "lost_l_star"):
            a1 = per_img_auc["ralr_adaptive"] if idx_sel is None else per_img_auc["ralr_adaptive"][idx_sel]
            a0 = per_img_auc[base_m] if idx_sel is None else per_img_auc[base_m][idx_sel]
            p1 = per_img_pxap["ralr_adaptive"] if idx_sel is None else per_img_pxap["ralr_adaptive"][idx_sel]
            p0 = per_img_pxap[base_m] if idx_sel is None else per_img_pxap[base_m][idx_sel]
            o1 = per_img_otsu_hit["ralr_adaptive"] if idx_sel is None else per_img_otsu_hit["ralr_adaptive"][idx_sel]
            o0 = per_img_otsu_hit[base_m] if idx_sel is None else per_img_otsu_hit[base_m][idx_sel]
            out_p[f"ralr_adaptive_minus_{base_m}"] = {
                "box_auc": paired_bootstrap_diff_ci(a1, a0, seed=seed),
                "pxap": paired_bootstrap_diff_ci(p1, p0, seed=seed),
                "corloc_otsu": paired_bootstrap_diff_ci(o1, o0, seed=seed),
            }
        return out_p

    res = {
        "model_id": model_id,
        "resolution": int(target_size),
        "n_images_full": int(n_img),
        "n_images_cal": int(len(cal_idx)),
        "n_images_heldout": int(len(heldout_idx)),
        "calibration": cal_sel,
        "layer_stats": {str(k): v for k, v in layer_stats.items()},
        "methods": _summarize(None),
        "paired_diffs": _paired(None),
        "methods_heldout": _summarize(heldout_idx) if len(heldout_idx) >= 50 else {},
        "paired_diffs_heldout": _paired(heldout_idx) if len(heldout_idx) >= 50 else {},
        "extract_sec": round(extract_sec, 2),
        "total_sec": round(time.perf_counter() - t0, 2),
    }

    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "models" / f"{model_id}.json").write_text(json.dumps(res, indent=2))
    print(
        f"[{model_id}] DONE in {res['total_sec']:.1f}s | "
        f"boxAUC: {res['methods']['cls_l_star']['box_auc']['mean']:.3f}->{res['methods']['ralr_adaptive']['box_auc']['mean']:.3f} | "
        f"PxAP: {res['methods']['cls_l_star']['pxap']['mean']:.3f}->{res['methods']['ralr_adaptive']['pxap']['mean']:.3f} | "
        f"Otsu: {res['methods']['cls_l_star']['corloc_otsu']['mean']:.3f}->{res['methods']['ralr_adaptive']['corloc_otsu']['mean']:.3f}",
        flush=True,
    )
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="Action A1: Fast GPU OpenImages30k + PxAP RALR evaluation.")
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--out_dir", default="/content/results/zoo12_ralr_openimages30k")
    ap.add_argument("--models", default="zoo12")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cal_n", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights_cache_dir", default=str(Path.home() / ".cache/queryblind_weights"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke_random", action="store_true")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    items = load_openimages_pairs(Path(args.img_dir), Path(args.mask_dir), limit=args.limit, seed=args.seed)
    model_ids = resolve_model_ids(args.models)
    print(f"Action A1 OpenImages30k: {len(items)} images, {len(model_ids)} models on {device}", flush=True)

    gt_cache = {}
    results = {}
    for mid in model_ids:
        done = out_dir / "models" / f"{mid}.json"
        if args.resume and done.exists():
            print(f"[{mid}] --resume: reusing {done}", flush=True)
            results[mid] = json.loads(done.read_text())
            continue
        results[mid] = run_one_model_openimages(
            mid, items, gt_cache, device, out_dir, Path(args.weights_cache_dir),
            batch_size=args.batch_size, cal_n=args.cal_n, seed=args.seed, smoke_random=args.smoke_random,
        )

    # Gate G3 summary (matched n=11 excl. MAE)
    non_mae = [m for m in results if m != "mae_vitb16"]
    n_nd = len(non_mae)
    wins_auc = sum(1 for m in non_mae if results[m]["paired_diffs"]["ralr_adaptive_minus_cls_l_star"]["box_auc"]["mean_diff"] > 0)
    wins_pxap = sum(1 for m in non_mae if results[m]["paired_diffs"]["ralr_adaptive_minus_cls_l_star"]["pxap"]["mean_diff"] > 0)
    wins_otsu = sum(1 for m in non_mae if results[m]["paired_diffs"]["ralr_adaptive_minus_cls_l_star"]["corloc_otsu"]["mean_diff"] > 0)

    payload = {
        "git_sha": _git_sha(),
        "dataset": "openimages30k_test",
        "n_images": len(items),
        "models": results,
        "gate_g3_n11_excl_mae": {
            "n_models": n_nd,
            "wins_box_auc_vs_cls_l_star": wins_auc,
            "p_box_auc_two_sided": float(binomtest(wins_auc, n_nd, 0.5, alternative="two-sided").pvalue) if n_nd else None,
            "wins_pxap_vs_cls_l_star": wins_pxap,
            "p_pxap_two_sided": float(binomtest(wins_pxap, n_nd, 0.5, alternative="two-sided").pvalue) if n_nd else None,
            "wins_otsu_vs_cls_l_star": wins_otsu,
            "p_otsu_two_sided": float(binomtest(wins_otsu, n_nd, 0.5, alternative="two-sided").pvalue) if n_nd else None,
        },
    }
    (out_dir / "ralr_openimages30k_report.json").write_text(json.dumps(payload, indent=2))
    print("\n=== GATE G3 OPENIMAGES30K SUMMARY (n=11 excl. MAE) ===")
    print(json.dumps(payload["gate_g3_n11_excl_mae"], indent=2))
    return payload


if __name__ == "__main__":
    main()
