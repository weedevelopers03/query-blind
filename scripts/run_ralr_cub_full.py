#!/usr/bin/env python3
"""Action A5 (§7): Fast GPU-vectorized RALR evaluation on the FULL CUB-200-2011 test set (N = 5,794).

Closes Dr. Zia's Priority Action A5 ("removes the 'you reported WSOL on your calibration split' objection"):
  - Calibrates unlabelled (l*, branch) AND Murtaza pseudo-box threshold tau* strictly on the frozen
    500-image unlabelled split (configs/splits/cub_pilot_500_seed0.txt; zero GT boxes/masks touched).
  - Evaluates BOTH the full official test set (N = 5,794) AND the strictly disjoint held-out test set
    (N = 5,294 = 5,794 - 500) with zero calibration overlap.
  - Reports all 6 metrics per method:
      * box_auc        (tunes_on_boxes = False, GPU)
      * pxap           (tunes_on_boxes = False, exact tie-safe GPU)
      * pointing       (tunes_on_boxes = False, GPU)
      * corloc_otsu    (tunes_on_boxes = False, + mean_fg_frac + degen_frac_outside_002_080)
      * corloc_cal_tau (tunes_on_boxes = False, Murtaza et al. WACV 2025 pseudo-box tau* fit on X_cal, eval on held-out)
      * maxboxacc_v2   (tunes_on_boxes = True, Choe et al. CVPR 2020 over deltas {0.3, 0.5, 0.7})
  - Low-RAM streaming (~3.5 GB peak CPU RAM) + per-model --resume so Colab T4 runs are 100% crash-safe.
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
from PIL import Image
from scipy.stats import binomtest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval import metrics as M
from eval.preprocess import (
    build_image_transform,
    get_transformed_gt_mask,
    load_cub_pilot_split,
)
from extract.cache import last_third_blocks
from measures import keydom, qii
from measures._common import box_auc_torch
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
from scripts.run_ralr_openimages30k import (
    METHODS_OI as METHODS_CUB,
    _lost_continuous_batch,
    _upsample_norm_batch,
    pxap_torch_batch,
)


def _compute_maxboxacc_from_iou_grid(
    iou_grid: np.ndarray,  # (T, N)
    thresholds: tuple[float, ...],
    deltas: tuple[float, ...] = M.DEFAULT_MAXBOXACC_DELTAS,
) -> dict:
    by_delta = {}
    best_accs = []
    for d in deltas:
        accs = np.nanmean(iou_grid >= float(d), axis=1)
        best_i = int(np.argmax(accs))
        best_acc = float(accs[best_i])
        best_accs.append(best_acc)
        by_delta[f"{d:.1f}"] = {"acc": round(best_acc, 4), "best_tau": round(float(thresholds[best_i]), 3)}
    return {
        "mean": round(float(np.mean(best_accs)), 4),
        "by_delta": by_delta,
        "tunes_on_boxes": True,
    }


@torch.no_grad()
def run_one_model_cub_full(
    model_id: str,
    samples: list,
    cal_id_set: set[int],
    gt_cache: dict[int, tuple],
    device: torch.device,
    out_dir: Path,
    weights_cache: Path,
    *,
    batch_size: int = 64,
    tau_q: float = R.FROZEN_TAU_Q,
    seed: int = 0,
    smoke_random: bool = False,
) -> dict:
    t0 = time.perf_counter()
    adapter, model, mean, std, target_size = load_pilot_model(
        model_id, device, weights_cache, smoke_random=smoke_random,
    )
    tf = build_image_transform(target_size, mean, std)
    n_img = len(samples)

    # Precompute transformed GT segmentation masks (prefer_seg=True) & tight boxes once per resolution (224 or 196)
    if target_size not in gt_cache:
        masks, boxes = [], []
        for s in samples:
            with Image.open(s.image_path) as im:
                ow, oh = im.size
            m, tbox = get_transformed_gt_mask(s, ow, oh, target_size, prefer_seg=True)
            masks.append(m.astype(bool))
            boxes.append([tbox])
        gt_cache[target_size] = (np.stack(masks, axis=0), boxes)
    gt_masks_np, gt_boxes_list = gt_cache[target_size]

    # Exact calibration split X_cal vs strictly disjoint held-out split
    cal_idx = np.array([i for i, s in enumerate(samples) if int(s.cub_id) in cal_id_set], dtype=np.int64)
    if len(cal_idx) == 0:
        rng_cal = np.random.default_rng(int(seed))
        cal_idx = np.sort(rng_cal.choice(n_img, size=min(500, n_img), replace=False))
    cal_mask_bool = np.zeros(n_img, dtype=bool)
    cal_mask_bool[cal_idx] = True
    heldout_idx = np.where(~cal_mask_bool)[0]
    cal_pos_map = {int(idx): pos for pos, idx in enumerate(cal_idx)}

    layers = last_third_blocks(adapter.n_blocks(model))
    last_l = layers[-1]

    # Pass 1: Extract on X_cal (unlabelled) to select l* & branch (zero GT used!)
    t_cal = time.perf_counter()
    cal_buf = {l: {k: [] for k in ("q_g", "k_patch", "q_patches", "cls_attn_patch", "head_weights")} for l in layers}
    scales, grids = {}, {}
    for st in range(0, len(cal_idx), batch_size):
        batch_ids = cal_idx[st : st + batch_size]
        xb = torch.stack([tf(Image.open(samples[i].image_path).convert("RGB")) for i in batch_ids], dim=0).to(device)
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
    print(f"[{model_id}] calibrated on {len(cal_idx)} unlabelled CUB imgs in {cal_sec:.1f}s -> l*={l_star}, branch={branch}", flush=True)

    # Pass 2: Extract ONLY {l_star, last_l} across all 5,794 test images
    extract_layers = sorted({l_star, last_l})
    gh, gw = grids[l_star]
    t_ext = time.perf_counter()

    buf = {l: {k: [] for k in ("q_g", "k_patch", "q_patches", "cls_attn_patch", "head_weights", "patch_norms")} for l in extract_layers}
    for st in range(0, n_img, batch_size):
        en = min(n_img, st + batch_size)
        xb = torch.stack([tf(Image.open(samples[i].image_path).convert("RGB")) for i in range(st, en)], dim=0).to(device)
        ex = extract_batch_ralr(adapter, model, xb, extract_layers)
        for l in extract_layers:
            for k in buf[l]:
                t_val = ex[l][k]
                buf[l][k].append(t_val.half() if k in ("k_patch", "q_patches") else t_val)
    extract_sec = time.perf_counter() - t_ext

    qg_ls = torch.cat(buf[l_star]["q_g"], 0).float()
    kp_ls = torch.cat(buf[l_star]["k_patch"], 0)
    qp_ls = torch.cat(buf[l_star]["q_patches"], 0)
    wh_ls = torch.cat(buf[l_star]["head_weights"], 0).float()
    ca_ls = torch.cat(buf[l_star]["cls_attn_patch"], 0).float()
    pn_ls = torch.cat(buf[l_star]["patch_norms"], 0).float()
    ca_last = torch.cat(buf[last_l]["cls_attn_patch"], 0).float()
    del buf

    # Compute M2 key-effect maps beta_i(l*, h) in low-RAM GPU chunks
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

    # Stream-score all 7 methods (exact GPU box-AUC/PxAP/pointing + Otsu + 19-tau strided CC for MaxBoxAccV2 & Murtaza Cal-tau*)
    thresholds = M.DEFAULT_SWEEP_THRESHOLDS
    n_tau = len(thresholds)
    per_img_auc = {m: np.empty(n_img, dtype=np.float64) for m in METHODS_CUB}
    per_img_pxap = {m: np.empty(n_img, dtype=np.float64) for m in METHODS_CUB}
    per_img_pt = {m: np.empty(n_img, dtype=np.float64) for m in METHODS_CUB}
    per_img_otsu_hit = {m: np.empty(n_img, dtype=np.float64) for m in METHODS_CUB}
    per_img_otsu_fg = {m: np.empty(n_img, dtype=np.float64) for m in METHODS_CUB}
    iou_gt_grid = {m: np.empty((n_tau, n_img), dtype=np.float64) for m in METHODS_CUB}
    cal_pseudo_hits = {m: np.empty((n_tau, len(cal_idx)), dtype=np.float64) for m in METHODS_CUB}

    score_bs = 128
    for st in range(0, n_img, score_bs):
        en = min(n_img, st + score_bs)
        gt_b = torch.from_numpy(gt_masks_np[st:en]).to(device)

        m_last = ca_last[st:en].mean(dim=1).to(device)
        m_ls = ca_ls[st:en].mean(dim=1).to(device)
        m_lsw = (ca_ls[st:en] * wh_ls[st:en].unsqueeze(-1)).sum(dim=1).to(device)
        q98 = torch.quantile(m_ls, 0.98, dim=1, keepdim=True)
        m_clip98 = torch.minimum(m_ls, q98)
        pn_b = pn_ls[st:en]
        med = pn_b.median(dim=1, keepdim=True).values
        mad = (pn_b - med).abs().median(dim=1, keepdim=True).values
        out_b = (pn_b > (med + 5.0 * mad)).to(device)
        m_noout = m_ls.masked_fill(out_b, 0.0)
        m_lost = _lost_continuous_batch(kp_ls[st:en].to(device).float())
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
            B_cur = maps_t.shape[0]
            argm = maps_t.reshape(B_cur, -1).argmax(dim=1)
            gt_flat = gt_b.reshape(B_cur, -1)
            per_img_pt[meth][st:en] = gt_flat[torch.arange(B_cur, device=device), argm].float().cpu().numpy()

            maps_np = maps_t.cpu().numpy()
            for bi in range(B_cur):
                gi = st + bi
                m2d = maps_np[bi]
                obox, omask, _ = M.otsu_box(m2d)
                per_img_otsu_fg[meth][gi] = float(omask.mean())
                per_img_otsu_hit[meth][gi] = 1.0 if M.max_iou_any_box(obox, gt_boxes_list[gi]) >= 0.5 else 0.0
                cpos = cal_pos_map.get(gi, -1)
                for ti, tau in enumerate(thresholds):
                    pbox = M._strided_cc_box(m2d, float(tau))
                    iou_gt_grid[meth][ti, gi] = M.max_iou_any_box(pbox, gt_boxes_list[gi])
                    if cpos >= 0:
                        cal_pseudo_hits[meth][ti, cpos] = 1.0 if M.bbox_iou(pbox, obox) >= 0.5 else 0.0

    del kp_ls, betas_ls, wh_ls, ca_ls, pn_ls, ca_last, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Select Murtaza pseudo-box scalar threshold tau* strictly on unlabelled X_cal (zero GT boxes!)
    cal_tau_idx = {}
    cal_tau_val = {}
    per_img_cal_hit = {}
    for meth in METHODS_CUB:
        best_ti = int(np.argmax(np.mean(cal_pseudo_hits[meth], axis=1)))
        cal_tau_idx[meth] = best_ti
        cal_tau_val[meth] = float(thresholds[best_ti])
        per_img_cal_hit[meth] = (iou_gt_grid[meth][best_ti] >= 0.5).astype(np.float64)

    def _summarize(idx_sel: np.ndarray | None = None) -> dict:
        ms_out = {}
        cal_eval_idx = heldout_idx if (idx_sel is None and len(heldout_idx) > 0) else (idx_sel if idx_sel is not None else np.arange(n_img))
        for meth in METHODS_CUB:
            a = per_img_auc[meth] if idx_sel is None else per_img_auc[meth][idx_sel]
            p = per_img_pxap[meth] if idx_sel is None else per_img_pxap[meth][idx_sel]
            pt = per_img_pt[meth] if idx_sel is None else per_img_pt[meth][idx_sel]
            oh = per_img_otsu_hit[meth] if idx_sel is None else per_img_otsu_hit[meth][idx_sel]
            of = per_img_otsu_fg[meth] if idx_sel is None else per_img_otsu_fg[meth][idx_sel]
            ch = per_img_cal_hit[meth][cal_eval_idx]
            ig = iou_gt_grid[meth] if idx_sel is None else iou_gt_grid[meth][:, idx_sel]

            am, alo, ahi = bootstrap_ci(a, seed=seed)
            pm, plo, phi = bootstrap_ci(p, seed=seed)
            ptm, ptlo, pthi = bootstrap_ci(pt, seed=seed)
            om, olo, ohi = bootstrap_ci(oh, seed=seed)
            cm, clo, chi = bootstrap_ci(ch, seed=seed)
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
                "corloc_cal_tau": {
                    "mean": round(cm, 4), "ci95": [round(clo, 4), round(chi, 4)],
                    "cal_tau": round(cal_tau_val[meth], 4),
                    "n_eval": int(len(cal_eval_idx)),
                    "tunes_on_boxes": False,
                },
                "maxboxacc_v2": _compute_maxboxacc_from_iou_grid(ig, thresholds),
            }
        return ms_out

    def _paired(idx_sel: np.ndarray | None = None) -> dict:
        out_p = {}
        cal_eval_idx = heldout_idx if (idx_sel is None and len(heldout_idx) > 0) else (idx_sel if idx_sel is not None else np.arange(n_img))
        for base_m in ("cls_l_star", "cls_l_star_w", "cls_last", "cls_l_star_clip98", "cls_l_star_no_out", "lost_l_star"):
            a1 = per_img_auc["ralr_adaptive"] if idx_sel is None else per_img_auc["ralr_adaptive"][idx_sel]
            a0 = per_img_auc[base_m] if idx_sel is None else per_img_auc[base_m][idx_sel]
            p1 = per_img_pxap["ralr_adaptive"] if idx_sel is None else per_img_pxap["ralr_adaptive"][idx_sel]
            p0 = per_img_pxap[base_m] if idx_sel is None else per_img_pxap[base_m][idx_sel]
            o1 = per_img_otsu_hit["ralr_adaptive"] if idx_sel is None else per_img_otsu_hit["ralr_adaptive"][idx_sel]
            o0 = per_img_otsu_hit[base_m] if idx_sel is None else per_img_otsu_hit[base_m][idx_sel]
            c1 = per_img_cal_hit["ralr_adaptive"][cal_eval_idx]
            c0 = per_img_cal_hit[base_m][cal_eval_idx]
            out_p[f"ralr_adaptive_minus_{base_m}"] = {
                "box_auc": paired_bootstrap_diff_ci(a1, a0, seed=seed),
                "pxap": paired_bootstrap_diff_ci(p1, p0, seed=seed),
                "corloc_otsu": paired_bootstrap_diff_ci(o1, o0, seed=seed),
                "corloc_cal_tau": paired_bootstrap_diff_ci(c1, c0, seed=seed),
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
        "methods_heldout": _summarize(heldout_idx) if len(heldout_idx) >= 2 else {},
        "paired_diffs_heldout": _paired(heldout_idx) if len(heldout_idx) >= 2 else {},
        "extract_sec": round(extract_sec, 2),
        "total_sec": round(time.perf_counter() - t0, 2),
    }

    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "models" / f"{model_id}.json").write_text(json.dumps(res, indent=2))
    print(
        f"[{model_id}] DONE in {res['total_sec']:.1f}s | "
        f"boxAUC: {res['methods']['cls_l_star']['box_auc']['mean']:.3f}->{res['methods']['ralr_adaptive']['box_auc']['mean']:.3f} | "
        f"PxAP: {res['methods']['cls_l_star']['pxap']['mean']:.3f}->{res['methods']['ralr_adaptive']['pxap']['mean']:.3f} | "
        f"Otsu: {res['methods']['cls_l_star']['corloc_otsu']['mean']:.3f}->{res['methods']['ralr_adaptive']['corloc_otsu']['mean']:.3f} | "
        f"CalTau: {res['methods']['cls_l_star']['corloc_cal_tau']['mean']:.3f}->{res['methods']['ralr_adaptive']['corloc_cal_tau']['mean']:.3f} | "
        f"MBAv2: {res['methods']['cls_l_star']['maxboxacc_v2']['mean']:.3f}->{res['methods']['ralr_adaptive']['maxboxacc_v2']['mean']:.3f}",
        flush=True,
    )
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="Action A5: Fast GPU full CUB-200-2011 test set (5,794) RALR evaluation.")
    ap.add_argument("--cub_dir", required=True, help="Path to CUB_200_2011 root.")
    ap.add_argument("--seg_dir", default=None, help="Optional path to CUB segmentations/ directory.")
    ap.add_argument("--cal_split_file", default=str(REPO_ROOT / "configs" / "splits" / "cub_pilot_500_seed0.txt"))
    ap.add_argument("--out_dir", default="/content/results/zoo12_ralr_cub_full5794")
    ap.add_argument("--models", default="zoo12")
    ap.add_argument("--limit", type=int, default=None, help="Optional cap for smoke testing.")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights_cache_dir", default=str(Path.home() / ".cache/queryblind_weights"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke_random", action="store_true")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load ALL official CUB test images (5,794 when split_file=None)
    samples = load_cub_pilot_split(
        args.cub_dir,
        n_images=args.limit or 5794,
        seed=args.seed,
        split_file=None,
        seg_dir=args.seg_dir,
    )
    cal_p = Path(args.cal_split_file)
    cal_id_set = {int(x.strip()) for x in cal_p.read_text().splitlines() if x.strip()} if cal_p.exists() else set()
    n_seg = sum(1 for s in samples if s.seg_path is not None and s.seg_path.exists())
    model_ids = resolve_model_ids(args.models)
    print(f"Action A5 CUB-full: {len(samples)} test images ({n_seg} with seg masks), {len(model_ids)} models on {device}", flush=True)

    gt_cache = {}
    results = {}
    for mid in model_ids:
        done = out_dir / "models" / f"{mid}.json"
        if args.resume and done.exists():
            print(f"[{mid}] --resume: reusing {done}", flush=True)
            results[mid] = json.loads(done.read_text())
            continue
        results[mid] = run_one_model_cub_full(
            mid, samples, cal_id_set, gt_cache, device, out_dir, Path(args.weights_cache_dir),
            batch_size=args.batch_size, seed=args.seed, smoke_random=args.smoke_random,
        )

    non_mae = [m for m in results if m != "mae_vitb16"]
    n_nd = len(non_mae)
    def _wins(key: str, split: str = "paired_diffs") -> int:
        return sum(1 for m in non_mae if results[m][split]["ralr_adaptive_minus_cls_l_star"][key]["mean_diff"] > 0)

    wins_mba_full = sum(
        1 for m in non_mae
        if results[m]["methods"]["ralr_adaptive"]["maxboxacc_v2"]["mean"] > results[m]["methods"]["cls_l_star"]["maxboxacc_v2"]["mean"]
    )
    wins_mba_ho = sum(
        1 for m in non_mae
        if results[m]["methods_heldout"] and results[m]["methods_heldout"]["ralr_adaptive"]["maxboxacc_v2"]["mean"] > results[m]["methods_heldout"]["cls_l_star"]["maxboxacc_v2"]["mean"]
    )

    payload = {
        "git_sha": _git_sha(),
        "dataset": "cub_200_2011_full_test",
        "n_images_full": len(samples),
        "n_with_seg_masks": n_seg,
        "cal_split_file": str(args.cal_split_file),
        "models": results,
        "summary_n11_excl_mae_full": {
            "n_models": n_nd,
            "wins_box_auc": _wins("box_auc"),
            "p_box_auc_two_sided": float(binomtest(_wins("box_auc"), n_nd, 0.5).pvalue) if n_nd else None,
            "wins_pxap": _wins("pxap"),
            "p_pxap_two_sided": float(binomtest(_wins("pxap"), n_nd, 0.5).pvalue) if n_nd else None,
            "wins_otsu": _wins("corloc_otsu"),
            "p_otsu_two_sided": float(binomtest(_wins("corloc_otsu"), n_nd, 0.5).pvalue) if n_nd else None,
            "wins_cal_tau": _wins("corloc_cal_tau"),
            "p_cal_tau_two_sided": float(binomtest(_wins("corloc_cal_tau"), n_nd, 0.5).pvalue) if n_nd else None,
            "wins_maxboxacc_v2": wins_mba_full,
        },
        "summary_n11_excl_mae_heldout": {
            "n_models": n_nd,
            "wins_box_auc": _wins("box_auc", "paired_diffs_heldout") if results[non_mae[0]]["paired_diffs_heldout"] else None,
            "wins_pxap": _wins("pxap", "paired_diffs_heldout") if results[non_mae[0]]["paired_diffs_heldout"] else None,
            "wins_otsu": _wins("corloc_otsu", "paired_diffs_heldout") if results[non_mae[0]]["paired_diffs_heldout"] else None,
            "wins_cal_tau": _wins("corloc_cal_tau", "paired_diffs_heldout") if results[non_mae[0]]["paired_diffs_heldout"] else None,
            "wins_maxboxacc_v2": wins_mba_ho,
        } if n_nd else {},
    }
    (out_dir / "ralr_cub_full5794_report.json").write_text(json.dumps(payload, indent=2))
    print("\n=== ACTION A5 CUB FULL-TEST SUMMARY (n=11 excl. MAE) ===")
    print(json.dumps(payload["summary_n11_excl_mae_full"], indent=2))
    print("=== HELD-OUT 5,294 (DISJOINT FROM X_cal=500) ===")
    print(json.dumps(payload["summary_n11_excl_mae_heldout"], indent=2))
    return payload


if __name__ == "__main__":
    main()
