#!/usr/bin/env python3
"""Pilot RALR runner: Routing-Aware Localization Readout & Mandatory Baselines (Guide §§7, 9, 11.1 #6).

Supports:
  --dataset cub       (exact 500-image CUB-200-2011 pilot split with pixel masks + boxes)
  --dataset imagenet  (1,000-image ImageNet-1K val pilot split with multi-object XML boxes)

Evaluates all §7.2 & §7.4 methods under the identical preprocessing & metric protocol (eval/metrics.py):
  - cls_last          : raw [CLS] attention, last block, head-averaged (DINO default)
  - cls_l_star        : raw [CLS] attention at l*, head-averaged (isolates layer selection alone — CRITICAL)
  - cls_l_star_w      : head-weighted [CLS] attention at l*
  - ralr_high         : always-HIGH (query-free key-effect map beta_i(l*, h), without shortcut removal)
  - ralr_low          : always-LOW (Step-7 shortcut knockout on [CLS] attention at l*)
  - ralr_high_cleaned : key-effect map with Step-7 shortcut suppression
  - ralr_adaptive     : frozen tau_q = 0.5 branch selection

Reports (with explicit 'tunes_on_boxes: yes/no' column per Guide §9.2 #3):
  - box-AUC, PxAP, Pointing (threshold-free, tunes_on_boxes=no)
  - CorLoc@0.5 Otsu (parameter-free, tunes_on_boxes=no)
  - CorLoc@0.5 Cal-tau* (Murtaza-style unlabelled pseudo-box fixed threshold, tunes_on_boxes=no)
  - MaxBoxAccV2 (Choe et al. CVPR 2020, tunes_on_boxes=yes)
  - Paired bootstrap 95% CIs vs cls_l_star & cls_last + across-backbone sign test (Gate G3).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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

from adapters.base import attn_probs
from eval import metrics as M
from eval.preprocess import (
    build_image_transform,
    get_transformed_gt_mask,
    get_transformed_imagenet_gt_mask,
    load_cub_pilot_split,
    load_imagenet_val_pilot_split,
    save_imagenet_box_overlays,
)
from extract.cache import last_third_blocks
from measures import keydom, qii, routing
from measures._common import (
    compute_head_weights,
    get_output_proj_and_layerscale,
    patch_attention,
    write_records,
)
from ralr import readout as R
from scripts.run_pilot_m1_m2 import (
    PILOT8_SPECS,
    _file_sha256_16,
    _git_sha,
    bootstrap_ci,
    load_pilot_model,
    paired_bootstrap_diff_ci,
    score_layer_gpu,
)

METHODS_ORDER = (
    "cls_last",
    "cls_l_star",
    "cls_l_star_w",
    "ralr_high",
    "ralr_low",
    "ralr_high_cleaned",
    "ralr_adaptive",
)


# --------------------------------------------------------------------------- Unified dataset wrapper
def load_eval_samples(args):
    """Return list of dicts with keys: id, image_path, kind ('cub'|'imagenet'), raw_sample."""
    if args.dataset == "cub":
        if not args.cub_dir:
            raise ValueError("--cub_dir is required for --dataset cub")
        samples = load_cub_pilot_split(
            args.cub_dir,
            n_images=args.limit or 500,
            seed=args.seed,
            split_file=args.split_file,
        )
        return [{"id": str(s.cub_id), "image_path": s.image_path, "kind": "cub", "raw": s} for s in samples]

    if args.dataset == "imagenet":
        if not args.images_dir or not args.imagenet_xml_dir:
            raise ValueError("--images_dir and --imagenet_xml_dir are required for --dataset imagenet")
        split_f = args.split_file
        if split_f == str(REPO_ROOT / "configs" / "splits" / "cub_pilot_500_seed0.txt"):
            split_f = str(REPO_ROOT / "configs" / "splits" / "imagenet_pilot_1000_seed0.txt")
        samples = load_imagenet_val_pilot_split(
            args.images_dir,
            args.imagenet_xml_dir,
            n_images=args.limit or 1000,
            seed=args.seed,
            split_file=split_f,
        )
        return [{"id": s.image_id, "image_path": s.image_path, "kind": "imagenet", "raw": s} for s in samples]

    raise ValueError(f"Unknown dataset {args.dataset!r}")


def precompute_gt_for_resolution(items: list[dict], target_size: int) -> tuple[list[np.ndarray], list[list[tuple[float, float, float, float]]]]:
    """Precompute transformed GT binary pixel masks (S, S) and multi-instance GT boxes once per resolution."""
    masks, boxes_list = [], []
    for it in items:
        with Image.open(it["image_path"]) as im:
            ow, oh = im.size
        if it["kind"] == "cub":
            mask, tbox = get_transformed_gt_mask(it["raw"], ow, oh, target_size, prefer_seg=True)
            masks.append(mask.astype(bool))
            boxes_list.append([tbox])
        else:
            mask, tboxes = get_transformed_imagenet_gt_mask(it["raw"], ow, oh, target_size)
            masks.append(mask.astype(bool))
            boxes_list.append(tboxes)
    return masks, boxes_list


# --------------------------------------------------------------------------- Batched extraction for RALR
@torch.no_grad()
def extract_batch_ralr(adapter, model, x: torch.Tensor, layers: list[int]) -> dict[int, dict]:
    """Single batched forward pass extracting q_g, k_patch, q_patches, cls_attn, w_h, patch_norms, patch_contrib."""
    layer_set = set(layers)
    max_l = max(layers)
    out: dict[int, dict] = {}

    if adapter.family == "timm_vit":
        layout = adapter.token_layout(model, x)
        cls_idx = layout["cls_idx"]
        patch_idx = layout["patch_idx"]
        z = model.patch_embed(x)
        z = model._pos_embed(z)
        z = model.patch_drop(z)
        z = model.norm_pre(z)

        for l, blk in enumerate(model.blocks[: max_l + 1]):
            z_in = z
            z_out = blk(z_in) if (l < max_l or l in layer_set) else z_in
            if l in layer_set:
                a = blk.attn
                B, T, _ = z_in.shape
                qkv = a.qkv(blk.norm1(z_in)).reshape(B, T, 3, a.num_heads, a.head_dim).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                q, k = a.q_norm(q), a.k_norm(k)
                scale = float(a.scale)
                attn_row = attn_probs(q[:, :, cls_idx : cls_idx + 1], k, scale).squeeze(2)  # (B, H, T)
                W_O, _, gamma = get_output_proj_and_layerscale(adapter, model, l)
                w_h = compute_head_weights(attn_row, v, W_O, gamma=gamma)  # (B, H)
                c_all = routing.token_contributions(attn_row, v, W_O, gamma=gamma, return_vectors=False)  # (B, T)
                a_patch = patch_attention(q[:, :, cls_idx, :], k[:, :, patch_idx, :], scale)  # (B, H, N)
                out[l] = {
                    "q_g": q[:, :, cls_idx, :].float().cpu(),
                    "k_patch": k[:, :, patch_idx, :].float().cpu(),
                    "q_patches": q[:, :, patch_idx, :].float().cpu(),
                    "cls_attn_patch": a_patch.float().cpu(),
                    "head_weights": w_h.float().cpu(),
                    "patch_norms": z_out[:, patch_idx, :].float().norm(dim=-1).cpu(),
                    "patch_contrib": c_all[:, patch_idx].float().cpu(),
                    "scale": scale,
                    "grid": layout["grid"],
                }
            z = z_out
        return out

    if adapter.family == "open_clip_vit":
        vt = adapter._visual(model)
        layout = adapter.token_layout(model)
        cls_idx = layout["cls_idx"]
        patch_idx = layout["patch_idx"]
        z = vt.conv1(x).reshape(x.shape[0], vt.conv1.out_channels, -1).permute(0, 2, 1)
        cls = vt.class_embedding.to(z.dtype).expand(z.shape[0], 1, -1)
        z = torch.cat([cls, z], dim=1) + vt.positional_embedding.to(z.dtype)
        z = vt.patch_dropout(z)
        z = vt.ln_pre(z)

        for l, blk in enumerate(vt.transformer.resblocks[: max_l + 1]):
            z_in = z
            z_out = blk(z_in) if (l < max_l or l in layer_set) else z_in
            if l in layer_set:
                a = blk.attn
                B, T, _ = z_in.shape
                H, d = a.num_heads, a.head_dim
                zn = blk.ln_1(z_in)
                Wq, Wk, Wv = a.in_proj_weight.chunk(3, dim=0)
                bq, bk, bv = a.in_proj_bias.chunk(3, dim=0)
                q = (zn @ Wq.T + bq).reshape(B, T, H, d).transpose(1, 2)
                k = (zn @ Wk.T + bk).reshape(B, T, H, d).transpose(1, 2)
                v = (zn @ Wv.T + bv).reshape(B, T, H, d).transpose(1, 2)
                scale = float(d ** -0.5)
                attn_row = attn_probs(q[:, :, cls_idx : cls_idx + 1], k, scale).squeeze(2)
                W_O, _, gamma = get_output_proj_and_layerscale(adapter, model, l)
                w_h = compute_head_weights(attn_row, v, W_O, gamma=gamma)
                c_all = routing.token_contributions(attn_row, v, W_O, gamma=gamma, return_vectors=False)
                a_patch = patch_attention(q[:, :, cls_idx, :], k[:, :, patch_idx, :], scale)
                out[l] = {
                    "q_g": q[:, :, cls_idx, :].float().cpu(),
                    "k_patch": k[:, :, patch_idx, :].float().cpu(),
                    "q_patches": q[:, :, patch_idx, :].float().cpu(),
                    "cls_attn_patch": a_patch.float().cpu(),
                    "head_weights": w_h.float().cpu(),
                    "patch_norms": z_out[:, patch_idx, :].float().norm(dim=-1).cpu(),
                    "patch_contrib": c_all[:, patch_idx].float().cpu(),
                    "scale": scale,
                    "grid": layout["grid"],
                }
            z = z_out
        return out

    raise NotImplementedError(f"RALR batched extractor not implemented for family={adapter.family!r}")


@torch.no_grad()
def compute_key_effect_maps_gpu(
    q_g_all: np.ndarray,      # (I, H, d_h)
    k_patch_all: np.ndarray,  # (I, H, N, d_h)
    q_patch_all: np.ndarray,  # (I, H, N, d_h)
    scale: float,
    device: torch.device,
    seed: int = 0,
    chunk_size: int = 64,
) -> np.ndarray:
    """Compute M2 key-effect maps beta_i(l*, h) on pre-softmax logits using fixed |Q|=64 queries (Guide §3.3)."""
    I, H, N, d_h = k_patch_all.shape
    Q_all = np.empty((I, H, keydom.NUM_QUERIES, d_h), dtype=np.float32)
    for i in range(I):
        Q_all[i] = keydom.build_query_set_64(q_g_all[i], q_patch_all[i], q_g_all, image_idx=i, seed=seed + i)

    betas = np.empty((I, H, N), dtype=np.float32)
    for s in range(0, I, chunk_size):
        e = min(I, s + chunk_size)
        Qt = torch.as_tensor(Q_all[s:e], dtype=torch.float32, device=device)      # (B, H, 64, d_h)
        Kt = torch.as_tensor(k_patch_all[s:e], dtype=torch.float32, device=device)  # (B, H, N, d_h)
        logits = (torch.einsum("bhqd,bhnd->bhqn", Qt, Kt) * float(scale)).to(torch.float64)
        mu = logits.mean(dim=(-2, -1), keepdim=True)
        beta_bar = logits.mean(dim=-2, keepdim=True) - mu  # (B, H, 1, N)
        betas[s:e] = beta_bar.squeeze(-2).float().cpu().numpy()
    return betas


def load_offline_calibration(model_id: str, m1_m2_dir: Path | None, layers: list[int]) -> dict[int, dict] | None:
    """Load unlabelled KD(l), QII_S4(l), entropy(l) from existing M1/M2 pilot output if available."""
    if m1_m2_dir is None:
        return None
    p = m1_m2_dir / "models" / f"{model_id}.json"
    if not p.exists():
        return None
    raw = json.loads(p.read_text())
    stats = {}
    for row in raw.get("layer_summaries", []):
        l = int(row["layer"])
        if l in layers:
            stats[l] = {
                "kd": float(row["omega2_key"]),
                "qii_s4": float(row["qii_s4"]),
                "entropy": float(row["entropy_norm"]),
            }
    return stats if len(stats) == len(layers) else None


# --------------------------------------------------------------------------- Run one model
def run_one_model_ralr(
    model_id: str,
    items: list[dict],
    gt_cache: dict[int, tuple[list[np.ndarray], list[list[tuple[float, float, float, float]]]]],
    device: torch.device,
    out_dir: Path,
    weights_cache: Path,
    *,
    m1_m2_dir: Path | None = None,
    batch_size: int = 32,
    tau_q: float = R.FROZEN_TAU_Q,
    k_clusters: int = 2,
    cal_frac: float = 0.20,
    seed: int = 0,
    smoke_random: bool = False,
) -> dict:
    t0 = time.perf_counter()
    adapter, model, mean, std, target_size = load_pilot_model(
        model_id, device, weights_cache, smoke_random=smoke_random
    )
    n_blocks = adapter.n_blocks(model)
    layers = last_third_blocks(n_blocks)
    last_layer = layers[-1]
    tf = build_image_transform(target_size, mean, std)

    if target_size not in gt_cache:
        gt_cache[target_size] = precompute_gt_for_resolution(items, target_size)
    gt_masks, gt_boxes_list = gt_cache[target_size]
    n_img = len(items)

    # Check if we already have unlabelled offline calibration (KD, QII_S4, entropy) from M1/M2
    cal_stats = load_offline_calibration(model_id, m1_m2_dir, layers)
    if cal_stats is not None:
        sel = R.select_layer_and_branch(cal_stats, tau_q=tau_q)
        extract_layers = sorted({sel["l_star"], last_layer})
    else:
        sel = None
        extract_layers = list(layers)

    print(
        f"\n[{model_id}] RALR | device={device}, res={target_size}, n={n_img}, "
        f"extract_layers={extract_layers} (cal={'cached' if cal_stats else 'compute_unlabelled'})",
        flush=True,
    )

    t_ext = time.perf_counter()
    collected = {
        l: {k: [] for k in ("q_g", "k_patch", "q_patches", "cls_attn_patch", "head_weights", "patch_norms", "patch_contrib")}
        for l in extract_layers
    }
    scales, grids = {}, {}
    for start in range(0, n_img, batch_size):
        batch = items[start : start + batch_size]
        imgs = []
        for it in batch:
            with Image.open(it["image_path"]) as im:
                imgs.append(tf(im.convert("RGB")))
        xb = torch.stack(imgs, dim=0).to(device)
        ex = extract_batch_ralr(adapter, model, xb, extract_layers)
        for l in extract_layers:
            d = ex[l]
            scales[l] = d["scale"]
            grids[l] = d["grid"]
            for k in collected[l]:
                collected[l][k].append(d[k])
    extract_sec = time.perf_counter() - t_ext
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    cat = {
        l: {k: torch.cat(v, dim=0).numpy() for k, v in collected[l].items()}
        for l in extract_layers
    }

    # If no cached M1/M2 calibration, run unlabelled offline calibration (Guide §7.1 Steps 1-4; zero boxes touched!)
    if sel is None:
        cal_stats = {}
        dummy_masks = np.ones((n_img, target_size, target_size), dtype=np.uint8)
        dummy_masks[:, 0, 0] = 0
        for l in extract_layers:
            pca = qii.fit_query_pca(cat[l]["q_g"], k=qii.DEFAULT_PCA_K)
            sc = score_layer_gpu(
                cat[l]["q_g"], cat[l]["k_patch"], cat[l]["q_patches"], cat[l]["head_weights"],
                dummy_masks, scales[l], grids[l], pca, device=device, seeds=(seed,),
            )
            cal_stats[l] = {
                "kd": float(np.nanmean(sc["omega2_weighted"])),
                "qii_s4": float(np.nanmean(sc["qii_weighted_by_s"]["S4"])),
                "entropy": float(np.nanmean(sc["ent_weighted"])),
            }
        sel = R.select_layer_and_branch(cal_stats, tau_q=tau_q)

    l_star = sel["l_star"]
    branch = sel["branch"]
    print(
        f"  Calibration (unlabelled): l*={l_star} (eligible={sel['eligible_layers']}, degen={sel['all_layers_degenerate']}) | "
        f"KD={sel['kd_at_l_star']:.3f}, QII_S4={sel['qii_s4_at_l_star']:.3f}, E={sel['entropy_at_l_star']:.3f} -> branch={branch}",
        flush=True,
    )

    # Compute M2 key-effect maps beta_i(l*, h) on GPU
    betas_lstar = compute_key_effect_maps_gpu(
        cat[l_star]["q_g"], cat[l_star]["k_patch"], cat[l_star]["q_patches"],
        scales[l_star], device=device, seed=seed,
    )

    # Generate all 7 saliency maps per image
    maps_by_method: dict[str, list[np.ndarray]] = {m: [] for m in METHODS_ORDER}
    diags = []
    for i in range(n_img):
        mp, dg = R.compute_ralr_maps_single_image(
            cls_attn_last=cat[last_layer]["cls_attn_patch"][i],
            cls_attn_lstar=cat[l_star]["cls_attn_patch"][i],
            head_weights_lstar=cat[l_star]["head_weights"][i],
            key_effect_lstar=betas_lstar[i],
            patch_norms_lstar=cat[l_star]["patch_norms"][i],
            patch_contrib_lstar=cat[l_star]["patch_contrib"][i],
            k_patch_lstar=cat[l_star]["k_patch"][i],
            branch=branch,
            grid=grids[l_star],
            target_size=target_size,
            k_clusters=k_clusters,
            seed=seed + i,
        )
        for m in METHODS_ORDER:
            maps_by_method[m].append(mp[m])
        diags.append(dg)

    # Calibrate Murtaza-style single fixed threshold tau* on first cal_n unlabelled maps (zero GT boxes!)
    cal_n = max(10, min(n_img // 2, int(round(n_img * cal_frac))))
    eval_start = cal_n if n_img >= 40 else 0  # hold out cal_n from Cal-tau* evaluation to prevent any self-overlap

    method_results: dict[str, dict] = {}
    per_image_box_auc: dict[str, np.ndarray] = {}
    per_image_otsu_hit: dict[str, np.ndarray] = {}

    for m in METHODS_ORDER:
        mlist = maps_by_method[m]
        aucs = np.array([M.box_auc(mlist[i], gt_masks[i]) for i in range(n_img)], dtype=np.float64)
        pxaps = np.array([M.pxap(mlist[i], gt_masks[i]) for i in range(n_img)], dtype=np.float64)
        pts = np.array([M.pointing_game(mlist[i], gt_masks[i]) for i in range(n_img)], dtype=np.float64)
        otsu_boxes = [M.otsu_box(mlist[i])[0] for i in range(n_img)]
        corloc_otsu, hits_otsu = M.corloc(otsu_boxes, gt_boxes_list, iou_thresh=0.5)

        # Murtaza-style fixed scalar threshold calibrated purely on unlabelled maps vs Otsu pseudo-boxes
        cal_tau = M.calibrate_pseudo_box_threshold(mlist[:cal_n])
        cal_boxes = [M.fixed_threshold_box(mlist[i], cal_tau) for i in range(eval_start, n_img)]
        corloc_cal, hits_cal = M.corloc(cal_boxes, gt_boxes_list[eval_start:], iou_thresh=0.5)

        # MaxBoxAccV2 (Choe et al. CVPR 2020 — tunes_on_boxes=True, explicitly flagged per §9.2 #3)
        mba = M.maxboxacc_v2(mlist, gt_boxes_list)

        auc_m, auc_lo, auc_hi = bootstrap_ci(aucs, seed=seed)
        px_m, px_lo, px_hi = bootstrap_ci(pxaps, seed=seed)
        pt_m, pt_lo, pt_hi = bootstrap_ci(pts, seed=seed)
        co_m, co_lo, co_hi = bootstrap_ci(hits_otsu, seed=seed)
        cc_m, cc_lo, cc_hi = bootstrap_ci(hits_cal, seed=seed)

        per_image_box_auc[m] = aucs
        per_image_otsu_hit[m] = hits_otsu

        method_results[m] = {
            "box_auc": {"mean": round(auc_m, 4), "ci95": [round(auc_lo, 4), round(auc_hi, 4)], "tunes_on_boxes": False},
            "pxap": {"mean": round(px_m, 4), "ci95": [round(px_lo, 4), round(px_hi, 4)], "tunes_on_boxes": False},
            "pointing": {"mean": round(pt_m, 4), "ci95": [round(pt_lo, 4), round(pt_hi, 4)], "tunes_on_boxes": False},
            "corloc_otsu": {"mean": round(co_m, 4), "ci95": [round(co_lo, 4), round(co_hi, 4)], "tunes_on_boxes": False},
            "corloc_cal_tau": {
                "mean": round(cc_m, 4), "ci95": [round(cc_lo, 4), round(cc_hi, 4)],
                "cal_tau": round(cal_tau, 4), "n_eval": int(n_img - eval_start), "tunes_on_boxes": False,
            },
            "maxboxacc_v2": {
                "mean": round(mba["maxboxacc_v2"], 4),
                "by_delta": {k: {"acc": round(v["acc"], 4), "best_tau": round(v["best_tau"], 3)} for k, v in mba["by_delta"].items()},
                "tunes_on_boxes": True,
            },
        }

    # Paired bootstrap diffs vs cls_l_star (CRITICAL §7.4) and cls_last
    paired_diffs = {}
    for m in ("ralr_adaptive", "ralr_high", "ralr_low", "ralr_high_cleaned"):
        paired_diffs[f"{m}_minus_cls_l_star"] = {
            "box_auc": paired_bootstrap_diff_ci(per_image_box_auc[m], per_image_box_auc["cls_l_star"], seed=seed),
            "corloc_otsu": paired_bootstrap_diff_ci(per_image_otsu_hit[m], per_image_otsu_hit["cls_l_star"], seed=seed),
        }
        paired_diffs[f"{m}_minus_cls_last"] = {
            "box_auc": paired_bootstrap_diff_ci(per_image_box_auc[m], per_image_box_auc["cls_last"], seed=seed),
            "corloc_otsu": paired_bootstrap_diff_ci(per_image_otsu_hit[m], per_image_otsu_hit["cls_last"], seed=seed),
        }
    paired_diffs["ralr_high_cleaned_minus_ralr_high"] = {
        "box_auc": paired_bootstrap_diff_ci(per_image_box_auc["ralr_high_cleaned"], per_image_box_auc["ralr_high"], seed=seed),
        "corloc_otsu": paired_bootstrap_diff_ci(per_image_otsu_hit["ralr_high_cleaned"], per_image_otsu_hit["ralr_high"], seed=seed),
    }

    # §7.3 sanity check summary
    bg_check_vals = [d["largest_is_lower_attn"] for d in diags if d["largest_is_lower_attn"] is not None]
    step7_diag = {
        "mean_n_shortcut": round(float(np.mean([d["n_shortcut"] for d in diags])), 2),
        "mean_n_out_norm": round(float(np.mean([d["n_out_norm"] for d in diags])), 2),
        "mean_n_out_keyaff": round(float(np.mean([d["n_out_keyaff"] for d in diags])), 2),
        "pct_largest_cluster_is_lower_attn": round(100.0 * float(np.mean(bg_check_vals)), 1) if bg_check_vals else None,
    }

    d_adapt = paired_diffs["ralr_adaptive_minus_cls_l_star"]["box_auc"]
    print(
        f"  box-AUC: cls_last={method_results['cls_last']['box_auc']['mean']:.3f}, "
        f"cls@l*={method_results['cls_l_star']['box_auc']['mean']:.3f}, "
        f"RALR={method_results['ralr_adaptive']['box_auc']['mean']:.3f} "
        f"(Δ vs cls@l* = {d_adapt['mean_diff']:+.4f} [{d_adapt['ci95_low']:+.4f}, {d_adapt['ci95_high']:+.4f}]) | "
        f"Otsu CorLoc: cls@l*={method_results['cls_l_star']['corloc_otsu']['mean']:.3f}, "
        f"RALR={method_results['ralr_adaptive']['corloc_otsu']['mean']:.3f}",
        flush=True,
    )

    total_sec = time.perf_counter() - t0
    res = {
        "model_id": model_id,
        "resolution": target_size,
        "n_images": n_img,
        "calibration": sel,
        "layer_stats": {f"l{k}": v for k, v in sorted(cal_stats.items())},
        "step7_diagnostics": step7_diag,
        "methods": method_results,
        "paired_diffs": paired_diffs,
        "extract_sec": round(extract_sec, 2),
        "img_per_sec": round(n_img / max(extract_sec, 1e-6), 2),
        "total_sec": round(total_sec, 2),
    }
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "models" / f"{model_id}.json.tmp"
    tmp.write_text(json.dumps(res, indent=2))
    os.replace(tmp, out_dir / "models" / f"{model_id}.json")
    return res


# --------------------------------------------------------------------------- Report builder
def write_ralr_report(results: dict[str, dict], dataset: str, out_dir: Path) -> str:
    models = list(results.keys())
    n_models = len(models)

    lines = [
        f"# Pilot RALR Readout vs Mandatory Baselines ({dataset})\n",
        "## 1. Headline Comparison (box-AUC & CorLoc@0.5 Otsu vs `[CLS]@l*` — Tunes on Boxes: **NO**)\n",
        "| Model | l* | Branch | cls_last AUC | cls@l* AUC | RALR AUC | Δ(RALR − cls@l*) [95% CI] | cls@l* Otsu | RALR Otsu | RALR Cal-τ* | RALR MaxBoxAccV2 *(tunes=YES)* | Step7 BG% |",
        "|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    rows_csv = []
    wins_auc_lstar = 0
    wins_auc_last = 0
    wins_otsu_lstar = 0

    for m, r in results.items():
        cal = r["calibration"]
        ms = r["methods"]
        d_ls = r["paired_diffs"]["ralr_adaptive_minus_cls_l_star"]["box_auc"]
        d_last = r["paired_diffs"]["ralr_adaptive_minus_cls_last"]["box_auc"]
        d_ot = r["paired_diffs"]["ralr_adaptive_minus_cls_l_star"]["corloc_otsu"]
        if d_ls["mean_diff"] > 0:
            wins_auc_lstar += 1
        if d_last["mean_diff"] > 0:
            wins_auc_last += 1
        if d_ot["mean_diff"] > 0:
            wins_otsu_lstar += 1
        bg_pct = r["step7_diagnostics"]["pct_largest_cluster_is_lower_attn"]
        bg_str = f"{bg_pct:.1f}%" if bg_pct is not None else "—"
        lines.append(
            f"| {m} | {cal['l_star']} | {cal['branch']} | "
            f"{ms['cls_last']['box_auc']['mean']:.3f} | {ms['cls_l_star']['box_auc']['mean']:.3f} | "
            f"{ms['ralr_adaptive']['box_auc']['mean']:.3f} | "
            f"{d_ls['mean_diff']:+.4f} [{d_ls['ci95_low']:+.4f}, {d_ls['ci95_high']:+.4f}] | "
            f"{ms['cls_l_star']['corloc_otsu']['mean']:.3f} | {ms['ralr_adaptive']['corloc_otsu']['mean']:.3f} | "
            f"{ms['ralr_adaptive']['corloc_cal_tau']['mean']:.3f} | "
            f"{ms['ralr_adaptive']['maxboxacc_v2']['mean']:.3f} | {bg_str} |"
        )
        for meth in METHODS_ORDER:
            mm = ms[meth]
            rows_csv.append({
                "model_id": m, "l_star": cal["l_star"], "branch": cal["branch"], "method": meth,
                "box_auc": mm["box_auc"]["mean"], "pxap": mm["pxap"]["mean"],
                "pointing": mm["pointing"]["mean"], "corloc_otsu": mm["corloc_otsu"]["mean"],
                "corloc_cal_tau": mm["corloc_cal_tau"]["mean"], "maxboxacc_v2_tunes_yes": mm["maxboxacc_v2"]["mean"],
            })

    lines.append("\n## 2. Full Method & Ablation Table (Mean Across Models, Explicitly Labeled `tunes_on_boxes`)\n")
    lines.append("| Method | Tunes on Boxes? | Mean box-AUC | Mean PxAP | Mean Pointing | Mean CorLoc (Otsu) | Mean CorLoc (Cal-τ*) | Mean MaxBoxAccV2 *(tunes=YES)* |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for meth in METHODS_ORDER:
        auc_avg = np.mean([r["methods"][meth]["box_auc"]["mean"] for r in results.values()])
        px_avg = np.mean([r["methods"][meth]["pxap"]["mean"] for r in results.values()])
        pt_avg = np.mean([r["methods"][meth]["pointing"]["mean"] for r in results.values()])
        ot_avg = np.mean([r["methods"][meth]["corloc_otsu"]["mean"] for r in results.values()])
        ct_avg = np.mean([r["methods"][meth]["corloc_cal_tau"]["mean"] for r in results.values()])
        mb_avg = np.mean([r["methods"][meth]["maxboxacc_v2"]["mean"] for r in results.values()])
        lines.append(
            f"| `{meth}` | NO *(except MaxBoxAccV2)* | {auc_avg:.3f} | {px_avg:.3f} | {pt_avg:.3f} | {ot_avg:.3f} | {ct_avg:.3f} | {mb_avg:.3f} |"
        )

    p_sign_auc = float(binomtest(wins_auc_lstar, n_models, 0.5, alternative="greater").pvalue) if n_models > 0 else float("nan")
    p_sign_otsu = float(binomtest(wins_otsu_lstar, n_models, 0.5, alternative="greater").pvalue) if n_models > 0 else float("nan")
    lines.append("\n## 3. Gate G3 Preview (Sign Test Across Backbones vs `[CLS]@l*`, Guide §11.2)\n")
    lines.append(f"- **RALR beats `[CLS]@l*` on box-AUC:** `{wins_auc_lstar}/{n_models}` backbones (one-sided sign test `p = {p_sign_auc:.4f}`)")
    lines.append(f"- **RALR beats `cls_last` on box-AUC:** `{wins_auc_last}/{n_models}` backbones")
    lines.append(f"- **RALR beats `[CLS]@l*` on Otsu CorLoc@0.5:** `{wins_otsu_lstar}/{n_models}` backbones (one-sided sign test `p = {p_sign_otsu:.4f}`)")

    md_text = "\n".join(lines) + "\n"
    (out_dir / "summary_ralr.md").write_text(md_text)
    if rows_csv:
        write_records(rows_csv, out_dir / "summary_ralr.csv", columns=list(rows_csv[0].keys()))
    (out_dir / "ralr_report.json").write_text(json.dumps({
        "dataset": dataset,
        "models": results,
        "gate_g3_preview": {
            "wins_auc_vs_cls_l_star": wins_auc_lstar,
            "wins_auc_vs_cls_last": wins_auc_last,
            "wins_otsu_vs_cls_l_star": wins_otsu_lstar,
            "n_models": n_models,
            "sign_test_p_auc": p_sign_auc,
            "sign_test_p_otsu": p_sign_otsu,
        },
    }, indent=2))
    return md_text


# --------------------------------------------------------------------------- CLI
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Run RALR + mandatory baselines on Pilot-8 (CUB or ImageNet).")
    ap.add_argument("--dataset", choices=["cub", "imagenet"], default="cub")
    ap.add_argument("--cub_dir", default=None)
    ap.add_argument("--images_dir", default=None)
    ap.add_argument("--imagenet_xml_dir", default=None)
    ap.add_argument("--split_file", default=str(REPO_ROOT / "configs" / "splits" / "cub_pilot_500_seed0.txt"))
    ap.add_argument("--m1_m2_dir", default=None, help="Optional existing pilot_m1_m2 dir to reuse unlabelled KD/QII/E calibration.")
    ap.add_argument("--out_dir", default="results/pilot_ralr")
    ap.add_argument("--models", default="all")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--tau_q", type=float, default=R.FROZEN_TAU_Q)
    ap.add_argument("--k_clusters", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights_cache_dir", default=str(Path.home() / ".cache" / "queryblind_weights"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke_random", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_ids = list(PILOT8_SPECS.keys()) if args.models == "all" else [m.strip() for m in args.models.split(",") if m.strip()]
    for mid in model_ids:
        if mid not in PILOT8_SPECS:
            raise ValueError(f"Unknown pilot model {mid!r}")

    items = load_eval_samples(args)
    print(f"Loaded {len(items)} {args.dataset} images for RALR on device={device}", flush=True)

    # For ImageNet, save mandatory 20-image visual box overlay check (§8.2)
    if args.dataset == "imagenet":
        raw_samples = [it["raw"] for it in items]
        save_imagenet_box_overlays(raw_samples, 224, out_dir / "imagenet_box_overlay_224.png", n_images=20)
        save_imagenet_box_overlays(raw_samples, 196, out_dir / "imagenet_box_overlay_196.png", n_images=20)

    weights_cache = Path(args.weights_cache_dir)
    m1_m2_dir = Path(args.m1_m2_dir) if args.m1_m2_dir else None
    gt_cache: dict[int, tuple] = {}
    results: dict[str, dict] = {}
    t_all = time.perf_counter()

    for mid in model_ids:
        done = out_dir / "models" / f"{mid}.json"
        if args.resume and done.exists():
            print(f"[{mid}] --resume: reusing {done}", flush=True)
            results[mid] = json.loads(done.read_text())
            continue
        results[mid] = run_one_model_ralr(
            mid, items, gt_cache, device, out_dir, weights_cache,
            m1_m2_dir=m1_m2_dir, batch_size=args.batch_size, tau_q=args.tau_q,
            k_clusters=args.k_clusters, seed=args.seed, smoke_random=args.smoke_random,
        )

    md_text = write_ralr_report(results, args.dataset, out_dir)
    cfg_payload = json.dumps({
        "dataset": args.dataset, "n_images": len(items), "models": model_ids,
        "tau_q": args.tau_q, "k_clusters": args.k_clusters, "seed": args.seed,
        "smoke_random": args.smoke_random,
    }, sort_keys=True)
    meta = {
        "git_sha": _git_sha(),
        "config_hash": hashlib.sha256(cfg_payload.encode()).hexdigest()[:16],
        "file_sha256_16": {
            rel: _file_sha256_16(REPO_ROOT / rel)
            for rel in ("eval/metrics.py", "ralr/readout.py", "scripts/run_pilot_ralr.py", "configs/checkpoints.yaml")
        },
        "device": str(device),
        "dataset": args.dataset,
        "n_images": len(items),
        "smoke_random": bool(args.smoke_random),
        "total_seconds": round(time.perf_counter() - t_all, 2),
        "throughput_by_model": {
            m: {"extract_sec": r["extract_sec"], "img_per_sec": r["img_per_sec"], "total_sec": r["total_sec"]}
            for m, r in results.items()
        },
        "unix_time": int(time.time()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print("\n" + md_text, flush=True)
    return results


if __name__ == "__main__":
    main()
