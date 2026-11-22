#!/usr/bin/env python3
"""Pilot Interventions I1 (Query Replacement) & I2 (Token-Group Knockout) runner (Guide §6).

Executes on Pilot-8 with exact matched controls and mean-ablation:
  - I1 (Query Replacement):
      * Sweeps L_int in {1, 2, 3, 4} (covers {1, 2, 4, L/4} for 12-block ViTs).
      * Surrogates S1 (isotropic Gaussian), S2 (empirical mean), S3 (cross-image [CLS]), S4 (PCA Gaussian).
      * 5 seeds for stochastic surrogates (S1, S3, S4) with seed variance reported separately from image CI (§10.2).
      * Read-outs: Δbox-AUC (primary, GPU), ΔOtsu CorLoc, 1 - cos(z_clean, z_int), ΔkNN@20 (cosine),
        and KL(p_clean || p_int) + top-1 agreement for classifier models (DeiT-III).
  - I2 (Token-Group Knockout at final block l=L-1):
      * Groups: FG, BG, OUT, REG (exact M3 partition).
      * Both zero-ablation (zero [CLS] attn + renormalize) AND mean-ablation (replace v_i with dataset mean v_bar).
      * Matched random control: drops exact same count m_i of random patch tokens, 5 seeds, per group & mode.
      * Reports group effect, random effect (mean ± seed-SD), and net specific effect (group - random) with 95% CI.

Fast GPU execution: caches pre-block-(L-max_L_int) hidden states z_start once per model so all I1/I2 sweeps
run only the last 1-4 transformer blocks with zero repeated image I/O.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.preprocess import (
    _resolve_cub_root,
    build_image_transform,
    get_transformed_gt_mask,
    load_cub_pilot_split,
    patch_fg_mask,
)
from interventions import forward_ops as fops
from interventions import i1_i2 as I12
from measures import qii
from measures.routing import detect_outlier_tokens
from scripts.run_pilot_m1_m2 import (
    PILOT8_SPECS,
    _file_sha256_16,
    _git_sha,
    _load_registry_meta,
    bootstrap_ci,
    load_pilot_model,
    paired_bootstrap_diff_ci,
)


def _load_cub_class_labels(cub_dir: str | Path, cub_ids: list[int]) -> np.ndarray:
    """Read ground-truth 200-way CUB species labels from image_class_labels.txt (or fallback zeros)."""
    root = _resolve_cub_root(cub_dir)
    lbl_file = root / "image_class_labels.txt"
    if not lbl_file.exists():
        return np.zeros(len(cub_ids), dtype=np.int64)
    id_to_cls = {}
    for ln in lbl_file.read_text().splitlines():
        if ln.strip():
            cid, cls_id = ln.strip().split()
            id_to_cls[int(cid)] = int(cls_id) - 1
    return np.array([id_to_cls.get(cid, 0) for cid in cub_ids], dtype=np.int64)


@torch.no_grad()
def _run_cached_tail_batched(
    adapter,
    model,
    z_start_all: torch.Tensor,  # (I, T, D) on CPU
    layout: dict,
    start_block: int,
    device: torch.device,
    batch_size: int = 32,
    *,
    q_cls_overrides_all: dict[int, np.ndarray] | None = None,
    ko_masks_all: dict[int, np.ndarray] | None = None,
    ko_mode: str = "zero",
    v_means: dict[int, torch.Tensor] | None = None,
    capture_clean_stats: bool = False,
) -> dict:
    """Run blocks [start_block..L-1] in mini-batches over cached z_start_all."""
    n_img = z_start_all.shape[0]
    embs, logits_list, wmaps = [], [], []
    q_g_acc: dict[int, list[torch.Tensor]] = {}
    v_sum_acc: dict[int, torch.Tensor] = {}
    v_cnt_acc: dict[int, int] = {}
    pnorm_acc: dict[int, list[torch.Tensor]] = {}

    for s in range(0, n_img, batch_size):
        e = min(n_img, s + batch_size)
        zb = z_start_all[s:e].to(device)
        q_ov_b = (
            {l: torch.as_tensor(arr[s:e], dtype=torch.float32, device=device) for l, arr in q_cls_overrides_all.items()}
            if q_cls_overrides_all else None
        )
        ko_b = (
            {l: torch.as_tensor(arr[s:e], dtype=torch.bool, device=device) for l, arr in ko_masks_all.items()}
            if ko_masks_all else None
        )
        res = fops.forward_from_block_with_interventions(
            adapter, model, zb, layout, start_block,
            q_cls_overrides=q_ov_b, ko_masks=ko_b, ko_mode=ko_mode, v_means=v_means,
            capture_clean_stats=capture_clean_stats,
        )
        embs.append(res["emb"].cpu().numpy())
        if res["logits"] is not None:
            logits_list.append(res["logits"].cpu().numpy())
        wmaps.append(res["last_weighted_map"].cpu().numpy())
        if capture_clean_stats:
            for l, qg in res["q_g_by_layer"].items():
                q_g_acc.setdefault(l, []).append(qg)
                v_sum_acc[l] = v_sum_acc[l] + res["v_sum_by_layer"][l] if l in v_sum_acc else res["v_sum_by_layer"][l].clone()
                v_cnt_acc[l] = v_cnt_acc.get(l, 0) + res["v_count_by_layer"][l]
                pnorm_acc.setdefault(l, []).append(res["patch_norms_by_l"][l])

    out = {
        "emb": np.concatenate(embs, axis=0),
        "logits": np.concatenate(logits_list, axis=0) if logits_list else None,
        "wmap": np.concatenate(wmaps, axis=0),
    }
    if capture_clean_stats:
        out["q_g_by_layer"] = {l: torch.cat(v, dim=0).numpy() for l, v in q_g_acc.items()}
        out["v_means"] = {l: (v_sum_acc[l] / float(v_cnt_acc[l])).float() for l in v_sum_acc}
        out["patch_norms_by_l"] = {l: torch.cat(v, dim=0).numpy() for l, v in pnorm_acc.items()}
    return out


def run_one_model_i1_i2(
    model_id: str,
    samples,
    class_labels: np.ndarray,
    gt_cache: dict[int, tuple],
    device: torch.device,
    out_dir: Path,
    weights_cache: Path,
    reg_meta: dict,
    *,
    batch_size: int = 32,
    l_int_sweep: tuple[int, ...] = I12.DEFAULT_L_INT_SWEEP,
    seeds: tuple[int, ...] = I12.DEFAULT_SEEDS,
    knn_k: int = 20,
    smoke_random: bool = False,
) -> dict:
    t0 = time.perf_counter()
    adapter, model, mean, std, target_size = load_pilot_model(
        model_id, device, weights_cache, smoke_random=smoke_random,
        keep_head=(model_id == "deit3_small_patch16_224"),
    )
    n_blocks = adapter.n_blocks(model)
    max_l_int = min(max(l_int_sweep), n_blocks)
    start_block = n_blocks - max_l_int
    last_block = n_blocks - 1
    tf = build_image_transform(target_size, mean, std)
    n_img = len(samples)

    if target_size not in gt_cache:
        masks, boxes_list, fg_patches = [], [], []
        for s in samples:
            with Image.open(s.image_path) as im:
                ow, oh = im.size
            mask, tbox = get_transformed_gt_mask(s, ow, oh, target_size, prefer_seg=True)
            masks.append(mask.astype(np.uint8))
            boxes_list.append([tbox])
            fg_patches.append(patch_fg_mask(mask, (14, 14)).reshape(-1).astype(bool))
        gt_cache[target_size] = (np.stack(masks, axis=0), boxes_list, np.stack(fg_patches, axis=0))
    gt_masks_np, gt_boxes_list, fg_patch_masks = gt_cache[target_size]

    print(f"\n[{model_id}] I1+I2 | device={device}, res={target_size}, n={n_img}, start_block={start_block}", flush=True)

    # 1. Cache z_start (input to start_block) once
    t_ext = time.perf_counter()
    z_chunks = []
    layout = None
    for s in range(0, n_img, batch_size):
        batch = samples[s : s + batch_size]
        imgs = [tf(Image.open(it.image_path).convert("RGB")) for it in batch]
        xb = torch.stack(imgs, dim=0).to(device)
        zb, lay = fops.forward_to_block_input(adapter, model, xb, start_block)
        layout = lay
        z_chunks.append(zb.cpu())
    z_start_all = torch.cat(z_chunks, dim=0)
    grid = layout["grid"]

    # 2. Clean baseline pass over blocks [start_block .. L-1]
    clean = _run_cached_tail_batched(
        adapter, model, z_start_all, layout, start_block, device, batch_size, capture_clean_stats=True,
    )
    clean_sec = time.perf_counter() - t_ext

    clean_auc, clean_otsu = I12.eval_maps_box_auc_and_otsu(
        clean["wmap"], grid, target_size, gt_masks_np, gt_boxes_list, device=device, compute_otsu=True,
    )
    clean_knn_acc, clean_knn_hits = I12.knn_accuracy_cosine(clean["emb"], class_labels, k=knn_k, device=device)
    has_logits = clean["logits"] is not None

    # Fit per-layer empirical PCA for I1 S2/S4 surrogates
    pca_by_l = {l: qii.fit_query_pca(qg, k=qii.DEFAULT_PCA_K) for l, qg in clean["q_g_by_layer"].items()}
    v_means = clean["v_means"]
    out_patch_last, _ = detect_outlier_tokens(
        clean["patch_norms_by_l"][last_block], np.arange(len(layout["patch_idx"])), k_mad=5.0
    )

    print(
        f"  Clean baseline: box-AUC={float(np.nanmean(clean_auc)):.3f}, Otsu={float(np.nanmean(clean_otsu)):.3f}, "
        f"kNN@{knn_k}={clean_knn_acc:.3f}, has_logits={has_logits} ({clean_sec:.1f}s)",
        flush=True,
    )

    # ----------------------------------------------------------------------- 3. Intervention I1 (Query Replacement)
    i1_results: list[dict] = []
    for l_int in l_int_sweep:
        if l_int > n_blocks:
            continue
        edit_layers = list(range(n_blocks - l_int, n_blocks))
        for fam in I12.DEFAULT_I1_SURROGATES:
            fam_seeds = (seeds[0],) if fam == "S2" else seeds
            auc_by_s, cos_by_s, knn_by_s, kl_by_s, t1_by_s = [], [], [], [], []
            otsu_s0 = None

            for si, sd in enumerate(fam_seeds):
                q_ov = {
                    l: I12.sample_surrogate_queries_batch(fam, clean["q_g_by_layer"][l], pca_by_l[l], seed=100 * sd + l)
                    for l in edit_layers
                }
                res = _run_cached_tail_batched(
                    adapter, model, z_start_all, layout, start_block, device, batch_size, q_cls_overrides_all=q_ov,
                )
                aucs, otsus = I12.eval_maps_box_auc_and_otsu(
                    res["wmap"], grid, target_size, gt_masks_np, gt_boxes_list, device=device, compute_otsu=(si == 0),
                )
                if si == 0:
                    otsu_s0 = otsus
                cos_v = I12.cosine_similarity_batch(clean["emb"], res["emb"])
                _, knn_h = I12.knn_accuracy_cosine(res["emb"], class_labels, feats_ref=clean["emb"], k=knn_k, device=device)

                auc_by_s.append(aucs)
                cos_by_s.append(cos_v)
                knn_by_s.append(knn_h)
                if has_logits:
                    kl_by_s.append(I12.kl_divergence_logits(clean["logits"], res["logits"]))
                    t1_by_s.append(I12.top1_agreement(clean["logits"], res["logits"]))

            # Average per image across seeds for paired image-level bootstrap CI, and log seed-SD separately (§10.2)
            auc_img = np.mean(auc_by_s, axis=0)
            cos_img = np.mean(cos_by_s, axis=0)
            knn_img = np.mean(knn_by_s, axis=0)
            d_auc = paired_bootstrap_diff_ci(auc_img, clean_auc, seed=0)
            d_otsu = paired_bootstrap_diff_ci(otsu_s0, clean_otsu, seed=0)
            d_knn = paired_bootstrap_diff_ci(knn_img, clean_knn_hits, seed=0)
            cos_drop_m, cos_drop_lo, cos_drop_hi = bootstrap_ci(1.0 - cos_img, seed=0)

            seed_auc_means = [float(np.nanmean(a - clean_auc)) for a in auc_by_s]
            entry = {
                "model_id": model_id,
                "objective": reg_meta.get(model_id, {}).get("objective", "unknown"),
                "L_int": int(l_int),
                "surrogate": fam,
                "n_seeds": len(fam_seeds),
                "box_auc_clean": round(float(np.nanmean(clean_auc)), 4),
                "box_auc_int": round(float(np.nanmean(auc_img)), 4),
                "delta_box_auc": round(d_auc["mean_diff"], 4),
                "delta_box_auc_ci95": [round(d_auc["ci95_low"], 4), round(d_auc["ci95_high"], 4)],
                "delta_box_auc_seed_sd": round(float(np.std(seed_auc_means)), 5),
                "delta_otsu_corloc": round(d_otsu["mean_diff"], 4),
                "cos_drop_mean": round(cos_drop_m, 4),
                "cos_drop_ci95": [round(cos_drop_lo, 4), round(cos_drop_hi, 4)],
                "knn_clean": round(float(clean_knn_acc), 4),
                "delta_knn": round(d_knn["mean_diff"], 4),
                "delta_knn_ci95": [round(d_knn["ci95_low"], 4), round(d_knn["ci95_high"], 4)],
            }
            if has_logits:
                kl_img = np.mean(kl_by_s, axis=0)
                t1_img = np.mean(t1_by_s, axis=0)
                kl_m, kl_lo, kl_hi = bootstrap_ci(kl_img, seed=0)
                t1_m, _, _ = bootstrap_ci(t1_img, seed=0)
                entry["kl_mean"] = round(kl_m, 4)
                entry["kl_ci95"] = [round(kl_lo, 4), round(kl_hi, 4)]
                entry["top1_agree"] = round(t1_m, 4)
            i1_results.append(entry)

        # Print compact S4 progress per L_int
        s4_row = [r for r in i1_results if r["L_int"] == l_int and r["surrogate"] == "S4"][0]
        s1_row = [r for r in i1_results if r["L_int"] == l_int and r["surrogate"] == "S1"][0]
        print(
            f"  I1 L_int={l_int} | S4: ΔAUC={s4_row['delta_box_auc']:+.4f}, 1-cos={s4_row['cos_drop_mean']:.4f}, ΔkNN={s4_row['delta_knn']:+.4f} | "
            f"S1: ΔAUC={s1_row['delta_box_auc']:+.4f}, 1-cos={s1_row['cos_drop_mean']:.4f}, ΔkNN={s1_row['delta_knn']:+.4f}",
            flush=True,
        )

    # ----------------------------------------------------------------------- 4. Intervention I2 (Token-Group Knockout)
    group_masks = I12.build_token_group_masks(fg_patch_masks, out_patch_last, layout)
    i2_results: list[dict] = []
    for grp in I12.DEFAULT_I2_GROUPS:
        gmask = group_masks[grp]
        mean_cnt = float(gmask.sum(axis=1).mean())
        if mean_cnt <= 0.0:
            continue
        for mode in ("zero", "mean_v"):
            # Group knockout
            res_g = _run_cached_tail_batched(
                adapter, model, z_start_all, layout, start_block, device, batch_size,
                ko_masks_all={last_block: gmask}, ko_mode=mode, v_means=v_means,
            )
            auc_g, _ = I12.eval_maps_box_auc_and_otsu(
                res_g["wmap"], grid, target_size, gt_masks_np, None, device=device, compute_otsu=False,
            )
            cos_drop_g = 1.0 - I12.cosine_similarity_batch(clean["emb"], res_g["emb"])
            _, knn_g = I12.knn_accuracy_cosine(res_g["emb"], class_labels, feats_ref=clean["emb"], k=knn_k, device=device)
            kl_g = I12.kl_divergence_logits(clean["logits"], res_g["logits"]) if has_logits else None

            # Matched random knockout (5 seeds, exact same count m_i per image, patch tokens only)
            auc_r_list, cos_r_list, knn_r_list, kl_r_list = [], [], [], []
            for sd in seeds:
                rmask = I12.make_matched_random_patch_mask(gmask, layout, seed=500 * sd + hash(grp) % 97)
                res_r = _run_cached_tail_batched(
                    adapter, model, z_start_all, layout, start_block, device, batch_size,
                    ko_masks_all={last_block: rmask}, ko_mode=mode, v_means=v_means,
                )
                ar, _ = I12.eval_maps_box_auc_and_otsu(
                    res_r["wmap"], grid, target_size, gt_masks_np, None, device=device, compute_otsu=False,
                )
                auc_r_list.append(ar)
                cos_r_list.append(1.0 - I12.cosine_similarity_batch(clean["emb"], res_r["emb"]))
                _, kr = I12.knn_accuracy_cosine(res_r["emb"], class_labels, feats_ref=clean["emb"], k=knn_k, device=device)
                knn_r_list.append(kr)
                if has_logits:
                    kl_r_list.append(I12.kl_divergence_logits(clean["logits"], res_r["logits"]))

            auc_r = np.mean(auc_r_list, axis=0)
            cos_drop_r = np.mean(cos_r_list, axis=0)
            knn_r = np.mean(knn_r_list, axis=0)

            # Paired net specific effects: (group_shift - matched_random_shift)
            d_auc_g = auc_g - clean_auc
            d_auc_r = auc_r - clean_auc
            net_auc = paired_bootstrap_diff_ci(d_auc_g, d_auc_r, seed=0)
            net_cos = paired_bootstrap_diff_ci(cos_drop_g, cos_drop_r, seed=0)
            net_knn = paired_bootstrap_diff_ci(knn_g - clean_knn_hits, knn_r - clean_knn_hits, seed=0)

            row = {
                "model_id": model_id,
                "objective": reg_meta.get(model_id, {}).get("objective", "unknown"),
                "layer": int(last_block),
                "group": grp,
                "mode": mode,
                "mean_tokens_ko": round(mean_cnt, 2),
                "n_random_seeds": len(seeds),
                # Functional shift (1 - cos)
                "cos_drop_group": round(float(np.mean(cos_drop_g)), 4),
                "cos_drop_rand": round(float(np.mean(cos_drop_r)), 4),
                "cos_drop_rand_seed_sd": round(float(np.std([np.mean(c) for c in cos_r_list])), 5),
                "net_cos_drop": round(net_cos["mean_diff"], 4),
                "net_cos_drop_ci95": [round(net_cos["ci95_low"], 4), round(net_cos["ci95_high"], 4)],
                "net_cos_excl0": bool(net_cos["ci_excludes_zero"]),
                # kNN accuracy shift
                "delta_knn_group": round(float(np.mean(knn_g - clean_knn_hits)), 4),
                "delta_knn_rand": round(float(np.mean(knn_r - clean_knn_hits)), 4),
                "net_delta_knn": round(net_knn["mean_diff"], 4),
                "net_delta_knn_ci95": [round(net_knn["ci95_low"], 4), round(net_knn["ci95_high"], 4)],
                # Map shift (box-AUC)
                "delta_auc_group": round(float(np.nanmean(d_auc_g)), 4),
                "delta_auc_rand": round(float(np.nanmean(d_auc_r)), 4),
                "net_delta_auc": round(net_auc["mean_diff"], 4),
                "net_delta_auc_ci95": [round(net_auc["ci95_low"], 4), round(net_auc["ci95_high"], 4)],
            }
            if has_logits and kl_g is not None:
                kl_r = np.mean(kl_r_list, axis=0)
                net_kl = paired_bootstrap_diff_ci(kl_g, kl_r, seed=0)
                row["kl_group"] = round(float(np.mean(kl_g)), 4)
                row["kl_rand"] = round(float(np.mean(kl_r)), 4)
                row["net_kl"] = round(net_kl["mean_diff"], 4)
                row["net_kl_ci95"] = [round(net_kl["ci95_low"], 4), round(net_kl["ci95_high"], 4)]
            i2_results.append(row)
            print(
                f"  I2 {grp:<3s} ({mode:<6s}, m={mean_cnt:5.1f}) | 1-cos: grp={row['cos_drop_group']:.4f} vs rnd={row['cos_drop_rand']:.4f} "
                f"(net={row['net_cos_drop']:+.4f}) | ΔAUC: grp={row['delta_auc_group']:+.4f} vs rnd={row['delta_auc_rand']:+.4f} "
                f"(net={row['net_delta_auc']:+.4f})",
                flush=True,
            )

    del model, z_start_all
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    total_sec = time.perf_counter() - t0
    res_obj = {
        "model_id": model_id,
        "objective": reg_meta.get(model_id, {}).get("objective", "unknown"),
        "n_images": n_img,
        "resolution": target_size,
        "clean_box_auc": round(float(np.nanmean(clean_auc)), 4),
        "clean_otsu_corloc": round(float(np.nanmean(clean_otsu)), 4),
        "clean_knn": round(float(clean_knn_acc), 4),
        "has_logits": has_logits,
        "i1_results": i1_results,
        "i2_results": i2_results,
        "clean_sec": round(clean_sec, 2),
        "total_sec": round(total_sec, 2),
    }
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "models" / f"{model_id}.json.tmp"
    tmp.write_text(json.dumps(res_obj, indent=2))
    os.replace(tmp, out_dir / "models" / f"{model_id}.json")
    return res_obj


def write_i1_i2_report(results: dict[str, dict], out_dir: Path) -> str:
    lines = ["# Pilot Interventions I1 (Query Replacement) & I2 (Token-Group Knockout) Summary\n"]

    lines.append("## 1. Intervention I1 — Query Replacement Curve (`S4` & `S1` across `L_int ∈ {1,2,3,4}`)\n")
    lines.append("| Model | Obj | Surr | L_int | Clean AUC | Δbox-AUC [95% CI] | Seed SD | ΔOtsu | 1−cos [95% CI] | ΔkNN@20 | KL (if cls) |")
    lines.append("|---|---|---|---:|---:|---|---:|---:|---|---:|---:|")
    for m, r in results.items():
        for row in r["i1_results"]:
            if row["surrogate"] not in ("S4", "S1"):
                continue
            kl_s = f"{row['kl_mean']:.4f}" if "kl_mean" in row else "—"
            lines.append(
                f"| {m} | {row['objective']} | {row['surrogate']} | {row['L_int']} | {row['box_auc_clean']:.3f} | "
                f"{row['delta_box_auc']:+.4f} [{row['delta_box_auc_ci95'][0]:+.4f}, {row['delta_box_auc_ci95'][1]:+.4f}] | "
                f"{row['delta_box_auc_seed_sd']:.4f} | {row['delta_otsu_corloc']:+.3f} | "
                f"{row['cos_drop_mean']:.4f} [{row['cos_drop_ci95'][0]:.4f}, {row['cos_drop_ci95'][1]:.4f}] | "
                f"{row['delta_knn']:+.4f} | {kl_s} |"
            )

    lines.append("\n## 2. Intervention I2 — Token-Group Knockout vs Matched Random Control (`zero` & `mean_v`, `l=11`)\n")
    lines.append("| Model | Obj | Group | Mode | m_KO | 1−cos (grp) | 1−cos (rnd) | Net 1−cos [95% CI] | Net ΔkNN | Net Δbox-AUC [95% CI] | Net KL |")
    lines.append("|---|---|---|---|---:|---:|---:|---|---:|---|---:|")
    for m, r in results.items():
        for row in r["i2_results"]:
            kl_s = f"{row['net_kl']:+.4f}" if "net_kl" in row else "—"
            lines.append(
                f"| {m} | {row['objective']} | {row['group']} | {row['mode']} | {row['mean_tokens_ko']:.1f} | "
                f"{row['cos_drop_group']:.4f} | {row['cos_drop_rand']:.4f} | "
                f"{row['net_cos_drop']:+.4f} [{row['net_cos_drop_ci95'][0]:+.4f}, {row['net_cos_drop_ci95'][1]:+.4f}] | "
                f"{row['net_delta_knn']:+.4f} | "
                f"{row['net_delta_auc']:+.4f} [{row['net_delta_auc_ci95'][0]:+.4f}, {row['net_delta_auc_ci95'][1]:+.4f}] | {kl_s} |"
            )

    md_text = "\n".join(lines) + "\n"
    (out_dir / "summary_i1_i2.md").write_text(md_text)
    (out_dir / "i1_i2_report.json").write_text(json.dumps(results, indent=2))
    return md_text


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Run Pilot Interventions I1 & I2 with matched controls on CUB-500.")
    ap.add_argument("--cub_dir", required=True)
    ap.add_argument("--split_file", default=str(REPO_ROOT / "configs" / "splits" / "cub_pilot_500_seed0.txt"))
    ap.add_argument("--out_dir", default="results/pilot_interventions_i1_i2_cub")
    ap.add_argument("--models", default="all")
    ap.add_argument("--n_images", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--l_int_sweep", default="1,2,3,4")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--knn_k", type=int, default=20)
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
    l_int_sweep = tuple(int(x.strip()) for x in args.l_int_sweep.split(",") if x.strip())
    seeds = tuple(int(x.strip()) for x in args.seeds.split(",") if x.strip())

    samples = load_cub_pilot_split(args.cub_dir, n_images=args.n_images, seed=0, split_file=args.split_file)
    class_labels = _load_cub_class_labels(args.cub_dir, [s.cub_id for s in samples])
    print(f"Loaded {len(samples)} CUB images ({len(np.unique(class_labels))} unique species) on device={device}", flush=True)

    reg_meta = _load_registry_meta()
    weights_cache = Path(args.weights_cache_dir)
    gt_cache: dict[int, tuple] = {}
    results: dict[str, dict] = {}
    t_all = time.perf_counter()

    for mid in model_ids:
        done = out_dir / "models" / f"{mid}.json"
        if args.resume and done.exists():
            print(f"[{mid}] --resume: reusing {done}", flush=True)
            results[mid] = json.loads(done.read_text())
            continue
        results[mid] = run_one_model_i1_i2(
            mid, samples, class_labels, gt_cache, device, out_dir, weights_cache, reg_meta,
            batch_size=args.batch_size, l_int_sweep=l_int_sweep, seeds=seeds, knn_k=args.knn_k,
            smoke_random=args.smoke_random,
        )

    md_text = write_i1_i2_report(results, out_dir)
    cfg_payload = json.dumps({
        "models": model_ids, "n_images": len(samples), "l_int_sweep": l_int_sweep,
        "seeds": seeds, "knn_k": args.knn_k, "smoke_random": args.smoke_random,
    }, sort_keys=True)
    meta = {
        "git_sha": _git_sha(),
        "config_hash": hashlib.sha256(cfg_payload.encode()).hexdigest()[:16],
        "file_sha256_16": {
            rel: _file_sha256_16(REPO_ROOT / rel)
            for rel in ("interventions/forward_ops.py", "interventions/i1_i2.py", "scripts/run_pilot_interventions_i1_i2.py")
        },
        "device": str(device),
        "dataset": "CUB-200-2011",
        "n_images": len(samples),
        "smoke_random": bool(args.smoke_random),
        "total_seconds": round(time.perf_counter() - t_all, 2),
        "unix_time": int(time.time()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print("\n" + md_text, flush=True)
    return results


if __name__ == "__main__":
    main()
