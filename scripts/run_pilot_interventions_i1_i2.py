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


def _load_cub_train_gallery(cub_dir: str | Path) -> tuple[list[Path], np.ndarray]:
    """Load official CUB-200-2011 train split (is_train == 1, 5,994 images) as full kNN gallery."""
    root = _resolve_cub_root(cub_dir)
    img_file = root / "images.txt"
    split_file = root / "train_test_split.txt"
    lbl_file = root / "image_class_labels.txt"
    if not (img_file.exists() and split_file.exists() and lbl_file.exists()):
        return [], np.zeros((0,), dtype=np.int64)

    id_to_rel = {}
    for ln in img_file.read_text().splitlines():
        if ln.strip():
            cid, rel = ln.strip().split(maxsplit=1)
            id_to_rel[int(cid)] = rel
    id_to_cls = {}
    for ln in lbl_file.read_text().splitlines():
        if ln.strip():
            cid, cls_id = ln.strip().split()
            id_to_cls[int(cid)] = int(cls_id) - 1

    paths: list[Path] = []
    labels: list[int] = []
    for ln in split_file.read_text().splitlines():
        if not ln.strip():
            continue
        cid_s, is_tr_s = ln.strip().split()
        if int(is_tr_s) == 1:
            cid = int(cid_s)
            p = root / "images" / id_to_rel[cid]
            if p.exists():
                paths.append(p)
                labels.append(id_to_cls.get(cid, 0))
    return paths, np.array(labels, dtype=np.int64)


@torch.no_grad()
def _extract_gallery_embeddings(
    adapter,
    model,
    gallery_paths: list[Path],
    tf,
    device: torch.device,
    cache_path: Path | None = None,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (or load cached) clean [CLS] and patch-mean embeddings for the full kNN gallery."""
    if cache_path is not None and cache_path.exists():
        dat = np.load(cache_path)
        if dat["emb"].shape[0] == len(gallery_paths):
            return dat["emb"], dat["emb_patch_mean"]

    embs, pmeans = [], []
    layout = None
    for s in range(0, len(gallery_paths), batch_size):
        batch = gallery_paths[s : s + batch_size]
        imgs = [tf(Image.open(p).convert("RGB")) for p in batch]
        xb = torch.stack(imgs, dim=0).to(device)
        zf, layout = fops.forward_to_block_input(adapter, model, xb, adapter.n_blocks(model))
        emb, _ = fops.read_global_embedding_and_logits(adapter, model, zf, layout)
        pm = fops.read_patch_mean_embedding(adapter, model, zf, layout)
        embs.append(emb.cpu().numpy().astype(np.float32))
        pmeans.append(pm.cpu().numpy().astype(np.float32))

    emb_all = np.concatenate(embs, axis=0)
    pm_all = np.concatenate(pmeans, axis=0)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".tmp.npz")
        np.savez(tmp, emb=emb_all, emb_patch_mean=pm_all)
        os.replace(tmp, cache_path)
    return emb_all, pm_all


def _safe_bootstrap_ci(vals: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap CI filtering NaNs cleanly; returns (nan, nan, nan) if <2 valid entries."""
    v = np.asarray(vals, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    if v.size == 1:
        x = float(v[0])
        return x, x, x
    return bootstrap_ci(v, seed=seed)


def _safe_paired_diff_ci(a: np.ndarray, b: np.ndarray, seed: int = 0) -> dict:
    """Paired bootstrap diff CI filtering positions where either a or b is NaN."""
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(aa) & np.isfinite(bb)
    n_ok = int(ok.sum())
    if n_ok == 0:
        return {"mean_diff": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "ci_excludes_zero": False, "n_valid": 0}
    if n_ok == 1:
        d = float(aa[ok][0] - bb[ok][0])
        return {"mean_diff": d, "ci95_low": d, "ci95_high": d, "ci_excludes_zero": False, "n_valid": 1}
    res = paired_bootstrap_diff_ci(aa[ok], bb[ok], seed=seed)
    res["n_valid"] = n_ok
    return res


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
    embs, pmeans, logits_list, wmaps = [], [], [], []
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
        pmeans.append(res["emb_patch_mean"].cpu().numpy())
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
        "emb_patch_mean": np.concatenate(pmeans, axis=0),
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
    gallery_paths: list[Path] | None = None,
    gallery_labels: np.ndarray | None = None,
    batch_size: int = 32,
    l_int_sweep: tuple[int, ...] = I12.DEFAULT_L_INT_SWEEP,
    seeds: tuple[int, ...] = I12.DEFAULT_SEEDS,
    knn_k: int = 20,
    i2_layers: tuple[int, ...] | None = None,
    smoke_random: bool = False,
) -> dict:
    t0 = time.perf_counter()
    adapter, model, mean, std, target_size = load_pilot_model(
        model_id, device, weights_cache, smoke_random=smoke_random,
        keep_head=(model_id == "deit3_small_patch16_224"),
    )
    n_blocks = adapter.n_blocks(model)
    max_l_int = min(max(l_int_sweep), n_blocks)
    last_block = n_blocks - 1
    eff_i2_layers = tuple(sorted(set(i2_layers))) if i2_layers is not None else (max(0, last_block - 1), last_block)
    start_block = min(n_blocks - max_l_int, min(eff_i2_layers))
    tf = build_image_transform(target_size, mean, std)
    n_img = len(samples)
    use_patch_mean = (model_id == "mae_vitb16")

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

    print(
        f"\n[{model_id}] I1+I2 | device={device}, res={target_size}, n={n_img}, "
        f"start_block={start_block}, i2_layers={eff_i2_layers}",
        flush=True,
    )

    # 0. Extract (or load cached) full training gallery embeddings for cosine kNN@20
    use_full_gallery = bool(gallery_paths) and gallery_labels is not None and len(gallery_paths) >= knn_k
    gal_emb, gal_pmean, gal_labels = None, None, None
    knn_gallery_mode = "loo_pilot"
    n_gallery = n_img
    if use_full_gallery:
        t_gal = time.perf_counter()
        cache_p = None if smoke_random else (out_dir / "gallery_cache" / f"{model_id}.npz")
        gal_emb, gal_pmean = _extract_gallery_embeddings(
            adapter, model, gallery_paths, tf, device, cache_path=cache_p, batch_size=max(batch_size, 64),
        )
        gal_labels = gallery_labels
        knn_gallery_mode = "full_train"
        n_gallery = len(gallery_paths)
        print(f"  kNN gallery: full_train (n={n_gallery}, {time.perf_counter() - t_gal:.1f}s)", flush=True)

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
    patch_idx = np.asarray(layout["patch_idx"], dtype=np.int64)

    # 2. Clean baseline pass over blocks [start_block .. L-1]
    clean = _run_cached_tail_batched(
        adapter, model, z_start_all, layout, start_block, device, batch_size, capture_clean_stats=True,
    )
    clean_sec = time.perf_counter() - t_ext

    clean_auc, clean_otsu = I12.eval_maps_box_auc_and_otsu(
        clean["wmap"], grid, target_size, gt_masks_np, gt_boxes_list, device=device, compute_otsu=True,
    )
    clean_patch_auc_all = I12.eval_patch_box_auc_unablated(clean["wmap"], fg_patch_masks, ablated_patch_mask=None)

    def _eval_knn(q_emb: np.ndarray, *, patch_mean: bool = False) -> tuple[float, np.ndarray]:
        if use_full_gallery:
            ref_e = gal_pmean if patch_mean else gal_emb
            return I12.knn_accuracy_cosine(
                q_emb, class_labels, feats_ref=ref_e, labels_ref=gal_labels, exclude_self=False, k=knn_k, device=device,
            )
        ref_e = clean["emb_patch_mean"] if patch_mean else clean["emb"]
        return I12.knn_accuracy_cosine(q_emb, class_labels, feats_ref=ref_e, exclude_self=True, k=knn_k, device=device)

    clean_knn_acc, clean_knn_hits = _eval_knn(clean["emb"], patch_mean=False)
    clean_pm_knn_acc, clean_pm_knn_hits = (
        _eval_knn(clean["emb_patch_mean"], patch_mean=True) if use_patch_mean else (float("nan"), None)
    )
    has_logits = clean["logits"] is not None

    # Fit per-layer empirical PCA for I1 S2/S4 surrogates
    pca_by_l = {l: qii.fit_query_pca(qg, k=qii.DEFAULT_PCA_K) for l, qg in clean["q_g_by_layer"].items()}
    v_means = clean["v_means"]
    out_patch_by_l = {
        l: detect_outlier_tokens(clean["patch_norms_by_l"][l], np.arange(len(patch_idx)), k_mad=5.0)[0]
        for l in eff_i2_layers
    }

    pm_note = f", patch_mean_kNN@{knn_k}={clean_pm_knn_acc:.3f}" if use_patch_mean else ""
    print(
        f"  Clean baseline: box-AUC={float(np.nanmean(clean_auc)):.3f} (patch={float(np.nanmean(clean_patch_auc_all)):.3f}), "
        f"Otsu={float(np.nanmean(clean_otsu)):.3f}, kNN@{knn_k}={clean_knn_acc:.3f} [{knn_gallery_mode}]{pm_note}, "
        f"has_logits={has_logits} ({clean_sec:.1f}s)",
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
            pm_cos_by_s, pm_knn_by_s = [], []
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
                _, knn_h = _eval_knn(res["emb"], patch_mean=False)

                auc_by_s.append(aucs)
                cos_by_s.append(cos_v)
                knn_by_s.append(knn_h)
                if use_patch_mean:
                    pm_cos_by_s.append(I12.cosine_similarity_batch(clean["emb_patch_mean"], res["emb_patch_mean"]))
                    _, pm_kh = _eval_knn(res["emb_patch_mean"], patch_mean=True)
                    pm_knn_by_s.append(pm_kh)
                if has_logits:
                    kl_by_s.append(I12.kl_divergence_logits(clean["logits"], res["logits"]))
                    t1_by_s.append(I12.top1_agreement(clean["logits"], res["logits"]))

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
            if use_patch_mean:
                pm_cos_img = np.mean(pm_cos_by_s, axis=0)
                pm_knn_img = np.mean(pm_knn_by_s, axis=0)
                pm_cm, pm_clo, pm_chi = bootstrap_ci(1.0 - pm_cos_img, seed=0)
                pm_dk = paired_bootstrap_diff_ci(pm_knn_img, clean_pm_knn_hits, seed=0)
                entry["patch_mean_cos_drop"] = round(pm_cm, 4)
                entry["patch_mean_cos_drop_ci95"] = [round(pm_clo, 4), round(pm_chi, 4)]
                entry["patch_mean_knn_clean"] = round(float(clean_pm_knn_acc), 4)
                entry["patch_mean_delta_knn"] = round(pm_dk["mean_diff"], 4)
                entry["patch_mean_delta_knn_ci95"] = [round(pm_dk["ci95_low"], 4), round(pm_dk["ci95_high"], 4)]
            if has_logits:
                kl_img = np.mean(kl_by_s, axis=0)
                t1_img = np.mean(t1_by_s, axis=0)
                kl_m, kl_lo, kl_hi = bootstrap_ci(kl_img, seed=0)
                t1_m, _, _ = bootstrap_ci(t1_img, seed=0)
                entry["kl_mean"] = round(kl_m, 4)
                entry["kl_ci95"] = [round(kl_lo, 4), round(kl_hi, 4)]
                entry["top1_agree"] = round(t1_m, 4)
            i1_results.append(entry)

        s4_row = [r for r in i1_results if r["L_int"] == l_int and r["surrogate"] == "S4"][0]
        s1_row = [r for r in i1_results if r["L_int"] == l_int and r["surrogate"] == "S1"][0]
        print(
            f"  I1 L_int={l_int} | S4: ΔAUC={s4_row['delta_box_auc']:+.4f}, 1-cos={s4_row['cos_drop_mean']:.4f}, ΔkNN={s4_row['delta_knn']:+.4f} | "
            f"S1: ΔAUC={s1_row['delta_box_auc']:+.4f}, 1-cos={s1_row['cos_drop_mean']:.4f}, ΔkNN={s1_row['delta_knn']:+.4f}",
            flush=True,
        )

    # ----------------------------------------------------------------------- 4. Intervention I2 (Token-Group Knockout)
    # Non-FG eligible pool for BG matched random control (prevents BG random control from hitting FG birds)
    T_tok = z_start_all.shape[1]
    non_fg_token_mask = np.zeros((n_img, T_tok), dtype=bool)
    non_fg_token_mask[:, patch_idx] = ~fg_patch_masks

    i2_results: list[dict] = []
    for i2_layer in eff_i2_layers:
        group_masks = I12.build_token_group_masks(fg_patch_masks, out_patch_by_l[i2_layer], layout)
        is_final_layer = (i2_layer == last_block)

        for grp in I12.DEFAULT_I2_GROUPS:
            gmask = group_masks[grp]
            mean_cnt = float(gmask.sum(axis=1).mean())
            if mean_cnt <= 0.0:
                continue
            elig_mask = non_fg_token_mask if grp == "BG" else None
            control_pool = "non_fg_patches" if grp == "BG" else "all_patches"

            for mode in ("zero", "mean_v"):
                # Group knockout
                res_g = _run_cached_tail_batched(
                    adapter, model, z_start_all, layout, start_block, device, batch_size,
                    ko_masks_all={i2_layer: gmask}, ko_mode=mode, v_means=v_means,
                )
                # Score box-AUC STRICTLY over non-ablated tokens:
                # - At i2_layer < last_block: l=last_block has 0 ablated tokens (downstream propagation!).
                # - At i2_layer == last_block & mode == "zero": exclude gmask[:, patch_idx] from BOTH int & clean.
                # - At i2_layer == last_block & mode == "mean_v": attention weights at last_block are structurally
                #   unchanged (v-only edit), so ΔAUC = 0.0 structurally; documented explicitly.
                if is_final_layer and mode == "zero":
                    abl_p_g = gmask[:, patch_idx]
                    auc_g = I12.eval_patch_box_auc_unablated(res_g["wmap"], fg_patch_masks, ablated_patch_mask=abl_p_g)
                    clean_auc_g_matched = I12.eval_patch_box_auc_unablated(clean["wmap"], fg_patch_masks, ablated_patch_mask=abl_p_g)
                    d_auc_g = auc_g - clean_auc_g_matched
                else:
                    auc_g = I12.eval_patch_box_auc_unablated(res_g["wmap"], fg_patch_masks, ablated_patch_mask=None)
                    d_auc_g = auc_g - clean_patch_auc_all

                cos_drop_g = 1.0 - I12.cosine_similarity_batch(clean["emb"], res_g["emb"])
                _, knn_g = _eval_knn(res_g["emb"], patch_mean=False)
                d_knn_g = knn_g - clean_knn_hits
                kl_g = I12.kl_divergence_logits(clean["logits"], res_g["logits"]) if has_logits else None

                pm_cos_g, pm_dknn_g = None, None
                if use_patch_mean:
                    pm_cos_g = 1.0 - I12.cosine_similarity_batch(clean["emb_patch_mean"], res_g["emb_patch_mean"])
                    _, pm_kg = _eval_knn(res_g["emb_patch_mean"], patch_mean=True)
                    pm_dknn_g = pm_kg - clean_pm_knn_hits

                # Matched random knockout (5 seeds, exact same count m_i per image; BG draws from non-FG pool)
                d_auc_r_list, cos_r_list, d_knn_r_list, kl_r_list = [], [], [], []
                pm_cos_r_list, pm_dknn_r_list = [], []
                for sd in seeds:
                    rmask = I12.make_matched_random_patch_mask(
                        gmask, layout, seed=500 * sd + 31 * i2_layer + (hash(grp) % 97), eligible_mask=elig_mask,
                    )
                    res_r = _run_cached_tail_batched(
                        adapter, model, z_start_all, layout, start_block, device, batch_size,
                        ko_masks_all={i2_layer: rmask}, ko_mode=mode, v_means=v_means,
                    )
                    if is_final_layer and mode == "zero":
                        abl_p_r = rmask[:, patch_idx]
                        ar = I12.eval_patch_box_auc_unablated(res_r["wmap"], fg_patch_masks, ablated_patch_mask=abl_p_r)
                        cr = I12.eval_patch_box_auc_unablated(clean["wmap"], fg_patch_masks, ablated_patch_mask=abl_p_r)
                        d_auc_r_list.append(ar - cr)
                    else:
                        ar = I12.eval_patch_box_auc_unablated(res_r["wmap"], fg_patch_masks, ablated_patch_mask=None)
                        d_auc_r_list.append(ar - clean_patch_auc_all)

                    cos_r_list.append(1.0 - I12.cosine_similarity_batch(clean["emb"], res_r["emb"]))
                    _, kr = _eval_knn(res_r["emb"], patch_mean=False)
                    d_knn_r_list.append(kr - clean_knn_hits)
                    if use_patch_mean:
                        pm_cos_r_list.append(1.0 - I12.cosine_similarity_batch(clean["emb_patch_mean"], res_r["emb_patch_mean"]))
                        _, pm_kr = _eval_knn(res_r["emb_patch_mean"], patch_mean=True)
                        pm_dknn_r_list.append(pm_kr - clean_pm_knn_hits)
                    if has_logits:
                        kl_r_list.append(I12.kl_divergence_logits(clean["logits"], res_r["logits"]))

                with np.errstate(all="ignore"):
                    d_auc_r = np.nanmean(np.stack(d_auc_r_list, axis=0), axis=0)
                cos_drop_r = np.mean(cos_r_list, axis=0)
                d_knn_r = np.mean(d_knn_r_list, axis=0)

                # Raw group CIs (critical especially for BG!) + paired net specific effects (group - random)
                cg_m, cg_lo, cg_hi = _safe_bootstrap_ci(cos_drop_g, seed=0)
                kg_m, kg_lo, kg_hi = _safe_bootstrap_ci(d_knn_g, seed=0)
                ag_m, ag_lo, ag_hi = _safe_bootstrap_ci(d_auc_g, seed=0)
                ar_m, _, _ = _safe_bootstrap_ci(d_auc_r, seed=0)

                net_auc = _safe_paired_diff_ci(d_auc_g, d_auc_r, seed=0)
                net_cos = paired_bootstrap_diff_ci(cos_drop_g, cos_drop_r, seed=0)
                net_knn = paired_bootstrap_diff_ci(d_knn_g, d_knn_r, seed=0)

                auc_note = (
                    "structural_zero_final_layer_v_only"
                    if (is_final_layer and mode == "mean_v")
                    else ("non_ablated_tokens_only" if (is_final_layer and mode == "zero") else "downstream_l11_all_tokens_unablated")
                )

                def _r4(x: float) -> float | None:
                    return None if (x is None or not np.isfinite(x)) else round(float(x), 4)

                row = {
                    "model_id": model_id,
                    "objective": reg_meta.get(model_id, {}).get("objective", "unknown"),
                    "layer": int(i2_layer),
                    "group": grp,
                    "mode": mode,
                    "control_pool": control_pool,
                    "auc_scoring": auc_note,
                    "mean_tokens_ko": round(mean_cnt, 2),
                    "n_random_seeds": len(seeds),
                    # Functional shift (1 - cos): raw group + CI, random, and net
                    "cos_drop_group": _r4(cg_m),
                    "cos_drop_group_ci95": [_r4(cg_lo), _r4(cg_hi)],
                    "cos_drop_rand": _r4(float(np.mean(cos_drop_r))),
                    "cos_drop_rand_seed_sd": round(float(np.std([np.mean(c) for c in cos_r_list])), 5),
                    "net_cos_drop": _r4(net_cos["mean_diff"]),
                    "net_cos_drop_ci95": [_r4(net_cos["ci95_low"]), _r4(net_cos["ci95_high"])],
                    "net_cos_excl0": bool(net_cos["ci_excludes_zero"]),
                    # kNN accuracy shift: raw group + CI, random, and net
                    "delta_knn_group": _r4(kg_m),
                    "delta_knn_group_ci95": [_r4(kg_lo), _r4(kg_hi)],
                    "delta_knn_rand": _r4(float(np.mean(d_knn_r))),
                    "net_delta_knn": _r4(net_knn["mean_diff"]),
                    "net_delta_knn_ci95": [_r4(net_knn["ci95_low"]), _r4(net_knn["ci95_high"])],
                    # Map shift (non-ablated patch box-AUC): raw group + CI, random, and net
                    "n_valid_auc_group": int(np.isfinite(d_auc_g).sum()),
                    "n_valid_auc_paired": int(net_auc["n_valid"]),
                    "delta_auc_group": _r4(ag_m),
                    "delta_auc_group_ci95": [_r4(ag_lo), _r4(ag_hi)],
                    "delta_auc_rand": _r4(ar_m),
                    "net_delta_auc": _r4(net_auc["mean_diff"]),
                    "net_delta_auc_ci95": [_r4(net_auc["ci95_low"]), _r4(net_auc["ci95_high"])],
                }
                if use_patch_mean and pm_cos_g is not None and pm_dknn_g is not None:
                    pm_cos_r = np.mean(pm_cos_r_list, axis=0)
                    pm_dknn_r = np.mean(pm_dknn_r_list, axis=0)
                    pm_net_c = paired_bootstrap_diff_ci(pm_cos_g, pm_cos_r, seed=0)
                    pm_net_k = paired_bootstrap_diff_ci(pm_dknn_g, pm_dknn_r, seed=0)
                    row["patch_mean_cos_drop_group"] = _r4(float(np.mean(pm_cos_g)))
                    row["patch_mean_cos_drop_rand"] = _r4(float(np.mean(pm_cos_r)))
                    row["patch_mean_net_cos_drop"] = _r4(pm_net_c["mean_diff"])
                    row["patch_mean_delta_knn_group"] = _r4(float(np.mean(pm_dknn_g)))
                    row["patch_mean_net_delta_knn"] = _r4(pm_net_k["mean_diff"])
                if has_logits and kl_g is not None:
                    kl_r = np.mean(kl_r_list, axis=0)
                    klg_m, klg_lo, klg_hi = _safe_bootstrap_ci(kl_g, seed=0)
                    net_kl = paired_bootstrap_diff_ci(kl_g, kl_r, seed=0)
                    row["kl_group"] = _r4(klg_m)
                    row["kl_group_ci95"] = [_r4(klg_lo), _r4(klg_hi)]
                    row["kl_rand"] = _r4(float(np.mean(kl_r)))
                    row["net_kl"] = _r4(net_kl["mean_diff"])
                    row["net_kl_ci95"] = [_r4(net_kl["ci95_low"]), _r4(net_kl["ci95_high"])]
                i2_results.append(row)

                auc_g_s = f"{row['delta_auc_group']:+.4f}" if row["delta_auc_group"] is not None else "   n/a"
                auc_n_s = f"{row['net_delta_auc']:+.4f}" if row["net_delta_auc"] is not None else "   n/a"
                print(
                    f"  I2 l={i2_layer:2d} {grp:<3s} ({mode:<6s}, m={mean_cnt:5.1f}, pool={control_pool:<14s}) | "
                    f"1-cos: grp={row['cos_drop_group']:.4f} vs rnd={row['cos_drop_rand']:.4f} (net={row['net_cos_drop']:+.4f}) | "
                    f"ΔAUC_unabl: grp={auc_g_s} net={auc_n_s} (n_ok={row['n_valid_auc_group']})",
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
        "knn_gallery_mode": knn_gallery_mode,
        "knn_gallery_size": int(n_gallery),
        "clean_box_auc": round(float(np.nanmean(clean_auc)), 4),
        "clean_patch_box_auc": round(float(np.nanmean(clean_patch_auc_all)), 4),
        "clean_otsu_corloc": round(float(np.nanmean(clean_otsu)), 4),
        "clean_knn": round(float(clean_knn_acc), 4),
        "clean_patch_mean_knn": round(float(clean_pm_knn_acc), 4) if use_patch_mean else None,
        "has_logits": has_logits,
        "i2_layers": list(eff_i2_layers),
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


def _fmt_opt(val: float | None, fmt: str = "+.4f") -> str:
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return "—"
    return format(val, fmt)


def _fmt_ci(val: float | None, ci: list | None, fmt: str = "+.4f") -> str:
    if val is None or ci is None or ci[0] is None or ci[1] is None:
        return "—"
    return f"{format(val, fmt)} [{format(ci[0], fmt)}, {format(ci[1], fmt)}]"


def write_i1_i2_report(results: dict[str, dict], out_dir: Path) -> str:
    first_r = next(iter(results.values()), {})
    gal_mode = first_r.get("knn_gallery_mode", "unknown")
    gal_size = first_r.get("knn_gallery_size", 0)

    lines = [
        "# Pilot Interventions I1 (Query Replacement) & I2 (Token-Group Knockout) Summary (v2 — Fixed Read-Outs)\n",
        f"- **kNN gallery:** `{gal_mode}` (`N = {gal_size}` reference images, `k = 20` cosine-weighted).",
        "- **I2 `box-AUC` scoring:** strictly over **non-ablated tokens** (`l=10` knockout propagates into `l=11` with 0 ablated tokens; `l=11` `zero` excludes knocked-out patches from both clean and intervened AUC; `l=11` `mean_v` is structurally `0.0000` on the `l=11` map because `V`-only edits cannot alter `l=11` attention weights, while `l=10` `mean_v` produces real downstream `Δbox-AUC`).",
        "- **I2 `BG` control:** matched random pool restricted to **non-FG patches** (`control_pool = non_fg_patches`), and raw group effects with 95% CIs are reported alongside net effects.\n",
    ]

    lines.append("## 1. Intervention I1 — Full Surrogate Curve (`S1, S2, S3, S4` across `L_int ∈ {1,2,3,4}`)\n")
    lines.append("| Model | Obj | Surr | Seeds | L_int | Clean AUC | Δbox-AUC [95% CI] | Seed SD | ΔOtsu | 1−cos [95% CI] | ΔkNN@20 | KL (if cls) |")
    lines.append("|---|---|---|---:|---:|---:|---|---:|---:|---|---:|---:|")
    for m, r in results.items():
        for row in r["i1_results"]:
            kl_s = f"{row['kl_mean']:.4f}" if "kl_mean" in row else "—"
            lines.append(
                f"| {m} | {row['objective']} | {row['surrogate']} | {row['n_seeds']} | {row['L_int']} | {row['box_auc_clean']:.3f} | "
                f"{_fmt_ci(row['delta_box_auc'], row['delta_box_auc_ci95'])} | "
                f"{row['delta_box_auc_seed_sd']:.4f} | {row['delta_otsu_corloc']:+.3f} | "
                f"{_fmt_ci(row['cos_drop_mean'], row['cos_drop_ci95'], '.4f')} | "
                f"{row['delta_knn']:+.4f} | {kl_s} |"
            )

    lines.append("\n## 2. Intervention I2 — Token-Group Knockout (Raw Group Effects + Matched Random Control, `l=10` & `l=11`)\n")
    lines.append("| Model | Layer | Group | Mode | Pool | m_KO | Raw 1−cos [95% CI] | Rnd 1−cos | Net 1−cos [95% CI] | Raw ΔkNN | Net ΔkNN | Raw ΔAUC_unabl [95% CI] (n_ok) | Net ΔAUC_unabl | Net KL |")
    lines.append("|---|---:|---|---|---|---:|---|---:|---|---:|---:|---|---|---:|")
    for m, r in results.items():
        for row in r["i2_results"]:
            kl_s = _fmt_opt(row.get("net_kl")) if "net_kl" in row else "—"
            raw_auc_s = (
                f"{_fmt_ci(row['delta_auc_group'], row['delta_auc_group_ci95'])} ({row['n_valid_auc_group']})"
                if row["delta_auc_group"] is not None
                else f"— ({row['n_valid_auc_group']})"
            )
            if row.get("auc_scoring") == "structural_zero_final_layer_v_only":
                raw_auc_s = "0.0000 *(v-only @ l_last)*"
            net_auc_s = _fmt_opt(row["net_delta_auc"]) if row.get("auc_scoring") != "structural_zero_final_layer_v_only" else "0.0000 *(struct)*"
            lines.append(
                f"| {m} | {row['layer']} | {row['group']} | {row['mode']} | {row['control_pool']} | {row['mean_tokens_ko']:.1f} | "
                f"{_fmt_ci(row['cos_drop_group'], row['cos_drop_group_ci95'], '.4f')} | {_fmt_opt(row['cos_drop_rand'], '.4f')} | "
                f"{_fmt_ci(row['net_cos_drop'], row['net_cos_drop_ci95'])} | "
                f"{_fmt_opt(row['delta_knn_group'])} | {_fmt_opt(row['net_delta_knn'])} | "
                f"{raw_auc_s} | {net_auc_s} | {kl_s} |"
            )

    if "mae_vitb16" in results:
        mr = results["mae_vitb16"]
        lines.append("\n## 3. MAE Patch-Mean Read-Out Supplement (`mae_vitb16`)\n")
        lines.append(
            f"- Clean `[CLS]` kNN@20 = `{mr['clean_knn']:.4f}` vs Clean **patch-mean** kNN@20 = `{mr.get('clean_patch_mean_knn')}` (`{gal_mode}` gallery).\n"
        )
        lines.append("| Intervention | Setting | Patch-Mean 1−cos | Patch-Mean ΔkNN@20 |")
        lines.append("|---|---|---:|---:|")
        for row in mr["i1_results"]:
            if row["L_int"] in (1, 4) and "patch_mean_cos_drop" in row:
                lines.append(
                    f"| I1 | {row['surrogate']} L_int={row['L_int']} | {row['patch_mean_cos_drop']:.4f} | {row['patch_mean_delta_knn']:+.4f} |"
                )
        for row in mr["i2_results"]:
            if "patch_mean_cos_drop_group" in row:
                lines.append(
                    f"| I2 | l={row['layer']} {row['group']} ({row['mode']}) | "
                    f"grp={row['patch_mean_cos_drop_group']:.4f} (net={row['patch_mean_net_cos_drop']:+.4f}) | "
                    f"grp={row['patch_mean_delta_knn_group']:+.4f} (net={row['patch_mean_net_delta_knn']:+.4f}) |"
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
    ap.add_argument("--i2_layers", default="", help="Comma-separated block indices for I2 (default: penultimate & final, e.g. 10,11).")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--knn_k", type=int, default=20)
    ap.add_argument("--knn_gallery", choices=["full_train", "loo_pilot"], default="full_train")
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
    i2_layers = tuple(int(x.strip()) for x in args.i2_layers.split(",") if x.strip()) if args.i2_layers.strip() else None
    seeds = tuple(int(x.strip()) for x in args.seeds.split(",") if x.strip())

    samples = load_cub_pilot_split(args.cub_dir, n_images=args.n_images, seed=0, split_file=args.split_file)
    class_labels = _load_cub_class_labels(args.cub_dir, [s.cub_id for s in samples])
    gallery_paths, gallery_labels = (
        _load_cub_train_gallery(args.cub_dir) if args.knn_gallery == "full_train" else ([], np.zeros((0,), dtype=np.int64))
    )
    print(
        f"Loaded {len(samples)} CUB pilot images ({len(np.unique(class_labels))} unique species) "
        f"+ {len(gallery_paths)} train gallery images on device={device}",
        flush=True,
    )

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
            gallery_paths=gallery_paths, gallery_labels=gallery_labels,
            batch_size=args.batch_size, l_int_sweep=l_int_sweep, seeds=seeds, knn_k=args.knn_k,
            i2_layers=i2_layers, smoke_random=args.smoke_random,
        )

    md_text = write_i1_i2_report(results, out_dir)
    cfg_payload = json.dumps({
        "models": model_ids, "n_images": len(samples), "l_int_sweep": l_int_sweep,
        "i2_layers": i2_layers, "seeds": seeds, "knn_k": args.knn_k, "knn_gallery": args.knn_gallery,
        "smoke_random": args.smoke_random,
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
        "knn_gallery": args.knn_gallery,
        "n_gallery": len(gallery_paths),
        "smoke_random": bool(args.smoke_random),
        "total_seconds": round(time.perf_counter() - t_all, 2),
        "unix_time": int(time.time()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print("\n" + md_text, flush=True)
    return results


if __name__ == "__main__":
    main()
