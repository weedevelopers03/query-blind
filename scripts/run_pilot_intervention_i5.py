#!/usr/bin/env python3
"""Pilot Intervention I5: Controlled [CLS] State Randomization & Across-Model Prediction Test (Guide §6).

Edit: Randomize the [CLS] hidden state entering block L_r (L_r in {8, 9, 10, 11}) while holding all patch
      and register tokens fixed, then run blocks L_r .. L-1 normally.
Modes:
  - cov_matched (primary): Empirical mean + top-k PCA covariance Gaussian matched to each image's ||z_cls||_2.
  - cross_img:             Real [CLS] hidden state from a random different image (derangement j != i).
  - iso_norm:              Isotropic Gaussian matched to each image's ||z_cls||_2 (off-manifold control).
Seeds: 5 seeds (0..4) per (L_r, mode); seed SD reported separately from image bootstrap CI (§10.2).
Prediction test: If --m1_m2_dir is provided, correlates M1 (QII_S4, QII_S1) and M2 (omega^2_key) at layer L_r
                 against I5 map survival (delta_box_auc and box_auc_retention) across models with bootstrap CIs.
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
    build_image_transform,
    get_transformed_gt_mask,
    get_transformed_imagenet_gt_mask,
    load_cub_pilot_split,
    load_imagenet_val_pilot_split,
)
from interventions import forward_ops as fops
from interventions import i1_i2 as I12
from interventions import i5_cls_rand as I5
from scripts.run_pilot_interventions_i1_i2 import (
    _extract_gallery_embeddings,
    _load_cub_class_labels,
    _load_cub_train_gallery,
    _load_imagenet_labels_and_gallery,
)
from scripts.run_pilot_m1_m2 import (
    PILOT8_SPECS,
    _file_sha256_16,
    _git_sha,
    _load_registry_meta,
    bootstrap_ci,
    load_pilot_model,
    paired_bootstrap_diff_ci,
)


@torch.no_grad()
def _run_tail_i5(
    adapter,
    model,
    z_start_all: torch.Tensor,
    layout: dict,
    start_block: int,
    device: torch.device,
    batch_size: int = 32,
    *,
    cls_state_overrides_all: dict[int, np.ndarray] | None = None,
    capture_clean_stats: bool = False,
) -> dict:
    n_img = z_start_all.shape[0]
    embs, logits_list, wmaps = [], [], []
    cls_in_acc: dict[int, list[torch.Tensor]] = {}

    for s in range(0, n_img, batch_size):
        e = min(n_img, s + batch_size)
        zb = z_start_all[s:e].to(device)
        cls_ov_b = (
            {l: torch.as_tensor(arr[s:e], dtype=torch.float32, device=device) for l, arr in cls_state_overrides_all.items()}
            if cls_state_overrides_all else None
        )
        res = fops.forward_from_block_with_interventions(
            adapter, model, zb, layout, start_block,
            cls_state_overrides=cls_ov_b, capture_clean_stats=capture_clean_stats,
        )
        embs.append(res["emb"].cpu().numpy())
        if res["logits"] is not None:
            logits_list.append(res["logits"].cpu().numpy())
        wmaps.append(res["last_weighted_map"].cpu().numpy())
        if capture_clean_stats:
            for l, zc in res["cls_state_in_by_l"].items():
                cls_in_acc.setdefault(l, []).append(zc)

    out = {
        "emb": np.concatenate(embs, axis=0),
        "logits": np.concatenate(logits_list, axis=0) if logits_list else None,
        "wmap": np.concatenate(wmaps, axis=0),
    }
    if capture_clean_stats:
        out["cls_state_in_by_l"] = {l: torch.cat(v, dim=0).numpy() for l, v in cls_in_acc.items()}
    return out


def run_one_model_i5(
    model_id: str,
    samples,
    class_labels: np.ndarray,
    gt_cache: dict[int, tuple],
    device: torch.device,
    out_dir: Path,
    weights_cache: Path,
    reg_meta: dict,
    *,
    dataset: str = "cub",
    gallery_paths: list[Path] | None = None,
    gallery_labels: np.ndarray | None = None,
    batch_size: int = 32,
    l_r_sweep: tuple[int, ...] = I5.DEFAULT_I5_LR_SWEEP,
    modes: tuple[str, ...] = I5.DEFAULT_I5_MODES,
    seeds: tuple[int, ...] = I5.DEFAULT_I5_SEEDS,
    knn_k: int = 20,
    smoke_random: bool = False,
) -> dict:
    t0 = time.perf_counter()
    adapter, model, mean, std, target_size = load_pilot_model(
        model_id, device, weights_cache, smoke_random=smoke_random,
        keep_head=(model_id == "deit3_small_patch16_224"),
    )
    n_blocks = adapter.n_blocks(model)
    eff_lr = tuple(sorted(l for l in l_r_sweep if 0 <= l < n_blocks))
    start_block = min(eff_lr)
    tf = build_image_transform(target_size, mean, std)
    n_img = len(samples)
    exclude_from_fn = (model_id == "mae_vitb16")

    if target_size not in gt_cache:
        masks, boxes_list = [], []
        for s in samples:
            with Image.open(s.image_path) as im:
                ow, oh = im.size
            if dataset == "cub":
                mask, tbox = get_transformed_gt_mask(s, ow, oh, target_size, prefer_seg=True)
                tboxes = [tbox]
            else:
                mask, tboxes = get_transformed_imagenet_gt_mask(s, ow, oh, target_size)
            masks.append(mask.astype(np.uint8))
            boxes_list.append(tboxes)
        gt_cache[target_size] = (np.stack(masks, axis=0), boxes_list)
    gt_masks_np, gt_boxes_list = gt_cache[target_size]

    print(f"\n[{model_id}] I5 ({dataset}) | device={device}, res={target_size}, n={n_img}, L_r={eff_lr}", flush=True)

    # Optional external kNN gallery
    use_full_gallery = bool(gallery_paths) and gallery_labels is not None and len(gallery_paths) >= knn_k
    gal_emb, gal_labels = None, None
    knn_gallery_mode = "loo_pilot"
    n_gallery = n_img
    if use_full_gallery:
        cache_p = None if smoke_random else (out_dir / "gallery_cache" / f"{model_id}.npz")
        gal_emb, _ = _extract_gallery_embeddings(
            adapter, model, gallery_paths, tf, device, cache_path=cache_p, batch_size=max(batch_size, 64),
        )
        gal_labels = gallery_labels
        knn_gallery_mode = "full_train" if dataset == "cub" else "disjoint_val"
        n_gallery = len(gallery_paths)

    # 1. Cache z_start at min(L_r)
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

    # 2. Clean baseline + capture clean [CLS] hidden states entering each L_r
    clean = _run_tail_i5(
        adapter, model, z_start_all, layout, start_block, device, batch_size, capture_clean_stats=True,
    )
    clean_auc, clean_otsu = I12.eval_maps_box_auc_and_otsu(
        clean["wmap"], grid, target_size, gt_masks_np, gt_boxes_list, device=device, compute_otsu=True,
    )
    has_logits = clean["logits"] is not None

    def _eval_knn(q_emb: np.ndarray) -> tuple[float, np.ndarray]:
        if use_full_gallery:
            return I12.knn_accuracy_cosine(
                q_emb, class_labels, feats_ref=gal_emb, labels_ref=gal_labels, exclude_self=False, k=knn_k, device=device,
            )
        return I12.knn_accuracy_cosine(q_emb, class_labels, feats_ref=clean["emb"], exclude_self=True, k=knn_k, device=device)

    clean_knn_acc, clean_knn_hits = _eval_knn(clean["emb"])
    pca_by_l = {l: I5.fit_cls_state_pca(clean["cls_state_in_by_l"][l], k=I5.DEFAULT_CLS_PCA_K) for l in eff_lr}

    # 3. Sweep L_r x mode x 5 seeds
    i5_results: list[dict] = []
    clean_auc_mean = float(np.nanmean(clean_auc))
    for lr in eff_lr:
        z_cls_clean = clean["cls_state_in_by_l"][lr]
        for mode in modes:
            auc_by_s, cos_by_s, knn_by_s, kl_by_s, t1_by_s = [], [], [], [], []
            otsu_s0 = None
            for si, sd in enumerate(seeds):
                cls_ov = I5.sample_cls_state_override(mode, z_cls_clean, pca_by_l[lr], seed=300 * sd + 17 * lr)
                res = _run_tail_i5(
                    adapter, model, z_start_all, layout, start_block, device, batch_size,
                    cls_state_overrides_all={lr: cls_ov},
                )
                aucs, otsus = I12.eval_maps_box_auc_and_otsu(
                    res["wmap"], grid, target_size, gt_masks_np, gt_boxes_list, device=device, compute_otsu=(si == 0),
                )
                if si == 0:
                    otsu_s0 = otsus
                auc_by_s.append(aucs)
                cos_by_s.append(I12.cosine_similarity_batch(clean["emb"], res["emb"]))
                _, kh = _eval_knn(res["emb"])
                knn_by_s.append(kh)
                if has_logits:
                    kl_by_s.append(I12.kl_divergence_logits(clean["logits"], res["logits"]))
                    t1_by_s.append(I12.top1_agreement(clean["logits"], res["logits"]))

            auc_img = np.mean(auc_by_s, axis=0)
            cos_img = np.mean(cos_by_s, axis=0)
            knn_img = np.mean(knn_by_s, axis=0)
            d_auc = paired_bootstrap_diff_ci(auc_img, clean_auc, seed=0)
            d_otsu = paired_bootstrap_diff_ci(otsu_s0, clean_otsu, seed=0)
            d_knn = paired_bootstrap_diff_ci(knn_img, clean_knn_hits, seed=0)
            cos_m, cos_lo, cos_hi = bootstrap_ci(1.0 - cos_img, seed=0)
            seed_auc_means = [float(np.nanmean(a - clean_auc)) for a in auc_by_s]
            auc_int_mean = float(np.nanmean(auc_img))

            row = {
                "model_id": model_id,
                "objective": reg_meta.get(model_id, {}).get("objective", "unknown"),
                "L_r": int(lr),
                "mode": mode,
                "n_seeds": len(seeds),
                "box_auc_clean": round(clean_auc_mean, 4),
                "box_auc_int": round(auc_int_mean, 4),
                "box_auc_retention": round(auc_int_mean / max(clean_auc_mean, 1e-12), 4),
                "delta_box_auc": round(d_auc["mean_diff"], 4),
                "delta_box_auc_ci95": [round(d_auc["ci95_low"], 4), round(d_auc["ci95_high"], 4)],
                "delta_box_auc_seed_sd": round(float(np.std(seed_auc_means)), 5),
                "delta_otsu_corloc": round(d_otsu["mean_diff"], 4),
                "cos_drop_mean": round(cos_m, 4),
                "cos_drop_ci95": [round(cos_lo, 4), round(cos_hi, 4)],
                "delta_knn": round(d_knn["mean_diff"], 4),
                "delta_knn_ci95": [round(d_knn["ci95_low"], 4), round(d_knn["ci95_high"], 4)],
            }
            if has_logits:
                kl_m, kl_lo, kl_hi = bootstrap_ci(np.mean(kl_by_s, axis=0), seed=0)
                t1_m, _, _ = bootstrap_ci(np.mean(t1_by_s, axis=0), seed=0)
                row["kl_mean"] = round(kl_m, 4)
                row["kl_ci95"] = [round(kl_lo, 4), round(kl_hi, 4)]
                row["top1_agree"] = round(t1_m, 4)
            i5_results.append(row)
            print(
                f"  I5 L_r={lr:2d} ({mode:<11s}) | ΔAUC={row['delta_box_auc']:+.4f} (ret={row['box_auc_retention']:.3f}) | "
                f"1-cos={row['cos_drop_mean']:.4f} | ΔkNN={row['delta_knn']:+.4f}",
                flush=True,
            )

    del model, z_start_all
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    res_obj = {
        "model_id": model_id,
        "objective": reg_meta.get(model_id, {}).get("objective", "unknown"),
        "dataset": dataset,
        "exclude_from_function_tests": exclude_from_fn,
        "n_images": n_img,
        "resolution": target_size,
        "knn_gallery_mode": knn_gallery_mode,
        "knn_gallery_size": int(n_gallery),
        "clean_box_auc": round(clean_auc_mean, 4),
        "clean_otsu_corloc": round(float(np.nanmean(clean_otsu)), 4),
        "clean_knn": round(float(clean_knn_acc), 4),
        "has_logits": has_logits,
        "i5_results": i5_results,
        "total_sec": round(time.perf_counter() - t0, 2),
    }
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "models" / f"{model_id}.json.tmp"
    tmp.write_text(json.dumps(res_obj, indent=2))
    os.replace(tmp, out_dir / "models" / f"{model_id}.json")
    return res_obj


def compute_prediction_table(results: dict[str, dict], m1_m2_dir: Path | None) -> list[dict]:
    """Guide §6 Prediction Test: does M1 (QII_S4, QII_S1) or M2 (w2_key) at L_r predict I5 map survival?"""
    if m1_m2_dir is None or not (m1_m2_dir / "models").is_dir():
        return []

    m12_by_model: dict[str, dict[int, dict]] = {}
    for mid in results:
        p = m1_m2_dir / "models" / f"{mid}.json"
        if not p.exists():
            continue
        dat = json.loads(p.read_text())
        layer_map: dict[int, dict] = {}
        # Real pilot_m1_m2 schema stores a list of dicts under "layer_summaries"
        if isinstance(dat.get("layer_summaries"), list):
            for lv in dat["layer_summaries"]:
                if isinstance(lv, dict) and "layer" in lv:
                    layer_map[int(lv["layer"])] = lv
        elif isinstance(dat.get("layers"), dict):
            for l_key, lv in dat["layers"].items():
                l_idx = int(l_key.lstrip("l")) if isinstance(l_key, str) else int(l_key)
                layer_map[l_idx] = lv
        m12_by_model[mid] = layer_map

    if len(m12_by_model) < 3:
        return []

    # Collect available (L_r, mode) pairs
    lr_modes = sorted({(row["L_r"], row["mode"]) for r in results.values() for row in r["i5_results"]})
    pred_rows: list[dict] = []
    for lr, mode in lr_modes:
        for scope in ("all_models", "excl_mae"):
            for predictor in ("qii_s4", "qii_s1", "w2_key"):
                for target in ("delta_box_auc", "box_auc_retention"):
                    xs, ys, mids = [], [], []
                    for mid, r in results.items():
                        if scope == "excl_mae" and mid == "mae_vitb16":
                            continue
                        if mid not in m12_by_model or lr not in m12_by_model[mid]:
                            continue
                        lv = m12_by_model[mid][lr]
                        if predictor == "qii_s4":
                            xval = lv.get("qii_s4", lv.get("qii", {}).get("S4", {}).get("mean"))
                        elif predictor == "qii_s1":
                            xval = lv.get("qii_s1", lv.get("qii", {}).get("S1", {}).get("mean"))
                        else:
                            xval = lv.get("omega2_key", lv.get("keydom", {}).get("omega2_key_mean"))
                        match = [rw for rw in r["i5_results"] if rw["L_r"] == lr and rw["mode"] == mode]
                        if xval is None or not match:
                            continue
                        xs.append(float(xval))
                        ys.append(float(match[0][target]))
                        mids.append(mid)
                    corr = I5.bootstrap_across_model_correlation(np.array(xs), np.array(ys), seed=0)
                    pred_rows.append({
                        "L_r": int(lr),
                        "mode": mode,
                        "scope": scope,
                        "predictor": predictor,
                        "target": target,
                        **corr,
                    })
    return pred_rows


def write_i5_report(results: dict[str, dict], pred_rows: list[dict], out_dir: Path) -> str:
    first_r = next(iter(results.values()), {})
    lines = [
        "# Pilot Intervention I5 — Controlled `[CLS]` State Randomization Summary (Guide §6)\n",
        f"- **Dataset:** `{first_r.get('dataset', 'unknown')}` (`N = {first_r.get('n_images', 0)}`); kNN gallery `{first_r.get('knn_gallery_mode')}` (`N = {first_r.get('knn_gallery_size', 0)}`).",
        "- **Modes:** `cov_matched` (primary: empirical mean + top-16 PCA covariance, norm-matched), `cross_img` (real `[CLS]` from another image), `iso_norm` (isotropic off-manifold, norm-matched); 5 seeds (`0..4`).\n",
        "## 1. Per-Model Map Survival & Functional Shift\n",
        "| Model | Obj | L_r | Mode | Clean AUC | Δbox-AUC [95% CI] | Seed SD | Retention | 1−cos [95% CI] | ΔkNN@20 | KL |",
        "|---|---|---:|---|---:|---|---:|---:|---|---:|---:|",
    ]
    for m, r in results.items():
        for row in r["i5_results"]:
            kl_s = f"{row['kl_mean']:.4f}" if "kl_mean" in row else "—"
            lines.append(
                f"| {m} | {row['objective']} | {row['L_r']} | {row['mode']} | {row['box_auc_clean']:.3f} | "
                f"{row['delta_box_auc']:+.4f} [{row['delta_box_auc_ci95'][0]:+.4f}, {row['delta_box_auc_ci95'][1]:+.4f}] | "
                f"{row['delta_box_auc_seed_sd']:.4f} | {row['box_auc_retention']:.3f} | "
                f"{row['cos_drop_mean']:.4f} [{row['cos_drop_ci95'][0]:.4f}, {row['cos_drop_ci95'][1]:.4f}] | "
                f"{row['delta_knn']:+.4f} | {kl_s} |"
            )

    if pred_rows:
        lines.append("\n## 2. Across-Model Prediction Test (`M1`/`M2` at `L_r` → `I5` Map Survival, Guide §6)\n")
        lines.append("| L_r | Mode | Scope | Predictor | Target | N | Spearman ρ [95% CI] | p | Pearson r [95% CI] |")
        lines.append("|---:|---|---|---|---|---:|---|---:|---|")
        for pr in pred_rows:
            if pr["target"] != "delta_box_auc":
                continue
            sp_ci = pr["spearman_ci95"]
            pr_ci = pr["pearson_ci95"]
            sp_s = f"{pr['spearman_rho']:+.3f} [{sp_ci[0]:+.3f}, {sp_ci[1]:+.3f}]" if sp_ci[0] is not None else f"{pr['spearman_rho']}"
            pr_s = f"{pr['pearson_r']:+.3f} [{pr_ci[0]:+.3f}, {pr_ci[1]:+.3f}]" if pr_ci[0] is not None else f"{pr['pearson_r']}"
            lines.append(
                f"| {pr['L_r']} | {pr['mode']} | {pr['scope']} | {pr['predictor']} | {pr['target']} | "
                f"{pr['n_models']} | {sp_s} | {pr['spearman_p']:.4f} | {pr_s} |"
            )

    md_text = "\n".join(lines) + "\n"
    (out_dir / "summary_i5.md").write_text(md_text)
    (out_dir / "i5_report.json").write_text(json.dumps({"models": results, "prediction_test": pred_rows}, indent=2))
    return md_text


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Run Intervention I5 (controlled [CLS] randomization) + prediction test.")
    ap.add_argument("--dataset", choices=["cub", "imagenet"], default="imagenet")
    ap.add_argument("--cub_dir", default="")
    ap.add_argument("--imagenet_images_dir", default="")
    ap.add_argument("--imagenet_xml_dir", default="")
    ap.add_argument("--imagenet_gallery_size", type=int, default=5000)
    ap.add_argument("--split_file", default="")
    ap.add_argument("--m1_m2_dir", default="", help="Optional path to pilot_m1_m2_{dataset} dir for across-model prediction test.")
    ap.add_argument("--out_dir", default="results/pilot_intervention_i5")
    ap.add_argument("--models", default="all")
    ap.add_argument("--n_images", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--l_r_sweep", default="8,9,10,11")
    ap.add_argument("--modes", default="cov_matched,cross_img,iso_norm")
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
    l_r_sweep = tuple(int(x.strip()) for x in args.l_r_sweep.split(",") if x.strip())
    modes = tuple(x.strip() for x in args.modes.split(",") if x.strip())
    seeds = tuple(int(x.strip()) for x in args.seeds.split(",") if x.strip())

    if args.dataset == "cub":
        if not args.cub_dir:
            raise ValueError("--cub_dir is required when --dataset cub")
        n_img = args.n_images if args.n_images > 0 else 500
        split_file = args.split_file or str(REPO_ROOT / "configs" / "splits" / "cub_pilot_500_seed0.txt")
        samples = load_cub_pilot_split(args.cub_dir, n_images=n_img, seed=0, split_file=split_file)
        class_labels = _load_cub_class_labels(args.cub_dir, [s.cub_id for s in samples])
        gallery_paths, gallery_labels = (
            _load_cub_train_gallery(args.cub_dir) if args.knn_gallery == "full_train" else ([], np.zeros((0,), dtype=np.int64))
        )
        ds_label = "CUB-200-2011"
    else:
        if not args.imagenet_images_dir or not args.imagenet_xml_dir:
            raise ValueError("--imagenet_images_dir and --imagenet_xml_dir are required when --dataset imagenet")
        n_img = args.n_images if args.n_images > 0 else 1000
        split_file = args.split_file or str(REPO_ROOT / "configs" / "splits" / "imagenet_pilot_1000_seed0.txt")
        samples = load_imagenet_val_pilot_split(
            args.imagenet_images_dir, args.imagenet_xml_dir, n_images=n_img, seed=0, split_file=split_file,
        )
        gal_n = args.imagenet_gallery_size if args.knn_gallery == "full_train" else 0
        class_labels, gallery_paths, gallery_labels = _load_imagenet_labels_and_gallery(
            args.imagenet_images_dir, args.imagenet_xml_dir, samples, gallery_size=gal_n, seed=1,
        )
        ds_label = "ImageNet-1K-val"

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
        results[mid] = run_one_model_i5(
            mid, samples, class_labels, gt_cache, device, out_dir, weights_cache, reg_meta,
            dataset=args.dataset, gallery_paths=gallery_paths, gallery_labels=gallery_labels,
            batch_size=args.batch_size, l_r_sweep=l_r_sweep, modes=modes, seeds=seeds, knn_k=args.knn_k,
            smoke_random=args.smoke_random,
        )

    m1_m2_dir = Path(args.m1_m2_dir) if args.m1_m2_dir else None
    pred_rows = compute_prediction_table(results, m1_m2_dir)
    md_text = write_i5_report(results, pred_rows, out_dir)
    cfg_payload = json.dumps({
        "dataset": args.dataset, "models": model_ids, "n_images": len(samples),
        "l_r_sweep": l_r_sweep, "modes": modes, "seeds": seeds, "knn_k": args.knn_k,
        "smoke_random": args.smoke_random,
    }, sort_keys=True)
    meta = {
        "git_sha": _git_sha(),
        "config_hash": hashlib.sha256(cfg_payload.encode()).hexdigest()[:16],
        "file_sha256_16": {
            rel: _file_sha256_16(REPO_ROOT / rel)
            for rel in ("interventions/forward_ops.py", "interventions/i5_cls_rand.py", "scripts/run_pilot_intervention_i5.py")
        },
        "device": str(device),
        "dataset": ds_label,
        "n_images": len(samples),
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
