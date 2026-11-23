#!/usr/bin/env python3
"""Pilot Measure M4 runner: Artifact-Controlled Similarity (Guide §5 & Gate G1 Criterion 2).

Supports:
  --dataset cub         (exact 500-image CUB-200-2011 pilot split; fast ~90s on T4)
  --dataset waterbirds  (Waterbirds test split, up to all 5,794 images)
  --dataset folder      (generic image folder, e.g. ImageNet-1K val pilot split)

Usage (CUB-500):
  PYTHONPATH=. python3 scripts/run_pilot_similarity_m4.py \
      --dataset cub --cub_dir /kaggle/working/datasets/cub \
      --out_dir /kaggle/working/results/pilot_similarity_m4_cub --resume

Usage (Waterbirds):
  PYTHONPATH=. python3 scripts/run_pilot_similarity_m4.py \
      --dataset waterbirds --waterbirds_dir /kaggle/working/datasets/waterbirds/waterbirds_v1.0 \
      --out_dir /kaggle/working/results/pilot_similarity_m4_wb --resume
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.base import attn_probs
from eval.preprocess import build_image_transform, load_cub_pilot_split, load_imagenet_val_pilot_split
from extract.cache import last_third_blocks
from measures import similarity as sim_mod
from measures._common import compute_head_weights, get_output_proj_and_layerscale, patch_attention
from measures.similarity import detect_outlier_patches
from scripts.run_pilot_m1_m2 import (
    PILOT8_SPECS,
    ZOO12_SPECS,
    resolve_model_ids,
    _file_sha256_16,
    _git_sha,
    load_pilot_model,
)
from scripts.run_pilot_waterbirds_m3 import _resolve_waterbirds_root


# --------------------------------------------------------------------------- Dataset loaders
def load_image_paths(args) -> list[tuple[int, Path]]:
    """Return list of (image_id, Path) according to --dataset."""
    if args.dataset == "cub":
        if not args.cub_dir:
            raise ValueError("--cub_dir is required for --dataset cub")
        samples = load_cub_pilot_split(
            args.cub_dir,
            n_images=args.limit or 500,
            seed=args.seed,
            split_file=args.split_file,
        )
        return [(s.cub_id, s.image_path) for s in samples]

    if args.dataset == "waterbirds":
        if not args.waterbirds_dir:
            raise ValueError("--waterbirds_dir is required for --dataset waterbirds")
        import csv
        root = _resolve_waterbirds_root(args.waterbirds_dir)
        rows = []
        with open(root / "metadata.csv", newline="") as f:
            for row in csv.DictReader(f):
                if int(row["split"]) != 2:
                    continue
                rows.append((int(row["img_id"]), root / row["img_filename"]))
        if args.limit is not None and args.limit < len(rows):
            rng = np.random.default_rng(args.seed)
            sel = np.sort(rng.choice(len(rows), size=args.limit, replace=False))
            rows = [rows[i] for i in sel]
        return rows

    if args.dataset == "imagenet":
        if not args.images_dir:
            raise ValueError("--images_dir is required for --dataset imagenet (ImageNet val directory)")
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
        return [(i, s.image_path) for i, s in enumerate(samples)]

    if args.dataset == "folder":
        if not args.images_dir:
            raise ValueError("--images_dir is required for --dataset folder")
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        paths = sorted(p for p in Path(args.images_dir).rglob("*") if p.suffix.lower() in exts)
        if not paths:
            raise FileNotFoundError(f"No images found under {args.images_dir}")
        if args.limit is not None and args.limit < len(paths):
            rng = np.random.default_rng(args.seed)
            sel = np.sort(rng.choice(len(paths), size=args.limit, replace=False))
            paths = [paths[i] for i in sel]
        return [(i, p) for i, p in enumerate(paths)]

    raise ValueError(f"Unknown dataset {args.dataset!r}")


# --------------------------------------------------------------------------- Batched extractor (timm + open_clip)
@torch.no_grad()
def extract_batch_m4(adapter, model, x: torch.Tensor, layers: list[int]) -> dict[int, dict]:
    """Single batched forward pass returning x_out, patch_idx, cls_idx, and head-weighted [CLS] patch attention."""
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
                attn_row = attn_probs(q[:, :, cls_idx : cls_idx + 1], k, scale).squeeze(2)
                W_O, _, gamma = get_output_proj_and_layerscale(adapter, model, l)
                w_h = compute_head_weights(attn_row, v, W_O, gamma=gamma)
                a_patch = patch_attention(q[:, :, cls_idx, :], k[:, :, patch_idx, :], scale)  # (B, H, N)
                a_weighted = (a_patch * w_h.unsqueeze(-1)).sum(dim=1)  # (B, N)
                out[l] = {
                    "x_out": z_out.float(),
                    "cls_idx": cls_idx,
                    "patch_idx": patch_idx,
                    "attn_map": a_weighted.float(),
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
                a_patch = patch_attention(q[:, :, cls_idx, :], k[:, :, patch_idx, :], scale)
                a_weighted = (a_patch * w_h.unsqueeze(-1)).sum(dim=1)
                out[l] = {
                    "x_out": z_out.float(),
                    "cls_idx": cls_idx,
                    "patch_idx": patch_idx,
                    "attn_map": a_weighted.float(),
                }
            z = z_out
        return out

    raise NotImplementedError(f"M4 batched extractor not implemented for family={adapter.family!r}")


# --------------------------------------------------------------------------- Feature extraction per model
@torch.no_grad()
def extract_model_m4_features(
    model_id: str,
    image_items: list[tuple[int, Path]],
    r3_patch_idx: np.ndarray,
    target_layers: list[int],
    device: torch.device,
    cache_dir: Path,
    weights_cache: Path,
    batch_size: int = 32,
    smoke_random: bool = False,
) -> dict:
    """Extract & cache R1, R2(C0/C1), R3(tokens + OUT mask), and head-weighted attn maps for one model."""
    t0 = time.perf_counter()
    adapter, model, mean, std, target_size = load_pilot_model(
        model_id, device, weights_cache, smoke_random=smoke_random
    )
    n_blocks = adapter.n_blocks(model)
    layers = [l for l in target_layers if 0 <= l < n_blocks]
    preprocess = build_image_transform(target_size, mean, std)

    n_img = len(image_items)
    print(f"\n[{model_id}] M4 extract | device={device}, res={target_size}, n={n_img}, layers={layers}", flush=True)

    storage = {
        l: {"r1": [], "r2_c0": [], "r2_c1": [], "r3": [], "r3_out": [], "n_out": [], "attn_map": []}
        for l in layers
    }

    t_ext = time.perf_counter()
    for start in range(0, n_img, batch_size):
        batch = image_items[start : start + batch_size]
        imgs = []
        for _, path in batch:
            with Image.open(path) as im:
                imgs.append(preprocess(im.convert("RGB")))
        x = torch.stack(imgs, dim=0).to(device)
        B = x.shape[0]
        p_idx_batch = torch.as_tensor(r3_patch_idx[start : start + B], dtype=torch.long, device=device)  # (B, N')
        b_arange = torch.arange(B, device=device)[:, None]

        extracted = extract_batch_m4(adapter, model, x, layers)
        for l in layers:
            out = extracted[l]
            x_out = out["x_out"]  # (B, S, d)
            g_idx = out["cls_idx"]
            p_idx = out["patch_idx"]
            patch_x = x_out[:, p_idx, :]  # (B, N, d)
            if patch_x.shape[1] != 196:
                raise AssertionError(f"[{model_id}] expected 14x14=196 matched patches, got {patch_x.shape[1]}")

            # R1: global token
            r1 = x_out[:, g_idx, :].cpu().numpy().astype(np.float32)

            # OUT tokens via exact M3 rule: median + 5 * MAD on ||x_out_i||_2 per image
            patch_norms = torch.linalg.norm(patch_x.to(torch.float64), dim=-1).cpu().numpy()  # (B, N)
            out_mask_np = detect_outlier_patches(patch_norms)  # (B, N) bool
            out_mask_t = torch.as_tensor(out_mask_np, dtype=torch.bool, device=device)

            # R2: C0 (all patches) and C1 (non-OUT patches)
            r2_c0 = patch_x.mean(dim=1).cpu().numpy().astype(np.float32)
            valid = ~out_mask_t
            counts = valid.sum(dim=1)
            if (counts == 0).any():
                valid = valid.clone()
                valid[counts == 0] = True
                counts = valid.sum(dim=1)
            r2_c1 = ((patch_x * valid.unsqueeze(-1)).sum(dim=1) / counts.unsqueeze(-1)).cpu().numpy().astype(np.float32)

            # R3: subsampled N' patches + matching OUT flags
            r3_tok = patch_x[b_arange, p_idx_batch].cpu().numpy().astype(np.float32)  # (B, N', d)
            r3_out = out_mask_np[np.arange(B)[:, None], r3_patch_idx[start : start + B]]  # (B, N')

            a_weighted = out["attn_map"].cpu().numpy().astype(np.float32)  # (B, N)

            storage[l]["r1"].append(r1)
            storage[l]["r2_c0"].append(r2_c0)
            storage[l]["r2_c1"].append(r2_c1)
            storage[l]["r3"].append(r3_tok)
            storage[l]["r3_out"].append(r3_out)
            storage[l]["n_out"].append(out_mask_np.sum(axis=1).astype(np.int32))
            storage[l]["attn_map"].append(a_weighted)

    extract_sec = time.perf_counter() - t_ext
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    cache_dir.mkdir(parents=True, exist_ok=True)
    out_stats = {}
    for l in layers:
        r1_all = np.concatenate(storage[l]["r1"], axis=0)
        r2_c0_all = np.concatenate(storage[l]["r2_c0"], axis=0)
        r2_c1_all = np.concatenate(storage[l]["r2_c1"], axis=0)
        r3_all = np.concatenate(storage[l]["r3"], axis=0)
        r3_out_all = np.concatenate(storage[l]["r3_out"], axis=0)
        n_out_all = np.concatenate(storage[l]["n_out"], axis=0)
        attn_all = np.concatenate(storage[l]["attn_map"], axis=0)

        # Atomic save per (model, layer) so analysis only loads one layer at a time (low RAM)
        npz_path = cache_dir / f"{model_id}_l{l}.npz"
        tmp_path = cache_dir / f"{model_id}_l{l}.tmp.npz"
        np.savez(
            tmp_path,
            r1=r1_all,
            r2_c0=r2_c0_all,
            r2_c1=r2_c1_all,
            r3=r3_all,
            r3_out=r3_out_all,
            n_out=n_out_all,
            attn_map=attn_all,
        )
        os.replace(tmp_path, npz_path)
        n_patches = attn_all.shape[1]
        out_stats[f"l{l}"] = {
            "layer": l,
            "layer_frac": round(float(l + 1) / float(n_blocks), 4),
            "dim": int(r1_all.shape[1]),
            "n_patches": int(n_patches),
            "n_out_mean": round(float(n_out_all.mean()), 3),
            "out_frac": round(float(n_out_all.mean() / n_patches), 5),
            "r3_out_frac": round(float(r3_out_all.mean()), 5),
        }

    total_sec = time.perf_counter() - t0
    meta_model = {
        "model_id": model_id,
        "n_blocks": n_blocks,
        "layers": layers,
        "resolution": target_size,
        "n_images": n_img,
        "extract_sec": round(extract_sec, 2),
        "img_per_sec": round(n_img / max(extract_sec, 1e-6), 2),
        "total_sec": round(total_sec, 2),
        "out_stats": out_stats,
    }
    tmp_json = cache_dir / f"{model_id}_meta.json.tmp"
    tmp_json.write_text(json.dumps(meta_model, indent=2))
    os.replace(tmp_json, cache_dir / f"{model_id}_meta.json")
    return meta_model


# --------------------------------------------------------------------------- Layer-wise M4 analysis
def analyze_layer_m4(
    layer: int,
    model_names: list[str],
    cache_dir: Path,
    device: torch.device,
    batch_size: int = 512,
    min_batches: int = 40,
    d_star: int = 256,
    knn_k: int = 10,
    n_splits_r2: int = 100,
    n_splits_r3: int = 40,
    seed: int = 0,
    compute_r3_splits: bool = True,
) -> dict:
    """Run full M4 analysis (R1, R2, R3 × C0/C1 × CKA/Procrustes/mKNN + tau_noise + Map IoU) for one layer."""
    r1_map, r2_c0_map, r2_c1_map, r3_map, r3_out_map, attn_map = {}, {}, {}, {}, {}, {}
    out_summary = {}
    for m in model_names:
        z = np.load(cache_dir / f"{m}_l{layer}.npz")
        r1_map[m] = z["r1"]
        r2_c0_map[m] = z["r2_c0"]
        r2_c1_map[m] = z["r2_c1"]
        r3_map[m] = z["r3"]
        r3_out_map[m] = z["r3_out"]
        attn_map[m] = z["attn_map"]
        n_out = z["n_out"]
        out_summary[m] = {
            "n_out_mean": round(float(n_out.mean()), 3),
            "r3_out_pct": round(100.0 * float(z["r3_out"].mean()), 3),
        }

    # §5.5 Map IoU grounding matrix (Otsu on head-weighted [CLS] attention maps)
    iou_mat = sim_mod.map_iou_matrix(attn_map, model_names)

    results_layer: dict = {
        "layer": layer,
        "models": model_names,
        "out_summary": out_summary,
        "map_iou_matrix": np.round(iou_mat, 4).tolist(),
        "levels": {},
    }

    # --- R1: global token (unchanged under C1; still compute C0 matrices + map IoU grounding)
    t_s = time.perf_counter()
    r1_cka = sim_mod.debiased_cka_matrix(r1_map, model_names, batch_size=batch_size, min_batches=min_batches, seed=seed, device=device)
    r1_proc, r1_evr = sim_mod.procrustes_matrix(r1_map, model_names, d_star=d_star, device=device)
    r1_mknn = sim_mod.mutual_knn_matrix(r1_map, model_names, k=knn_k, device=device)
    print(f"  [l={layer}] R1 done ({time.perf_counter() - t_s:.1f}s)", flush=True)
    results_layer["levels"]["R1"] = {
        "note": "R1 (global token) is unchanged under C1 patch masking (Guide §5.3); tau(C0, C1) = 1.0 by construction.",
        "procrustes_evr": {m: round(v, 4) for m, v in r1_evr.items()},
        "metrics": {
            "cka": {
                "mat_c0": np.round(r1_cka, 4).tolist(),
                "map_iou_spearman_c0": sim_mod.correlate_similarity_with_map_iou(r1_cka, iou_mat),
            },
            "procrustes": {
                "mat_c0": np.round(r1_proc, 4).tolist(),
                "map_iou_spearman_c0": sim_mod.correlate_similarity_with_map_iou(r1_proc, iou_mat),
            },
            "mknn": {
                "mat_c0": np.round(r1_mknn, 4).tolist(),
                "map_iou_spearman_c0": sim_mod.correlate_similarity_with_map_iou(r1_mknn, iou_mat),
            },
        },
    }

    # --- R2: mean-pooled patches (C0 vs C1 + full split-half tau_noise for all 3 metrics)
    _, r2_evr_c0 = sim_mod.procrustes_matrix(r2_c0_map, model_names, d_star=d_star, device=device)
    _, r2_evr_c1 = sim_mod.procrustes_matrix(r2_c1_map, model_names, d_star=d_star, device=device)
    r2_metrics = {}
    for metric in ("cka", "procrustes", "mknn"):
        t_s = time.perf_counter()
        n_sp = n_splits_r2 if metric == "cka" else min(n_splits_r2, 40)
        st = sim_mod.split_half_tau_test(
            r2_c0_map, r2_c1_map, model_names, metric=metric,
            n_splits=n_sp, batch_size=batch_size, min_batches=min_batches,
            d_star=d_star, knn_k=knn_k, seed=seed, device=device,
        )
        m_c0 = st.pop("mat_c0_full")
        m_c1 = st.pop("mat_c1_full")
        st["mat_c0"] = np.round(m_c0, 4).tolist()
        st["mat_c1"] = np.round(m_c1, 4).tolist()
        st["max_abs_shift"] = round(float(np.nanmax(np.abs(m_c0 - m_c1))), 4)
        st["mean_abs_shift"] = round(float(np.nanmean(np.abs(sim_mod.upper_tri_vector(m_c0) - sim_mod.upper_tri_vector(m_c1)))), 4)
        st["map_iou_spearman_c0"] = sim_mod.correlate_similarity_with_map_iou(m_c0, iou_mat)
        st["map_iou_spearman_c1"] = sim_mod.correlate_similarity_with_map_iou(m_c1, iou_mat)
        r2_metrics[metric] = st
        print(
            f"  [l={layer}] R2 {metric:<10s} tau={st['tau_c0_c1_full']:.3f} vs noise={st['tau_noise']['mean']:.3f} "
            f"({time.perf_counter() - t_s:.1f}s)",
            flush=True,
        )

    results_layer["levels"]["R2"] = {
        "procrustes_evr_c0": {m: round(v, 4) for m, v in r2_evr_c0.items()},
        "procrustes_evr_c1": {m: round(v, 4) for m, v in r2_evr_c1.items()},
        "metrics": r2_metrics,
    }

    # --- R3: subsampled individual patch tokens (N'=8 per image; C1 drops union OUT rows per pair)
    t_s = time.perf_counter()
    r3_c0_mats = sim_mod.r3_pairwise_matrices(
        r3_map, r3_out_map, model_names, condition="C0",
        batch_size=batch_size, min_batches=min_batches, d_star=d_star, knn_k=knn_k,
        seed=seed, device=device, metrics=("cka", "procrustes", "mknn"),
    )
    r3_c1_mats = sim_mod.r3_pairwise_matrices(
        r3_map, r3_out_map, model_names, condition="C1",
        batch_size=batch_size, min_batches=min_batches, d_star=d_star, knn_k=knn_k,
        seed=seed, device=device, metrics=("cka", "procrustes", "mknn"),
    )
    print(f"  [l={layer}] R3 full C0/C1 matrices (CKA+Procrustes+mKNN) done ({time.perf_counter() - t_s:.1f}s)", flush=True)

    r3_metrics = {}
    for metric in ("cka", "procrustes", "mknn"):
        m_c0 = r3_c0_mats[metric]
        m_c1 = r3_c1_mats[metric]
        tau_full, p_full = sim_mod.kendall_tau_b_matrices(m_c0, m_c1)
        entry = {
            "metric": metric,
            "tau_c0_c1_full": float(tau_full),
            "tau_c0_c1_full_p": float(p_full),
            "mat_c0": np.round(m_c0, 4).tolist(),
            "mat_c1": np.round(m_c1, 4).tolist(),
            "max_abs_shift": round(float(np.nanmax(np.abs(m_c0 - m_c1))), 4),
            "mean_abs_shift": round(float(np.nanmean(np.abs(sim_mod.upper_tri_vector(m_c0) - sim_mod.upper_tri_vector(m_c1)))), 4),
            "map_iou_spearman_c0": sim_mod.correlate_similarity_with_map_iou(m_c0, iou_mat),
            "map_iou_spearman_c1": sim_mod.correlate_similarity_with_map_iou(m_c1, iou_mat),
        }
        # Split-half tau_noise for R3 CKA (primary H3/G1 test, fast shared-Gram)
        if compute_r3_splits and metric == "cka":
            t_s = time.perf_counter()
            st = sim_mod.split_half_tau_test(
                r3_map, r3_map, model_names, metric=metric,
                n_splits=n_splits_r3, batch_size=batch_size, min_batches=min_batches,
                d_star=d_star, knn_k=knn_k, seed=seed, device=device,
                r3_out_by_model=r3_out_map,
            )
            st.pop("mat_c0_full", None)
            st.pop("mat_c1_full", None)
            entry.update(st)
            print(
                f"  [l={layer}] R3 cka splits ({n_splits_r3}) tau={st['tau_c0_c1_full']:.3f} "
                f"vs noise={st['tau_noise']['mean']:.3f} ({time.perf_counter() - t_s:.1f}s)",
                flush=True,
            )
        r3_metrics[metric] = entry

    results_layer["levels"]["R3"] = {"metrics": r3_metrics}

    # §5.6 Shuffled-feature calibration floor on first two models (R2)
    if len(model_names) >= 2:
        results_layer["shuffled_floor_r2"] = {
            "pair": [model_names[0], model_names[1]],
            **sim_mod.shuffled_floor_calibration(
                r2_c0_map[model_names[0]], r2_c0_map[model_names[1]],
                batch_size=batch_size, min_batches=min_batches, d_star=d_star, knn_k=knn_k,
                seed=seed + 42, device=device,
            ),
        }
    return results_layer


# --------------------------------------------------------------------------- Report formatting
def format_m4_markdown(report: dict) -> str:
    lines = [
        f"# Pilot Measure M4 — Artifact-Controlled Similarity ({report['dataset']}, N={report['n_images']})\n",
        "## 1. Rank Stability: C0 (Raw) vs C1 (OUT-Masked) & Split-Half Noise Floor\n",
        "| Layer | Level | Metric | tau(C0,C1) | tau_noise [95% CI] | Same-Half Δ [95% CI] | Cross-Half Δ [95% CI] | G1 (<noise)? | Max |Δsim| | ρ(C0,IoU) | ρ(C1,IoU) |",
        "|---|---|---|---:|---|---|---|---|---:|---:|---:|",
    ]
    for l_key, lr in report["layers"].items():
        for level in ("R2", "R3"):
            for metric in ("cka", "procrustes", "mknn"):
                d = lr["levels"][level]["metrics"][metric]
                tau_f = d["tau_c0_c1_full"]
                if "tau_noise" in d:
                    tn = d["tau_noise"]
                    ds = d["diff_same_minus_noise"]
                    dc = d["diff_cross_minus_noise"]
                    tn_s = f"{tn['mean']:.3f} [{tn['ci95_low']:.3f}, {tn['ci95_high']:.3f}]"
                    ds_s = f"{ds['mean']:+.3f} [{ds['ci95_low']:+.3f}, {ds['ci95_high']:+.3f}]"
                    dc_s = f"{dc['mean']:+.3f} [{dc['ci95_low']:+.3f}, {dc['ci95_high']:+.3f}]"
                    g1 = "YES" if (ds["below_noise_ci_excl0"] or dc["below_noise_ci_excl0"]) else "no"
                else:
                    tn_s = ds_s = dc_s = "—"
                    g1 = "—"
                r0 = d["map_iou_spearman_c0"]["spearman_rho"]
                r1 = d["map_iou_spearman_c1"]["spearman_rho"]
                lines.append(
                    f"| {l_key} | {level} | {metric} | {tau_f:.3f} | {tn_s} | {ds_s} | {dc_s} | {g1} | "
                    f"{d['max_abs_shift']:.3f} | {r0:+.3f} | {r1:+.3f} |"
                )

    lines.append("\n## 2. R1 (Global Token) Map-IoU Grounding (Unchanged by C1)\n")
    lines.append("| Layer | CKA ρ(IoU) | Procrustes ρ(IoU) | mKNN ρ(IoU) |")
    lines.append("|---|---:|---:|---:|")
    for l_key, lr in report["layers"].items():
        r1m = lr["levels"]["R1"]["metrics"]
        lines.append(
            f"| {l_key} | {r1m['cka']['map_iou_spearman_c0']['spearman_rho']:+.3f} | "
            f"{r1m['procrustes']['map_iou_spearman_c0']['spearman_rho']:+.3f} | "
            f"{r1m['mknn']['map_iou_spearman_c0']['spearman_rho']:+.3f} |"
        )

    lines.append("\n## 3. Gate G1 Criterion 2 Summary (Guide §11.2)\n")
    g1s = report["g1_criterion2_summary"]
    lines.append(f"- **Any CKA (R2 or R3) significantly below tau_noise (same-half 95% CI < 0):** `{g1s['any_cka_same_half_below_noise']}`")
    lines.append(f"- **Any CKA (R2 or R3) significantly below tau_noise (cross-half 95% CI < 0):** `{g1s['any_cka_cross_half_below_noise']}`")
    lines.append(f"- **Triggered cells:** `{g1s['triggered_cells']}`")

    if "shuffled_floor_r2" in list(report["layers"].values())[-1]:
        sf = list(report["layers"].values())[-1]["shuffled_floor_r2"]
        lines.append(
            f"\n**Shuffled-feature null floor (final layer R2, {sf['pair'][0]} vs {sf['pair'][1]}):** "
            f"CKA={sf['cka']:.4f}, Procrustes={sf['procrustes']:.4f}, mKNN={sf['mknn']:.4f}"
        )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- CLI
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Run Measure M4 (Artifact-Controlled Similarity) on Pilot-8.")
    ap.add_argument("--dataset", choices=["cub", "waterbirds", "imagenet", "folder"], default="cub")
    ap.add_argument("--cub_dir", default=None)
    ap.add_argument("--waterbirds_dir", default=None)
    ap.add_argument("--images_dir", default=None)
    ap.add_argument("--imagenet_xml_dir", default=None)
    ap.add_argument("--split_file", default=str(REPO_ROOT / "configs" / "splits" / "cub_pilot_500_seed0.txt"))
    ap.add_argument("--out_dir", default="results/pilot_similarity_m4")
    ap.add_argument("--models", default="all")
    ap.add_argument("--layers", default="8,9,10,11", help="Comma-separated block indices (for 12-block pilot models).")
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on number of images.")
    ap.add_argument("--batch_size", type=int, default=32, help="Extraction forward-pass batch size.")
    ap.add_argument("--cka_batch_size", type=int, default=512, help="Minibatch size for debiased CKA.")
    ap.add_argument("--cka_min_batches", type=int, default=40, help="Minimum batches for minibatch CKA when N > cka_batch_size.")
    ap.add_argument("--d_star", type=int, default=256, help="PCA dimension for Orthogonal Procrustes.")
    ap.add_argument("--knn_k", type=int, default=10, help="k for mutual k-NN overlap.")
    ap.add_argument("--r3_patches", type=int, default=8, help="Number of random patches sampled per image for R3.")
    ap.add_argument("--n_splits_r2", type=int, default=100, help="Disjoint 50/50 image splits for R2 tau_noise.")
    ap.add_argument("--n_splits_r3", type=int, default=40, help="Disjoint 50/50 image splits for R3 tau_noise.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights_cache_dir", default=str(Path.home() / ".cache" / "queryblind_weights"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke_random", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "feature_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_ids = resolve_model_ids(args.models)

    image_items = load_image_paths(args)
    n_img = len(image_items)
    print(f"Loaded {n_img} images for M4 ({args.dataset}) on device={device}", flush=True)

    # All 8 pilot models share 14x14 = 196 patches (224/16 and 196/14)
    r3_patch_idx = sim_mod.sample_r3_patch_indices(n_img, 196, patches_per_image=args.r3_patches, seed=args.seed)
    np.save(out_dir / "r3_patch_indices.npy", r3_patch_idx)

    weights_cache = Path(args.weights_cache_dir)
    target_layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    t_all = time.perf_counter()

    model_metas = {}
    for mid in model_ids:
        meta_path = cache_dir / f"{mid}_meta.json"
        all_npz_exist = all((cache_dir / f"{mid}_l{l}.npz").exists() for l in target_layers)
        if args.resume and meta_path.exists() and all_npz_exist:
            mm = json.loads(meta_path.read_text())
            if mm.get("n_images") == n_img:
                print(f"[{mid}] --resume: reusing cached M4 features ({n_img} imgs)", flush=True)
                model_metas[mid] = mm
                continue
        model_metas[mid] = extract_model_m4_features(
            mid, image_items, r3_patch_idx, target_layers, device, cache_dir, weights_cache,
            batch_size=args.batch_size, smoke_random=args.smoke_random,
        )

    layers_report = {}
    triggered_same = []
    triggered_cross = []
    for l in target_layers:
        t_l = time.perf_counter()
        print(f"\n=== Analyzing Layer l={l} (R1, R2, R3 x C0/C1 x CKA/Procrustes/mKNN + tau_noise) ===", flush=True)
        lr = analyze_layer_m4(
            l, model_ids, cache_dir, device,
            batch_size=args.cka_batch_size, min_batches=args.cka_min_batches,
            d_star=args.d_star, knn_k=args.knn_k,
            n_splits_r2=args.n_splits_r2, n_splits_r3=args.n_splits_r3,
            seed=args.seed,
        )
        layers_report[f"l{l}"] = lr
        print(f"  Layer l={l} done in {time.perf_counter() - t_l:.1f}s", flush=True)
        for level in ("R2", "R3"):
            d = lr["levels"][level]["metrics"]["cka"]
            if "diff_same_minus_noise" in d:
                if d["diff_same_minus_noise"]["below_noise_ci_excl0"]:
                    triggered_same.append(f"l{l}:{level}:cka")
                if d["diff_cross_minus_noise"]["below_noise_ci_excl0"]:
                    triggered_cross.append(f"l{l}:{level}:cka")

    report = {
        "dataset": args.dataset,
        "n_images": n_img,
        "models": model_ids,
        "layers": layers_report,
        "g1_criterion2_summary": {
            "any_cka_same_half_below_noise": len(triggered_same) > 0,
            "any_cka_cross_half_below_noise": len(triggered_cross) > 0,
            "triggered_same_half": triggered_same,
            "triggered_cross_half": triggered_cross,
            "triggered_cells": sorted(set(triggered_same + triggered_cross)),
        },
    }
    (out_dir / "similarity_m4_report.json").write_text(json.dumps(report, indent=2))
    md_text = format_m4_markdown(report)
    (out_dir / "summary_similarity_m4.md").write_text(md_text)

    cfg_payload = json.dumps({
        "dataset": args.dataset, "n_images": n_img, "models": model_ids, "layers": target_layers,
        "cka_batch_size": args.cka_batch_size, "cka_min_batches": args.cka_min_batches,
        "d_star": args.d_star, "knn_k": args.knn_k, "r3_patches": args.r3_patches,
        "n_splits_r2": args.n_splits_r2, "n_splits_r3": args.n_splits_r3, "seed": args.seed,
        "smoke_random": args.smoke_random,
    }, sort_keys=True)
    meta = {
        "git_sha": _git_sha(),
        "config_hash": hashlib.sha256(cfg_payload.encode()).hexdigest()[:16],
        "file_sha256_16": {
            rel: _file_sha256_16(REPO_ROOT / rel)
            for rel in ("measures/similarity.py", "measures/routing.py", "scripts/run_pilot_similarity_m4.py", "configs/checkpoints.yaml")
        },
        "device": str(device),
        "dataset": args.dataset,
        "n_images": n_img,
        "smoke_random": bool(args.smoke_random),
        "total_seconds": round(time.perf_counter() - t_all, 2),
        "throughput_by_model": {
            m: {"extract_sec": r["extract_sec"], "img_per_sec": r["img_per_sec"], "total_sec": r["total_sec"]}
            for m, r in model_metas.items()
        },
        "unix_time": int(time.time()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print("\n" + md_text, flush=True)
    return report


if __name__ == "__main__":
    main()
