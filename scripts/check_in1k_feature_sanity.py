#!/usr/bin/env python3
"""ImageNet-1K feature sanity check (DeiT-III top-1 + IN-1K kNN) and frozen seed-0 ID list exporter (§8.4).

Purpose:
  1. Empirically verifies that post-norm [CLS] features and DeiT-III classifier logits are healthy on the
     1,000 ImageNet-1K pilot images (DeiT-III-S top-1 ~80% from its own logits; IN-1K cosine kNN@20 for
     DeiT-III, MoCo v3, DINO, DINOv2+reg), confirming CUB kNN differences are transfer behavior, not bugs.
  2. Exports canonical frozen seed-0 ID lists (CUB-500, ImageNet-1K-1000, and RANDOM 20-image eye-check
     subsets — never "first 20") with SHA-256 digests for committing to configs/splits/.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
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
    load_cub_pilot_split,
    load_imagenet_val_pilot_split,
)
from interventions import forward_ops as fops
from interventions import i1_i2 as I12
from scripts.run_pilot_interventions_i1_i2 import (
    _extract_gallery_embeddings,
    _load_imagenet_labels_and_gallery,
)
from scripts.run_pilot_m1_m2 import load_pilot_model


def export_random_eyecheck_ids(ids: list[str], out_path: Path, n_eye: int = 20, seed: int = 0) -> list[str]:
    """Deterministically sample `n_eye` random IDs (NOT the first 20) from `ids` with `seed`."""
    rng = np.random.default_rng(int(seed))
    k = min(int(n_eye), len(ids))
    idx = np.sort(rng.choice(len(ids), size=k, replace=False))
    chosen = [ids[int(i)] for i in idx]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(chosen) + "\n")
    return chosen


def _sha256_16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


@torch.no_grad()
def evaluate_in1k_sanity(
    model_ids: list[str],
    pilot_samples,
    pilot_labels: np.ndarray,
    gallery_paths: list[Path],
    gallery_labels: np.ndarray,
    device: torch.device,
    weights_cache: Path,
    *,
    batch_size: int = 64,
    knn_k: int = 20,
    smoke_random: bool = False,
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for mid in model_ids:
        t0 = time.perf_counter()
        keep_head = (mid == "deit3_small_patch16_224")
        adapter, model, mean, std, target_size = load_pilot_model(
            mid, device, weights_cache, smoke_random=smoke_random, keep_head=keep_head,
        )
        tf = build_image_transform(target_size, mean, std)

        embs, logits_all = [], []
        for s in range(0, len(pilot_samples), batch_size):
            batch = pilot_samples[s : s + batch_size]
            imgs = [tf(Image.open(it.image_path).convert("RGB")) for it in batch]
            xb = torch.stack(imgs, dim=0).to(device)
            zf, layout = fops.forward_to_block_input(adapter, model, xb, adapter.n_blocks(model))
            emb, logits = fops.read_global_embedding_and_logits(adapter, model, zf, layout)
            embs.append(emb.cpu().numpy().astype(np.float32))
            if logits is not None:
                logits_all.append(logits.cpu().numpy().astype(np.float32))
        pilot_emb = np.concatenate(embs, axis=0)

        top1_acc = None
        if logits_all:
            logits_np = np.concatenate(logits_all, axis=0)
            preds = np.argmax(logits_np, axis=-1)
            valid = pilot_labels >= 0
            top1_acc = float((preds[valid] == pilot_labels[valid]).mean()) if valid.any() else float("nan")

        if gallery_paths and len(gallery_paths) >= knn_k:
            gal_emb, _ = _extract_gallery_embeddings(
                adapter, model, gallery_paths, tf, device, cache_path=None, batch_size=batch_size,
            )
            knn_acc, _ = I12.knn_accuracy_cosine(
                pilot_emb, pilot_labels, feats_ref=gal_emb, labels_ref=gallery_labels,
                exclude_self=False, k=knn_k, device=device,
            )
            gal_mode = f"disjoint_val_{len(gallery_paths)}"
        else:
            knn_acc, _ = I12.knn_accuracy_cosine(pilot_emb, pilot_labels, k=knn_k, device=device)
            gal_mode = "loo_pilot"

        row = {
            "model_id": mid,
            "n_pilot": len(pilot_samples),
            "in1k_top1_logits": round(top1_acc, 4) if top1_acc is not None else None,
            "in1k_knn20": round(float(knn_acc), 4),
            "gallery_mode": gal_mode,
            "seconds": round(time.perf_counter() - t0, 1),
        }
        results[mid] = row
        top1_s = f"{top1_acc:.4f}" if top1_acc is not None else "—"
        print(
            f"[{mid}] IN-1K top-1 (logits)={top1_s} | IN-1K kNN@{knn_k}={knn_acc:.4f} ({gal_mode}) [{row['seconds']}s]",
            flush=True,
        )

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="ImageNet-1K feature sanity check + frozen seed-0 ID exporter.")
    ap.add_argument("--cub_dir", default="")
    ap.add_argument("--imagenet_images_dir", required=True)
    ap.add_argument("--imagenet_xml_dir", required=True)
    ap.add_argument("--export_splits_dir", default=str(REPO_ROOT / "configs" / "splits"))
    ap.add_argument("--out_dir", default="results/in1k_feature_sanity")
    ap.add_argument("--models", default="deit3_small_patch16_224,mocov3_vitb16,dino_vits16,dinov2_vitb14_reg")
    ap.add_argument("--n_in1k", type=int, default=1000)
    ap.add_argument("--n_cub", type=int, default=500)
    ap.add_argument("--gallery_size", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--knn_k", type=int, default=20)
    ap.add_argument("--weights_cache_dir", default=str(Path.home() / ".cache" / "queryblind_weights"))
    ap.add_argument("--smoke_random", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = Path(args.export_splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exported: dict[str, dict] = {}
    if args.cub_dir:
        cub_split_path = splits_dir / "cub_pilot_500_seed0.txt"
        cub_samples = load_cub_pilot_split(args.cub_dir, n_images=args.n_cub, seed=0, split_file=cub_split_path)
        cub_ids = [str(s.cub_id) for s in cub_samples]
        cub_eye_path = splits_dir / "cub_eyecheck_20_seed0.txt"
        cub_eye = export_random_eyecheck_ids(cub_ids, cub_eye_path, n_eye=20, seed=0)
        exported["cub_pilot_500_seed0.txt"] = {"n": len(cub_ids), "sha256_16": _sha256_16(cub_split_path)}
        exported["cub_eyecheck_20_seed0.txt"] = {"n": len(cub_eye), "sha256_16": _sha256_16(cub_eye_path), "ids": cub_eye}

    in1k_split_path = splits_dir / "imagenet_pilot_1000_seed0.txt"
    in1k_samples = load_imagenet_val_pilot_split(
        args.imagenet_images_dir, args.imagenet_xml_dir, n_images=args.n_in1k, seed=0, split_file=in1k_split_path,
    )
    in1k_ids = [s.image_id for s in in1k_samples]
    in1k_eye_path = splits_dir / "imagenet_eyecheck_20_seed0.txt"
    in1k_eye = export_random_eyecheck_ids(in1k_ids, in1k_eye_path, n_eye=20, seed=0)
    exported["imagenet_pilot_1000_seed0.txt"] = {"n": len(in1k_ids), "sha256_16": _sha256_16(in1k_split_path)}
    exported["imagenet_eyecheck_20_seed0.txt"] = {"n": len(in1k_eye), "sha256_16": _sha256_16(in1k_eye_path), "ids": in1k_eye}

    pilot_labels, gal_paths, gal_labels = _load_imagenet_labels_and_gallery(
        args.imagenet_images_dir, args.imagenet_xml_dir, in1k_samples, gallery_size=args.gallery_size, seed=1,
    )
    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    sanity = evaluate_in1k_sanity(
        model_ids, in1k_samples, pilot_labels, gal_paths, gal_labels,
        device, Path(args.weights_cache_dir),
        batch_size=args.batch_size, knn_k=args.knn_k, smoke_random=args.smoke_random,
    )

    payload = {"exported_splits": exported, "in1k_sanity": sanity}
    (out_dir / "in1k_feature_sanity.json").write_text(json.dumps(payload, indent=2))
    print("\nExported split files:", json.dumps(exported, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
