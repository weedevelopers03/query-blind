#!/usr/bin/env python3
"""Run 5-seed matched random-drop control (C1_rand) on all 12 models for M4 ImageNet-1K (OSF §3 H3).

Completes the registered H3 condition: tau(C0, C1_rand) - tau(C0, C1) >= 0.10 on the exact 12-model zoo.
Reuses cached feature_cache/<model>_l<layer>.npz if present; otherwise extracts R3 patches + OUT masks
in ~5-8 minutes on a single T4 GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.preprocess import load_imagenet_val_pilot_split
from measures import similarity as sim_mod
from scripts.run_pilot_m1_m2 import resolve_model_ids
from scripts.run_pilot_similarity_m4 import extract_model_m4_features


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="12-model M4 matched random-drop control (C1_rand, 5 seeds).")
    ap.add_argument("--images_dir", required=True, help="ImageNet-1K val images directory.")
    ap.add_argument("--imagenet_xml_dir", required=True, help="ImageNet-1K val XML annotations directory.")
    ap.add_argument("--split_file", default=str(REPO_ROOT / "configs" / "splits" / "imagenet_pilot_1000_seed0.txt"))
    ap.add_argument("--m4_dir", default="results/zoo12_similarity_m4_in1k", help="M4 directory (reuses feature_cache if present).")
    ap.add_argument("--out_json", default="results/zoo12_similarity_m4_in1k/zoo12_m4_c1_rand_in1k.json")
    ap.add_argument("--models", default="zoo12")
    ap.add_argument("--layers", default="8,9,10,11")
    ap.add_argument("--r3_patches", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--weights_cache_dir", default=str(Path.home() / ".cache" / "queryblind_weights"))
    ap.add_argument("--smoke_random", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_ids = resolve_model_ids(args.models)
    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    seeds = tuple(int(x.strip()) for x in args.seeds.split(",") if x.strip())

    samples = load_imagenet_val_pilot_split(
        args.images_dir, args.imagenet_xml_dir, n_images=1000, seed=0, split_file=Path(args.split_file),
    )
    image_items = [(s.image_id, s.image_path) for s in samples]
    n_img = len(image_items)
    rng = np.random.default_rng(0)
    r3_patch_idx = np.stack([np.sort(rng.choice(196, size=args.r3_patches, replace=False)) for _ in range(n_img)], axis=0)

    cache_dir = Path(args.m4_dir) / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    weights_cache = Path(args.weights_cache_dir)

    for mid in model_ids:
        if all((cache_dir / f"{mid}_l{l}.npz").exists() for l in layers):
            print(f"[{mid}] reusing cached M4 features in {cache_dir}", flush=True)
            continue
        extract_model_m4_features(
            mid, image_items, r3_patch_idx, layers, device, cache_dir, weights_cache,
            batch_size=args.batch_size, smoke_random=args.smoke_random,
        )

    per_layer = {}
    print(f"\n=== 12-Model Matched Random-Drop Control (C1_rand, seeds={seeds}, R3 CKA, N={n_img}) ===", flush=True)
    for l in layers:
        r3_tok, r3_out = {}, {}
        for mid in model_ids:
            z = np.load(cache_dir / f"{mid}_l{l}.npz")
            r3_tok[mid] = z["r3"]
            r3_out[mid] = z["r3_out"]
        res = sim_mod.r3_matched_random_drop_control(r3_tok, r3_out, model_ids, seeds=seeds, device=device)
        per_layer[f"l{l}"] = res
        gap = res["tau_c0_c1_rand_mean"] - res["tau_c0_c1_out"]
        print(
            f"  l={l:2d} | tau(C0, C1_OUT)={res['tau_c0_c1_out']:.4f} | "
            f"tau(C0, C1_rand)={res['tau_c0_c1_rand_mean']:.4f} ± {res['tau_c0_c1_rand_std']:.4f} | "
            f"gap(rand - OUT)={gap:+.4f} (PASS>=0.10: {gap >= 0.10})",
            flush=True,
        )

    payload = {
        "dataset": "imagenet_1k",
        "n_images": n_img,
        "n_models": len(model_ids),
        "models": model_ids,
        "seeds": list(seeds),
        "elapsed_sec": round(time.perf_counter() - t0, 1),
        "m4_matched_random_drop": {"imagenet_1k": per_layer},
    }
    out_p = Path(args.out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved 12-model C1_rand control -> {out_p} ({payload['elapsed_sec']}s)", flush=True)


if __name__ == "__main__":
    main()
