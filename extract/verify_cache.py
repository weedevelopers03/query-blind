"""Guide §1.6: "write a verify_cache.py that re-runs 32 random images through the live model and
asserts agreement with the cache to 1e-3 - run it after every extraction job, because a
half-written shard is indistinguishable from a good one until your plots look strange."

The guide doesn't store per-image identifiers in the cache (see cache.py's shapes), so this
assumes whatever calls it can re-supply the SAME images in the SAME order used to build the
shard - the natural case is a driver script re-loading its own fixed-order dataset split, so it
passes those images straight back in here rather than this module re-deriving them.

Usage: verify_cache(adapter, model, images, shard_dir) -> raises AssertionError with the first
mismatching field/image/layer on any disagreement > atol; returns silently (no printed "OK") if
everything matches, so it fits a `python verify_cache.py && echo OK` / CI-style check.
"""
import random

import numpy as np
import torch

from extract.cache import _cls_and_pooled, _extract_layer, _residual_out, last_third_blocks, load_shard

FIELDS = ["q", "k", "v", "attn", "norms", "cls", "pooled"]


@torch.no_grad()
def verify_cache(adapter, model, images: torch.Tensor, shard_dir, n_samples: int = 32,
                  atol: float = 1e-3, seed: int = 0):
    meta, cached = load_shard(shard_dir)
    n_images = meta["n_images"]
    image_list = [images[i:i + 1] for i in range(len(images))] if isinstance(images, torch.Tensor) else [
        image.unsqueeze(0) if image.ndim == 3 else image for image in images
    ]
    assert len(image_list) == n_images, (
        f"{shard_dir}: cache has {n_images} images, but {len(image_list)} were passed in - "
        "wrong dataset/split, or the cache is from a different run"
    )
    layers = meta["layers"]
    assert layers == last_third_blocks(adapter.n_blocks(model)), (
        f"{shard_dir}: cache's own recorded layers {layers} don't match this model/adapter's "
        f"last_third_blocks() {last_third_blocks(adapter.n_blocks(model))} - wrong checkpoint?"
    )
    d_model = cached["cls"].shape[-1]

    rng = random.Random(seed)
    sample = rng.sample(range(n_images), min(n_samples, n_images))

    for i in sample:
        x = image_list[i]
        for li, l in enumerate(layers):
            q, k, v, a = _extract_layer(adapter, model, x, l)
            cls, pooled = _cls_and_pooled(adapter, model, x, l, d_model)
            norms = _residual_out(adapter, model, x, l).norm(dim=-1)
            live = dict(q=q, k=k, v=v, attn=a, norms=norms, cls=cls, pooled=pooled)
            for field in FIELDS:
                # cache.py stores fp16; compare like-for-like so this catches real bugs
                # (corruption, wrong image/layer) and not just expected fp16 rounding
                live_arr = live[field].half().float().numpy()
                cached_arr = cached[field][i, li].astype(np.float32)
                default_count = cached_arr.shape[1] if field in ("k", "v") else (
                    cached_arr.shape[-1] if field == "attn" else cached_arr.shape[0]
                )
                token_count = meta.get("token_counts", [default_count] * n_images)[i]
                if field in ("k", "v"):
                    cached_arr = cached_arr[:, :token_count, :]
                elif field == "attn":
                    cached_arr = cached_arr[:, :token_count]
                elif field == "norms":
                    cached_arr = cached_arr[:token_count]
                if field == "cls" and np.isnan(cached_arr).all():
                    continue  # MAP-head models: cls is NaN by design, see cache.py
                diff = np.abs(live_arr - cached_arr).max()
                assert diff <= atol, (
                    f"{shard_dir}: image {i}, layer {l}, field '{field}' disagrees with the live "
                    f"model by {diff:.2e} (> {atol}) - re-extract this shard, don't trust it"
                )
