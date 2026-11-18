"""Smoke test for extract/cache.py + extract/verify_cache.py (guide §1.6).

cache.py and verify_cache.py are library modules, not standalone scripts - that's why there was
no obvious way to "run" them directly. This is the thing that actually runs: it calls both,
proving the write -> verify round trip works in this environment.

Not a real extraction: there's no dataset or §2 driver yet (that's the next real step - this only
tests the caching mechanics), so it uses a real checkpoint but throwaway random-noise images. The
mechanics being tested (do the shapes/dtypes/fields round-trip correctly) don't depend on the
images being real photos.

Usage: python extract/smoke_test.py
"""
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import timm
import torch

from adapters.timm_vit import TimmViTAdapter
from extract.cache import extract_shard
from extract.verify_cache import verify_cache

adapter = TimmViTAdapter()
model = timm.create_model("vit_small_patch16_224.dino", pretrained=True)
adapter.prepare(model)

torch.manual_seed(0)
images = torch.randn(5, 3, 224, 224)  # throwaway noise - only the caching mechanics are on test

shard = extract_shard(adapter, model, images, "dino_vits16", "smoke_test", "test", "/tmp/cache", d_model=384)
print("wrote shard:", shard)

verify_cache(adapter, model, images, shard)
print("verify_cache: OK - cache agrees with the live model")
