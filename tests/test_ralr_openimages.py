"""Unit & end-to-end smoke tests for Action A1 (scripts/run_ralr_openimages30k.py)."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from eval import metrics as M
from scripts.run_ralr_openimages30k import (
    load_openimages_pairs,
    pxap_torch_batch,
    run_one_model_openimages,
)


def test_pxap_torch_matches_cpu_exact_including_ties():
    """Verify GPU/torch pxap_torch_batch matches eval/metrics.py::pxap to <1e-5 even with heavy score ties."""
    rng = np.random.default_rng(42)
    B, H, W = 6, 32, 32
    maps = rng.uniform(0.0, 1.0, size=(B, H, W)).astype(np.float32)
    # Inject quantized ties on half the batch
    maps[:3] = np.round(maps[:3] * 10.0) / 10.0
    masks = rng.uniform(0.0, 1.0, size=(B, H, W)) > 0.65

    gpu_vals = pxap_torch_batch(torch.from_numpy(maps), torch.from_numpy(masks)).numpy()
    cpu_vals = np.array([M.pxap(maps[i], masks[i]) for i in range(B)], dtype=np.float64)
    np.testing.assert_allclose(gpu_vals, cpu_vals, atol=1e-5)


def test_openimages_end_to_end_smoke_random(tmp_path):
    """End-to-end CPU smoke test of run_one_model_openimages on synthetic images + masks."""
    img_dir = tmp_path / "images"
    msk_dir = tmp_path / "masks"
    img_dir.mkdir()
    msk_dir.mkdir()
    for i in range(6):
        Image.new("RGB", (240, 200), (60 + i * 25, 90, 120)).save(img_dir / f"oi_{i:03d}.jpg")
        msk = np.zeros((200, 240), dtype=np.uint8)
        msk[40:150, 60:180] = 255
        Image.fromarray(msk).save(msk_dir / f"oi_{i:03d}.png")

    items = load_openimages_pairs(img_dir, msk_dir)
    assert len(items) == 6
    gt_cache = {}
    res = run_one_model_openimages(
        "dino_vits16", items, gt_cache, torch.device("cpu"),
        tmp_path / "out", tmp_path / "w",
        batch_size=3, cal_n=4, seed=0, smoke_random=True,
    )
    assert res["n_images_full"] == 6
    assert "pxap" in res["methods"]["ralr_adaptive"]
    assert "degen_frac_outside_002_080" in res["methods"]["cls_l_star"]["corloc_otsu"]
    assert "pxap" in res["paired_diffs"]["ralr_adaptive_minus_cls_l_star"]
