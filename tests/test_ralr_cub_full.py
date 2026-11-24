"""Unit & end-to-end smoke test for Action A5 (scripts/run_ralr_cub_full.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval import metrics as M
from scripts.run_ralr_cub_full import _compute_maxboxacc_from_iou_grid, main as a5_main


def test_maxboxacc_grid_matches_eval_metrics_exact():
    rng = np.random.default_rng(7)
    maps = [rng.random((64, 64), dtype=np.float32) for _ in range(10)]
    gt_boxes = [[(12.0, 10.0, 48.0, 50.0)] for _ in range(10)]
    ref = M.maxboxacc_v2(maps, gt_boxes)

    normed = [M.minmax_normalize_map(m) for m in maps]
    iou_grid = np.empty((len(M.DEFAULT_SWEEP_THRESHOLDS), len(maps)), dtype=np.float64)
    for ti, tau in enumerate(M.DEFAULT_SWEEP_THRESHOLDS):
        for i in range(len(maps)):
            pbox = M._strided_cc_box(normed[i], float(tau))
            iou_grid[ti, i] = M.max_iou_any_box(pbox, gt_boxes[i])
    got = _compute_maxboxacc_from_iou_grid(iou_grid, M.DEFAULT_SWEEP_THRESHOLDS)
    assert abs(got["mean"] - ref["maxboxacc_v2"]) < 1e-3
    assert got["tunes_on_boxes"] is True


def test_cub_full_end_to_end_smoke_random(tmp_path):
    """End-to-end smoke test on synthetic CUB directory (images + boxes + seg masks + X_cal split)."""
    cub_root = tmp_path / "CUB_200_2011"
    img_dir = cub_root / "images" / "001.Test_Bird"
    seg_dir = tmp_path / "segmentations" / "001.Test_Bird"
    img_dir.mkdir(parents=True)
    seg_dir.mkdir(parents=True)

    img_lines, split_lines, box_lines = [], [], []
    rng = np.random.default_rng(42)
    for cid in range(1, 11):
        fname = f"bird_{cid:03d}.jpg"
        rel = f"001.Test_Bird/{fname}"
        arr = rng.integers(0, 255, size=(120, 140, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / fname)
        mask = np.zeros((120, 140), dtype=np.uint8)
        mask[25:95, 30:110] = 255
        Image.fromarray(mask).save((seg_dir / fname).with_suffix(".png"))
        img_lines.append(f"{cid} {rel}")
        split_lines.append(f"{cid} 0")  # all official test
        box_lines.append(f"{cid} 31.0 26.0 80.0 70.0")  # 1-based x y w h

    (cub_root / "images.txt").write_text("\n".join(img_lines) + "\n")
    (cub_root / "train_test_split.txt").write_text("\n".join(split_lines) + "\n")
    (cub_root / "bounding_boxes.txt").write_text("\n".join(box_lines) + "\n")
    cal_file = tmp_path / "cal_ids.txt"
    cal_file.write_text("1\n2\n3\n4\n")  # 4 calibration IDs, 6 strictly disjoint held-out

    out_dir = tmp_path / "out_a5"
    payload = a5_main([
        "--cub_dir", str(cub_root),
        "--seg_dir", str(tmp_path / "segmentations"),
        "--cal_split_file", str(cal_file),
        "--out_dir", str(out_dir),
        "--models", "dino_vits16",
        "--batch_size", "4",
        "--smoke_random",
    ])
    rep = json.loads((out_dir / "ralr_cub_full5794_report.json").read_text())
    m = rep["models"]["dino_vits16"]
    assert m["n_images_full"] == 10
    assert m["n_images_cal"] == 4
    assert m["n_images_heldout"] == 6
    assert rep["n_with_seg_masks"] == 10
    for k in ("box_auc", "pxap", "pointing", "corloc_otsu", "corloc_cal_tau", "maxboxacc_v2"):
        assert k in m["methods"]["ralr_adaptive"]
        assert k in m["methods_heldout"]["ralr_adaptive"]
    assert m["methods"]["ralr_adaptive"]["corloc_cal_tau"]["n_eval"] == 6
    assert m["methods"]["ralr_adaptive"]["maxboxacc_v2"]["tunes_on_boxes"] is True
    assert m["methods"]["ralr_adaptive"]["corloc_cal_tau"]["tunes_on_boxes"] is False
    assert payload["summary_n11_excl_mae_full"]["n_models"] == 1
