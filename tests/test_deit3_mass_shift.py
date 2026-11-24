"""Unit tests for Action A2 (DeiT-III I1-S4 attention-mass shift) and Action A1 (OpenImages30k RALR loader)."""
from __future__ import annotations

import numpy as np
from PIL import Image

from scripts.run_deit3_i1_s4_mass_shift import analyze_attention_mass_shift, run_one_model
from scripts.run_pilot_ralr import load_eval_samples, parse_args, precompute_gt_for_resolution
from eval.preprocess import load_imagenet_val_pilot_split
import torch


def test_a2_attention_mass_shift_conservation_and_shortcut_detection():
    """Verify mass conservation (FG+BG+OUT == 0) and recovery of BG_high_c shortcut mass loss."""
    rng = np.random.default_rng(0)
    I, N, S = 20, 196, 5

    # Construct synthetic maps: true [CLS] puts high mass on 10 high-contrib BG shortcut tokens (idx 100..109)
    # and less on FG (idx 0..49); S4 removes mass from BG shortcut tokens and shifts it onto FG!
    fg_masks = np.zeros((I, N), dtype=bool)
    fg_masks[:, :50] = True
    out_masks = np.zeros((I, N), dtype=bool)
    out_masks[:, 190:194] = True  # 4 OUT tokens

    contrib = rng.uniform(0.1, 0.5, size=(I, N))
    contrib[:, 100:110] = rng.uniform(3.0, 5.0, size=(I, 10))  # top-10% high-BSS BG shortcuts

    a_true = np.full((I, N), 0.002, dtype=np.float64)
    a_true[:, :50] = 0.005
    a_true[:, 100:110] = 0.045  # heavy shortcut concentration
    a_true /= a_true.sum(axis=1, keepdims=True)

    a_s4_seeds = np.zeros((S, I, N), dtype=np.float64)
    for s in range(S):
        arr = np.full((I, N), 0.002, dtype=np.float64)
        arr[:, :50] = 0.012 + rng.uniform(-0.0005, 0.0005, size=(I, 50))
        arr[:, 100:110] = 0.005 + rng.uniform(-0.0005, 0.0005, size=(I, 10))  # shortcuts lose mass!
        arr /= arr.sum(axis=1, keepdims=True)
        a_s4_seeds[s] = arr

    res = analyze_attention_mass_shift(
        a_true=a_true,
        a_s4_seeds=a_s4_seeds,
        fg_masks=fg_masks,
        out_masks=out_masks,
        patch_contrib=contrib,
        seed=0,
    )
    assert res["max_mass_conservation_err"] < 1e-12
    g = res["groups"]
    total_delta = g["FG"]["delta_mass_mean"] + g["BG"]["delta_mass_mean"] + g["OUT"]["delta_mass_mean"]
    assert abs(total_delta) < 1e-4
    # Mass moved OFF BG_high_c and ONTO FG
    assert g["FG"]["delta_mass_mean"] > 0.20
    assert g["BG_high_c_top10pct"]["delta_mass_mean"] < -0.20
    assert res["shortcut_loss_test"]["spearman_rho_delta_a_vs_contrib_in_BG"]["mean"] < -0.25
    assert res["shortcut_loss_test"]["frac_of_lost_mass_on_BG_high_c"]["mean"] > 0.75


def test_a1_openimages_loader_and_mask_transform(tmp_path):
    """Verify --dataset openimages pairs image/mask strictly by stem and transforms pixel masks cleanly."""
    img_dir = tmp_path / "images"
    msk_dir = tmp_path / "masks"
    img_dir.mkdir()
    msk_dir.mkdir()

    # Distractor image without mask must be skipped cleanly
    Image.new("RGB", (320, 240), (10, 10, 10)).save(img_dir / "unmatched_000.jpg")
    for stem in ("oi_001", "oi_002", "oi_003"):
        Image.new("RGB", (320, 240), (100, 120, 140)).save(img_dir / f"{stem}.jpg")
        msk = np.zeros((240, 320), dtype=np.uint8)
        msk[60:180, 100:220] = 255
        Image.fromarray(msk).save(msk_dir / f"{stem}.png")

    args = parse_args([
        "--dataset", "openimages",
        "--openimages_img_dir", str(img_dir),
        "--openimages_mask_dir", str(msk_dir),
        "--split_file", "none",
    ])
    items = load_eval_samples(args)
    assert [it["id"] for it in items] == ["oi_001", "oi_002", "oi_003"]
    masks, boxes = precompute_gt_for_resolution(items, 224)
    assert len(masks) == 3 and masks[0].shape == (224, 224)
    assert masks[0].sum() > 0
    x1, y1, x2, y2 = boxes[0][0]
    assert 0 <= x1 < x2 <= 224 and 0 <= y1 < y2 <= 224


def test_a2_end_to_end_smoke_random(tmp_path):
    """End-to-end CPU smoke test of run_one_model on DeiT-III architecture (catches adapter/unpack/routing bugs)."""
    img_dir = tmp_path / "img"
    xml_dir = tmp_path / "xml"
    img_dir.mkdir()
    xml_dir.mkdir()
    for i in range(4):
        st = f"im_{i}"
        Image.new("RGB", (240, 240), (80 + i * 30, 90, 110)).save(img_dir / f"{st}.JPEG")
        (xml_dir / f"{st}.xml").write_text(
            "<annotation><size><width>240</width><height>240</height></size>"
            "<object><bndbox><xmin>40</xmin><ymin>40</ymin><xmax>180</xmax><ymax>180</ymax></bndbox></object></annotation>"
        )
    samples = load_imagenet_val_pilot_split(img_dir, xml_dir, n_images=4)
    res = run_one_model(
        "deit3_small_patch16_224",
        samples,
        torch.device("cpu"),
        tmp_path / "weights",
        batch_size=2,
        seeds=(0, 1),
        k_pca=4,
        smoke_random=True,
    )
    assert res["max_mass_conservation_err"] < 1e-5
    assert "FG" in res["groups"] and "BG_high_c_top10pct" in res["groups"]
    assert "clean_box_auc" in res["box_auc_check"]
