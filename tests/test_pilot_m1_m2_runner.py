"""Tests for scripts/run_pilot_m1_m2.py:
  1. Single-pass batched extractor matches qii.extract_m1_inputs_from_adapter() to < 1e-5 across
     TimmViTAdapter (standard + DINOv2-reg at 196px) and OpenCLIPViTAdapter (ViT-B-16).
  2. Official iBOT & MoCo v3 state_dict remappers load strictly with 0 missing backbone keys.
  3. Full CLI end-to-end smoke run on a synthetic CUB-200-2011 directory (--smoke_random),
     verifying split persistence, 20-image overlays, summary.md, fix3_and_g1.json, meta.json, keymaps.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import timm
import torch
from PIL import Image

from adapters.open_clip_vit import OpenCLIPViTAdapter
from adapters.timm_vit import TimmViTAdapter
from measures import keydom, qii
from measures._common import box_auc, box_auc_torch
from scripts.run_pilot_m1_m2 import (
    clean_ibot_state_dict,
    clean_mocov3_state_dict,
    extract_last_third_batch,
    main as run_pilot_main,
    score_layer_gpu,
)


@pytest.mark.parametrize("kind", ["dino_s16", "dinov2_b14_reg", "openclip_b16"])
def test_single_pass_batched_extractor_matches_adapter_reference(kind):
    torch.manual_seed(0)
    if kind == "dino_s16":
        model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0)
        adapter, res = TimmViTAdapter(), 224
    elif kind == "dinov2_b14_reg":
        model = timm.create_model(
            "vit_base_patch14_reg4_dinov2", pretrained=False, num_classes=0, img_size=196, dynamic_img_size=True
        )
        adapter, res = TimmViTAdapter(), 196
    else:
        import open_clip

        model = open_clip.create_model("ViT-B-16", pretrained=None).visual
        adapter, res = OpenCLIPViTAdapter(), 224

    adapter.prepare(model)
    x = torch.randn(2, 3, res, res)
    layers = [8, 9, 10, 11]
    fast = extract_last_third_batch(adapter, model, x, layers)

    for l in (8, 11):
        ref = qii.extract_m1_inputs_from_adapter(adapter, model, x, l)
        for key in ("q_g", "k_patch", "q_patches", "head_weights"):
            diff = (fast[l][key] - ref[key].float().cpu()).abs().max().item()
            assert diff < 1e-5, f"{kind} l={l} {key} diff={diff}"
        assert fast[l]["grid"] == (14, 14)


def test_ibot_and_mocov3_state_dict_remappers_strict_backbone():
    # Build fake official iBOT checkpoint (module.backbone.* + head.* + masked_embed)
    m_s = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0)
    fake_ibot = {"state_dict": {f"backbone.{k}": v.clone() for k, v in m_s.state_dict().items()}}
    fake_ibot["state_dict"]["head.mlp.weight"] = torch.zeros(4, 4)
    fake_ibot["state_dict"]["masked_embed"] = torch.zeros(1, 384)
    sd_ibot = clean_ibot_state_dict(fake_ibot)
    msg_i = m_s.load_state_dict(sd_ibot, strict=False)
    assert msg_i.missing_keys == [] and msg_i.unexpected_keys == []

    # Build fake official MoCo v3 checkpoint (module.base_encoder.* + module.momentum_encoder.* + head.*)
    m_b = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
    fake_moco = {"state_dict": {f"module.base_encoder.{k}": v.clone() for k, v in m_b.state_dict().items()}}
    fake_moco["state_dict"]["module.base_encoder.head.weight"] = torch.zeros(4, 4)
    fake_moco["state_dict"]["module.momentum_encoder.cls_token"] = torch.zeros(1, 1, 768)
    sd_moco = clean_mocov3_state_dict(fake_moco)
    msg_m = m_b.load_state_dict(sd_moco, strict=False)
    assert msg_m.missing_keys == [] and msg_m.unexpected_keys == []


def test_cli_end_to_end_smoke_on_synthetic_cub(tmp_path):
    cub = tmp_path / "CUB_200_2011"
    (cub / "images" / "001.Bird").mkdir(parents=True)
    img_lines, split_lines, box_lines = [], [], []
    rng = np.random.default_rng(0)
    for i in range(1, 13):
        rel = f"001.Bird/img_{i:03d}.jpg"
        arr = rng.integers(0, 255, size=(240, 320, 3), dtype=np.uint8)
        Image.fromarray(arr).save(cub / "images" / rel)
        img_lines.append(f"{i} {rel}")
        # Mark first 2 as train (1), rest 10 as test (0)
        split_lines.append(f"{i} {1 if i <= 2 else 0}")
        box_lines.append(f"{i} 60.0 30.0 180.0 160.0")
    (cub / "images.txt").write_text("\n".join(img_lines) + "\n")
    (cub / "train_test_split.txt").write_text("\n".join(split_lines) + "\n")
    (cub / "bounding_boxes.txt").write_text("\n".join(box_lines) + "\n")

    out_dir = tmp_path / "out"
    split_file = tmp_path / "split.txt"
    run_pilot_main([
        "--cub_dir", str(cub),
        "--out_dir", str(out_dir),
        "--split_file", str(split_file),
        "--n_images", "6",
        "--models", "dino_vits16,deit3_small_patch16_224,dinov2_vitb14_reg",
        "--batch_size", "3",
        "--smoke_random",
    ])

    assert split_file.exists() and len(split_file.read_text().splitlines()) == 6
    assert (out_dir / "box_overlays_20_224px.png").exists()
    assert (out_dir / "box_overlays_20_196px.png").exists()
    assert (out_dir / "summary.md").exists()
    assert (out_dir / "summary_by_model_layer.csv").exists()

    meta = json.loads((out_dir / "meta.json").read_text())
    assert meta["sanity_checks_all_passed"] is True
    assert meta["n_images"] == 6

    f3g1 = json.loads((out_dir / "fix3_and_g1.json").read_text())
    assert f3g1["fix3_spearman_l_star"]["n_models"] == 3
    assert f3g1["g1_dino_vs_deit3"]["at_l_star"]["qii_s4_diff_dino_minus_deit3"]["n_images"] == 6

    km = np.load(out_dir / "keymaps" / "dinov2_vitb14_reg.npy")
    assert km.dtype == np.float16 and km.shape == (6, 4, 12, 14, 14)


def test_box_auc_torch_exact_ties_and_score_layer_gpu_equivalence():
    # 1. Exact tie handling (constant map, discrete quantized ties, random continuous, empty mask)
    rng = np.random.default_rng(7)
    maps = np.stack([
        rng.standard_normal((16, 16)).astype(np.float32),
        rng.integers(0, 4, size=(16, 16)).astype(np.float32),  # heavy ties!
        np.ones((16, 16), dtype=np.float32),                   # all tied -> 0.5
        rng.standard_normal((16, 16)).astype(np.float32),
    ])
    masks = np.zeros((4, 16, 16), dtype=np.uint8)
    masks[0, 4:12, 4:12] = 1
    masks[1, 2:10, 6:14] = 1
    masks[2, 0:8, 0:8] = 1
    # masks[3] all-zero -> NaN

    got_t = box_auc_torch(torch.from_numpy(maps), torch.from_numpy(masks)).numpy()
    for i in range(3):
        exp = box_auc(maps[i], masks[i])
        assert got_t[i] == pytest.approx(exp, abs=1e-12)
    assert np.isnan(got_t[3])

    # 2. score_layer_gpu matches single-image reference qii.compute_qii_image + keydom.compute_key_dominance_image
    I, H, G, d = 8, 3, 6, 8
    N = G * G
    scale = d ** -0.5
    qg = rng.standard_normal((I, H, d)).astype(np.float32)
    kp = rng.standard_normal((I, H, N, d)).astype(np.float32)
    qp = rng.standard_normal((I, H, N, d)).astype(np.float32)
    wh = rng.uniform(0.1, 1.0, size=(I, H)).astype(np.float32)
    gt = np.zeros((I, 24, 24), dtype=np.uint8)
    gt[:, 6:18, 6:18] = 1
    pca = qii.fit_query_pca(qg, k=4)

    fast = score_layer_gpu(qg, kp, qp, wh, gt, scale, (G, G), pca, torch.device("cpu"), chunk_size=3)

    for i in range(I):
        r1 = qii.compute_qii_image(
            qg[i], kp[i], scale, (G, G),
            q_global_pool=qg, image_idx=i, q_patches=qp[i], pca=pca, head_weights=wh[i], gt_mask=gt[i],
        )
        assert fast["ent_weighted"][i] == pytest.approx(r1["entropy_weighted"], abs=1e-6)
        assert fast["auc_attn_weighted"][i] == pytest.approx(r1["box_auc_weighted"], abs=1e-5)
        np.testing.assert_allclose(fast["auc_attn_per_head"][i], r1["box_auc_per_head"], atol=1e-5)
        for S in qii.SURROGATES:
            assert fast["qii_weighted_by_s"][S][i] == pytest.approx(r1["per_surrogate"][S]["qii_weighted"], abs=1e-5)
            np.testing.assert_allclose(fast["qii_by_s"][S][i], r1["per_surrogate"][S]["qii"], atol=1e-5)

        Q64 = keydom.build_query_set_64(qg[i], qp[i], qg, image_idx=i, seed=0)
        r2 = keydom.compute_key_dominance_image(Q64, kp[i], scale, (G, G), head_weights=wh[i], gt_mask=gt[i])
        assert fast["omega2_weighted"][i] == pytest.approx(r2["omega2_weighted"], abs=1e-6)
        assert fast["auc_km_weighted"][i] == pytest.approx(r2["box_auc_agg"], abs=1e-5)
        np.testing.assert_allclose(fast["keymaps_fp16"][i], r2["keymap"].astype(np.float16), atol=1e-3)

