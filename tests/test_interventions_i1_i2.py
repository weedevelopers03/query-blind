"""Unit, invariance, matched-control, and end-to-end smoke tests for Interventions I1 & I2 (Guide §6)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from interventions import forward_ops as fops
from interventions import i1_i2 as I12
from scripts import run_pilot_interventions_i1_i2 as runner_i12
from scripts.run_pilot_m1_m2 import load_pilot_model


@pytest.mark.parametrize("model_id", ["dino_vits16", "deit3_small_patch16_224", "dinov2_vitb14_reg", "openclip_vitb16"])
def test_clean_block_forward_matches_native_model_exact(model_id: str, tmp_path: Path):
    """Critical correctness test: zero-edit intervened forward must reproduce native model output to <1e-4."""
    torch.manual_seed(0)
    dev = torch.device("cpu")
    adapter, model, _, _, target_size = load_pilot_model(
        model_id, dev, tmp_path, smoke_random=True, keep_head=(model_id == "deit3_small_patch16_224")
    )
    x = torch.randn(2, 3, target_size, target_size)

    with torch.no_grad():
        if adapter.family == "timm_vit":
            ref_feat = model.forward_features(x)
            ref_emb = ref_feat[:, 0, :].float()
            ref_logits = model(x).float() if model.num_classes > 0 and not isinstance(model.head, torch.nn.Identity) else None
        else:
            ref_emb = model(x).float()
            ref_logits = None

        z8, layout = fops.forward_to_block_input(adapter, model, x, target_block=8)
        out = fops.forward_from_block_with_interventions(adapter, model, z8, layout, start_block=8, capture_clean_stats=True)

    max_emb_diff = float((out["emb"] - ref_emb).abs().max().item())
    assert max_emb_diff < 1e-4, f"[{model_id}] clean embedding drift {max_emb_diff}"
    if ref_logits is not None:
        assert out["logits"] is not None
        max_log_diff = float((out["logits"] - ref_logits).abs().max().item())
        assert max_log_diff < 1e-4, f"[{model_id}] clean logits drift {max_log_diff}"


def test_i1_and_i2_preserve_patch_stream_exactly(tmp_path: Path):
    """Guide §6: I1 and I2 (zero & mean_v) must edit ONLY [CLS] read-out while leaving all patch tokens identical."""
    torch.manual_seed(1)
    dev = torch.device("cpu")
    adapter, model, _, _, ts = load_pilot_model("dinov2_vitb14_reg", dev, tmp_path, smoke_random=True)
    x = torch.randn(2, 3, ts, ts)
    z8, layout = fops.forward_to_block_input(adapter, model, x, target_block=8)
    patch_idx = layout["patch_idx"]

    clean = fops.forward_from_block_with_interventions(adapter, model, z8, layout, start_block=8, capture_clean_stats=True)
    v_means = {l: (clean["v_sum_by_layer"][l] / float(clean["v_count_by_layer"][l])).float() for l in clean["v_sum_by_layer"]}

    # I1: replace [CLS] query in last 2 blocks (10, 11) with random norm-matched surrogate
    q_ov = {l: torch.randn_like(clean["q_g_by_layer"][l]) * 5.0 for l in (10, 11)}
    res_i1 = fops.forward_from_block_with_interventions(adapter, model, z8, layout, start_block=8, q_cls_overrides=q_ov)

    # Patch tokens at blocks 10..11 must be untouched at final block (Wait: in block 11, patches attend to [CLS] too!
    # So at L_int=1 (last block only), patch outputs at block 11 are bit-for-bit identical!)
    res_i1_l1 = fops.forward_from_block_with_interventions(adapter, model, z8, layout, start_block=8, q_cls_overrides={11: q_ov[11]})
    patch_diff_i1 = float((res_i1_l1["z_final"][:, patch_idx, :] - clean["z_final"][:, patch_idx, :]).abs().max().item())
    cls_diff_i1 = float((res_i1_l1["emb"] - clean["emb"]).abs().max().item())
    assert patch_diff_i1 < 1e-5, f"I1 L_int=1 modified patch stream: {patch_diff_i1}"
    assert cls_diff_i1 > 1e-4, "I1 failed to change [CLS] embedding"
    assert float((res_i1["emb"] - clean["emb"]).abs().max().item()) > 1e-4

    # I2: zero-ablation AND mean_v ablation on 30 patch tokens at final block 11
    T = z8.shape[1]
    ko = torch.zeros((2, T), dtype=torch.bool)
    ko[:, patch_idx[:30]] = True
    for mode in ("zero", "mean_v"):
        res_i2 = fops.forward_from_block_with_interventions(
            adapter, model, z8, layout, start_block=8, ko_masks={11: ko}, ko_mode=mode, v_means=v_means,
        )
        p_diff = float((res_i2["z_final"][:, patch_idx, :] - clean["z_final"][:, patch_idx, :]).abs().max().item())
        c_diff = float((res_i2["emb"] - clean["emb"]).abs().max().item())
        assert p_diff < 1e-5, f"I2 ({mode}) modified patch stream: {p_diff}"
        assert c_diff > 1e-5, f"I2 ({mode}) failed to change [CLS] embedding"
        if mode == "zero":
            # Knocked-out patch attention weights must be identically 0.0
            assert float(res_i2["last_attn_patch"][:, :, :30].abs().max().item()) < 1e-7
            # Remaining patch weights sum to 1.0
            assert torch.allclose(res_i2["last_attn_patch"].sum(dim=-1), torch.ones(2, res_i2["last_attn_patch"].shape[1]), atol=1e-5)


def test_matched_random_mask_exact_count_and_no_cls_or_reg():
    layout = {"cls_idx": 0, "register_idx": [1, 2, 3, 4], "patch_idx": list(range(5, 201)), "n_tokens": 201}
    fg = np.zeros((10, 196), dtype=bool)
    out = np.zeros((10, 196), dtype=bool)
    fg[:, :40] = True
    out[:, :3] = True  # OUT overrides FG
    masks = I12.build_token_group_masks(fg, out, layout)
    assert (masks["OUT"].sum(axis=1) == 3).all()
    assert (masks["FG"].sum(axis=1) == 37).all()
    assert (masks["BG"].sum(axis=1) == 156).all()
    assert (masks["REG"].sum(axis=1) == 4).all()

    r0 = I12.make_matched_random_patch_mask(masks["FG"], layout, seed=0)
    r1 = I12.make_matched_random_patch_mask(masks["FG"], layout, seed=1)
    assert (r0.sum(axis=1) == 37).all()
    assert not r0[:, :5].any(), "Random patch mask must never select [CLS] or register tokens!"
    assert not np.array_equal(r0, r1)


def test_i1_i2_cli_end_to_end_smoke(tmp_path: Path):
    cub = tmp_path / "CUB_200_2011"
    (cub / "images" / "001.Bird").mkdir(parents=True)
    rng = np.random.default_rng(0)
    img_lines, split_lines, box_lines, cls_lines = [], [], [], []
    for i in range(1, 9):
        rel = f"001.Bird/img_{i:02d}.jpg"
        Image.fromarray(rng.integers(0, 255, size=(80, 100, 3), dtype=np.uint8)).save(cub / "images" / rel)
        img_lines.append(f"{i} {rel}")
        split_lines.append(f"{i} 0")
        box_lines.append(f"{i} 10 10 50 50")
        cls_lines.append(f"{i} {(i % 2) + 1}")
    (cub / "images.txt").write_text("\n".join(img_lines) + "\n")
    (cub / "train_test_split.txt").write_text("\n".join(split_lines) + "\n")
    (cub / "bounding_boxes.txt").write_text("\n".join(box_lines) + "\n")
    (cub / "image_class_labels.txt").write_text("\n".join(cls_lines) + "\n")

    out_dir = tmp_path / "i1_i2_out"
    runner_i12.main([
        "--cub_dir", str(cub),
        "--split_file", str(tmp_path / "split.txt"),
        "--out_dir", str(out_dir),
        "--models", "deit3_small_patch16_224,dinov2_vitb14_reg",
        "--n_images", "6",
        "--batch_size", "3",
        "--l_int_sweep", "1,2",
        "--seeds", "0,1",
        "--knn_k", "3",
        "--smoke_random",
    ])
    assert (out_dir / "summary_i1_i2.md").exists()
    assert (out_dir / "i1_i2_report.json").exists()
    assert (out_dir / "meta.json").exists()
    meta = json.loads((out_dir / "meta.json").read_text())
    assert meta["git_sha"] is not None and len(meta["git_sha"]) >= 7
    rep = json.loads((out_dir / "i1_i2_report.json").read_text())
    assert rep["deit3_small_patch16_224"]["has_logits"] is True
    assert "kl_mean" in rep["deit3_small_patch16_224"]["i1_results"][0]
    assert any(r["group"] == "REG" for r in rep["dinov2_vitb14_reg"]["i2_results"])
