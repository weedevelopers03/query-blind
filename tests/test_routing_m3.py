"""Unit, seeded-bug, and integration tests for Measure M3 (measures/routing.py + run_pilot_waterbirds_m3.py).

Tests every §4 & §12 invariant:
  1. Sum over heads BEFORE taking norm (seeded bug: summing per-head norms overstates c_i when heads cancel).
  2. Exact reconstruction sum_i u_i + (gamma * b_O) == adapter.global_update() to < 1e-5 on DeiT-III (with LayerScale),
     DINOv2-reg (with LayerScale + 4 registers), and OpenCLIP ViT-B-16 — AND omitting LayerScale fails!
  3. Uniform contributions -> BSS_norm == 1.0 regardless of FG/BG area ratio (proves area confound removed, §4.3),
     while BSS_raw scales with background area!
  4. OUT outlier detection (median + 5 * MAD) and OUT-overrides-FG/BG primary vs spatial-retained alt ordering (§4.2).
  5. Residual ratio RR = ||x_g^(l)|| / ||Delta_g^(l)|| exact check + Waterbirds congruence gap & tercile stratification.
  6. ImageNet normalization override check for ibot_vits16 & mocov3_vitb16 in load_pilot_model().
  7. End-to-end CLI smoke test of run_pilot_waterbirds_m3.py on synthetic Waterbirds + CUB segmentations.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import timm
import torch
from PIL import Image

from adapters.open_clip_vit import OpenCLIPViTAdapter
from adapters.timm_vit import TimmViTAdapter
from measures import routing
from measures._common import get_output_proj_and_layerscale
from scripts.run_pilot_m1_m2 import IMAGENET_MEAN, IMAGENET_STD, load_pilot_model
from scripts.run_pilot_waterbirds_m3 import extract_m3_last_third_batch, main as run_wb_main


def test_1_sum_over_heads_before_norm_catches_cancellation_bug():
    # Two heads pointing in EXACT opposite directions for token 0 -> true c_0 = 0.0
    # Seeded bug (sum_h ||u_{h,0}||_2) would wrongly report 2.0!
    attn = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])   # (B=1, H=2, T=2)
    v = torch.zeros(1, 2, 2, 2)
    v[0, 0, 0] = torch.tensor([1.0, 0.0])
    v[0, 1, 0] = torch.tensor([-1.0, 0.0])
    # W_O maps each head's 2D vector identically into shared 2D output space
    W_O = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])

    c = routing.token_contributions(attn, v, W_O)
    assert c[0, 0].item() == pytest.approx(0.0, abs=1e-6)

    # Contrast against the §4.1/§12 anti-pattern (norm per head then sum):
    c_head0 = routing.token_contributions(attn[:, 0:1], v[:, 0:1], W_O[:, :2])
    c_head1 = routing.token_contributions(attn[:, 1:2], v[:, 1:2], W_O[:, 2:])
    wrong_sum_of_norms = (c_head0 + c_head1)[0, 0].item()
    assert wrong_sum_of_norms == pytest.approx(2.0, abs=1e-6)


@pytest.mark.parametrize("kind", ["deit3_s16", "dinov2_b14_reg", "openclip_b16"])
def test_2_token_contributions_reconstruct_global_update_and_catch_layerscale(kind):
    torch.manual_seed(0)
    if kind == "deit3_s16":
        model = timm.create_model("deit3_small_patch16_224", pretrained=False, num_classes=0)
        with torch.no_grad():
            for blk in model.blocks:
                blk.ls1.gamma.uniform_(0.3, 0.8)
        adapter, res = TimmViTAdapter(), 224
    elif kind == "dinov2_b14_reg":
        model = timm.create_model(
            "vit_base_patch14_reg4_dinov2", pretrained=False, num_classes=0, img_size=196, dynamic_img_size=True
        )
        with torch.no_grad():
            for blk in model.blocks:
                blk.ls1.gamma.uniform_(0.3, 0.8)
        adapter, res = TimmViTAdapter(), 196
    else:
        import open_clip

        model = open_clip.create_model("ViT-B-16", pretrained=None).visual
        adapter, res = OpenCLIPViTAdapter(), 224

    adapter.prepare(model)
    x = torch.randn(2, 3, res, res)
    l = 11
    _, _, v_full = adapter.get_qkv(model, x, l)
    attn_row = adapter.attention_probs(model, x, l)[:, :, 0, :]
    W_O, b_O, gamma = get_output_proj_and_layerscale(adapter, model, l)

    c, u = routing.token_contributions(attn_row, v_full, W_O, gamma=gamma, return_vectors=True)
    b_eff = (b_O * (gamma if gamma is not None else 1.0)) if b_O is not None else 0.0
    recon = u.sum(dim=1) + b_eff
    truth = adapter.global_update(model, x, l)
    assert (recon - truth).abs().max().item() < 1e-5

    # Batched extractor matches exact c_all and delta_g_norm
    batched = extract_m3_last_third_batch(adapter, model, x, [8, 11])[11]
    assert (batched["c_all"] - c).abs().max().item() < 1e-5
    assert (batched["delta_g_norm"] - truth.norm(dim=-1)).abs().max().item() < 1e-5

    # Seeded bug: omitting LayerScale on DeiT-III / DINOv2-reg must fail reconstruction
    if gamma is not None:
        _, u_bad = routing.token_contributions(attn_row, v_full, W_O, gamma=None, return_vectors=True)
        bad_recon = u_bad.sum(dim=1) + (b_O if b_O is not None else 0.0)
        assert (bad_recon - truth).abs().max().item() > 1e-2


def test_3_uniform_attention_gives_bss_norm_1_and_outlier_override_works():
    # 1 CLS + 4 REG + 16 patches = 21 tokens; 4 FG patches (25% area), 12 BG patches (75% area)
    layout = dict(cls_idx=0, register_idx=[1, 2, 3, 4], patch_idx=list(range(5, 21)), dist_idx=[], grid=(4, 4))
    fg_mask = np.zeros((1, 16), dtype=np.uint8)
    fg_mask[0, :4] = 1  # patches 0..3 are FG (25%), 4..15 are BG (75%)

    # Case A: Uniform contributions across all patches -> BSS_norm MUST be 1.0, while BSS_raw = 0.75 (area confounded!)
    c_unif = np.ones((1, 21), dtype=np.float64)
    c_unif[0, 0] = 0.0   # zero self so BSS_raw = 12/16 = 0.75
    c_unif[0, 1:5] = 0.5  # registers
    norms_clean = np.full((1, 21), 10.0)
    norms_clean[0, 5::2] = 10.5  # small non-zero MAD = 0.25 -> threshold 11.25 -> 0 outliers

    dec = routing.routing_decomposition_batch(c_unif, norms_clean, layout, fg_mask, np.array([20.0]), np.array([4.0]))
    assert dec["n_out_tokens"][0] == 0
    assert dec["bss_norm"][0] == pytest.approx(1.0, abs=1e-12)
    assert dec["bss_raw"][0] == pytest.approx(0.75, abs=1e-12)
    assert dec["residual_ratio"][0] == pytest.approx(5.0, abs=1e-12)
    # All 5 shares sum to 1.0
    total_share = dec["share_fg"][0] + dec["share_bg"][0] + dec["share_out"][0] + dec["share_reg"][0] + dec["share_self"][0]
    assert total_share == pytest.approx(1.0, abs=1e-12)

    # Case B: Make patch 0 (inside FG!) a massive high-norm outlier (norm=100, c=10)
    # Primary ordering: OUT overrides FG -> patch 0 moves to OUT -> BSS_norm >> 1, whereas alt keeps it in FG -> BSS_norm_alt < 1
    c_out = c_unif.copy()
    c_out[0, 5] = 10.0  # patch 0 (spatial FG)
    norms_out = norms_clean.copy()
    norms_out[0, 5] = 100.0
    dec_o = routing.routing_decomposition_batch(c_out, norms_out, layout, fg_mask, np.array([20.0]), np.array([4.0]))
    assert dec_o["n_out_tokens"][0] == 1
    assert dec_o["bss_norm"][0] > 1.5       # (12*1 + 10)/13 / 1.0 = 22/13 ≈ 1.692
    assert dec_o["bss_norm_alt"][0] < 0.4   # 1.0 / ((10 + 3)/4) = 4/13 ≈ 0.308


def test_4_waterbirds_congruence_analysis_and_terciles():
    rng = np.random.default_rng(0)
    n = 120
    y = np.array([0] * 60 + [1] * 60)
    place = np.array([0] * 30 + [1] * 30 + [0] * 30 + [1] * 30)
    cong = y == place
    # Synthetic shortcut model: congruent BSS_norm ~ 1.8, incongruent ~ 1.1 -> gap ~ +0.7
    bss = np.where(cong, 1.8, 1.1) + 0.05 * rng.standard_normal(n)
    areas = rng.uniform(0.05, 0.6, size=n)
    terc = routing.assign_area_terciles(areas)
    assert set(terc) == {"small", "medium", "large"}

    wb = routing.waterbirds_congruence_analysis(bss, cong, area_terciles=terc, y_bird=y, place_bg=place)
    assert wb["congruence_gap_cong_minus_incong"]["mean_diff"] == pytest.approx(0.7, abs=0.05)
    assert wb["congruence_gap_cong_minus_incong"]["ci_excludes_zero"] is True
    assert len(wb["by_group"]) == 4


def test_5_ibot_and_mocov3_use_imagenet_norm_override(tmp_path):
    dev = torch.device("cpu")
    for mid in ("ibot_vits16", "mocov3_vitb16"):
        _, _, mean, std, ts = load_pilot_model(mid, dev, tmp_path, smoke_random=True)
        assert mean == IMAGENET_MEAN and std == IMAGENET_STD and ts == 224
    # OpenCLIP keeps OpenAI CLIP norm (not ImageNet)
    _, _, oc_m, oc_s, _ = load_pilot_model("openclip_vitb16", dev, tmp_path, smoke_random=True)
    assert abs(oc_m[0] - 0.48145466) < 1e-4


def test_6_waterbirds_m3_cli_end_to_end_smoke(tmp_path):
    wb = tmp_path / "waterbird_complete95_forest2water2"
    seg = tmp_path / "segmentations"
    (wb / "001.Bird").mkdir(parents=True)
    (seg / "001.Bird").mkdir(parents=True)

    rng = np.random.default_rng(1)
    rows = ["img_id,img_filename,y,split,place,place_filename"]
    for i in range(1, 13):
        rel = f"001.Bird/img_{i:03d}.jpg"
        Image.fromarray(rng.integers(0, 255, size=(240, 300, 3), dtype=np.uint8)).save(wb / rel)
        m = np.zeros((240, 300), dtype=np.uint8)
        m[60:180, 90:210] = 255
        Image.fromarray(m).save((seg / rel).with_suffix(".png"))
        y = i % 2
        pl = (i // 2) % 2
        rows.append(f"{i},{rel},{y},2,{pl},bg.jpg")
    (wb / "metadata.csv").write_text("\n".join(rows) + "\n")

    out_dir = tmp_path / "out_m3"
    run_wb_main([
        "--waterbirds_dir", str(wb),
        "--cub_seg_dir", str(seg),
        "--out_dir", str(out_dir),
        "--models", "dino_vits16,deit3_small_patch16_224,dinov2_vitb14_reg",
        "--batch_size", "4",
        "--smoke_random",
    ])
    assert (out_dir / "summary_waterbirds.md").exists()
    assert (out_dir / "summary_waterbirds_m3.csv").exists()
    assert (out_dir / "waterbirds_congruence.json").exists()
    meta = json.loads((out_dir / "meta.json").read_text())
    assert meta["n_images"] == 12
    assert meta["n_congruent"] + meta["n_incongruent"] == 12
