"""Unit, norm/covariance, exactness, prediction-correlation, and CLI smoke tests for Intervention I5 (Guide §6)."""
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
from interventions import i5_cls_rand as I5
from scripts import run_pilot_intervention_i5 as runner_i5
from scripts.run_pilot_m1_m2 import load_pilot_model


def test_cls_state_override_exact_identity_and_perturbation(tmp_path: Path):
    """Overriding [CLS] with its clean state must match clean output (<1e-5); random override must move [CLS]."""
    torch.manual_seed(0)
    dev = torch.device("cpu")
    for mid in ("dino_vits16", "dinov2_vitb14_reg", "openclip_vitb16"):
        adapter, model, _, _, ts = load_pilot_model(mid, dev, tmp_path, smoke_random=True)
        x = torch.randn(2, 3, ts, ts)
        z8, layout = fops.forward_to_block_input(adapter, model, x, target_block=8)
        clean = fops.forward_from_block_with_interventions(
            adapter, model, z8, layout, start_block=8, capture_clean_stats=True,
        )
        # 1. Identity override at L_r=10 reproduces clean exactly
        z_cls_10 = clean["cls_state_in_by_l"][10]
        ident = fops.forward_from_block_with_interventions(
            adapter, model, z8, layout, start_block=8, cls_state_overrides={10: z_cls_10},
        )
        assert float((ident["emb"] - clean["emb"]).abs().max().item()) < 1e-5
        assert float((ident["last_weighted_map"] - clean["last_weighted_map"]).abs().max().item()) < 1e-5

        # 2. Norm-matched random override genuinely alters [CLS] embedding & final attention map
        rand_cls = torch.randn_like(z_cls_10)
        rand_cls = rand_cls / rand_cls.norm(dim=-1, keepdim=True) * z_cls_10.norm(dim=-1, keepdim=True)
        pert = fops.forward_from_block_with_interventions(
            adapter, model, z8, layout, start_block=8, cls_state_overrides={10: rand_cls},
        )
        assert float((pert["emb"] - clean["emb"]).abs().max().item()) > 1e-4
        # Note: random-init dinov2 LayerScale gamma=1e-5 yields map diff ~2.6e-6 (vs 0.0 for identity)
        assert float((pert["last_weighted_map"] - clean["last_weighted_map"]).abs().max().item()) > 1e-6


def test_i5_samplers_norm_covariance_and_derangement():
    """Verify cov_matched stays in-manifold & norm-matched, iso_norm is off-manifold & norm-matched, cross_img has j!=i."""
    rng = np.random.default_rng(42)
    I, D = 80, 128
    # Construct low-rank concentrated [CLS] cone (mean along e_0 + 4-dim fluctuations)
    mu = np.zeros(D, dtype=np.float64)
    mu[0] = 10.0
    low_rank = rng.standard_normal((I, 4)) @ np.eye(4, D) * 1.5
    z_cls = mu[None, :] + low_rank
    true_norms = np.linalg.norm(z_cls, axis=-1)

    pca = I5.fit_cls_state_pca(z_cls, k=16)

    # 1. cov_matched: exact norm match + high cosine alignment with empirical mean mu
    cov_s = I5.sample_cls_state_override("cov_matched", z_cls, pca, seed=0)
    assert np.allclose(np.linalg.norm(cov_s, axis=-1), true_norms, rtol=1e-4, atol=1e-4)
    cos_cov = (cov_s @ mu) / (np.linalg.norm(cov_s, axis=-1) * np.linalg.norm(mu))
    assert float(cos_cov.mean()) > 0.90, f"cov_matched left empirical cone: cos={cos_cov.mean()}"

    # 2. iso_norm: exact norm match + near-orthogonal to mu
    iso_s = I5.sample_cls_state_override("iso_norm", z_cls, pca, seed=0)
    assert np.allclose(np.linalg.norm(iso_s, axis=-1), true_norms, rtol=1e-4, atol=1e-4)
    cos_iso = (iso_s @ mu) / (np.linalg.norm(iso_s, axis=-1) * np.linalg.norm(mu))
    assert abs(float(cos_iso.mean())) < 0.25

    # 3. cross_img: zero self-matches (every row equals some OTHER row j != i)
    cr_s = I5.sample_cls_state_override("cross_img", z_cls, pca, seed=0)
    for i in range(I):
        assert not np.allclose(cr_s[i], z_cls[i]), f"cross_img self-matched index {i}"


def test_across_model_prediction_correlation_exact():
    x = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99])
    y = -0.5 + 0.5 * x  # exact positive linear relationship
    res = I5.bootstrap_across_model_correlation(x, y, n_boot=500, seed=0)
    assert res["n_models"] == 8
    assert np.isclose(res["spearman_rho"], 1.0)
    assert np.isclose(res["pearson_r"], 1.0)
    assert res["spearman_ci95"][0] > 0.99


def test_i5_cli_end_to_end_smoke_with_prediction_test(tmp_path: Path):
    cub = tmp_path / "CUB_200_2011"
    (cub / "images" / "001.Bird").mkdir(parents=True)
    rng = np.random.default_rng(0)
    img_lines, split_lines, box_lines, cls_lines = [], [], [], []
    for i in range(1, 13):
        rel = f"001.Bird/img_{i:02d}.jpg"
        Image.fromarray(rng.integers(0, 255, size=(80, 100, 3), dtype=np.uint8)).save(cub / "images" / rel)
        img_lines.append(f"{i} {rel}")
        split_lines.append(f"{i} {0 if i <= 6 else 1}")
        box_lines.append(f"{i} 10 10 50 50")
        cls_lines.append(f"{i} {(i % 2) + 1}")
    (cub / "images.txt").write_text("\n".join(img_lines) + "\n")
    (cub / "train_test_split.txt").write_text("\n".join(split_lines) + "\n")
    (cub / "bounding_boxes.txt").write_text("\n".join(box_lines) + "\n")
    (cub / "image_class_labels.txt").write_text("\n".join(cls_lines) + "\n")

    # Mock M1/M2 directory for 3 models to verify the across-model prediction test table
    m12_dir = tmp_path / "m12_mock"
    (m12_dir / "models").mkdir(parents=True)
    for idx, mid in enumerate(("dino_vits16", "deit3_small_patch16_224", "dinov2_vitb14_reg")):
        mock_m12 = {
            "layers": {
                "l10": {"qii": {"S4": {"mean": 0.7 + 0.1 * idx}, "S1": {"mean": 0.5 + 0.1 * idx}}, "keydom": {"omega2_key_mean": 0.4 + 0.2 * idx}},
                "l11": {"qii": {"S4": {"mean": 0.75 + 0.08 * idx}, "S1": {"mean": 0.55 + 0.08 * idx}}, "keydom": {"omega2_key_mean": 0.45 + 0.2 * idx}},
            }
        }
        (m12_dir / "models" / f"{mid}.json").write_text(json.dumps(mock_m12))

    out_dir = tmp_path / "i5_out"
    runner_i5.main([
        "--dataset", "cub",
        "--cub_dir", str(cub),
        "--split_file", str(tmp_path / "split.txt"),
        "--m1_m2_dir", str(m12_dir),
        "--out_dir", str(out_dir),
        "--models", "dino_vits16,deit3_small_patch16_224,dinov2_vitb14_reg",
        "--n_images", "6",
        "--batch_size", "3",
        "--l_r_sweep", "10,11",
        "--seeds", "0,1",
        "--knn_k", "3",
        "--smoke_random",
    ])
    md_text = (out_dir / "summary_i5.md").read_text()
    assert (out_dir / "i5_report.json").exists()
    assert (out_dir / "meta.json").exists()
    assert "## 1. Per-Model Map Survival" in md_text
    assert "## 2. Across-Model Prediction Test" in md_text
    rep = json.loads((out_dir / "i5_report.json").read_text())
    assert len(rep["prediction_test"]) > 0
    assert rep["models"]["deit3_small_patch16_224"]["has_logits"] is True
