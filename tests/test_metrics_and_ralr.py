"""Hand-computed toy tests for eval/metrics.py (§9), RALR (§7), and ImageNet-1K XML loader (§8)."""
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
from eval.preprocess import (
    get_transformed_imagenet_gt_mask,
    load_imagenet_val_pilot_split,
    parse_ilsvrc_xml_boxes,
    save_imagenet_box_overlays,
)
from ralr import readout as R
from scripts import run_pilot_ralr as ralr_runner


# --------------------------------------------------------------------------- eval/metrics.py hand-computed tests (§9)
def test_hand_computed_box_auc_pxap_pointing():
    # 2x2 toy saliency map: scores [0.9, 0.7, 0.5, 0.1], labels [1, 0, 1, 0]
    smap = np.array([[0.9, 0.7], [0.5, 0.1]], dtype=np.float64)
    gt = np.array([[1, 0], [1, 0]], dtype=bool)

    # Hand-computed ROC AUC: 3 of 4 (pos, neg) pairs correctly ordered -> 3/4 = 0.75
    assert abs(M.box_auc(smap, gt) - 0.75) < 1e-12
    assert abs(M.box_auc(-smap, gt) - 0.25) < 1e-12

    # Hand-computed PxAP: sorted y = [1, 0, 1, 0]; recall steps at idx 0 (P=1/1, dR=0.5) & idx 2 (P=2/3, dR=0.5)
    # AP = 0.5 * 1.0 + 0.5 * (2/3) = 5/6
    assert abs(M.pxap(smap, gt) - (5.0 / 6.0)) < 1e-12

    # Tie-handling check: all equal scores -> AUC = 0.5, PxAP = prevalence = 0.5
    flat = np.ones((2, 2), dtype=np.float64)
    assert abs(M.box_auc(flat, gt) - 0.5) < 1e-12
    assert abs(M.pxap(flat, gt) - 0.5) < 1e-12

    # Pointing game: argmax is (0, 0) -> inside gt (1.0); inverted argmax is (1, 1) -> outside (0.0)
    assert M.pointing_game(smap, gt) == 1.0
    assert M.pointing_game(-smap, gt) == 0.0


def test_largest_cc_otsu_multi_object_corloc_and_maxboxacc():
    # 20x20 map: small 2x2 noise blob at top-left, large 8x8 object at [10:18, 10:18]
    smap = np.zeros((20, 20), dtype=np.float64)
    smap[0:2, 0:2] = 1.0
    smap[10:18, 10:18] = 1.0

    # Otsu + largest CC MUST select the 8x8 object [10, 10, 18, 18], NOT the union [0, 0, 18, 18]
    box, mask, _ = M.otsu_box(smap)
    assert box == (10.0, 10.0, 18.0, 18.0), f"Expected largest CC box (10,10,18,18), got {box}"
    assert abs(M.bbox_iou(box, (10.0, 10.0, 18.0, 18.0)) - 1.0) < 1e-12

    # Multi-instance CorLoc: hit if predicted box matches ANY GT box (second box here!)
    gt_multi = [(0.0, 0.0, 4.0, 4.0), (10.0, 10.0, 18.0, 18.0)]
    cl, hits = M.corloc([box], [gt_multi], iou_thresh=0.5)
    assert cl == 1.0 and hits[0] == 1.0

    # Murtaza-style pseudo-box calibration uses zero GT boxes and finds a valid threshold in (0, 1)
    tau_cal = M.calibrate_pseudo_box_threshold([smap, smap])
    assert 0.0 < tau_cal < 1.0
    assert M.fixed_threshold_box(smap, tau_cal) == (10.0, 10.0, 18.0, 18.0)

    # MaxBoxAccV2: perfect 1.0 across all deltas {0.3, 0.5, 0.7} and explicitly flags tunes_on_boxes=True
    mba = M.maxboxacc_v2([smap, smap], [gt_multi, gt_multi])
    assert mba["tunes_on_boxes"] is True
    assert abs(mba["maxboxacc_v2"] - 1.0) < 1e-12


# --------------------------------------------------------------------------- RALR unit & seeded-bug tests (§7)
def test_ralr_layer_selection_rejects_degenerate_entropy_and_freezes_branch():
    # Layer 11 has highest KD (0.95) BUT entropy=0.94 > 0.90 -> MUST be rejected!
    stats = {
        8: {"kd": 0.40, "qii_s4": 0.80, "entropy": 0.72},
        9: {"kd": 0.65, "qii_s4": 0.42, "entropy": 0.78},
        10: {"kd": 0.55, "qii_s4": 0.90, "entropy": 0.81},
        11: {"kd": 0.95, "qii_s4": 0.99, "entropy": 0.94},
    }
    sel = R.select_layer_and_branch(stats, tau_q=0.5, entropy_ceiling=0.90)
    assert sel["l_star"] == 9, f"Failed to reject degenerate layer 11: {sel}"
    assert sel["branch"] == "LOW"  # QII_S4(9) = 0.42 < 0.50
    assert sel["all_layers_degenerate"] is False

    # If all layers exceed entropy ceiling (e.g. MAE), flag all_layers_degenerate=True cleanly
    all_deg = {l: {**v, "entropy": 0.95} for l, v in stats.items()}
    sel_deg = R.select_layer_and_branch(all_deg, tau_q=0.5, entropy_ceiling=0.90)
    assert sel_deg["all_layers_degenerate"] is True
    assert sel_deg["l_star"] == 11
    assert sel_deg["branch"] == "HIGH"


def test_ralr_step7_shortcut_detection_and_knockout():
    rng = np.random.default_rng(0)
    N, H, d_h = 196, 6, 32
    # 150 background patches cluster around e0 (low CLS attn), 46 FG patches around e1 (high CLS attn)
    k_concat = np.zeros((N, H * d_h), dtype=np.float64)
    k_concat[:150, 0] = 5.0
    k_concat[150:, 1] = 5.0
    k_concat += 0.1 * rng.standard_normal((N, H * d_h))

    cls_attn = np.full((H, N), 0.002, dtype=np.float64)
    cls_attn[:, 150:] = 0.015
    # Inject 2 high-norm OUT tokens at indices 0, 1 (in dominant background cluster)
    patch_norms = np.full(N, 10.0, dtype=np.float64) + 0.1 * rng.standard_normal(N)
    patch_norms[[0, 1]] = 200.0
    # Inject 3 high-contrib background shortcut tokens at indices 2, 3, 4 (dominant background cluster -> high keyaff)
    # and 3 high-contrib foreground tokens at 150, 151, 152 (non-dominant cluster -> low keyaff)
    patch_contrib = np.full(N, 1.0, dtype=np.float64) + 0.05 * rng.standard_normal(N)
    patch_contrib[[2, 3, 4]] = 50.0
    patch_contrib[[150, 151, 152]] = 40.0
    cls_attn[:, [0, 1, 2, 3, 4]] = 0.08
    cls_attn /= cls_attn.sum(axis=-1, keepdims=True)

    # Default pilot-fixed bg_high rule: catches background shortcuts (2, 3, 4) and does NOT drop foreground (150..152)
    det = R.detect_shortcut_tokens(patch_norms, patch_contrib, k_concat, cls_patch_attn=cls_attn.mean(axis=0), seed=0)
    assert det["largest_is_lower_attn"] is True
    assert det["out_norm_mask"][0] and det["out_norm_mask"][1]
    assert det["out_shortcut_mask"][[2, 3, 4]].any()
    assert not det["out_shortcut_mask"][[150, 151, 152]].any()

    # Literal guide rule (ablation): catches low-keyaff tokens (150..152)
    det_lit = R.detect_shortcut_tokens(
        patch_norms, patch_contrib, k_concat, cls_patch_attn=cls_attn.mean(axis=0), seed=0, shortcut_rule="guide_literal"
    )
    assert det_lit["out_shortcut_mask"][[150, 151, 152]].all()

    # Full single-image readout check (including cls_l_star_clip98 & cls_l_star_no_out controls)
    k_patch = k_concat.reshape(N, H, d_h).transpose(1, 0, 2)
    key_eff = rng.standard_normal((H, N))
    w_h = np.ones(H, dtype=np.float64) / H
    maps_hi, diag = R.compute_ralr_maps_single_image(
        cls_attn_last=cls_attn, cls_attn_lstar=cls_attn, head_weights_lstar=w_h,
        key_effect_lstar=key_eff, patch_norms_lstar=patch_norms, patch_contrib_lstar=patch_contrib,
        k_patch_lstar=k_patch, branch="HIGH", grid=(14, 14), target_size=56, seed=0,
    )
    maps_lo, _ = R.compute_ralr_maps_single_image(
        cls_attn_last=cls_attn, cls_attn_lstar=cls_attn, head_weights_lstar=w_h,
        key_effect_lstar=key_eff, patch_norms_lstar=patch_norms, patch_contrib_lstar=patch_contrib,
        k_patch_lstar=k_patch, branch="LOW", grid=(14, 14), target_size=56, seed=0,
    )
    assert "cls_l_star_clip98" in maps_hi and "cls_l_star_no_out" in maps_hi
    assert "lost_l_star" in maps_hi and "tokencut_l_star" in maps_hi
    assert "lost_l_star" in ralr_runner.METHODS_ORDER and "tokencut_l_star" in ralr_runner.METHODS_ORDER
    assert np.allclose(maps_hi["ralr_adaptive"], maps_hi["ralr_high"])
    assert np.allclose(maps_lo["ralr_adaptive"], maps_lo["ralr_low"])
    assert not np.allclose(maps_lo["ralr_low"], maps_lo["cls_l_star_w"])
    for mname, m in maps_hi.items():
        assert m.shape == (56, 56), f"{mname} wrong shape {m.shape}"
        assert m.min() >= -1e-9 and m.max() <= 1.0 + 1e-9
    assert diag["n_shortcut"] >= 2


def test_lost_and_tokencut_recover_synthetic_fg_object():
    """Failing-if-broken test: LOST and TokenCut must recover a 4x4 central FG object on a 14x14 grid."""
    rng = np.random.default_rng(42)
    gh, gw = 14, 14
    N, D = gh * gw, 64
    gt = np.zeros((gh, gw), dtype=bool)
    gt[5:9, 5:9] = True  # 16 FG patches vs 180 BG patches (all 4 corners are BG)
    fg_flat = gt.reshape(-1)

    # Distinct orthogonal clusters for BG (around e0) and FG (around e1)
    k_concat = 0.05 * rng.standard_normal((N, D))
    k_concat[~fg_flat, 0] += 3.0
    k_concat[fg_flat, 1] += 3.0

    lost_1d = R.lost_map_from_keys(k_concat, k_patches=50)
    tc_1d = R.tokencut_map_from_keys(k_concat, (gh, gw), tau=0.2)

    assert lost_1d.shape == (N,) and tc_1d.shape == (N,)
    assert M.box_auc(lost_1d.reshape(gh, gw), gt) > 0.99, "LOST failed to localize synthetic FG object"
    assert M.box_auc(tc_1d.reshape(gh, gw), gt) > 0.99, "TokenCut failed to localize synthetic FG object"
    # Verify TokenCut sign orientation puts FG > BG (not inverted!)
    assert tc_1d[fg_flat].mean() > tc_1d[~fg_flat].mean()
    # Verify upsampled maps are normalized in [0, 1]
    for arr in (lost_1d, tc_1d):
        up = R.upsample_patch_map(arr, (gh, gw), 56)
        assert up.shape == (56, 56) and up.min() >= -1e-9 and up.max() <= 1.0 + 1e-9


# --------------------------------------------------------------------------- ImageNet XML + RALR CLI smoke (§8)
def _write_synthetic_imagenet(root: Path, n_images: int = 8) -> tuple[Path, Path]:
    img_dir = root / "val" / "n01440764"
    xml_dir = root / "val_xml"
    img_dir.mkdir(parents=True)
    xml_dir.mkdir(parents=True)
    rng = np.random.default_rng(123)
    for i in range(n_images):
        stem = f"ILSVRC2012_val_{i + 1:08d}"
        arr = rng.integers(0, 255, size=(120, 160, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / f"{stem}.JPEG")
        xml = f"""<annotation>
  <size><width>160</width><height>120</height><depth>3</depth></size>
  <object><name>n01440764</name><bndbox><xmin>20</xmin><ymin>10</ymin><xmax>80</xmax><ymax>90</ymax></bndbox></object>
  <object><name>n01440764</name><bndbox><xmin>90</xmin><ymin>30</ymin><xmax>140</xmax><ymax>100</ymax></bndbox></object>
</annotation>"""
        (xml_dir / f"{stem}.xml").write_text(xml)
    return root / "val", xml_dir


def test_imagenet_xml_loader_multi_object_and_ralr_cli_smoke(tmp_path: Path):
    img_dir, xml_dir = _write_synthetic_imagenet(tmp_path, n_images=8)
    split_file = tmp_path / "in_split.txt"

    samples = load_imagenet_val_pilot_split(img_dir, xml_dir, n_images=6, seed=0, split_file=split_file)
    assert len(samples) == 6
    assert split_file.exists()
    assert len(samples[0].boxes_0based) == 2

    # Reloading from existing split_file must return identical stems verbatim (nested subfolder fallback)
    samples2 = load_imagenet_val_pilot_split(img_dir, xml_dir, n_images=2, seed=99, split_file=split_file)
    assert [s.image_id for s in samples2] == [s.image_id for s in samples]

    # Flat layout + existing split_file MUST use O(N) direct fast path without calling Path.rglob
    flat_img = tmp_path / "flat_val"
    flat_img.mkdir()
    for s in samples:
        (flat_img / f"{s.image_id}.JPEG").write_bytes(s.image_path.read_bytes())
    orig_rglob = Path.rglob
    def _forbid_rglob(self, pattern):
        raise AssertionError("Flat split_file fast path must NOT call Path.rglob!")
    Path.rglob = _forbid_rglob  # type: ignore[assignment]
    try:
        samples_fast = load_imagenet_val_pilot_split(flat_img, xml_dir, n_images=6, seed=0, split_file=split_file)
    finally:
        Path.rglob = orig_rglob  # type: ignore[assignment]
    assert [s.image_id for s in samples_fast] == [s.image_id for s in samples]
    assert all(len(s.boxes_0based) == 2 for s in samples_fast)

    # Union GT mask covers both boxes
    mask, tboxes = get_transformed_imagenet_gt_mask(samples[0], 160, 120, 224)
    assert len(tboxes) == 2
    assert mask.shape == (224, 224) and mask.sum() > 0

    ov = save_imagenet_box_overlays(samples, 224, tmp_path / "ov.png", n_images=4)
    assert ov.exists()

    # End-to-end RALR CLI smoke test on synthetic ImageNet
    out_dir = tmp_path / "ralr_out"
    ralr_runner.main([
        "--dataset", "imagenet",
        "--images_dir", str(img_dir),
        "--imagenet_xml_dir", str(xml_dir),
        "--split_file", str(split_file),
        "--out_dir", str(out_dir),
        "--models", "dino_vits16,dinov2_vitb14_reg",
        "--batch_size", "4",
        "--smoke_random",
    ])
    assert (out_dir / "summary_ralr.md").exists()
    assert (out_dir / "ralr_report.json").exists()
    assert (out_dir / "meta.json").exists()
    assert (out_dir / "imagenet_box_overlay_224.png").exists()
    meta = json.loads((out_dir / "meta.json").read_text())
    assert meta["git_sha"] is not None and len(meta["git_sha"]) >= 7
