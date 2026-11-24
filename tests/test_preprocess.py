"""Unit tests for box geometry, 1-based->0-based shifts, clipping, and Resize+CenterCrop (Guide §8.2, §9, Review (3)).

Explicitly tests the three hand-computed toy cases Dr. Zia requested in Review (3):
  1. Off-by-one shift (VOC 1-based XML & CUB 1-based bounding_boxes.txt -> 0-based).
  2. Clipping to the ORIGINAL image frame versus a patch-padded frame (and catches the padded-clip bug).
  3. Box transformation under Resize(shorter_side) + CenterCrop, verified BOTH analytically by hand
     AND directly against rasterizing a bright white rectangle on a PIL image and running torchvision's
     actual Resize + CenterCrop pipeline.
"""
from __future__ import annotations

import numpy as np
import pytest
import torchvision.transforms as T
from PIL import Image, ImageDraw

from eval.preprocess import (
    box_to_pixel_mask,
    clip_box_to_frame,
    cub_box_1based_to_0based,
    load_cub_pilot_split,
    load_imagenet_val_pilot_split,
    patch_fg_mask,
    save_box_overlays,
    transform_box_resize_center_crop,
    voc_box_1based_to_0based,
    CUBSample,
)


def test_1_off_by_one_shift_voc_and_cub():
    # VOC 1-based inclusive: top-left pixel (1, 1) to bottom-right pixel (10, 20) covers 10x20 pixels
    # 0-based half-open must be [0, 0, 10, 20) -> width=10, height=20
    x1, y1, x2, y2 = voc_box_1based_to_0based(1, 1, 10, 20)
    assert (x1, y1, x2, y2) == (0.0, 0.0, 10.0, 20.0)
    assert (x2 - x1, y2 - y1) == (10.0, 20.0)

    # Interior VOC box [5, 8, 15, 28] -> 0-based [4, 7, 15, 28) -> width=11, height=21
    assert voc_box_1based_to_0based(5, 8, 15, 28) == (4.0, 7.0, 15.0, 28.0)

    # CUB 1-based [x=1, y=1, w=10, h=20] -> 0-based [0, 0, 10, 20)
    assert cub_box_1based_to_0based(1.0, 1.0, 10.0, 20.0) == (0.0, 0.0, 10.0, 20.0)
    assert cub_box_1based_to_0based(60.0, 27.0, 325.0, 304.0) == (59.0, 26.0, 384.0, 330.0)


def test_2_clip_to_original_frame_vs_padded_frame():
    # Image is 500x375; padded to multiple of 16 -> 512x384.
    # A predicted box extending into the padding [400, 300, 512, 384] MUST clip to (500, 375), NOT (512, 384)!
    orig_w, orig_h = 500, 375
    pad_w, pad_h = 512, 384
    pred = (-5.0, -2.0, 512.0, 384.0)

    clipped = clip_box_to_frame(pred, orig_w, orig_h)
    assert clipped == (0.0, 0.0, 500.0, 375.0)

    # Seeded bug: clipping to padded frame leaks 12px horizontally and 9px vertically past real image
    bad_clipped = clip_box_to_frame(pred, pad_w, pad_h)
    assert bad_clipped != clipped
    assert bad_clipped[2] - clipped[2] == 12.0
    assert bad_clipped[3] - clipped[3] == 9.0


@pytest.mark.parametrize(
    "orig_w,orig_h,target,box,expected",
    [
        # Landscape 400x200 -> Resize(200) keeps 400x200 -> CenterCrop(200) crops x in [100, 300)
        (400, 200, 200, (120, 40, 280, 160), (20.0, 40.0, 180.0, 160.0)),
        # Landscape box partially outside left/right crop boundary -> clipped to [0, 200]
        (400, 200, 200, (50, 20, 350, 180), (0.0, 20.0, 200.0, 180.0)),
        # Portrait 200x400 -> Resize(100) gives 100x200 (scale=0.5) -> CenterCrop(100) crops y in [50, 150)
        (200, 400, 100, (40, 120, 160, 280), (20.0, 10.0, 80.0, 90.0)),
        # Square 448x448 -> Resize(224) gives 224x224 (scale=0.5), crop=(0,0)
        (448, 448, 224, (100, 200, 300, 400), (50.0, 100.0, 150.0, 200.0)),
        # /14 protocol: 392x196 -> Resize(196) keeps 392x196 -> CenterCrop(196) crops x in [98, 294)
        (392, 196, 196, (108, 10, 284, 186), (10.0, 10.0, 186.0, 186.0)),
    ],
)
def test_3_box_transform_resize_center_crop_hand_computed(orig_w, orig_h, target, box, expected):
    got = transform_box_resize_center_crop(box, orig_w, orig_h, target)
    np.testing.assert_allclose(got, expected, atol=1e-6)


@pytest.mark.parametrize("orig_w,orig_h,target", [(500, 333, 224), (333, 500, 224), (500, 375, 196), (311, 417, 196)])
def test_3b_box_transform_matches_actual_torchvision_pipeline(orig_w, orig_h, target):
    """Cross-check our box math against torchvision Resize + CenterCrop on a real rasterized rectangle."""
    box = (int(0.25 * orig_w), int(0.2 * orig_h), int(0.75 * orig_w), int(0.8 * orig_h))
    img = Image.new("L", (orig_w, orig_h), 0)
    # PIL rectangle inclusive endpoints -> [x1, y1, x2-1, y2-1]
    ImageDraw.Draw(img).rectangle([box[0], box[1], box[2] - 1, box[3] - 1], fill=255)

    tf = T.Compose([T.Resize(target, interpolation=T.InterpolationMode.NEAREST), T.CenterCrop(target)])
    arr = np.asarray(tf(img))
    ys, xs = np.nonzero(arr > 127)
    tv_box = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))

    got = transform_box_resize_center_crop(box, orig_w, orig_h, target)
    # Within <= 1.0 pixel of nearest-neighbor rasterization boundary
    np.testing.assert_allclose(got, tv_box, atol=1.0)


def test_patch_fg_mask_50pct_rule_and_overlay_grid(tmp_path):
    # 224x224 mask, 14x14 grid -> 16x16 pixels per patch
    # Fill top-left patch (0,0) by 100%, patch (0,1) by 50% (8 cols), patch (0,2) by 43.75% (7 cols)
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[0:16, 0:16] = 1
    mask[0:16, 16:24] = 1
    mask[0:16, 32:39] = 1
    fg = patch_fg_mask(mask, (14, 14), threshold=0.5)
    assert fg[0, 0] == 1
    assert fg[0, 1] == 1
    assert fg[0, 2] == 0
    assert fg.sum() == 2

    # 20-image overlay smoke test
    samples = []
    for i in range(4):
        p = tmp_path / f"im_{i}.jpg"
        Image.fromarray(np.full((240, 320, 3), 100 + i * 20, dtype=np.uint8)).save(p)
        samples.append(CUBSample(cub_id=i + 1, rel_path=p.name, image_path=p, box_0based=(80, 40, 240, 200), seg_path=None))
    out = save_box_overlays(samples, 224, tmp_path / "overlays.png", n_images=4)
    assert out.exists()
    assert box_to_pixel_mask((10, 10, 20, 20), 224).sum() == 100


def test_4_image_id_keying_ignores_directory_sort_order(tmp_path):
    """Review (2) & (3): verify explicit image-ID pairing replaces positional 'sorted filenames match'."""
    # --- ImageNet-1K explicit stem lookup (with distractor images + unsorted split order) ---
    img_dir = tmp_path / "in1k_val"
    xml_dir = tmp_path / "in1k_xml"
    img_dir.mkdir()
    xml_dir.mkdir()

    # Distractor file that sorts BEFORE the split stems — would shift every positional index!
    Image.new("RGB", (100, 100), (0, 0, 0)).save(img_dir / "ILSVRC2012_val_00000001.JPEG")

    specs = [
        ("ILSVRC2012_val_00000050", (10, 20, 60, 80)),
        ("ILSVRC2012_val_00000012", (5, 15, 45, 55)),
        ("ILSVRC2012_val_00000099", (30, 40, 90, 95)),
    ]
    for stem, (x1, y1, x2, y2) in specs:
        Image.new("RGB", (100, 100), (128, 128, 128)).save(img_dir / f"{stem}.JPEG")
        (xml_dir / f"{stem}.xml").write_text(
            f"<annotation><size><width>100</width><height>100</height></size>"
            f"<object><bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin><xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox></object></annotation>"
        )

    split_file = tmp_path / "split.txt"
    split_file.write_text("\n".join(s for s, _ in specs) + "\n")
    loaded = load_imagenet_val_pilot_split(img_dir, xml_dir, split_file=split_file)
    assert [(s.image_id, s.boxes_0based[0]) for s in loaded] == [(st, tuple(float(v) for v in b)) for st, b in specs]

    # Missing split ID must raise FileNotFoundError rather than silently aligning another image
    bad_split = tmp_path / "bad_split.txt"
    bad_split.write_text("ILSVRC2012_val_00000050\nILSVRC2012_val_99999999\n")
    with pytest.raises(FileNotFoundError):
        load_imagenet_val_pilot_split(img_dir, xml_dir, split_file=bad_split)

    # --- CUB-200-2011 explicit cub_id lookup (non-monotonic IDs + out-of-order files) ---
    cub_root = tmp_path / "CUB_200_2011"
    (cub_root / "images" / "cls").mkdir(parents=True)
    cub_specs = [(42, "cls/z_img.jpg", (11.0, 21.0, 30.0, 40.0)), (7, "cls/a_img.jpg", (5.0, 6.0, 20.0, 25.0))]
    for cid, rel, _ in cub_specs:
        Image.new("RGB", (100, 100), (50, 50, 50)).save(cub_root / "images" / rel)
    (cub_root / "images.txt").write_text("\n".join(f"{cid} {rel}" for cid, rel, _ in cub_specs) + "\n")
    (cub_root / "train_test_split.txt").write_text("\n".join(f"{cid} 0" for cid, _, _ in cub_specs) + "\n")
    (cub_root / "bounding_boxes.txt").write_text(
        "\n".join(f"{cid} {b[0]} {b[1]} {b[2]} {b[3]}" for cid, _, b in cub_specs) + "\n"
    )
    cub_split = tmp_path / "cub_split.txt"
    cub_split.write_text("42\n7\n")
    cub_loaded = load_cub_pilot_split(cub_root, split_file=cub_split)
    assert [s.cub_id for s in cub_loaded] == [42, 7]
    assert cub_loaded[0].box_0based == (10.0, 20.0, 40.0, 60.0)
    assert cub_loaded[1].box_0based == (4.0, 5.0, 24.0, 30.0)
