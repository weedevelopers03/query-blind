"""Dataset preprocessing, box/mask geometry, and CUB-200-2011 pilot split (Guide §§1.5, 8.2-8.4, Review (3)).

Enforced conventions:
  * 1-based -> 0-based conversion for VOC XML (xmin-1, ymin-1, xmax, ymax) and CUB
    bounding_boxes.txt (x-1, y-1, x-1+w, y-1+h).
  * Box clipping uses the ORIGINAL image frame (W, H), never a patch-padded frame.
  * 14x14 primary protocol (§1.5): target_size = 224 for patch-16 models, 196 for patch-14 models.
  * Images and GT boxes/masks undergo the EXACT same Resize(target_size) + CenterCrop(target_size)
    transformation, matching torchvision.transforms.Resize + CenterCrop pixel-for-pixel.
  * Normalization constants are read from the checkpoint config (timm / open_clip), never hardcoded.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw


# --------------------------------------------------------------------------- Box geometry (Review (3))
def voc_box_1based_to_0based(xmin: float, ymin: float, xmax: float, ymax: float) -> tuple[float, float, float, float]:
    """Convert PASCAL VOC 1-based inclusive [xmin, ymin, xmax, ymax] to 0-based half-open [x1, y1, x2, y2)."""
    return (float(xmin) - 1.0, float(ymin) - 1.0, float(xmax), float(ymax))


def cub_box_1based_to_0based(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
    """Convert CUB-200-2011 1-based [x, y, width, height] to 0-based half-open [x1, y1, x2, y2)."""
    x1 = float(x) - 1.0
    y1 = float(y) - 1.0
    return (x1, y1, x1 + float(w), y1 + float(h))


def clip_box_to_frame(
    box: Sequence[float],
    orig_w: int,
    orig_h: int,
) -> tuple[float, float, float, float]:
    """Clip a 0-based [x1, y1, x2, y2] box to the ORIGINAL image frame [0, orig_w] x [0, orig_h]."""
    x1, y1, x2, y2 = (float(v) for v in box)
    x1 = max(0.0, min(float(orig_w), x1))
    y1 = max(0.0, min(float(orig_h), y1))
    x2 = max(x1, min(float(orig_w), x2))
    y2 = max(y1, min(float(orig_h), y2))
    return (x1, y1, x2, y2)


def _resize_shorter_side_size(orig_w: int, orig_h: int, target_size: int) -> tuple[int, int]:
    """Exact output (new_w, new_h) of torchvision.transforms.Resize(target_size) on a PIL image."""
    if orig_w <= orig_h:
        new_w = target_size
        new_h = int(target_size * orig_h / orig_w)
    else:
        new_h = target_size
        new_w = int(target_size * orig_w / orig_h)
    return new_w, new_h


def transform_box_resize_center_crop(
    box: Sequence[float],
    orig_w: int,
    orig_h: int,
    target_size: int,
) -> tuple[float, float, float, float]:
    """Transform a 0-based [x1, y1, x2, y2] box through Resize(target_size) + CenterCrop(target_size).

    Matches torchvision.transforms.functional.resize + center_crop exactly:
      - Shorter side resized to target_size (integer truncation per torchvision F.resize).
      - Center crop top-left offset: crop_x = int(round((new_w - target_size) / 2.0)),
        crop_y = int(round((new_h - target_size) / 2.0)).
    """
    x1, y1, x2, y2 = clip_box_to_frame(box, orig_w, orig_h)
    new_w, new_h = _resize_shorter_side_size(orig_w, orig_h, target_size)
    sx = new_w / float(orig_w)
    sy = new_h / float(orig_h)
    crop_x = int(round((new_w - target_size) / 2.0))
    crop_y = int(round((new_h - target_size) / 2.0))

    tx1 = max(0.0, min(float(target_size), x1 * sx - crop_x))
    ty1 = max(0.0, min(float(target_size), y1 * sy - crop_y))
    tx2 = max(tx1, min(float(target_size), x2 * sx - crop_x))
    ty2 = max(ty1, min(float(target_size), y2 * sy - crop_y))
    return (tx1, ty1, tx2, ty2)


def transform_mask_resize_center_crop(mask_pil: Image.Image, target_size: int) -> np.ndarray:
    """Apply identical Resize(target_size, NEAREST) + CenterCrop(target_size) to a GT segmentation mask."""
    tf = T.Compose([
        T.Resize(target_size, interpolation=T.InterpolationMode.NEAREST),
        T.CenterCrop(target_size),
    ])
    arr = np.asarray(tf(mask_pil.convert("L")), dtype=np.uint8)
    return (arr >= 128).astype(np.uint8)


def box_to_pixel_mask(box_xyxy: Sequence[float], target_size: int) -> np.ndarray:
    """Rasterize a transformed 0-based [x1, y1, x2, y2) box into a binary (target_size, target_size) mask."""
    mask = np.zeros((target_size, target_size), dtype=np.uint8)
    x1, y1, x2, y2 = box_xyxy
    ix1 = max(0, min(target_size, int(np.floor(x1))))
    iy1 = max(0, min(target_size, int(np.floor(y1))))
    ix2 = max(ix1, min(target_size, int(np.ceil(x2))))
    iy2 = max(iy1, min(target_size, int(np.ceil(y2))))
    if ix2 > ix1 and iy2 > iy1:
        mask[iy1:iy2, ix1:ix2] = 1
    return mask


def patch_fg_mask(pixel_mask: np.ndarray, grid: tuple[int, int], threshold: float = 0.5) -> np.ndarray:
    """Guide §8.3: patch is FG iff >= 50% of its area lies inside the GT mask/box."""
    h, w = pixel_mask.shape[:2]
    gh, gw = grid
    if h % gh != 0 or w % gw != 0:
        raise ValueError(f"mask shape {(h, w)} not divisible by grid {grid}")
    ph, pw = h // gh, w // gw
    blocks = (pixel_mask > 0).astype(np.float32).reshape(gh, ph, gw, pw)
    frac = blocks.mean(axis=(1, 3))  # (gh, gw)
    return (frac >= threshold).astype(np.uint8)


# --------------------------------------------------------------------------- Image transform & norm
def target_size_for_patch_size(patch_size: int) -> int:
    """Guide §1.5 primary 14x14 protocol: 224 px for /16, 196 px for /14."""
    return 14 * int(patch_size)


def get_model_norm_stats(model, family: str, openclip_preprocess=None) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Read per-model normalization (mean, std) from checkpoint config — never hardcoded (§8.2)."""
    if family == "open_clip_vit" and openclip_preprocess is not None:
        for t in getattr(openclip_preprocess, "transforms", []):
            if isinstance(t, T.Normalize):
                return tuple(float(v) for v in t.mean), tuple(float(v) for v in t.std)
    if family in ("timm_vit", "map_head"):
        import timm

        cfg = timm.data.resolve_data_config({}, model=model)
        return tuple(float(v) for v in cfg["mean"]), tuple(float(v) for v in cfg["std"])
    if family == "open_clip_vit":
        # OpenCLIP's own default constants module
        from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

        return tuple(float(v) for v in OPENAI_DATASET_MEAN), tuple(float(v) for v in OPENAI_DATASET_STD)
    raise ValueError(f"Cannot resolve normalization constants for family={family!r}")


def build_image_transform(target_size: int, mean: Sequence[float], std: Sequence[float]) -> T.Compose:
    """Guide §8.2: Resize shorter side -> CenterCrop -> ToTensor -> per-model Normalize (no TTA)."""
    return T.Compose([
        T.Resize(target_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=list(mean), std=list(std)),
    ])


# --------------------------------------------------------------------------- CUB-200-2011 loader (§8.4)
@dataclass
class CUBSample:
    cub_id: int
    rel_path: str
    image_path: Path
    box_0based: tuple[float, float, float, float]   # 0-based [x1, y1, x2, y2) in original image coords
    seg_path: Path | None                           # optional official segmentation mask if present


def _resolve_cub_root(cub_dir: str | Path) -> Path:
    p = Path(cub_dir)
    if (p / "images.txt").exists():
        return p
    if (p / "CUB_200_2011" / "images.txt").exists():
        return p / "CUB_200_2011"
    raise FileNotFoundError(f"Could not find CUB images.txt under {p} or {p / 'CUB_200_2011'}")


def load_cub_pilot_split(
    cub_dir: str | Path,
    *,
    n_images: int = 500,
    seed: int = 0,
    split_file: str | Path | None = None,
    seg_dir: str | Path | None = None,
) -> list[CUBSample]:
    """Load the Guide §8.4 CUB-200-2011 test pilot split (500 test images, seed 0).

    If split_file exists, reads those exact CUB image IDs verbatim (Guide §8.1: never regenerate).
    Otherwise draws n_images from the 5,794 official test images with seed=0 and writes split_file.
    """
    root = _resolve_cub_root(cub_dir)

    id_to_rel: dict[int, str] = {}
    for line in (root / "images.txt").read_text().splitlines():
        if line.strip():
            cid, rel = line.strip().split(None, 1)
            id_to_rel[int(cid)] = rel

    test_ids: list[int] = []
    for line in (root / "train_test_split.txt").read_text().splitlines():
        if line.strip():
            cid, is_train = line.strip().split()
            if int(is_train) == 0:
                test_ids.append(int(cid))
    test_ids.sort()

    id_to_box: dict[int, tuple[float, float, float, float]] = {}
    for line in (root / "bounding_boxes.txt").read_text().splitlines():
        if line.strip():
            cid, x, y, w, h = line.strip().split()
            id_to_box[int(cid)] = cub_box_1based_to_0based(float(x), float(y), float(w), float(h))

    # Optional segmentation directory (CUB segmentations live either in root/segmentations or sibling)
    seg_root: Path | None = None
    for cand in (
        Path(seg_dir) if seg_dir else None,
        root / "segmentations",
        root.parent / "segmentations",
    ):
        if cand is not None and cand.is_dir():
            seg_root = cand
            break

    split_path = Path(split_file) if split_file is not None else None
    if split_path is not None and split_path.exists():
        chosen_ids = [int(x.strip()) for x in split_path.read_text().splitlines() if x.strip()]
    else:
        rng = np.random.default_rng(int(seed))
        k = min(int(n_images), len(test_ids))
        chosen_ids = sorted(int(v) for v in rng.choice(test_ids, size=k, replace=False))
        if split_path is not None:
            split_path.parent.mkdir(parents=True, exist_ok=True)
            split_path.write_text("\n".join(str(i) for i in chosen_ids) + "\n")

    samples: list[CUBSample] = []
    for cid in chosen_ids:
        rel = id_to_rel[cid]
        img_p = root / "images" / rel
        seg_p = None
        if seg_root is not None:
            cand_seg = (seg_root / rel).with_suffix(".png")
            if cand_seg.exists():
                seg_p = cand_seg
        samples.append(CUBSample(cub_id=cid, rel_path=rel, image_path=img_p, box_0based=id_to_box[cid], seg_path=seg_p))
    return samples


def get_transformed_gt_mask(sample: CUBSample, orig_w: int, orig_h: int, target_size: int, prefer_seg: bool = True) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Return (binary_mask (S, S), transformed_box_xyxy) for a CUBSample under Resize+CenterCrop."""
    tbox = transform_box_resize_center_crop(sample.box_0based, orig_w, orig_h, target_size)
    if prefer_seg and sample.seg_path is not None and sample.seg_path.exists():
        mask = transform_mask_resize_center_crop(Image.open(sample.seg_path), target_size)
        if mask.sum() > 0:
            return mask, tbox
    return box_to_pixel_mask(tbox, target_size), tbox


# --------------------------------------------------------------------------- 20-image overlay check (§8.2)
def save_box_overlays(
    samples: Sequence[CUBSample],
    target_size: int,
    out_path: str | Path,
    n_images: int = 20,
) -> Path:
    """Guide §8.2 & Review (3): save a 4x5 visual spot-check grid of transformed boxes on transformed images."""
    chosen = list(samples)[:n_images]
    cols = 5
    rows = max(1, int(np.ceil(len(chosen) / cols)))
    pad = 4
    canvas = Image.new("RGB", (cols * (target_size + pad) + pad, rows * (target_size + pad) + pad), (30, 30, 30))
    vis_tf = T.Compose([T.Resize(target_size, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(target_size)])

    for idx, s in enumerate(chosen):
        img = Image.open(s.image_path).convert("RGB")
        ow, oh = img.size
        cropped = vis_tf(img)
        tbox = transform_box_resize_center_crop(s.box_0based, ow, oh, target_size)
        draw = ImageDraw.Draw(cropped)
        draw.rectangle([tbox[0], tbox[1], tbox[2], tbox[3]], outline=(255, 40, 40), width=2)
        draw.text((4, 4), f"id={s.cub_id}", fill=(255, 255, 0))
        r, c = divmod(idx, cols)
        canvas.paste(cropped, (pad + c * (target_size + pad), pad + r * (target_size + pad)))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


# --------------------------------------------------------------------------- ImageNet-1K val + ILSVRC XML (§8.1, §8.4)
@dataclass(frozen=True)
class ImageNetSample:
    image_id: str  # e.g. "ILSVRC2012_val_00000001"
    image_path: Path
    xml_path: Path | None
    boxes_0based: tuple[tuple[float, float, float, float], ...]  # multi-object support (§8.3)


def parse_ilsvrc_xml_boxes(xml_path: str | Path) -> list[tuple[float, float, float, float]]:
    """Parse all <object><bndbox> annotations from an ILSVRC2012 validation XML file.

    ILSVRC2012 val XMLs use 0-based coordinates (xmin >= 0, ymin >= 0). Clamps to [0, W] x [0, H]
    when <size> is present and drops degenerate zero-area boxes.
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    root = tree.getroot()
    size_el = root.find("size")
    w_lim = float(size_el.findtext("width")) if (size_el is not None and size_el.findtext("width")) else None
    h_lim = float(size_el.findtext("height")) if (size_el is not None and size_el.findtext("height")) else None

    boxes: list[tuple[float, float, float, float]] = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        x1 = float(bb.findtext("xmin", "0"))
        y1 = float(bb.findtext("ymin", "0"))
        x2 = float(bb.findtext("xmax", "0"))
        y2 = float(bb.findtext("ymax", "0"))
        if w_lim is not None:
            x1 = max(0.0, min(x1, w_lim))
            x2 = max(0.0, min(x2, w_lim))
        if h_lim is not None:
            y1 = max(0.0, min(y1, h_lim))
            y2 = max(0.0, min(y2, h_lim))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def load_imagenet_val_pilot_split(
    images_dir: str | Path,
    xml_dir: str | Path | None = None,
    *,
    n_images: int = 1000,
    seed: int = 0,
    split_file: str | Path | None = None,
) -> list[ImageNetSample]:
    """Load Guide §8.4 ImageNet-1K val pilot split (1,000 images, seed 0) with multi-object XML boxes.

    Supports both flat (val/*.JPEG) and class-subfolder (val/n01440764/*.JPEG) directory layouts.
    If split_file exists, reads those exact image stems verbatim (Guide §8.1: never regenerate).
    """
    img_root = Path(images_dir)
    if not img_root.is_dir():
        raise FileNotFoundError(f"ImageNet val images_dir not found: {img_root}")

    exts = (".JPEG", ".jpeg", ".jpg", ".JPG", ".png", ".PNG")
    ext_set = {".jpeg", ".jpg", ".png"}
    split_path = Path(split_file) if split_file is not None else None
    xr = Path(xml_dir) if xml_dir is not None else None
    if xr is not None and not xr.is_dir():
        raise FileNotFoundError(f"ImageNet XML dir not found: {xr}")

    # Fast path: if frozen split_file already exists and directory is flat (e.g. Kaggle ILSVRC val/*.JPEG),
    # resolve the exact split stems directly in O(N) stat calls instead of scanning 50,000 files via rglob("*").
    if split_path is not None and split_path.exists():
        chosen_stems = [ln.strip() for ln in split_path.read_text().splitlines() if ln.strip()]
        fast_samples: list[ImageNetSample] = []
        fast_ok = True
        for st in chosen_stems:
            img_p = None
            for ext in exts:
                cand = img_root / f"{st}{ext}"
                if cand.is_file():
                    img_p = cand
                    break
            if img_p is None:
                fast_ok = False
                break
            xp = None
            if xr is not None:
                cand_x = xr / f"{st}.xml"
                if not cand_x.is_file():
                    fast_ok = False
                    break
                xp = cand_x
            boxes = tuple(parse_ilsvrc_xml_boxes(xp)) if xp is not None else ()
            fast_samples.append(ImageNetSample(image_id=st, image_path=img_p, xml_path=xp, boxes_0based=boxes))
        if fast_ok:
            return fast_samples

    stem_to_img: dict[str, Path] = {}
    for p in sorted(img_root.rglob("*")):
        if p.is_file() and p.suffix.lower() in ext_set:
            stem_to_img[p.stem] = p
    if not stem_to_img:
        raise FileNotFoundError(f"No ImageNet images found under {img_root}")

    stem_to_xml: dict[str, Path] = {}
    if xr is not None:
        for xp in sorted(xr.rglob("*.xml")):
            stem_to_xml[xp.stem] = xp
        eligible_stems = sorted(s for s in stem_to_img if s in stem_to_xml)
    else:
        eligible_stems = sorted(stem_to_img.keys())

    if not eligible_stems:
        raise FileNotFoundError(f"No matching image/XML pairs found under {img_root} and {xml_dir}")

    if split_path is not None and split_path.exists():
        chosen_stems = [ln.strip() for ln in split_path.read_text().splitlines() if ln.strip()]
    else:
        rng = np.random.default_rng(int(seed))
        k = min(int(n_images), len(eligible_stems))
        idx = np.sort(rng.choice(len(eligible_stems), size=k, replace=False))
        chosen_stems = [eligible_stems[int(i)] for i in idx]
        if split_path is not None:
            split_path.parent.mkdir(parents=True, exist_ok=True)
            split_path.write_text("\n".join(chosen_stems) + "\n")

    samples: list[ImageNetSample] = []
    for st in chosen_stems:
        if st not in stem_to_img:
            raise FileNotFoundError(f"Split stem {st!r} not found under {img_root}")
        xp = stem_to_xml.get(st)
        boxes = tuple(parse_ilsvrc_xml_boxes(xp)) if xp is not None else ()
        samples.append(ImageNetSample(image_id=st, image_path=stem_to_img[st], xml_path=xp, boxes_0based=boxes))
    return samples


def get_transformed_imagenet_gt_mask(
    sample: ImageNetSample,
    orig_w: int,
    orig_h: int,
    target_size: int,
) -> tuple[np.ndarray, list[tuple[float, float, float, float]]]:
    """Guide §8.3: union of all transformed boxes for FG mask + list of transformed boxes for multi-instance CorLoc."""
    tboxes = [transform_box_resize_center_crop(b, orig_w, orig_h, target_size) for b in sample.boxes_0based]
    tboxes = [b for b in tboxes if b[2] > b[0] and b[3] > b[1]]
    union_mask = np.zeros((target_size, target_size), dtype=np.uint8)
    for tb in tboxes:
        union_mask = np.maximum(union_mask, box_to_pixel_mask(tb, target_size))
    return union_mask, tboxes


def save_imagenet_box_overlays(
    samples: Sequence[ImageNetSample],
    target_size: int,
    out_path: str | Path,
    n_images: int = 20,
) -> Path:
    """Guide §8.2: 4x5 visual spot-check grid of transformed ImageNet multi-object XML boxes."""
    chosen = list(samples)[:n_images]
    cols = 5
    rows = max(1, int(np.ceil(len(chosen) / cols)))
    pad = 4
    canvas = Image.new("RGB", (cols * (target_size + pad) + pad, rows * (target_size + pad) + pad), (30, 30, 30))
    vis_tf = T.Compose([T.Resize(target_size, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(target_size)])

    for idx, s in enumerate(chosen):
        img = Image.open(s.image_path).convert("RGB")
        ow, oh = img.size
        cropped = vis_tf(img)
        _, tboxes = get_transformed_imagenet_gt_mask(s, ow, oh, target_size)
        draw = ImageDraw.Draw(cropped)
        for tb in tboxes:
            draw.rectangle([tb[0], tb[1], tb[2], tb[3]], outline=(40, 255, 80), width=2)
        draw.text((4, 4), s.image_id[-8:], fill=(255, 255, 0))
        r, c = divmod(idx, cols)
        canvas.paste(cropped, (pad + c * (target_size + pad), pad + r * (target_size + pad)))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path
