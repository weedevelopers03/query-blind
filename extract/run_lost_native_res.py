"""Fix for Experiment 4's gap (58.19% vs 61.9%): run_extraction.py's transforms.Resize((224,224))
forces every image into a square, distorting aspect ratio - unlike LOST's own main_lost.py, which
keeps each image at its native size and only zero-pads up to the next multiple of the patch size.

This script matches LOST's approach exactly: one image at a time, at its own (padded) native
size, no batching into a single fixed-shape tensor (which is why the old script resized in the
first place - torch.stack() needs every item the same shape). Needs dynamic_img_size=True on the
timm model, since its position embeddings don't auto-adjust to a non-224 input otherwise - the
adapter's own token_layout() also assumes a square grid (the guide's own stated assumption), so
this script computes the (non-square, per-image) grid itself instead of calling that method.

2026-10-11 fix: added explicit CUDA device handling (model/input were never moved off CPU before,
so a GPU accelerator had no effect) and an optional --split flag that filters to the official
VOC ImageSets/Main/{split}.txt list, matching run_extraction.py's --split behavior, so the image
count is directly comparable (e.g. exactly the 5,011-image trainval set).

2026-10-12 fix: closes the remaining 60.73% vs 61.9% gap. timm's own dynamic_img_size
interpolation (model._pos_embed, called via the adapter) is NOT numerically identical to the
original facebookresearch/dino repo's interpolate_pos_encoding, even though both resize the same
pos_embed to the same target shape - confirmed by reading timm 1.0.30's actual installed source
(VisionTransformer._pos_embed -> timm.layers.pos_embed.resample_abs_pos_embed): it calls
F.interpolate(..., size=new_size, mode='bicubic', antialias=True), while DINO's own method calls
F.interpolate(..., scale_factor=(w0/sqrt(N), h0/sqrt(N)), mode='bicubic') with no antialias (so
antialias=False) and its own "+0.1" scale_factor epsilon (see
https://github.com/facebookresearch/dino/issues/8) - two concrete numerical differences, not a
hunch. dino_pos_embed() below reproduces DINO's exact method (source confirmed against the
official repo, freshly cloned) and is monkey-patched onto this one model instance in main(),
leaving the shared adapters/timm_vit.py untouched.

2026-10-19 fix: this is that monkey-patch, actually wired in. The paragraph above already
described it and the math/types/F imports below were already in place for it, but dino_pos_embed()
itself was never defined and main() never patched it onto the model - so every run so far
(including the 60.73% one) was still going through timm's resample_abs_pos_embed, not this fix.
Result: 60.77% - a real but small move (closes ~0.04 of the 0.17pt gap), still 0.13pt under the
60.9% tolerance floor - about 7 images (net) out of 5011.

2026-10-20 addition: logs per-image IoU/correctness to a CSV (--csv_out, on by default) so that
remaining ~7-image gap can be inspected directly - which images are borderline, what they have in
common - instead of guessed at. Doesn't change the CorLoc computation itself, only adds visibility
into it.

2026-10-21 fix: two evaluation-convention differences from LOST's official main_lost.py, found by
reading the official source (valeoai/LOST @ fcedbec) against this script:
(1) Clip size. main_lost.py takes init_image_size = img.shape BEFORE padding (line 175) and passes
    it to lost(), so detect_box clips predicted boxes to the ORIGINAL image size. This script
    passed the padded size, so 693/5011 boxes (13.8%) extended up to 15 px past the image edge into
    the zero padding, which can only lower IoU. (The comment that used to sit in the main loop
    claiming the official code clips to the padded size was wrong.)
(2) GT coordinates. LOST's extract_gt_VOC converts VOC's 1-based inclusive xmin/ymin to 0-based by
    subtracting 1 (datasets.py lines 275-276); parse_voc_xml used the raw values.
IoU now uses LOST's own bbox_iou, and the boxes are saved as preds.pkl in main_lost.py's
{image_name: box} format, so LOST's own main_corloc_evaluation.py can score them independently.
The per-image CSV also records the seed patch.

Usage: PYTHONPATH=.:/path/to/LOST python3 extract/run_lost_native_res.py \\
  --image_dir /path/to/VOC2007/JPEGImages --split trainval
"""
import csv
import math
import pickle
import sys
import pathlib
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from adapters.timm_vit import TimmViTAdapter
from object_discovery import lost  # LOST's own; see lost_corloc_from_cache.py's docstring for why
from datasets import bbox_iou  # LOST's own IoU (object_discovery already imports it from here)

PATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TO_TENSOR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def dino_pos_embed(self, x):
    """Drop-in replacement for timm's VisionTransformer._pos_embed (bound via
    types.MethodType in main(), below), reproducing facebookresearch/dino's
    interpolate_pos_encoding exactly: bicubic interpolation with NO antialiasing
    and DINO's own "+0.1" scale-factor epsilon (see
    https://github.com/facebookresearch/dino/issues/8) - instead of timm's
    resample_abs_pos_embed, which uses antialias=True and an exact target size
    with no epsilon. This is the "2026-10-12 fix" described in this file's
    module docstring: documented there but never actually implemented until now
    (math/types/F were imported for it but unused).

    Assumes dynamic_img_size=True, under which timm hands _pos_embed an
    unflattened (B, H, W, C) tensor from patch_embed (so this can resample
    using the true H, W) - and assumes this checkpoint's architecture: one
    cls_token, no register tokens (true for DINO ViT-S/16). Confirmed working:
    the 2026-10-19 run moved CorLoc from 60.73% to 60.77% without crashing.
    """
    B, H, W, C = x.shape
    x = x.view(B, H * W, C)

    N = self.pos_embed.shape[1] - 1  # patches at pretrain resolution (224/16=14 -> 196)
    if H * W == N:
        pos_embed = self.pos_embed
    else:
        class_pos_embed = self.pos_embed[:, :1]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = self.pos_embed.shape[-1]
        w0, h0 = W + 0.1, H + 0.1  # DINO's own epsilon - see issue #8 above
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(h0 / math.sqrt(N), w0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        pos_embed = torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    cls_token = self.cls_token.expand(B, -1, -1)
    x = torch.cat((cls_token, x), dim=1) + pos_embed
    return self.pos_drop(x)


def load_native_padded(path, patch_size=PATCH_SIZE):
    """LOST's own padding formula (main_lost.py), applied to one image: keep native size, zero-pad
    up to the next multiple of patch_size. No resize, so no aspect-ratio distortion."""
    img = TO_TENSOR(Image.open(path).convert("RGB"))
    c, h, w = img.shape
    gh, gw = -(-h // patch_size), -(-w // patch_size)  # ceil division
    padded = torch.zeros(c, gh * patch_size, gw * patch_size)
    padded[:, :h, :w] = img
    # also return the UNPADDED (h, w): lost() must clip boxes to this, not to the padded size
    return padded.unsqueeze(0), (gh, gw), (h, w)


def load_split_ids(image_dir, split):
    """Reads the official VOCdevkit/VOC2007/ImageSets/Main/{split}.txt list (one image ID per
    line), matching run_extraction.py's --split filtering so counts are directly comparable."""
    split_file = Path(image_dir).parent / "ImageSets" / "Main" / f"{split}.txt"
    with open(split_file) as f:
        return {line.strip() for line in f if line.strip()}


def parse_voc_xml(xml_path):
    """GT boxes exactly as LOST's datasets.extract_gt_VOC builds them: integer coordinates, with
    VOC's 1-based inclusive xmin/ymin shifted to 0-based (xmin-1, ymin-1; xmax/ymax unchanged)."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w, h = float(size.find("width").text), float(size.find("height").text)
    boxes = []
    for o in root.findall("object"):
        b = o.find("bndbox")
        xmin, ymin, xmax, ymax = (int(b.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append([xmin - 1, ymin - 1, xmax, ymax])
    return np.array(boxes, dtype=np.int64).reshape(-1, 4), w, h


@torch.no_grad()
def patch_keys(adapter, model, x):
    """Last-layer patch keys (drop [CLS]) as (1, N, D), heads concatenated - lost()'s expected
    input. Does not call adapter.token_layout() (asserts a square grid; not true here).
    Returned on CPU: the ViT forward pass is the only part that benefits from the GPU - LOST's
    own seed-expansion/box-detection code is numpy-based and assumes CPU-side arrays."""
    l = adapter.n_blocks(model) - 1
    _, k, _ = adapter.get_qkv(model, x, l)       # (1, H, T, d_h), T = 1 + N patches
    assert k.dtype == torch.float32, f"keys are {k.dtype}; LOST computes similarities in fp32"
    B, H, T, d_h = k.shape
    k = k[:, :, 1:, :].transpose(1, 2).reshape(B, T - 1, H * d_h)
    return k.float().cpu()


def main(img_dir, gt_dir=None, pretrained=True, limit=None, split=None, csv_out=None, preds_out=None):
    gt_dir = gt_dir or (Path(img_dir).parent / "Annotations")

    print(f"--> Using device: {DEVICE}")
    adapter = TimmViTAdapter()
    model = timm.create_model("vit_small_patch16_224.dino", pretrained=pretrained, dynamic_img_size=True)
    model = model.to(DEVICE).eval()
    adapter.prepare(model)
    model._pos_embed = types.MethodType(dino_pos_embed, model)
    print("--> Patched _pos_embed: reproducing DINO's exact interpolate_pos_encoding (bicubic, no antialias, +0.1 epsilon)")

    img_paths = sorted(Path(img_dir).glob("*.jpg"))
    if split:
        split_ids = load_split_ids(img_dir, split)
        img_paths = [p for p in img_paths if p.stem in split_ids]
        print(f"--> Split '{split}': {len(img_paths)} images (from {len(split_ids)} IDs listed)")
    if limit:
        img_paths = img_paths[:limit]

    csv_writer = csv_file = None
    if csv_out:
        csv_file = open(csv_out, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "image_id", "orig_w", "orig_h", "gh", "gw", "num_gt",
            "best_iou", "correct",
            "pred_x1", "pred_y1", "pred_x2", "pred_y2",
            "gt_x1", "gt_y1", "gt_x2", "gt_y2", "seed",
        ])

    correct = total = n_size_mismatch = 0
    preds_dict = {}
    for i, path in enumerate(img_paths, 1):
        xml_path = Path(gt_dir) / f"{path.stem}.xml"
        if not xml_path.exists():
            continue
        gt_boxes, orig_w, orig_h = parse_voc_xml(xml_path)
        if len(gt_boxes) == 0:
            continue

        x, (gh, gw), (img_h, img_w) = load_native_padded(path)
        if (img_w, img_h) != (int(orig_w), int(orig_h)):
            n_size_mismatch += 1  # XML <size> vs the JPEG actually loaded; clipping uses the JPEG's
        x = x.to(DEVICE)
        feats = patch_keys(adapter, model, x)
        # init_image_size must be the UNPADDED (C, H, W): main_lost.py takes img.shape BEFORE padding
        # (line 175) and lost() -> detect_box clips the box to it (object_discovery.py lines 112-114).
        pred, _, _, seed = lost(feats, dims=(gh, gw), scales=(PATCH_SIZE, PATCH_SIZE),
                                init_image_size=(3, img_h, img_w))
        pred = np.asarray(pred)
        assert pred[2] <= img_w and pred[3] <= img_h, f"{path.stem}: box {pred} exceeds image {img_w}x{img_h}"
        preds_dict[path.name] = pred  # same {image_name: box} format as main_lost.py's preds.pkl

        ious = bbox_iou(torch.from_numpy(pred), torch.from_numpy(gt_boxes)).numpy()  # LOST's own IoU
        best_idx = int(ious.argmax())
        best_iou = float(ious[best_idx])
        is_correct = best_iou >= 0.5
        if is_correct:
            correct += 1
        total += 1

        if csv_writer:
            gt_best = gt_boxes[best_idx]
            csv_writer.writerow([
                path.stem, orig_w, orig_h, gh, gw, len(gt_boxes),
                f"{best_iou:.4f}", int(is_correct),
                *[f"{float(v):.1f}" for v in pred],
                *[f"{float(v):.1f}" for v in gt_best],
                int(seed),
            ])

        if i % 500 == 0:
            print(f"  --> {i}/{len(img_paths)} ({100*correct/max(total,1):.2f}% so far)")

    if csv_file:
        csv_file.close()
        print(f"--> Per-image results written to {csv_out}")
    if preds_out:
        with open(preds_out, "wb") as f:
            pickle.dump(preds_dict, f)
        print(f"--> Predictions written to {preds_out} (main_lost.py's preds.pkl format)")
    print(f"--> Images whose XML <size> differs from the JPEG: {n_size_mismatch}")

    corloc = 100 * correct / max(total, 1)
    print(f"--> CorLoc, native-resolution extraction ({total} images): {corloc:.2f}%  ({correct}/{total})")
    print("--> Published LOST: 61.9%; pre-registered pass band 60.9-62.9% (+-1.0 pt): "
          + ("INSIDE" if 60.9 <= corloc <= 62.9 else "OUTSIDE"))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", required=True)
    p.add_argument("--gt_dir", default=None, help="Defaults to a sibling 'Annotations' folder next to --image_dir")
    p.add_argument("--split", default=None, help="e.g. trainval - filters to VOCdevkit's ImageSets/Main/<split>.txt")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--csv_out", default="corloc_per_image.csv", help="Per-image IoU/correctness log (pass '' to disable)")
    p.add_argument("--preds_out", default="preds.pkl", help="Predicted boxes in main_lost.py's preds.pkl format (pass '' to disable)")
    args = p.parse_args()
    main(args.image_dir, args.gt_dir, limit=args.limit, split=args.split, csv_out=(args.csv_out or None),
         preds_out=(args.preds_out or None))
