"""Paired per-image comparison of two LOST runs (my pipeline vs the official main_lost.py), both
scored here with LOST's own GT conventions from their preds.pkl files ({image_name: [x1,y1,x2,y2]}).

  python3 compare_corloc_runs.py --ours preds.pkl \\
      --official /path/to/LOST/outputs/VOC07_trainval/LOST-vit_small16_k/preds.pkl \\
      --voc_root /path/to/VOC2007 --split trainval --out discordant.csv

Reports: CorLoc of each run; the paired 2x2 table (b = official hit & mine miss, c = mine hit &
official miss); exact McNemar p; box agreement (identical, IoU >= 0.9); and, for the discordant
images, how close to the 0.5 threshold they sit and whether multi-object images are over-represented
against the base rate. Seeds are not in main_lost.py's preds.pkl, so same-seed agreement is not here.
Needs numpy only.
"""
import argparse
import csv
import math
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def load_gt(xml_path):
    """As LOST's datasets.extract_gt_VOC: integer boxes, xmin-1 and ymin-1 (1-based inclusive -> 0-based)."""
    boxes = []
    for o in ET.parse(xml_path).getroot().findall("object"):
        b = o.find("bndbox")
        xmin, ymin, xmax, ymax = (int(b.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append([xmin - 1, ymin - 1, xmax, ymax])
    return np.array(boxes, dtype=np.float64).reshape(-1, 4)


def iou_many(box, boxes):
    ix = np.clip(np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0]), 0, None)
    iy = np.clip(np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1]), 0, None)
    inter = ix * iy
    union = (box[2] - box[0]) * (box[3] - box[1]) + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) - inter
    return inter / union


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def binom_tail_ge(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--official", required=True)
    ap.add_argument("--voc_root", required=True, help="folder containing Annotations/ and ImageSets/")
    ap.add_argument("--split", default="trainval")
    ap.add_argument("--out", default="discordant.csv")
    a = ap.parse_args()

    ours = {Path(k).stem: np.asarray(v, dtype=np.float64) for k, v in pickle.load(open(a.ours, "rb")).items()}
    off = {Path(k).stem: np.asarray(v, dtype=np.float64) for k, v in pickle.load(open(a.official, "rb")).items()}
    root = Path(a.voc_root)
    ids = [l.strip() for l in open(root / "ImageSets" / "Main" / f"{a.split}.txt") if l.strip()]

    rows, miss_o, miss_f = [], 0, 0
    for i in ids:
        gt = load_gt(root / "Annotations" / f"{i}.xml")
        if len(gt) == 0:
            continue
        if i not in ours:
            miss_o += 1
            continue
        if i not in off:
            miss_f += 1
            continue
        io, iff = iou_many(ours[i], gt).max(), iou_many(off[i], gt).max()
        rows.append(dict(image_id=i, num_gt=len(gt), mine_iou=io, official_iou=iff,
                         mine_hit=int(io >= 0.5), official_hit=int(iff >= 0.5),
                         box_iou=iou_many(ours[i], off[i][None, :])[0],
                         identical=int(np.array_equal(ours[i], off[i]))))
    N = len(rows)
    mh = np.array([r["mine_hit"] for r in rows]); oh = np.array([r["official_hit"] for r in rows])
    A = int(((mh == 1) & (oh == 1)).sum()); B = int(((oh == 1) & (mh == 0)).sum())
    C = int(((mh == 1) & (oh == 0)).sum()); D = int(((mh == 0) & (oh == 0)).sum())
    print(f"images compared: {N}  (missing from mine: {miss_o}, from official: {miss_f})")
    print(f"CorLoc  mine: {mh.sum()}/{N} = {100 * mh.mean():.2f}%   official: {oh.sum()}/{N} = {100 * oh.mean():.2f}%   "
          f"(official - mine = {int(oh.sum() - mh.sum())} images)")
    print(f"paired: both hit {A} | official-only (b) {B} | mine-only (c) {C} | both miss {D}   ->  b + c = {B + C}")
    print(f"McNemar exact two-sided p = {mcnemar_exact(B, C):.4g}")
    ident = sum(r["identical"] for r in rows)
    agree90 = sum(r["box_iou"] >= 0.9 for r in rows)
    print(f"boxes identical: {ident}/{N} ({100 * ident / N:.1f}%)   IoU(mine, official) >= 0.9: {agree90}/{N} ({100 * agree90 / N:.1f}%)")

    disc = [r for r in rows if r["mine_hit"] != r["official_hit"]]
    if disc:
        near = sum(min(abs(r["mine_iou"] - 0.5), abs(r["official_iou"] - 0.5)) < 0.05 for r in disc)
        multi = sum(r["num_gt"] > 1 for r in disc)
        base = float(np.mean([r["num_gt"] > 1 for r in rows]))
        print(f"discordant images ({len(disc)}): within 0.05 IoU of the 0.5 threshold on at least one side: {near}/{len(disc)}; "
              f"median IoU between the two boxes: {np.median([r['box_iou'] for r in disc]):.3f}")
        print(f"  multi-object: {multi}/{len(disc)} = {100 * multi / len(disc):.1f}% vs base rate {100 * base:.1f}% "
              f"(exact one-sided binomial p = {binom_tail_ge(multi, len(disc), base):.3g})")
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(disc[0].keys()))
            w.writeheader()
            w.writerows(disc)
        print(f"discordant images written to {a.out}")


if __name__ == "__main__":
    main()
