"""Pilot runner for Measure M3 (Routing Decomposition & Waterbirds BSS_norm, Guide §4 & Review (3)).

Runs all 8 pilot models on the official Waterbirds test split (5,794 images, or --limit N),
paired with CUB-200-2011 pixel segmentation masks (§8.1 & §8.3) under the primary 14x14 protocol
(224 px for /16, 196 px for /14), with per-model normalization constants.

Computes per (model, layer, image):
  * Kobayashi norm-based attribution c_i = ||gamma * sum_h a_i^(h) W_O^(h) v_i^(h)||_2
  * 5 token groups (FG / BG / OUT / REG / SELF), OUT = median + 5*MAD (primary & alt ordering)
  * BSS_raw, BSS_norm (area-normalized, primary for H2), BSS_norm_alt, Residual Ratio RR
  * Waterbirds congruent (y == place) vs incongruent (y != place) gap + 95% bootstrap CIs
    overall, by area tercile (small/medium/large, §4.4), and by 4 subgroups.

Usage on Kaggle:
    PYTHONPATH=. python3 scripts/run_pilot_waterbirds_m3.py \\
        --waterbirds_dir /kaggle/working/datasets/waterbirds/waterbird_complete95_forest2water2 \\
        --cub_seg_dir /kaggle/working/datasets/cub/segmentations \\
        --out_dir /kaggle/working/results/pilot_waterbirds_m3 \\
        --batch_size 32 \\
        --resume
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.base import attn_probs
from eval.preprocess import (
    build_image_transform,
    patch_fg_mask,
    transform_mask_resize_center_crop,
)
from extract.cache import last_third_blocks
from measures import routing
from measures._common import get_output_proj_and_layerscale, write_records
from scripts.run_pilot_m1_m2 import (
    PILOT8_SPECS,
    ZOO12_SPECS,
    resolve_model_ids,
    _file_sha256_16,
    _git_sha,
    _load_registry_meta,
    bootstrap_ci,
    load_pilot_model,
)


@dataclass
class WaterbirdsSample:
    img_id: int
    img_filename: str
    image_path: Path
    seg_path: Path
    y_bird: int          # 0 = landbird, 1 = waterbird
    place_bg: int        # 0 = land, 1 = water
    congruent: bool      # y_bird == place_bg


def _resolve_waterbirds_root(wb_dir: str | Path) -> Path:
    p = Path(wb_dir)
    if (p / "metadata.csv").exists():
        return p
    nested = p / "waterbird_complete95_forest2water2"
    if (nested / "metadata.csv").exists():
        return nested
    raise FileNotFoundError(f"Could not find Waterbirds metadata.csv under {p}")


def load_waterbirds_test_split(
    waterbirds_dir: str | Path,
    cub_seg_dir: str | Path,
    *,
    split_code: int = 2,
    limit: int | None = None,
) -> list[WaterbirdsSample]:
    """Load Waterbirds test split (split==2, 5,794 images) paired with CUB segmentation masks (§8.1)."""
    root = _resolve_waterbirds_root(waterbirds_dir)
    seg_root = Path(cub_seg_dir)
    if (seg_root / "segmentations").is_dir():
        seg_root = seg_root / "segmentations"

    samples: list[WaterbirdsSample] = []
    missing_seg = 0
    with (root / "metadata.csv").open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["split"]) != int(split_code):
                continue
            rel = row["img_filename"]
            img_p = root / rel
            seg_p = (seg_root / rel).with_suffix(".png")
            if not seg_p.exists():
                missing_seg += 1
                continue
            y = int(row["y"])
            pl = int(row["place"])
            samples.append(
                WaterbirdsSample(
                    img_id=int(row["img_id"]),
                    img_filename=rel,
                    image_path=img_p,
                    seg_path=seg_p,
                    y_bird=y,
                    place_bg=pl,
                    congruent=(y == pl),
                )
            )
    if missing_seg > 0:
        raise FileNotFoundError(f"Missing {missing_seg} CUB segmentation masks under {seg_root}")
    samples.sort(key=lambda s: s.img_id)
    if limit is not None:
        samples = samples[: int(limit)]
    return samples


@torch.no_grad()
def extract_m3_last_third_batch(
    adapter,
    model,
    x: torch.Tensor,
    layers: list[int],
) -> dict[int, dict[str, torch.Tensor]]:
    """Single-pass batched GPU extraction of M3 inputs at every block l in layers.

    Returns per layer:
      - c_all:           (B, T) token contribution norms ||gamma * sum_h a_i^(h) W_O^(h) v_i^(h)||_2
      - token_norms_all: (B, T) residual-stream L2 norms leaving block l (for OUT outlier detection)
      - x_g_in_norm:     (B,) ||x_g^(l)||_2 entering block l
      - delta_g_norm:    (B,) ||Delta_g^(l)||_2 attention update (incl. gamma & bias)
      - layout:          adapter token layout dict
    """
    layer_set = set(layers)
    max_l = max(layers)
    out: dict[int, dict] = {}

    if adapter.family == "timm_vit":
        layout = adapter.token_layout(model, x)
        cls_idx = layout["cls_idx"]
        z = model.patch_embed(x)
        z = model._pos_embed(z)
        z = model.patch_drop(z)
        z = model.norm_pre(z)

        for l, blk in enumerate(model.blocks[: max_l + 1]):
            z_in = z
            z_out = blk(z_in) if (l < max_l or l in layer_set) else z_in
            if l in layer_set:
                a = blk.attn
                B, T, _ = z_in.shape
                qkv = a.qkv(blk.norm1(z_in)).reshape(B, T, 3, a.num_heads, a.head_dim).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                q, k = a.q_norm(q), a.k_norm(k)
                attn_row = attn_probs(q[:, :, cls_idx : cls_idx + 1], k, float(a.scale)).squeeze(2)
                W_O, b_O, gamma = get_output_proj_and_layerscale(adapter, model, l)
                c_all, u_all = routing.token_contributions(attn_row, v, W_O, gamma=gamma, return_vectors=True)
                delta_g = u_all.sum(dim=1)
                if b_O is not None:
                    b_eff = b_O.to(delta_g.device) * (gamma.to(delta_g.device) if gamma is not None else 1.0)
                    delta_g = delta_g + b_eff
                out[l] = {
                    "c_all": c_all.float().cpu(),
                    "token_norms_all": z_out.float().norm(dim=-1).cpu(),
                    "x_g_in_norm": z_in[:, cls_idx].float().norm(dim=-1).cpu(),
                    "delta_g_norm": delta_g.float().norm(dim=-1).cpu(),
                    "layout": layout,
                }
            z = z_out
        return out

    if adapter.family == "open_clip_vit":
        vt = adapter._visual(model)
        layout = adapter.token_layout(model)
        cls_idx = layout["cls_idx"]
        z = vt.conv1(x).reshape(x.shape[0], vt.conv1.out_channels, -1).permute(0, 2, 1)
        cls = vt.class_embedding.to(z.dtype).expand(z.shape[0], 1, -1)
        z = torch.cat([cls, z], dim=1) + vt.positional_embedding.to(z.dtype)
        z = vt.patch_dropout(z)
        z = vt.ln_pre(z)

        for l, blk in enumerate(vt.transformer.resblocks[: max_l + 1]):
            z_in = z
            z_out = blk(z_in) if (l < max_l or l in layer_set) else z_in
            if l in layer_set:
                a = blk.attn
                B, T, _ = z_in.shape
                H, d = a.num_heads, a.head_dim
                zn = blk.ln_1(z_in)
                Wq, Wk, Wv = a.in_proj_weight.chunk(3, dim=0)
                bq, bk, bv = a.in_proj_bias.chunk(3, dim=0)
                q = (zn @ Wq.T + bq).reshape(B, T, H, d).transpose(1, 2)
                k = (zn @ Wk.T + bk).reshape(B, T, H, d).transpose(1, 2)
                v = (zn @ Wv.T + bv).reshape(B, T, H, d).transpose(1, 2)
                attn_row = attn_probs(q[:, :, cls_idx : cls_idx + 1], k, float(d ** -0.5)).squeeze(2)
                W_O, b_O, gamma = get_output_proj_and_layerscale(adapter, model, l)
                c_all, u_all = routing.token_contributions(attn_row, v, W_O, gamma=gamma, return_vectors=True)
                delta_g = u_all.sum(dim=1)
                if b_O is not None:
                    b_eff = b_O.to(delta_g.device) * (gamma.to(delta_g.device) if gamma is not None else 1.0)
                    delta_g = delta_g + b_eff
                out[l] = {
                    "c_all": c_all.float().cpu(),
                    "token_norms_all": z_out.float().norm(dim=-1).cpu(),
                    "x_g_in_norm": z_in[:, cls_idx].float().norm(dim=-1).cpu(),
                    "delta_g_norm": delta_g.float().norm(dim=-1).cpu(),
                    "layout": layout,
                }
            z = z_out
        return out

    raise NotImplementedError(f"M3 batched extractor not implemented for family={adapter.family!r}")


def _precompute_fg_patch_masks(samples: list[WaterbirdsSample], target_size: int, grid: tuple[int, int] = (14, 14)) -> np.ndarray:
    """Guide §8.3: store patch-level FG masks once per (dataset, grid/resolution) and reuse across models."""
    masks = np.empty((len(samples), grid[0], grid[1]), dtype=np.uint8)
    for i, s in enumerate(samples):
        pix = transform_mask_resize_center_crop(Image.open(s.seg_path), target_size)
        masks[i] = patch_fg_mask(pix, grid, threshold=0.5)
    return masks


def _write_m3_columns(col_chunks: list[dict[str, np.ndarray]], out_dir: Path, model_id: str) -> Path:
    cols = {k: np.concatenate([c[k] for c in col_chunks], axis=0) for k in routing.M3_SCHEMA_COLUMNS}
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
        import pyarrow  # noqa: F401

        dest = out_dir / f"{model_id}.parquet"
        tmp = dest.with_suffix(".tmp.parquet")
        pd.DataFrame(cols).to_parquet(tmp, index=False)
        os.replace(tmp, dest)
        return dest
    except Exception:
        dest = out_dir / f"{model_id}.csv"
        tmp = dest.with_suffix(".tmp.csv")
        n = len(cols["bss_norm"])
        with tmp.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(routing.M3_SCHEMA_COLUMNS)
            for idx in range(n):
                w.writerow([cols[k][idx] for k in routing.M3_SCHEMA_COLUMNS])
        os.replace(tmp, dest)
        return dest


def run_one_model_m3(
    model_id: str,
    samples: list[WaterbirdsSample],
    fg_masks_cache: dict[int, np.ndarray],
    device: torch.device,
    out_dir: Path,
    weights_cache_dir: Path,
    reg_meta: dict,
    *,
    batch_size: int = 32,
    smoke_random: bool = False,
) -> dict:
    t0 = time.perf_counter()
    adapter, model, mean, std, target_size = load_pilot_model(
        model_id, device, weights_cache_dir, smoke_random=smoke_random
    )
    n_blocks = adapter.n_blocks(model)
    layers = last_third_blocks(n_blocks)
    tf = build_image_transform(target_size, mean, std)

    if target_size not in fg_masks_cache:
        fg_masks_cache[target_size] = _precompute_fg_patch_masks(samples, target_size, grid=(14, 14))
    fg_masks = fg_masks_cache[target_size]

    n_img = len(samples)
    print(f"\n[{model_id}] M3 Waterbirds | device={device}, res={target_size}, n={n_img}, norm=({[round(m,3) for m in mean]}, {[round(s,3) for s in std]})", flush=True)

    t_ext0 = time.perf_counter()
    collected: dict[int, dict[str, list[torch.Tensor]]] = {
        l: {"c_all": [], "token_norms_all": [], "x_g_in_norm": [], "delta_g_norm": []} for l in layers
    }
    layouts: dict[int, dict] = {}

    for start in range(0, n_img, batch_size):
        batch = samples[start : start + batch_size]
        xb = torch.stack([tf(Image.open(s.image_path).convert("RGB")) for s in batch], dim=0).to(device)
        bo = extract_m3_last_third_batch(adapter, model, xb, layers)
        for l in layers:
            for k in ("c_all", "token_norms_all", "x_g_in_norm", "delta_g_norm"):
                collected[l][k].append(bo[l][k])
            layouts[l] = bo[l]["layout"]

    extract_sec = time.perf_counter() - t_ext0
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    row_meta = reg_meta.get(model_id, {})
    objective = row_meta.get("objective", "unknown")
    set_name = row_meta.get("set", "pilot")
    img_ids = np.array([s.img_id for s in samples], dtype=np.int64)
    congruent = np.array([s.congruent for s in samples], dtype=bool)
    y_bird = np.array([s.y_bird for s in samples], dtype=np.int64)
    place_bg = np.array([s.place_bg for s in samples], dtype=np.int64)

    col_chunks: list[dict[str, np.ndarray]] = []
    layer_summaries: list[dict] = []
    congruence_by_layer: dict[str, dict] = {}

    for l in layers:
        c_all = torch.cat(collected[l]["c_all"], dim=0).numpy()
        norms_all = torch.cat(collected[l]["token_norms_all"], dim=0).numpy()
        xg_n = torch.cat(collected[l]["x_g_in_norm"], dim=0).numpy()
        dg_n = torch.cat(collected[l]["delta_g_norm"], dim=0).numpy()
        lay = layouts[l]
        grid = lay["grid"]
        grid_str = f"{grid[0]}x{grid[1]}"

        dec = routing.routing_decomposition_batch(c_all, norms_all, lay, fg_masks, xg_n, dg_n)
        terciles = routing.assign_area_terciles(dec["gt_area_frac"])
        wb = routing.waterbirds_congruence_analysis(
            dec["bss_norm"], congruent, area_terciles=terciles, y_bird=y_bird, place_bg=place_bg
        )
        congruence_by_layer[f"l{l}"] = wb

        l_frac = float(l + 1) / float(n_blocks)
        col_chunks.append({
            "model_id": np.full(n_img, model_id, dtype=object),
            "objective": np.full(n_img, objective, dtype=object),
            "set": np.full(n_img, set_name, dtype=object),
            "layer": np.full(n_img, l, dtype=np.int64),
            "layer_frac": np.full(n_img, l_frac, dtype=np.float64),
            "image_id": img_ids,
            "bss_raw": dec["bss_raw"],
            "bss_norm": dec["bss_norm"],
            "bss_norm_alt": dec["bss_norm_alt"],
            "share_fg": dec["share_fg"],
            "share_bg": dec["share_bg"],
            "share_out": dec["share_out"],
            "share_reg": dec["share_reg"],
            "share_self": dec["share_self"],
            "share_fg_alt": dec["share_fg_alt"],
            "share_bg_alt": dec["share_bg_alt"],
            "residual_ratio": dec["residual_ratio"],
            "n_out_tokens": dec["n_out_tokens"],
            "gt_area_frac": dec["gt_area_frac"],
            "area_tercile": terciles,
            "waterbirds_congruent": congruent,
            "y_bird": y_bird,
            "place_bg": place_bg,
            "resolution": np.full(n_img, target_size, dtype=np.int64),
            "grid": np.full(n_img, grid_str, dtype=object),
        })

        rr_m, _, _ = bootstrap_ci(dec["residual_ratio"])
        gap = wb["congruence_gap_cong_minus_incong"]
        layer_summaries.append({
            "model_id": model_id,
            "objective": objective,
            "layer": l,
            "layer_frac": round(l_frac, 4),
            "resolution": target_size,
            "n_images": n_img,
            "bss_norm_all": round(wb["overall"]["mean"], 4),
            "bss_norm_cong": round(wb["congruent"]["mean"], 4),
            "bss_norm_incong": round(wb["incongruent"]["mean"], 4),
            "cong_gap": round(gap["mean_diff"], 4),
            "cong_gap_ci95_lo": round(gap["ci95_low"], 4),
            "cong_gap_ci95_hi": round(gap["ci95_high"], 4),
            "cong_gap_excl0": bool(gap["ci_excludes_zero"]),
            "bss_raw": round(float(np.nanmean(dec["bss_raw"])), 4),
            "bss_norm_alt": round(float(np.nanmean(dec["bss_norm_alt"])), 4),
            "share_fg": round(float(np.nanmean(dec["share_fg"])), 4),
            "share_bg": round(float(np.nanmean(dec["share_bg"])), 4),
            "share_out": round(float(np.nanmean(dec["share_out"])), 4),
            "share_reg": round(float(np.nanmean(dec["share_reg"])), 4),
            "share_self": round(float(np.nanmean(dec["share_self"])), 4),
            "n_out_mean": round(float(np.mean(dec["n_out_tokens"])), 2),
            "residual_ratio": round(rr_m, 3),
        })
        print(
            f"  l={l:2d} ({l_frac:.2f}) | BSS_norm={wb['overall']['mean']:.3f} "
            f"(cong={wb['congruent']['mean']:.3f}, incong={wb['incongruent']['mean']:.3f}, "
            f"gap={gap['mean_diff']:+.4f} [{gap['ci95_low']:+.4f}, {gap['ci95_high']:+.4f}]) | "
            f"shares FG={np.nanmean(dec['share_fg']):.2f} BG={np.nanmean(dec['share_bg']):.2f} "
            f"OUT={np.nanmean(dec['share_out']):.2f} REG={np.nanmean(dec['share_reg']):.2f} SELF={np.nanmean(dec['share_self']):.2f} | "
            f"RR={rr_m:.2f}",
            flush=True,
        )

    _write_m3_columns(col_chunks, out_dir / "routing", model_id)
    total_sec = time.perf_counter() - t0
    model_result = {
        "model_id": model_id,
        "objective": objective,
        "n_blocks": n_blocks,
        "layers": layers,
        "resolution": target_size,
        "norm_mean": list(mean),
        "norm_std": list(std),
        "n_images": n_img,
        "extract_sec": round(extract_sec, 2),
        "extract_img_per_sec": round(n_img / max(extract_sec, 1e-6), 2),
        "total_sec": round(total_sec, 2),
        "layer_summaries": layer_summaries,
        "congruence_by_layer": congruence_by_layer,
    }
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    tmp_json = out_dir / "models" / f"{model_id}.json.tmp"
    tmp_json.write_text(json.dumps(model_result, indent=2))
    os.replace(tmp_json, out_dir / "models" / f"{model_id}.json")
    return model_result


def write_waterbirds_report(results: dict[str, dict], out_dir: Path) -> None:
    rows = [r for mr in results.values() for r in mr["layer_summaries"]]
    cols = list(rows[0].keys()) if rows else []
    write_records(rows, out_dir / "summary_waterbirds_m3.csv", columns=cols)

    lines = [
        "# Pilot Measure M3 — Waterbirds Routing & BSS_norm Summary\n",
        "| Model | Obj | Layer (frac) | BSS_norm (all) | Cong | Incong | Gap (C−I) [95% CI] | FG | BG | OUT | REG | SELF | n_OUT | RR |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model_id']} | {r['objective']} | {r['layer']} ({r['layer_frac']:.2f}) | "
            f"{r['bss_norm_all']:.3f} | {r['bss_norm_cong']:.3f} | {r['bss_norm_incong']:.3f} | "
            f"{r['cong_gap']:+.4f} [{r['cong_gap_ci95_lo']:+.4f}, {r['cong_gap_ci95_hi']:+.4f}] | "
            f"{r['share_fg']:.2f} | {r['share_bg']:.2f} | {r['share_out']:.2f} | "
            f"{r['share_reg']:.2f} | {r['share_self']:.2f} | {r['n_out_mean']:.2f} | {r['residual_ratio']:.2f} |"
        )
    (out_dir / "summary_waterbirds.md").write_text("\n".join(lines) + "\n")

    congruence_dump = {m: r["congruence_by_layer"] for m, r in results.items()}
    (out_dir / "waterbirds_congruence.json").write_text(json.dumps(congruence_dump, indent=2))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Run Measure M3 (routing decomposition & Waterbirds BSS_norm) on Pilot-8.")
    ap.add_argument("--waterbirds_dir", required=True, help="Path to Waterbirds folder containing metadata.csv.")
    ap.add_argument("--cub_seg_dir", required=True, help="Path to CUB-200-2011 segmentations/ directory.")
    ap.add_argument("--out_dir", default="results/pilot_waterbirds_m3")
    ap.add_argument("--models", default="all", help="'all' for all 8 pilot models, or comma-separated IDs.")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on test images (None = full 5,794 test set).")
    ap.add_argument("--weights_cache_dir", default=str(Path.home() / ".cache" / "queryblind_weights"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke_random", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_ids = resolve_model_ids(args.models)

    samples = load_waterbirds_test_split(args.waterbirds_dir, args.cub_seg_dir, split_code=2, limit=args.limit)
    n_cong = sum(s.congruent for s in samples)
    print(f"Loaded {len(samples)} Waterbirds test images ({n_cong} congruent, {len(samples) - n_cong} incongruent)", flush=True)

    reg_meta = _load_registry_meta()
    weights_cache = Path(args.weights_cache_dir)
    fg_masks_cache: dict[int, np.ndarray] = {}
    results: dict[str, dict] = {}
    t_all = time.perf_counter()

    for mid in model_ids:
        done_file = out_dir / "models" / f"{mid}.json"
        if args.resume and done_file.exists():
            print(f"[{mid}] --resume: loaded existing {done_file}", flush=True)
            results[mid] = json.loads(done_file.read_text())
            continue
        results[mid] = run_one_model_m3(
            mid, samples, fg_masks_cache, device, out_dir, weights_cache, reg_meta,
            batch_size=args.batch_size, smoke_random=args.smoke_random,
        )

    write_waterbirds_report(results, out_dir)
    cfg_payload = json.dumps({"models": model_ids, "n_images": len(samples), "smoke_random": args.smoke_random}, sort_keys=True)
    meta = {
        "git_sha": _git_sha(),
        "config_hash": hashlib.sha256(cfg_payload.encode()).hexdigest()[:16],
        "file_sha256_16": {
            rel: _file_sha256_16(REPO_ROOT / rel)
            for rel in ("measures/routing.py", "eval/preprocess.py", "scripts/run_pilot_waterbirds_m3.py", "configs/checkpoints.yaml")
        },
        "device": str(device),
        "dataset": "Waterbirds-test",
        "n_images": len(samples),
        "n_congruent": n_cong,
        "n_incongruent": len(samples) - n_cong,
        "smoke_random": bool(args.smoke_random),
        "total_seconds": round(time.perf_counter() - t_all, 2),
        "throughput_by_model": {
            m: {"extract_sec": r["extract_sec"], "img_per_sec": r["extract_img_per_sec"], "total_sec": r["total_sec"]}
            for m, r in results.items()
        },
        "unix_time": int(time.time()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print("\n" + (out_dir / "summary_waterbirds.md").read_text(), flush=True)
    return results


if __name__ == "__main__":
    main()
