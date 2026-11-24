#!/usr/bin/env python3
"""Action A2 (§6.2): Mechanistic per-token [CLS] attention-mass shift under I1-S4 on DeiT-III checkpoints.

Supervisor Review (4) §6.2 exact 3-step protocol (exploratory mechanism check):
  1. For each image, compute per-token change in head-weighted [CLS] patch attention mass at final block l=11
     under I1-S4 (surrogate minus true):  delta_a_i = a_S4(i) - a_true(i), averaged over 5 seeds (0..4).
  2. Partition patches into disjoint M3 groups (OUT = > median + 5*MAD overrides FG/BG; FG = M_fg \\ OUT;
     BG = M_fg^c \\ OUT) and report total mass shift (delta_mass_FG, delta_mass_BG, delta_mass_OUT) plus
     per-token average shift and paired image-level bootstrap 95% CIs (B = 2,000).
  3. Test whether tokens losing mass (delta_a_i < 0) are the high-BSS background shortcut tokens identified in M3:
       - Split BG into top-decile contribution tokens BG_high_c (c_i > q90(c) among patches) vs BG_rest.
       - Compute mean mass shift on BG_high_c vs BG_rest, fraction of total negative mass shift absorbed by
         BG_high_c + OUT, and within-image Spearman correlation rho(delta_a_i, c_i) across BG tokens.

Default models: deit3_small_patch16_224, deit3_small_patch16_224_in21k, plus dino_vits16 as an SSL negative control.
Runs in ~90 seconds on 1 T4 GPU (ImageNet-1K 1,000 frozen split) and supports --smoke_random for local CPU testing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.base import attn_probs
from eval.preprocess import (
    build_image_transform,
    get_transformed_imagenet_gt_mask,
    load_imagenet_val_pilot_split,
    patch_fg_mask,
)
from interventions.i1_i2 import sample_surrogate_queries_batch
from measures import qii, routing
from measures._common import (
    box_auc_torch,
    compute_head_weights,
    get_output_proj_and_layerscale,
    patch_attention,
)
from scripts.run_pilot_m1_m2 import (
    _git_sha,
    bootstrap_ci,
    load_pilot_model,
)

DEFAULT_MODELS = ("deit3_small_patch16_224", "deit3_small_patch16_224_in21k", "dino_vits16")


def analyze_attention_mass_shift(
    a_true: np.ndarray,         # (I, N) head-weighted true [CLS] patch attention (rows sum to 1)
    a_s4_seeds: np.ndarray,     # (S, I, N) head-weighted S4 [CLS] patch attention (rows sum to 1)
    fg_masks: np.ndarray,       # (I, N) bool
    out_masks: np.ndarray,      # (I, N) bool (median + 5*MAD)
    patch_contrib: np.ndarray,  # (I, N) M3 Kobayashi contribution norms c_i
    gt_pixel_masks: list[np.ndarray] | None = None,
    grid: tuple[int, int] = (14, 14),
    target_size: int = 224,
    seed: int = 0,
) -> dict:
    """Exact 3-step mechanistic decomposition of delta_a = a_S4 - a_true (pure numpy, unit-testable)."""
    a_true = np.asarray(a_true, dtype=np.float64)
    a_s4_seeds = np.asarray(a_s4_seeds, dtype=np.float64)
    fg_m = np.asarray(fg_masks, dtype=bool)
    out_m = np.asarray(out_masks, dtype=bool)
    contrib = np.asarray(patch_contrib, dtype=np.float64)

    S, I, N = a_s4_seeds.shape
    a_s4_mean = a_s4_seeds.mean(axis=0)          # (I, N)
    delta_a = a_s4_mean - a_true                 # (I, N); row sums == 0.0

    # Disjoint M3 groups (OUT overrides FG/BG)
    g_out = out_m
    g_fg = fg_m & (~out_m)
    g_bg = (~fg_m) & (~out_m)

    # M3 high-BSS shortcut tokens inside BG: c_i > q90(c_i) per image (same rule as RALR Step 7)
    q90 = np.quantile(contrib, 0.90, axis=1, keepdims=True)
    g_bg_high = g_bg & (contrib > q90)
    g_bg_rest = g_bg & (~g_bg_high)

    # Step 2: per-image mass shifts by group
    def _group_stats(mask: np.ndarray) -> dict:
        true_mass = (a_true * mask).sum(axis=1)
        s4_mass = (a_s4_mean * mask).sum(axis=1)
        d_mass = (delta_a * mask).sum(axis=1)
        counts = mask.sum(axis=1)
        valid = counts > 0
        per_tok = np.where(valid, d_mass / np.maximum(counts, 1), np.nan)
        m_d, lo_d, hi_d = bootstrap_ci(d_mass, seed=seed)
        m_t, _, _ = bootstrap_ci(true_mass, seed=seed)
        m_s, _, _ = bootstrap_ci(s4_mass, seed=seed)
        # Seed SD of mean mass shift
        seed_means = [(a_s4_seeds[s] * mask - a_true * mask).sum(axis=1).mean() for s in range(S)]
        return {
            "mean_tokens": round(float(counts.mean()), 2),
            "true_mass_mean": round(float(m_t), 5),
            "s4_mass_mean": round(float(m_s), 5),
            "delta_mass_mean": round(float(m_d), 5),
            "delta_mass_ci95": [round(float(lo_d), 5), round(float(hi_d), 5)],
            "delta_mass_seed_sd": round(float(np.std(seed_means, ddof=1)) if S > 1 else 0.0, 6),
            "delta_per_token_mean": round(float(np.nanmean(per_tok)), 6) if valid.any() else 0.0,
            "n_images_with_group": int(valid.sum()),
        }

    # Step 3: Within-image Spearman rho(delta_a_i, c_i) across BG tokens + share of lost mass on shortcuts
    rhos_bg = []
    frac_lost_on_bg_high = []
    frac_lost_on_bg_high_or_out = []
    frac_lost_on_fg = []
    for i in range(I):
        neg = np.maximum(-delta_a[i], 0.0)
        tot_neg = float(neg.sum())
        if tot_neg > 1e-12:
            frac_lost_on_bg_high.append(float(neg[g_bg_high[i]].sum() / tot_neg))
            frac_lost_on_bg_high_or_out.append(float(neg[g_bg_high[i] | g_out[i]].sum() / tot_neg))
            frac_lost_on_fg.append(float(neg[g_fg[i]].sum() / tot_neg))
        if g_bg[i].sum() >= 5:
            r_val = spearmanr(delta_a[i, g_bg[i]], contrib[i, g_bg[i]]).statistic
            if np.isfinite(r_val):
                rhos_bg.append(float(r_val))

    rho_arr = np.array(rhos_bg, dtype=np.float64)
    rho_m, rho_lo, rho_hi = bootstrap_ci(rho_arr, seed=seed) if len(rho_arr) > 1 else (0.0, 0.0, 0.0)
    fl_bh_m, fl_bh_lo, fl_bh_hi = bootstrap_ci(np.array(frac_lost_on_bg_high), seed=seed)
    fl_bho_m, fl_bho_lo, fl_bho_hi = bootstrap_ci(np.array(frac_lost_on_bg_high_or_out), seed=seed)
    fl_fg_m, fl_fg_lo, fl_fg_hi = bootstrap_ci(np.array(frac_lost_on_fg), seed=seed)

    # Optional box-AUC verification on the exact same maps
    auc_info = {}
    if gt_pixel_masks is not None:
        gh, gw = grid
        gt_t = torch.from_numpy(np.stack(gt_pixel_masks, axis=0)).float()
        def _up(arr_2d: np.ndarray) -> torch.Tensor:
            t = torch.from_numpy(arr_2d.reshape(I, 1, gh, gw)).float()
            return F.interpolate(t, size=(target_size, target_size), mode="bilinear", align_corners=False).squeeze(1)

        t_true = _up(a_true)
        auc_true = box_auc_torch(t_true, gt_t).numpy()
        auc_s4_all = []
        for s in range(S):
            t_s4 = _up(a_s4_seeds[s])
            auc_s4_all.append(box_auc_torch(t_s4, gt_t).numpy())
        auc_s4_mean_img = np.mean(auc_s4_all, axis=0)
        d_auc = auc_s4_mean_img - auc_true
        da_m, da_lo, da_hi = bootstrap_ci(d_auc, seed=seed)
        auc_info = {
            "clean_box_auc": round(float(auc_true.mean()), 4),
            "s4_box_auc": round(float(auc_s4_mean_img.mean()), 4),
            "delta_box_auc": round(float(da_m), 4),
            "delta_box_auc_ci95": [round(float(da_lo), 4), round(float(da_hi), 4)],
        }

    # Conservation sanity check (must be ~0)
    max_row_sum_err = float(np.max(np.abs(delta_a.sum(axis=1))))

    return {
        "n_images": int(I),
        "n_seeds": int(S),
        "max_mass_conservation_err": max_row_sum_err,
        "box_auc_check": auc_info,
        "groups": {
            "FG": _group_stats(g_fg),
            "BG": _group_stats(g_bg),
            "OUT": _group_stats(g_out),
            "BG_high_c_top10pct": _group_stats(g_bg_high),
            "BG_rest": _group_stats(g_bg_rest),
        },
        "shortcut_loss_test": {
            "spearman_rho_delta_a_vs_contrib_in_BG": {
                "mean": round(float(rho_m), 4),
                "ci95": [round(float(rho_lo), 4), round(float(rho_hi), 4)],
                "n_images": int(len(rho_arr)),
            },
            "frac_of_lost_mass_on_BG_high_c": {
                "mean": round(float(fl_bh_m), 4),
                "ci95": [round(float(fl_bh_lo), 4), round(float(fl_bh_hi), 4)],
            },
            "frac_of_lost_mass_on_BG_high_c_or_OUT": {
                "mean": round(float(fl_bho_m), 4),
                "ci95": [round(float(fl_bho_lo), 4), round(float(fl_bho_hi), 4)],
            },
            "frac_of_lost_mass_on_FG": {
                "mean": round(float(fl_fg_m), 4),
                "ci95": [round(float(fl_fg_lo), 4), round(float(fl_fg_hi), 4)],
            },
        },
    }


@torch.no_grad()
def run_one_model(
    model_id: str,
    samples,
    device: torch.device,
    weights_cache: Path,
    *,
    batch_size: int = 32,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    k_pca: int = 32,
    smoke_random: bool = False,
) -> dict:
    t0 = time.perf_counter()
    adapter, model, mean, std, target_size = load_pilot_model(
        model_id, device, weights_cache, smoke_random=smoke_random,
    )
    if adapter.family != "timm_vit":
        raise ValueError(f"Expected timm_vit for {model_id}, got {adapter.family}")

    tf = build_image_transform(target_size, mean, std)
    n_img = len(samples)
    last_l = adapter.n_blocks(model) - 1

    # Precompute GT masks & patch FG masks
    gt_pixel_masks = []
    fg_patch_list = []
    grid = (target_size // 16, target_size // 16)
    imgs_tensor = []
    for s in samples:
        with Image.open(s.image_path) as im:
            rgb = im.convert("RGB")
            ow, oh = rgb.size
            imgs_tensor.append(tf(rgb))
        pmask, _ = get_transformed_imagenet_gt_mask(s, ow, oh, target_size)
        gt_pixel_masks.append(pmask.astype(bool))
        fg_patch_list.append(patch_fg_mask(pmask, grid, threshold=0.5).reshape(-1).astype(bool))

    fg_masks = np.stack(fg_patch_list, axis=0)  # (I, N)
    x_all = torch.stack(imgs_tensor, dim=0)

    q_g_list, k_p_list, w_h_list, a_true_list, out_list, contrib_list = [], [], [], [], [], []
    scale = None

    for start in range(0, n_img, batch_size):
        xb = x_all[start : start + batch_size].to(device)
        layout = adapter.token_layout(model, xb)
        cls_idx = layout["cls_idx"]
        patch_idx = layout["patch_idx"]
        grid = layout["grid"]

        z = model.patch_embed(xb)
        z = model._pos_embed(z)
        z = model.patch_drop(z)
        z = model.norm_pre(z)
        for l, blk in enumerate(model.blocks[: last_l + 1]):
            z_in = z
            z_out = blk(z_in)
            if l == last_l:
                a = blk.attn
                B, T, _ = z_in.shape
                qkv = a.qkv(blk.norm1(z_in)).reshape(B, T, 3, a.num_heads, a.head_dim).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                q, k = a.q_norm(q), a.k_norm(k)
                scale = float(a.scale)
                attn_row = attn_probs(q[:, :, cls_idx : cls_idx + 1], k, scale).squeeze(2)  # (B, H, T)
                W_O, _, gamma = get_output_proj_and_layerscale(adapter, model, l)
                w_h = compute_head_weights(attn_row, v, W_O, gamma=gamma)  # (B, H)
                c_all = routing.token_contributions(attn_row, v, W_O, gamma=gamma, return_vectors=False)  # (B, T)
                a_p = patch_attention(q[:, :, cls_idx, :], k[:, :, patch_idx, :], scale)  # (B, H, N)
                a_w = (a_p * w_h.unsqueeze(-1)).sum(dim=1)  # (B, N)
                # Outlier mask on post-block patch token norms (M3 rule: > median + 5*MAD)
                tok_norms = z_out.float().norm(dim=-1).cpu().numpy()  # (B, T)
                out_b, _ = routing.detect_outlier_tokens(tok_norms, patch_idx, k_mad=5.0)  # (B, N) bool

                q_g_list.append(q[:, :, cls_idx, :].float().cpu().numpy())
                k_p_list.append(k[:, :, patch_idx, :].float().cpu())
                w_h_list.append(w_h.float().cpu())
                a_true_list.append(a_w.float().cpu().numpy())
                out_list.append(out_b)
                contrib_list.append(c_all[:, patch_idx].float().cpu().numpy())
            z = z_out

    q_g_all = np.concatenate(q_g_list, axis=0)          # (I, H, d_h)
    k_p_all = torch.cat(k_p_list, dim=0)                # (I, H, N, d_h)
    w_h_all = torch.cat(w_h_list, dim=0)                # (I, H)
    a_true = np.concatenate(a_true_list, axis=0)        # (I, N)
    out_masks = np.concatenate(out_list, axis=0)        # (I, N)
    patch_contrib = np.concatenate(contrib_list, axis=0)

    # Fit exact same QueryPCA used in I1-S4 (k = 32)
    pca = qii.fit_query_pca(torch.from_numpy(q_g_all), k=k_pca)

    # Evaluate S4 surrogate queries across 5 seeds using exact same head weights w_h
    a_s4_seeds = []
    for sd in seeds:
        q_s4 = sample_surrogate_queries_batch("S4", q_g_all, pca, seed=sd)  # (I, H, d_h)
        q_s4_t = torch.from_numpy(q_s4)
        a_s4_chunks = []
        for start in range(0, n_img, batch_size):
            end = min(n_img, start + batch_size)
            ap = patch_attention(q_s4_t[start:end].to(device), k_p_all[start:end].to(device), scale)
            aw = (ap * w_h_all[start:end].to(device).unsqueeze(-1)).sum(dim=1)
            a_s4_chunks.append(aw.float().cpu().numpy())
        a_s4_seeds.append(np.concatenate(a_s4_chunks, axis=0))

    res = analyze_attention_mass_shift(
        a_true=a_true,
        a_s4_seeds=np.stack(a_s4_seeds, axis=0),
        fg_masks=fg_masks,
        out_masks=out_masks,
        patch_contrib=patch_contrib,
        gt_pixel_masks=gt_pixel_masks,
        grid=grid,
        target_size=target_size,
        seed=seeds[0],
    )
    res["model_id"] = model_id
    res["layer"] = int(last_l)
    res["elapsed_sec"] = round(time.perf_counter() - t0, 2)
    return res


def format_markdown(results: dict[str, dict]) -> str:
    lines = [
        "# Action A2 (§6.2): DeiT-III I1-S4 Per-Token Attention-Mass Shift Decomposition",
        "",
        "| Model | Clean AUC → S4 AUC (Δ) | Δmass FG [95% CI] | Δmass BG [95% CI] | Δmass OUT [95% CI] | Δmass BG_high_c (top 10% c_i) | ρ(Δa, c_i) in BG [95% CI] | % Lost Mass on BG_high_c+OUT |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mid, r in results.items():
        auc = r["box_auc_check"]
        g = r["groups"]
        st = r["shortcut_loss_test"]
        auc_s = f"{auc['clean_box_auc']:.3f}→{auc['s4_box_auc']:.3f} ({auc['delta_box_auc']:+.4f})" if auc else "—"
        lines.append(
            f"| `{mid}` | {auc_s} | "
            f"{g['FG']['delta_mass_mean']:+.4f} [{g['FG']['delta_mass_ci95'][0]:+.4f}, {g['FG']['delta_mass_ci95'][1]:+.4f}] | "
            f"{g['BG']['delta_mass_mean']:+.4f} [{g['BG']['delta_mass_ci95'][0]:+.4f}, {g['BG']['delta_mass_ci95'][1]:+.4f}] | "
            f"{g['OUT']['delta_mass_mean']:+.4f} [{g['OUT']['delta_mass_ci95'][0]:+.4f}, {g['OUT']['delta_mass_ci95'][1]:+.4f}] | "
            f"{g['BG_high_c_top10pct']['delta_mass_mean']:+.4f} | "
            f"{st['spearman_rho_delta_a_vs_contrib_in_BG']['mean']:+.3f} [{st['spearman_rho_delta_a_vs_contrib_in_BG']['ci95'][0]:+.3f}, {st['spearman_rho_delta_a_vs_contrib_in_BG']['ci95'][1]:+.3f}] | "
            f"{100 * st['frac_of_lost_mass_on_BG_high_c_or_OUT']['mean']:.1f}% |"
        )
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Action A2 (§6.2): DeiT-III I1-S4 attention-mass decomposition.")
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--imagenet_xml_dir", required=True)
    ap.add_argument("--split_file", default=str(REPO_ROOT / "configs" / "splits" / "imagenet_pilot_1000_seed0.txt"))
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--out_dir", default="results/deit3_i1_s4_mass_shift")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--weights_cache_dir", default=str(Path.home() / ".cache" / "queryblind_weights"))
    ap.add_argument("--smoke_random", action="store_true")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = load_imagenet_val_pilot_split(
        args.images_dir, args.imagenet_xml_dir,
        n_images=args.limit or 1000, seed=0, split_file=args.split_file,
    )
    if args.limit is not None:
        samples = samples[: args.limit]
    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]

    results = {}
    for mid in model_ids:
        print(f"[A2] Running {mid} on {len(samples)} images (device={device})...", flush=True)
        results[mid] = run_one_model(
            mid, samples, device, Path(args.weights_cache_dir),
            batch_size=args.batch_size, smoke_random=args.smoke_random,
        )

    payload = {
        "git_sha": _git_sha(),
        "dataset": "imagenet_1k",
        "n_images": len(samples),
        "models": results,
    }
    (out_dir / "deit3_i1_s4_mass_shift.json").write_text(json.dumps(payload, indent=2))
    md = format_markdown(results)
    (out_dir / "summary_deit3_i1_s4_mass_shift.md").write_text(md)
    print("\n" + md, flush=True)
    print("=== COPY THIS JSON BACK ===")
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    main()
