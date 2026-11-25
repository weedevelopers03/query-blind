#!/usr/bin/env python3
"""Figure 5 (H5 & Gate G3): RALR vs. [CLS]@l* across all 3 benchmarks + 3-benchmark Floor-Raiser Law (n=11 excl. MAE).

Reads verbatim from:
  - <results_dir>/zoo12_ralr_cub_full5794/ralr_cub_full5794_report.json (fallback: zoo12_ralr_cub/ralr_report.json)
  - <results_dir>/zoo12_ralr_in1k/ralr_report.json
  - <results_dir>/zoo12_ralr_openimages30k/ralr_openimages30k_report.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import binomtest, linregress

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import (
    MODEL_SHORT,
    ZOO12_ORDER,
    apply_pub_style,
    parse_fig_args,
    save_figure,
)


def _plot_dumbbell_panel(ax, report: dict, models: list[str], title: str, show_pxap: bool = False) -> None:
    y = np.arange(len(models))
    off = 0.18

    auc_wins = 0
    px_wins = 0
    otsu_wins = 0
    for i, mid in enumerate(models):
        m = report["models"][mid]["methods"]
        cls_auc = m["cls_l_star"]["box_auc"]["mean"]
        ralr_auc = m["ralr_adaptive"]["box_auc"]["mean"]
        cls_otsu = m["cls_l_star"]["corloc_otsu"]["mean"]
        ralr_otsu = m["ralr_adaptive"]["corloc_otsu"]["mean"]
        if ralr_auc > cls_auc:
            auc_wins += 1
        if ralr_otsu > cls_otsu:
            otsu_wins += 1
        if show_pxap and m["ralr_adaptive"]["pxap"]["mean"] > m["cls_l_star"]["pxap"]["mean"]:
            px_wins += 1

        # Primary box-AUC dumbbell (top row, navy / red loss)
        c_auc = "#1D4ED8" if ralr_auc >= cls_auc else "#DC2626"
        ax.plot([cls_auc, ralr_auc], [i - off, i - off], color=c_auc, lw=1.8, alpha=0.88)
        ax.scatter(cls_auc, i - off, s=34, facecolors="white", edgecolors="#1D4ED8", lw=1.2, zorder=4)
        ax.scatter(ralr_auc, i - off, s=40, color=c_auc, edgecolors="#0F172A", lw=0.6, zorder=5)

        # Secondary Otsu CorLoc@0.5 dumbbell (bottom row, teal)
        c_ot = "#0F766E" if ralr_otsu >= cls_otsu else "#DC2626"
        ax.plot([cls_otsu, ralr_otsu], [i + off, i + off], color=c_ot, lw=1.8, alpha=0.88)
        ax.scatter(cls_otsu, i + off, s=34, marker="s", facecolors="white", edgecolors="#0F766E", lw=1.2, zorder=4)
        ax.scatter(ralr_otsu, i + off, s=40, marker="s", color=c_ot, edgecolors="#0F172A", lw=0.6, zorder=5)

    p_auc = binomtest(auc_wins, len(models), 0.5, alternative="two-sided").pvalue
    p_otsu = binomtest(otsu_wins, len(models), 0.5, alternative="two-sided").pvalue
    sub = f"box-AUC: {auc_wins}/11 (p={p_auc:.3f}) | Otsu: {otsu_wins}/11 (p={p_otsu:.4f})"
    if show_pxap:
        p_px = binomtest(px_wins, len(models), 0.5, alternative="two-sided").pvalue
        sub = f"box-AUC: {auc_wins}/11 (p={p_auc:.3f}) | PxAP: {px_wins}/11 (p={p_px:.3f}) | Otsu: {otsu_wins}/11"

    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_SHORT[m] for m in models], fontsize=8.0)
    ax.invert_yaxis()
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("Score (Hollow = [CLS]@l*, Filled = RALR)")
    ax.set_title(f"{title}\n{sub}", fontsize=8.8)


def main() -> None:
    args = parse_fig_args("Generate Figure 5: RALR vs [CLS]@l* across 3 benchmarks + Floor-Raiser regression.")
    apply_pub_style()

    cub_full = args.results_dir / "zoo12_ralr_cub_full5794" / "ralr_cub_full5794_report.json"
    cub_path = cub_full if cub_full.exists() else (args.results_dir / "zoo12_ralr_cub" / "ralr_report.json")
    cub = json.loads(cub_path.read_text())
    cub_n = cub.get("n_images_full", 500)
    cub_lbl = f"CUB-{cub_n:,}"
    in1k = json.loads((args.results_dir / "zoo12_ralr_in1k" / "ralr_report.json").read_text())
    oi = json.loads((args.results_dir / "zoo12_ralr_openimages30k" / "ralr_openimages30k_report.json").read_text())

    non_mae = [m for m in ZOO12_ORDER if m != "mae_vitb16"]

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.2), constrained_layout=True)

    _plot_dumbbell_panel(
        axes[0, 0], cub, non_mae,
        f"A. CUB-200-2011 Full Test (N = {cub_n:,}, n = 11)",
        show_pxap=(cub_n >= 5000),
    )
    _plot_dumbbell_panel(axes[0, 1], in1k, non_mae, "B. ImageNet-1K (N = 1,000, n = 11)")
    axes[0, 1].set_yticklabels([])

    _plot_dumbbell_panel(axes[1, 0], oi, non_mae, "C. OpenImages30k (N = 4,999 masks, n = 11)", show_pxap=True)

    # --- Panel D: 3-Benchmark RALR Floor-Raiser Regression (Delta box-AUC vs Clean Baseline box-AUC) ---
    ax = axes[1, 1]
    ax.axhline(0.0, color="#64748B", ls="--", lw=1.1, zorder=1)

    benches = [
        (cub_lbl, cub, "#1B5E8A", "o"),
        ("ImageNet-1K", in1k, "#D97706", "s"),
        ("OpenImages30k", oi, "#0F766E", "^"),
    ]
    x_grid = np.linspace(0.57, 0.94, 100)
    for bname, rep, col, mk in benches:
        xs = np.array([rep["models"][m]["methods"]["cls_l_star"]["box_auc"]["mean"] for m in non_mae], dtype=float)
        ys = np.array([
            rep["models"][m]["methods"]["ralr_adaptive"]["box_auc"]["mean"]
            - rep["models"][m]["methods"]["cls_l_star"]["box_auc"]["mean"]
            for m in non_mae
        ], dtype=float)
        reg = linregress(xs, ys)
        r2 = reg.rvalue ** 2
        lbl = f"{bname}: slope={reg.slope:.2f}, $R^2$={r2:.2f} (p={reg.pvalue:.4f})"
        ax.scatter(xs, ys, color=col, marker=mk, s=52, alpha=0.88, edgecolors="#0F172A", lw=0.6, zorder=4)
        x_line = np.linspace(xs.min() - 0.015, xs.max() + 0.015, 50)
        ax.plot(x_line, reg.slope * x_line + reg.intercept, color=col, lw=2.0, label=lbl, zorder=3)

    ax.set_xlim(0.57, 1.00)
    ax.set_ylim(-0.042, 0.165)
    ax.set_xlabel("Clean Baseline [CLS]@l* box-AUC")
    ax.set_ylabel("RALR Gain $\\Delta$box-AUC (RALR $-$ [CLS]@l*)")
    ax.set_title("D. Dataset-Invariant Floor-Raiser Law (n = 11 per benchmark)\nRALR lifts weak baselines (+5–11 pt) & saturates near ceiling", fontsize=8.8)
    ax.legend(loc="upper right", fontsize=7.4, framealpha=0.95)

    # Annotate gain vs ceiling zones in open whitespace
    ax.text(0.585, 0.116, "Weak / contaminated\nbaselines: +5 to +11 pt", fontsize=7.5, color="#0F766E", fontweight="bold")
    ax.text(0.865, -0.032, "Strong SSL ceiling:\n~0 to -0.02 pt", fontsize=7.5, color="#B91C1C", fontweight="bold")

    # Shared legend for Panels A-C at bottom
    leg_elems = [
        Line2D([0], [0], marker="o", color="#1D4ED8", mfc="white", mec="#1D4ED8", lw=0, ms=6, label="Primary box-AUC: [CLS]@l*"),
        Line2D([0], [0], marker="o", color="#1D4ED8", mfc="#1D4ED8", mec="#0F172A", lw=0, ms=6, label="Primary box-AUC: RALR (red = loss)"),
        Line2D([0], [0], marker="s", color="#0F766E", mfc="white", mec="#0F766E", lw=0, ms=6, label="Zero-tuning Otsu: [CLS]@l*"),
        Line2D([0], [0], marker="s", color="#0F766E", mfc="#0F766E", mec="#0F172A", lw=0, ms=6, label="Zero-tuning Otsu: RALR (33/33 wins)"),
    ]
    fig.legend(handles=leg_elems, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=4, fontsize=7.8)

    fig.suptitle(
        "RALR Across All Three Benchmarks (Matched n = 11 excl. MAE) — Gate G3 Verdict & Floor-Raiser Characterization",
        fontsize=10.8, fontweight="bold", y=1.02,
    )
    save_figure(fig, args.out_dir, "fig5_ralr_vs_cls")


if __name__ == "__main__":
    main()
