#!/usr/bin/env python3
"""Figure 5 (H5): RALR vs. [CLS]@l* on primary box-AUC and secondary Otsu CorLoc@0.5 (matched n=11 excl. MAE).

Reads verbatim from:
  - <results_dir>/zoo12_ralr_cub/ralr_report.json
  - <results_dir>/zoo12_ralr_in1k/ralr_report.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import (
    FAMILY_COLORS,
    MODEL_FAMILY,
    MODEL_SHORT,
    ZOO12_ORDER,
    apply_pub_style,
    parse_fig_args,
    save_figure,
)


def _plot_dumbbell_panel(ax, report: dict, models: list[str], title: str) -> None:
    y = np.arange(len(models))
    off = 0.17

    auc_wins = 0
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

        # box-AUC dumbbell (top row, navy)
        c_auc = "#1D4ED8" if ralr_auc >= cls_auc else "#DC2626"
        ax.plot([cls_auc, ralr_auc], [i - off, i - off], color=c_auc, lw=1.8, alpha=0.85)
        ax.scatter(cls_auc, i - off, s=36, facecolors="white", edgecolors="#1D4ED8", lw=1.3, zorder=4)
        ax.scatter(ralr_auc, i - off, s=42, color=c_auc, edgecolors="#0F172A", lw=0.6, zorder=5)

        # Otsu dumbbell (bottom row, teal/emerald)
        c_ot = "#0F766E" if ralr_otsu >= cls_otsu else "#DC2626"
        ax.plot([cls_otsu, ralr_otsu], [i + off, i + off], color=c_ot, lw=1.8, alpha=0.85)
        ax.scatter(cls_otsu, i + off, s=36, marker="s", facecolors="white", edgecolors="#0F766E", lw=1.3, zorder=4)
        ax.scatter(ralr_otsu, i + off, s=42, marker="s", color=c_ot, edgecolors="#0F172A", lw=0.6, zorder=5)

    p_auc = binomtest(auc_wins, len(models), 0.5, alternative="two-sided").pvalue
    p_otsu = binomtest(otsu_wins, len(models), 0.5, alternative="two-sided").pvalue

    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_SHORT[m] for m in models], fontsize=8.1)
    ax.invert_yaxis()
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("Score (Hollow = [CLS]@l*, Filled = RALR)")
    ax.set_title(
        f"{title}\nbox-AUC: {auc_wins}/{len(models)} (p={p_auc:.4f})  |  Otsu: {otsu_wins}/{len(models)} (p={p_otsu:.4f})",
        fontsize=9.3,
    )


def main() -> None:
    args = parse_fig_args("Generate Figure 5: RALR vs [CLS]@l* (box-AUC and Otsu CorLoc@0.5).")
    apply_pub_style()

    cub = json.loads((args.results_dir / "zoo12_ralr_cub" / "ralr_report.json").read_text())
    in1k = json.loads((args.results_dir / "zoo12_ralr_in1k" / "ralr_report.json").read_text())

    # Matched n = 11 non-degenerate models (excluding MAE) across BOTH datasets
    non_mae = [m for m in ZOO12_ORDER if m != "mae_vitb16"]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.4), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.05, 1.05, 1.1]})

    _plot_dumbbell_panel(axes[0], cub, non_mae, "A. CUB-200-2011 (N = 500, n = 11)")
    _plot_dumbbell_panel(axes[1], in1k, non_mae, "B. ImageNet-1K (N = 1,000, n = 11)")
    axes[1].set_yticklabels([])

    # Shared legend below figure so Panels A & B remain 100% clear
    leg_elems = [
        Line2D([0], [0], marker="o", color="#1D4ED8", mfc="white", mec="#1D4ED8", lw=0, ms=6, label="Primary box-AUC: [CLS]@l*"),
        Line2D([0], [0], marker="o", color="#1D4ED8", mfc="#1D4ED8", mec="#0F172A", lw=0, ms=6, label="Primary box-AUC: RALR (red = loss)"),
        Line2D([0], [0], marker="s", color="#0F766E", mfc="white", mec="#0F766E", lw=0, ms=6, label="Secondary Otsu: [CLS]@l*"),
        Line2D([0], [0], marker="s", color="#0F766E", mfc="#0F766E", mec="#0F172A", lw=0, ms=6, label="Secondary Otsu: RALR"),
    ]
    fig.legend(handles=leg_elems, loc="lower center", bbox_to_anchor=(0.35, -0.065), ncol=4, fontsize=7.5)

    # --- Panel C: Mean Zero-Box-Tuning Otsu CorLoc@0.5 Ladder (matched n = 11) ---
    ax = axes[2]
    methods = [
        ("cls_last", "[CLS]@last"),
        ("cls_l_star", "[CLS]@l*"),
        ("cls_l_star_clip98", "[CLS]@l* clip98"),
        ("cls_l_star_no_out", "[CLS]@l* no_out"),
        ("lost_l_star", "Cont. LOST*"),
        ("tokencut_l_star", "Cont. TokenCut*"),
        ("ralr_adaptive", "RALR (ours)"),
    ]
    cub_means = [float(np.mean([cub["models"][m]["methods"][k]["corloc_otsu"]["mean"] for m in non_mae])) for k, _ in methods]
    in1k_means = [float(np.mean([in1k["models"][m]["methods"][k]["corloc_otsu"]["mean"] for m in non_mae])) for k, _ in methods]

    y = np.arange(len(methods))
    h = 0.36
    bars_cub = ax.barh(y - h / 2, cub_means, height=h, color="#1B5E8A", alpha=0.90, edgecolor="#0F172A", lw=0.6, label="CUB-500 (n = 11)")
    bars_in = ax.barh(y + h / 2, in1k_means, height=h, color="#D97706", alpha=0.88, edgecolor="#0F172A", lw=0.6, label="ImageNet-1K (n = 11)")

    # Emphasize RALR row
    bars_cub[-1].set_edgecolor("#0F172A")
    bars_cub[-1].set_linewidth(1.5)
    bars_in[-1].set_edgecolor("#0F172A")
    bars_in[-1].set_linewidth(1.5)

    for i in range(len(methods)):
        fw = "bold" if i == len(methods) - 1 else "normal"
        ax.text(cub_means[i] + 0.012, i - h / 2, f"{cub_means[i]:.3f}", va="center", fontsize=7.6, fontweight=fw)
        ax.text(in1k_means[i] + 0.012, i + h / 2, f"{in1k_means[i]:.3f}", va="center", fontsize=7.6, fontweight=fw)

    ax.set_yticks(y)
    ax.set_yticklabels([lbl for _, lbl in methods], fontsize=8.3)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 0.90)
    ax.set_xlabel("Mean Zero-Box-Tuning Otsu CorLoc@0.5 (tunes_on_boxes = NO)")
    ax.set_title("C. Secondary Otsu Baseline Ladder (n = 11)\n(*Continuous adaptations; cannot rescue primary H5)", fontsize=9.1)
    ax.legend(loc="upper right", fontsize=7.6)

    fig.suptitle(
        "Figure 5. RALR vs. [CLS]@l* & Zero-Box-Tuning Baselines (Matched n = 11 excl. MAE; H5 Fails Gate G3 on IN-1K Primary box-AUC)",
        fontsize=10.8, fontweight="bold", y=1.03,
    )
    save_figure(fig, args.out_dir, "fig5_ralr_vs_cls")


if __name__ == "__main__":
    main()
