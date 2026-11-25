#!/usr/bin/env python3
"""Figure 7 (Action A2, §6.2 — Thesis Defense Mechanism Figure):
Causal per-token [CLS] attention-mass redistribution under I1-S4 on DeiT-III vs. DINO S/16 control.

Reads verbatim from:
  - <results_dir>/deit3_i1_s4_mass_shift/deit3_i1_s4_mass_shift.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import apply_pub_style, parse_fig_args, save_figure


def main() -> None:
    args = parse_fig_args("Generate Figure 7: Action A2 DeiT-III I1-S4 causal attention-mass shift.")
    apply_pub_style()

    p = args.results_dir / "deit3_i1_s4_mass_shift" / "deit3_i1_s4_mass_shift.json"
    data = json.loads(p.read_text())["models"]

    order = [
        ("deit3_small_patch16_224", "DeiT-III S/16\n(IN-1K)"),
        ("deit3_small_patch16_224_in21k", "DeiT-III S/16*\n(IN-21K)"),
        ("dino_vits16", "DINO S/16\n(SSL Control)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.7), constrained_layout=True)

    # --- Panel A: Net Attention Mass Shift by Disjoint Group (FG, BG_high_c, BG_rest, OUT) ---
    ax = axes[0]
    x = np.arange(len(order))
    w = 0.20
    groups = [
        ("FG", "FG (Object)", "#0F766E"),
        ("OUT", "OUT (High-Norm Sinks)", "#DC2626"),
        ("BG_high_c_top10pct", "BG_high_c (Top-10% M3)", "#D97706"),
        ("BG_rest", "BG_rest", "#94A3B8"),
    ]
    for g_idx, (gk, glbl, gcol) in enumerate(groups):
        vals = [data[m]["groups"][gk]["delta_mass_mean"] * 100 for m, _ in order]
        los = [(data[m]["groups"][gk]["delta_mass_mean"] - data[m]["groups"][gk]["delta_mass_ci95"][0]) * 100 for m, _ in order]
        his = [(data[m]["groups"][gk]["delta_mass_ci95"][1] - data[m]["groups"][gk]["delta_mass_mean"]) * 100 for m, _ in order]
        pos = x + (g_idx - 1.5) * w
        ax.bar(pos, vals, width=w, color=gcol, edgecolor="#0F172A", lw=0.65, label=glbl)
        ax.errorbar(pos, vals, yerr=[los, his], fmt="none", ecolor="#0F172A", elinewidth=0.9, capsize=2)
        for i, v in enumerate(vals):
            if abs(v) >= 1.4:
                dy = 0.7 if v >= 0 else -0.9
                va = "bottom" if v >= 0 else "top"
                ax.text(pos[i], v + (his[i] if v >= 0 else -los[i]) + dy, f"{v:+.1f}%",
                        ha="center", va=va, fontsize=6.9, fontweight="bold", color="#0F172A")

    ax.axhline(0.0, color="#0F172A", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in order])
    ax.set_ylim(-20.5, 21.5)
    ax.set_ylabel("Net [CLS] Attention Mass Shift (S4 − True, %)")
    ax.set_title("A. Where Attention Moves Under I1-S4 (l = 11)")
    ax.legend(loc="lower right", fontsize=7.0, ncol=1)

    # --- Panel B: True vs. S4 Total Mass Breakdown (Stacked: FG / BG / OUT) ---
    ax = axes[1]
    w2 = 0.32
    for i, (m, _) in enumerate(order):
        g = data[m]["groups"]
        for k_idx, (key, dx, alpha) in enumerate([("true_mass_mean", -w2 / 2 - 0.02, 0.92), ("s4_mass_mean", w2 / 2 + 0.02, 0.62)]):
            fg = g["FG"][key] * 100
            bg = g["BG"][key] * 100
            ou = g["OUT"][key] * 100
            lbl_fg = "FG" if (i == 0 and k_idx == 0) else None
            lbl_bg = "BG" if (i == 0 and k_idx == 0) else None
            lbl_ou = "OUT" if (i == 0 and k_idx == 0) else None
            ax.bar(i + dx, fg, w2, color="#0F766E", alpha=alpha, edgecolor="#0F172A", lw=0.6, label=lbl_fg)
            ax.bar(i + dx, bg, w2, bottom=fg, color="#F59E0B", alpha=alpha, edgecolor="#0F172A", lw=0.6, label=lbl_bg)
            ax.bar(i + dx, ou, w2, bottom=fg + bg, color="#DC2626", alpha=alpha, edgecolor="#0F172A", lw=0.6, label=lbl_ou)
            ax.text(i + dx, 102, "True" if k_idx == 0 else "S4", ha="center", va="bottom", fontsize=7.2,
                    fontweight="bold" if k_idx == 0 else "normal", color="#334155")
            if fg >= 15:
                ax.text(i + dx, fg / 2, f"{fg:.0f}%", ha="center", va="center", fontsize=7.3, fontweight="bold", color="white")
            if ou >= 10:
                ax.text(i + dx, fg + bg + ou / 2, f"{ou:.0f}%", ha="center", va="center", fontsize=7.3, fontweight="bold", color="white")

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in order])
    ax.set_ylim(0, 114)
    ax.set_ylabel("Share of Total Patch Attention Mass (%)")
    ax.set_title("B. True Query vs. In-Manifold S4 Mass Composition")
    ax.legend(loc="lower right", fontsize=7.5, ncol=3)

    # --- Panel C: Where Does LOST Attention Mass Come From? (BG_high_c + OUT vs FG) ---
    ax = axes[2]
    w3 = 0.28
    lost_bho = [data[m]["shortcut_loss_test"]["frac_of_lost_mass_on_BG_high_c_or_OUT"]["mean"] * 100 for m, _ in order]
    lost_bho_lo = [(data[m]["shortcut_loss_test"]["frac_of_lost_mass_on_BG_high_c_or_OUT"]["mean"] -
                    data[m]["shortcut_loss_test"]["frac_of_lost_mass_on_BG_high_c_or_OUT"]["ci95"][0]) * 100 for m, _ in order]
    lost_bho_hi = [(data[m]["shortcut_loss_test"]["frac_of_lost_mass_on_BG_high_c_or_OUT"]["ci95"][1] -
                    data[m]["shortcut_loss_test"]["frac_of_lost_mass_on_BG_high_c_or_OUT"]["mean"]) * 100 for m, _ in order]

    lost_fg = [data[m]["shortcut_loss_test"]["frac_of_lost_mass_on_FG"]["mean"] * 100 for m, _ in order]
    lost_fg_lo = [(data[m]["shortcut_loss_test"]["frac_of_lost_mass_on_FG"]["mean"] -
                   data[m]["shortcut_loss_test"]["frac_of_lost_mass_on_FG"]["ci95"][0]) * 100 for m, _ in order]
    lost_fg_hi = [(data[m]["shortcut_loss_test"]["frac_of_lost_mass_on_FG"]["ci95"][1] -
                   data[m]["shortcut_loss_test"]["frac_of_lost_mass_on_FG"]["mean"]) * 100 for m, _ in order]

    ax.bar(x - w3 / 2, lost_bho, w3, color="#DC2626", edgecolor="#0F172A", lw=0.7, label="Lost from OUT + BG_high_c")
    ax.errorbar(x - w3 / 2, lost_bho, yerr=[lost_bho_lo, lost_bho_hi], fmt="none", ecolor="#0F172A", lw=1.0, capsize=2.5)
    ax.bar(x + w3 / 2, lost_fg, w3, color="#1B5E8A", edgecolor="#0F172A", lw=0.7, label="Lost from FG (Object)")
    ax.errorbar(x + w3 / 2, lost_fg, yerr=[lost_fg_lo, lost_fg_hi], fmt="none", ecolor="#0F172A", lw=1.0, capsize=2.5)

    for i in range(len(order)):
        ax.text(x[i] - w3 / 2, lost_bho[i] + lost_bho_hi[i] + 1.8, f"{lost_bho[i]:.1f}%",
                ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#991B1B")
        ax.text(x[i] + w3 / 2, lost_fg[i] + lost_fg_hi[i] + 1.8, f"{lost_fg[i]:.1f}%",
                ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#1B5E8A")

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in order])
    ax.set_ylim(0, 108)
    ax.set_ylabel("Share of Total Negative Shift max(−Δa, 0) (%)")
    ax.set_title("C. Source of Lost Mass: Shortcuts vs. Object")
    ax.legend(loc="upper center", fontsize=7.4, ncol=2)

    fig.suptitle(
        "Causal Mechanism of DeiT-III I1-S4 Gain (ImageNet-1K, N = 1,000, 5 Seeds, Exploratory Single-Recipe Analysis)",
        fontsize=10.8, fontweight="bold", y=1.03,
    )
    save_figure(fig, args.out_dir, "fig7_deit3_i1_s4_mechanism")


if __name__ == "__main__":
    main()
