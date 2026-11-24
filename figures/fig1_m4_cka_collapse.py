#!/usr/bin/env python3
"""Figure 1 (Main Result — H3): M4 CKA collapse vs. split-half noise floor and 12-model C1_rand control.

Reads verbatim from:
  - <results_dir>/zoo12_similarity_m4_in1k/similarity_m4_report.json
  - <results_dir>/zoo12_similarity_m4_in1k/zoo12_m4_c1_rand_in1k.json
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
    args = parse_fig_args("Generate Figure 1: M4 CKA collapse vs noise floor & C1_rand control.")
    apply_pub_style()

    m4_path = args.results_dir / "zoo12_similarity_m4_in1k" / "similarity_m4_report.json"
    cr_path = args.results_dir / "zoo12_similarity_m4_in1k" / "zoo12_m4_c1_rand_in1k.json"
    m4 = json.loads(m4_path.read_text())
    cr = json.loads(cr_path.read_text())["m4_matched_random_drop"]["imagenet_1k"]

    layers = ["l8", "l9", "l10", "l11"]
    layer_fracs = [9 / 12, 10 / 12, 11 / 12, 12 / 12]
    x = np.arange(len(layers))
    x_labels = ["l=8\n(0.75)", "l=9\n(0.83)", "l=10\n(0.92)", "l=11\n(1.00)"]

    def get_metric(lv: str, metric: str, key: str):
        return [m4["layers"][l]["levels"][lv]["metrics"][metric][key] for l in layers]

    # Panel A data (R3 patch tokens)
    r3_cka = get_metric("R3", "cka", "tau_c0_c1_full")
    r3_noise = [m4["layers"][l]["levels"]["R3"]["metrics"]["cka"]["tau_noise"]["mean"] for l in layers]
    r3_noise_lo = [m4["layers"][l]["levels"]["R3"]["metrics"]["cka"]["tau_noise"]["ci95_low"] for l in layers]
    r3_noise_hi = [m4["layers"][l]["levels"]["R3"]["metrics"]["cka"]["tau_noise"]["ci95_high"] for l in layers]
    r3_rand = [cr[l]["tau_c0_c1_rand_mean"] for l in layers]
    r3_rand_sd = [cr[l]["tau_c0_c1_rand_std"] for l in layers]
    r3_mknn = get_metric("R3", "mknn", "tau_c0_c1_full")
    r3_proc = get_metric("R3", "procrustes", "tau_c0_c1_full")

    # Panel B data (R2 mean-pooled patches)
    r2_cka = get_metric("R2", "cka", "tau_c0_c1_full")
    r2_noise = [m4["layers"][l]["levels"]["R2"]["metrics"]["cka"]["tau_noise"]["mean"] for l in layers]
    r2_noise_lo = [m4["layers"][l]["levels"]["R2"]["metrics"]["cka"]["tau_noise"]["ci95_low"] for l in layers]
    r2_noise_hi = [m4["layers"][l]["levels"]["R2"]["metrics"]["cka"]["tau_noise"]["ci95_high"] for l in layers]
    r2_mknn = get_metric("R2", "mknn", "tau_c0_c1_full")
    r2_proc = get_metric("R2", "procrustes", "tau_c0_c1_full")

    # Panel C data (Grounding: Spearman rho with localization map IoU)
    cka_iou_c0 = [d["spearman_rho"] for d in get_metric("R3", "cka", "map_iou_spearman_c0")]
    cka_iou_c1 = [d["spearman_rho"] for d in get_metric("R3", "cka", "map_iou_spearman_c1")]
    mknn_iou_c0 = [d["spearman_rho"] for d in get_metric("R3", "mknn", "map_iou_spearman_c0")]
    mknn_iou_c1 = [d["spearman_rho"] for d in get_metric("R3", "mknn", "map_iou_spearman_c1")]

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6), constrained_layout=True)

    # --- Panel A: R3 Patch-Level CKA vs Noise Floor & C1_rand ---
    ax = axes[0]
    ax.fill_between(x, r3_noise_lo, r3_noise_hi, color="#94A3B8", alpha=0.25, label="Split-half $\\tau_{\\mathrm{noise}}$ (95% CI)")
    ax.plot(x, r3_noise, "--", color="#475569", lw=1.6, marker="s", ms=4.5)
    ax.errorbar(
        x, r3_rand, yerr=r3_rand_sd, fmt="-^", color="#0F766E", lw=2.0, ms=6.5, capsize=3,
        label="$\\tau$(C0, C1$_{\\mathrm{rand}}$) (5-seed control)",
    )
    ax.plot(x, r3_mknn, "-D", color="#2563EB", lw=1.8, ms=5.5, label="Mutual $k$-NN ($k=10$) $\\tau$(C0, C1)")
    ax.plot(x, r3_proc, ":o", color="#7C3AED", lw=1.5, ms=4.8, label="Procrustes ($d^*=256$) $\\tau$(C0, C1)")
    ax.plot(x, r3_cka, "-o", color="#C0392B", lw=2.5, ms=7.0, label="Debiased CKA $\\tau$(C0, C1$_{\\mathrm{OUT}}$)")

    for i, val in enumerate(r3_cka):
        ax.annotate(f"{val:.3f}", (x[i], val), textcoords="offset points", xytext=(0, -14),
                    ha="center", fontsize=8.0, fontweight="bold", color="#991B1B")

    # Annotate the l8 gap cleanly with vertical bracket between C1_OUT and C1_rand
    ax.annotate(
        "", xy=(0.0, r3_rand[0] - 0.02), xytext=(0.0, r3_cka[0] + 0.03),
        arrowprops=dict(arrowstyle="<->", color="#C0392B", lw=1.4),
    )
    ax.text(
        0.12, 0.56,
        "OUT vs. rand gap:\n$-0.836$ to $-0.631$\n(req. $\\geq 0.10$)",
        fontsize=7.7, color="#7F1D1D", va="center",
        bbox=dict(boxstyle="round,pad=0.35", fc="#FEF2F2", ec="#FCA5A5", lw=0.9),
    )
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(-0.02, 1.07)
    ax.set_ylabel("Pairwise Rank Correlation (Kendall's $\\tau_b$)")
    ax.set_xlabel("Transformer Block (Layer Fraction)")
    ax.set_title("A. Patch Tokens (R3): CKA Collapse vs. Controls")
    ax.legend(loc="lower right", fontsize=7.4)

    # --- Panel B: R2 Mean-Pooled Patches (showing 3/4 collapse + l11:R2 exception) ---
    ax = axes[1]
    ax.fill_between(x, r2_noise_lo, r2_noise_hi, color="#94A3B8", alpha=0.25, label="Split-half $\\tau_{\\mathrm{noise}}$ (95% CI)")
    ax.plot(x, r2_noise, "--", color="#475569", lw=1.6, marker="s", ms=4.5)
    ax.plot(x, r2_mknn, "-D", color="#2563EB", lw=1.8, ms=5.5, label="Mutual $k$-NN $\\tau$(C0, C1)")
    ax.plot(x, r2_proc, ":o", color="#7C3AED", lw=1.5, ms=4.8, label="Procrustes $\\tau$(C0, C1)")
    ax.plot(x, r2_cka, "-o", color="#C0392B", lw=2.5, ms=7.0, label="Debiased CKA $\\tau$(C0, C1$_{\\mathrm{OUT}}$)")

    for i, val in enumerate(r2_cka):
        dy = -14 if i < 3 else -16
        dx = 0 if i < 3 else -10
        ax.annotate(f"{val:.3f}", (x[i], val), textcoords="offset points", xytext=(dx, dy),
                    ha="center", fontsize=8.0, fontweight="bold",
                    color="#991B1B" if i < 3 else "#1E293B")

    ax.annotate(
        "l11:R2 does NOT collapse\n(7/8 cells pass 8× Bonf.;\n$p_{\\mathrm{adj}}=0.0032$)",
        xy=(3, r2_cka[3]), xytext=(1.05, 0.20),
        fontsize=7.7, color="#1E293B",
        bbox=dict(boxstyle="round,pad=0.35", fc="#F8FAFC", ec="#94A3B8", lw=0.9),
        arrowprops=dict(arrowstyle="->", color="#475569", lw=1.1),
    )
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(-0.02, 1.07)
    ax.set_ylabel("Pairwise Rank Correlation (Kendall's $\\tau_b$)")
    ax.set_xlabel("Transformer Block (Layer Fraction)")
    ax.set_title("B. Mean-Pooled Patches (R2): 3/4 Layers Collapse")
    ax.legend(loc="lower left", fontsize=7.4)

    # --- Panel C: Grounding — Spearman rho(Similarity, Localization Map IoU) ---
    ax = axes[2]
    w = 0.19
    ax.bar(x - 1.5 * w, cka_iou_c0, width=w, color="#F87171", edgecolor="#991B1B", lw=0.7, label="R3 CKA (Raw C0)")
    ax.bar(x - 0.5 * w, cka_iou_c1, width=w, color="#991B1B", edgecolor="#450A0A", lw=0.7, label="R3 CKA (OUT-Masked C1)")
    ax.bar(x + 0.5 * w, mknn_iou_c0, width=w, color="#93C5FD", edgecolor="#1D4ED8", lw=0.7, label="R3 mKNN (Raw C0)")
    ax.bar(x + 1.5 * w, mknn_iou_c1, width=w, color="#1D4ED8", edgecolor="#1E3A8A", lw=0.7, label="R3 mKNN (OUT-Masked C1)")
    ax.axhline(0.0, color="#334155", lw=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(-0.10, 1.05)
    ax.set_ylabel("Spearman $\\rho$ with Otsu Map IoU (66 Pairs)")
    ax.set_xlabel("Transformer Block (Layer Fraction)")
    ax.set_title("C. Grounding vs. Map IoU: CKA vs. Mutual $k$-NN")
    ax.legend(loc="upper left", fontsize=7.2, ncol=2)

    fig.suptitle(
        "Figure 1. Artifact-Controlled Inter-Model Similarity on ImageNet-1K (12 Models, 66 Pairs, N = 1,000)",
        fontsize=11.5, fontweight="bold", y=1.03,
    )
    save_figure(fig, args.out_dir, "fig1_m4_cka_collapse")


if __name__ == "__main__":
    main()
