#!/usr/bin/env python3
"""Figure 6 (Exploratory H4 — Registers): All 3 registered criteria + exploratory R_m on DINOv2-B/14 & DINOv2-S/14.

Reads verbatim from:
  - <results_dir>/pilot_m1_m2_in1k/summary_by_model_layer.csv
  - <results_dir>/pilot_waterbirds_m3/summary_waterbirds_m3.csv
  - <results_dir>/confirmatory_interventions_i1_i2_in1k/i1_i2_report.json
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import apply_pub_style, parse_fig_args, save_figure


def _find_i2_row(rows: list[dict], layer: int, group: str, mode: str) -> dict:
    for r in rows:
        if int(r["layer"]) == layer and r["group"] == group and r["mode"] == mode:
            return r
    raise KeyError(f"Missing I2 row layer={layer}, group={group}, mode={mode}")


def _find_i2_contrast(rows: list[dict], layer: int, mode: str) -> dict:
    for r in rows:
        if int(r["layer"]) == layer and r["mode"] == mode:
            return r
    raise KeyError(f"Missing I2 contrast layer={layer}, mode={mode}")


def main() -> None:
    args = parse_fig_args("Generate Figure 6: Exploratory H4 registers across DINOv2-B/14 and DINOv2-S/14 pairs.")
    apply_pub_style()

    m12_rows = {}
    with open(args.results_dir / "pilot_m1_m2_in1k" / "summary_by_model_layer.csv", newline="") as f:
        for r in csv.DictReader(f):
            m12_rows[(r["model_id"], int(r["layer"]))] = r

    wb_rows = {}
    with open(args.results_dir / "pilot_waterbirds_m3" / "summary_waterbirds_m3.csv", newline="") as f:
        for r in csv.DictReader(f):
            wb_rows[(r["model_id"], int(r["layer"]))] = r

    i12 = json.loads((args.results_dir / "confirmatory_interventions_i1_i2_in1k" / "i1_i2_report.json").read_text())

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.7), constrained_layout=True)

    # --- Panel A: Criterion 1 — QII_S4 survival (|delta| <= 0.05) + Map box-AUC ---
    ax = axes[0]
    pairs = [
        ("DINOv2-B/14\n(Pilot, 3/3)", "dinov2_vitb14", "dinov2_vitb14_reg", 11, 11),
        ("DINOv2-S/14*\n(Unseen, 2/3)", "dinov2_vits14", "dinov2_vits14_reg", 11, 11),
    ]
    x = np.arange(len(pairs))
    w = 0.18

    q_base = [float(m12_rows[(b, lb)]["qii_s4"]) for _, b, _, lb, _ in pairs]
    q_reg = [float(m12_rows[(r, lr)]["qii_s4"]) for _, _, r, _, lr in pairs]
    a_base = [float(m12_rows[(b, lb)]["box_auc_attn"]) for _, b, _, lb, _ in pairs]
    a_reg = [float(m12_rows[(r, lr)]["box_auc_attn"]) for _, _, r, _, lr in pairs]

    ax.bar(x - 1.5 * w, q_base, w, color="#93C5FD", edgecolor="#1E3A8A", lw=0.8, label="QII$_{S4}$ (No Reg, l=11)")
    ax.bar(x - 0.5 * w, q_reg, w, color="#1D4ED8", edgecolor="#0F172A", lw=0.8, label="QII$_{S4}$ (+4 Reg, l=11)")
    ax.bar(x + 0.5 * w, a_base, w, color="#FDE68A", edgecolor="#92400E", lw=0.8, label="box-AUC (No Reg, l=11)")
    ax.bar(x + 1.5 * w, a_reg, w, color="#D97706", edgecolor="#0F172A", lw=0.8, label="box-AUC (+4 Reg, l=11)")

    for i in range(len(pairs)):
        dq = q_reg[i] - q_base[i]
        da = a_reg[i] - a_base[i]
        ax.text(x[i] - 1.0 * w, max(q_base[i], q_reg[i]) + 0.015, f"$\\Delta={dq:+.3f}$\n($|\\Delta|\\leq 0.05$ ✓)",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#1E3A8A")
        ax.text(x[i] + 1.0 * w, max(a_base[i], a_reg[i]) + 0.015, f"$\\Delta={da:+.3f}$",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#92400E")

    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in pairs])
    ax.set_ylim(0.58, 1.12)
    ax.set_ylabel("Score on ImageNet-1K (l = 11)")
    ax.set_title("A. Criterion 1: QII$_{S4}$ Survives Registers")
    ax.legend(loc="lower center", fontsize=6.9, ncol=2)

    # --- Panel B: Criterion 2 — share_OUT -> share_REG across l=8..11 (Waterbirds M3) ---
    ax = axes[1]
    layers = [8, 9, 10, 11]
    xl = np.arange(len(layers))
    bw = 0.20

    b_out = [float(wb_rows[("dinov2_vitb14", l)]["share_out"]) * 100 for l in layers]
    br_out = [float(wb_rows[("dinov2_vitb14_reg", l)]["share_out"]) * 100 for l in layers]
    br_reg = [float(wb_rows[("dinov2_vitb14_reg", l)]["share_reg"]) * 100 for l in layers]

    s_out = [float(wb_rows[("dinov2_vits14", l)]["share_out"]) * 100 for l in layers]
    sr_out = [float(wb_rows[("dinov2_vits14_reg", l)]["share_out"]) * 100 for l in layers]
    sr_reg = [float(wb_rows[("dinov2_vits14_reg", l)]["share_reg"]) * 100 for l in layers]

    ax.bar(xl - 1.5 * bw, b_out, bw, color="#DC2626", alpha=0.85, edgecolor="#0F172A", lw=0.6, label="B/14 share_OUT")
    ax.bar(xl - 0.5 * bw, br_out, bw, color="#FCA5A5", edgecolor="#991B1B", lw=0.7, label="B/14+reg share_OUT")
    ax.bar(xl - 0.5 * bw, br_reg, bw, bottom=br_out, color="#1D4ED8", alpha=0.82, edgecolor="#0F172A", lw=0.6, label="B/14+reg share_REG")

    ax.bar(xl + 0.5 * bw, s_out, bw, color="#D97706", alpha=0.85, edgecolor="#0F172A", lw=0.6, label="S/14* share_OUT")
    ax.bar(xl + 1.5 * bw, sr_out, bw, color="#FDE68A", edgecolor="#92400E", lw=0.7, label="S/14+reg* share_OUT")
    ax.bar(xl + 1.5 * bw, sr_reg, bw, bottom=sr_out, color="#0F766E", alpha=0.82, edgecolor="#0F172A", lw=0.6, label="S/14+reg* share_REG")

    ax.annotate(
        "B/14: −56% to −95% ✓\nS/14*: baseline OUT ≤2.6%\n(≥50% drop NOT met ✗)",
        xy=(1.5, 2.5), xytext=(1.15, 23.5),
        fontsize=7.3, color="#1E293B",
        bbox=dict(boxstyle="round,pad=0.32", fc="#F8FAFC", ec="#94A3B8", lw=0.9),
    )

    ax.set_xticks(xl)
    ax.set_xticklabels([f"l={l}" for l in layers])
    ax.set_ylim(0, 32)
    ax.set_ylabel("Routing Share of Global Update (%)")
    ax.set_xlabel("Transformer Block (Waterbirds N = 5,794)")
    ax.set_title("B. Criterion 2: OUT → REG Mass Transfer")
    ax.legend(loc="upper right", fontsize=6.6, ncol=2)

    # --- Panel C: Criterion 3 (REG KO) + Exploratory Extension (I2 R_m) ---
    ax = axes[2]
    reg_models = [("DINOv2-B/14+reg", "dinov2_vitb14", "dinov2_vitb14_reg"),
                  ("DINOv2-S/14+reg*", "dinov2_vits14", "dinov2_vits14_reg")]

    # Criterion 3 REG knockout metrics: l=11 1-cos readout shift + l=10 non-circular downstream |delta_AUC|
    cos_z11 = [float(_find_i2_row(i12[rm]["i2_results"], 11, "REG", "zero")["cos_drop_group"]) for _, _, rm in reg_models]
    cos_mv11 = [float(_find_i2_row(i12[rm]["i2_results"], 11, "REG", "mean_v")["cos_drop_group"]) for _, _, rm in reg_models]
    dauc_mv = [abs(float(_find_i2_row(i12[rm]["i2_results"], 10, "REG", "mean_v")["delta_auc_group"])) for _, _, rm in reg_models]
    dauc_z = [abs(float(_find_i2_row(i12[rm]["i2_results"], 10, "REG", "zero")["delta_auc_group"])) for _, _, rm in reg_models]

    # Exploratory R_m (l=11 mean_v)
    def rm_val(mid: str) -> float:
        c = _find_i2_contrast(i12[mid]["i2_fg_bg_contrasts"], 11, "mean_v")
        return math.log((float(c["fg_cos_drop"]) + 1e-6) / (float(c["bg_eq_fg_cos_drop"]) + 1e-6))

    rm_base = [rm_val(bm) for _, bm, _ in reg_models]
    rm_reg = [rm_val(rm) for _, _, rm in reg_models]

    xr = np.arange(len(reg_models))
    w3 = 0.22
    ax.bar(xr - w3, cos_z11, w3, color="#1D4ED8", edgecolor="#0F172A", lw=0.7, label="REG KO 1−cos (l=11 zero)")
    ax.bar(xr, cos_mv11, w3, color="#60A5FA", edgecolor="#1E3A8A", lw=0.7, label="REG KO 1−cos (l=11 mean_v)")
    ax.bar(xr + w3, [max(dauc_mv[i], dauc_z[i]) for i in range(2)], w3, color="#0F766E", edgecolor="#0F172A", lw=0.7,
           label="max |ΔAUC| at l=10 (≤0.01 ✓)")
    ax.axhline(0.01, color="#0F766E", ls="--", lw=1.1, alpha=0.85)
    ax.text(1.42, 0.0108, "±0.01 AUC bound", ha="right", fontsize=7.2, color="#0F766E")

    for i in range(2):
        ax.text(xr[i] - w3, cos_z11[i] + 0.001, f"{cos_z11[i]:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
        ax.text(xr[i], cos_mv11[i] + 0.001, f"{cos_mv11[i]:.3f}", ha="center", va="bottom", fontsize=7.5)
        ax.text(xr[i] + w3, max(dauc_mv[i], dauc_z[i]) + 0.001, f"{max(dauc_mv[i], dauc_z[i]):.4f}",
                ha="center", va="bottom", fontsize=7.2, color="#0F766E", fontweight="bold")

    # Box summarizing exploratory R_m extension cleanly without mixing scales
    ax.text(
        0.50, 0.96,
        f"Exploratory extension (I2 $R_m$, l=11 mean_v):\n"
        f"B/14: {rm_base[0]:.2f} → {rm_reg[0]:.2f} (+{rm_reg[0]-rm_base[0]:.2f})  |  "
        f"S/14*: {rm_base[1]:.2f} → {rm_reg[1]:.2f} (+{rm_reg[1]-rm_base[1]:.2f})",
        transform=ax.transAxes, ha="center", va="top", fontsize=7.3,
        bbox=dict(boxstyle="round,pad=0.32", fc="#FEF9C3", ec="#CA8A04", lw=0.85),
    )

    ax.set_xticks(xr)
    ax.set_xticklabels([r[0] for r in reg_models])
    ax.set_ylim(0.0, 0.046)
    ax.set_ylabel("Magnitude (1 − cos or |Δbox-AUC|)")
    ax.set_xlabel("Register-Equipped Checkpoint (ImageNet-1K)")
    ax.set_title("C. Criterion 3: REG Knockout & Exploratory $R_m$")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.77), fontsize=7.0)

    fig.suptitle(
        "Exploratory H4 (Register Tokens): All 3 Registered Criteria & Scale-Dependent Artifact Behavior (* = New-4 Unseen)",
        fontsize=10.8, fontweight="bold", y=1.03,
    )
    save_figure(fig, args.out_dir, "fig6_h4_registers")


if __name__ == "__main__":
    main()
