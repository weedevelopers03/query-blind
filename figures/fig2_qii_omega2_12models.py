#!/usr/bin/env python3
"""Figure 2 (H1 Falsified & Fix 3 Non-Redundancy): QII_S4, box-AUC, and omega^2_key across all 12 models.

Reads verbatim from:
  - <results_dir>/pilot_m1_m2_in1k/summary_by_model_layer.csv
  - <results_dir>/pilot_m1_m2_in1k/fix3_and_g1.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import (
    FAMILY_COLORS,
    FAMILY_LABELS,
    MODEL_FAMILY,
    MODEL_SHORT,
    ZOO12_ORDER,
    apply_pub_style,
    parse_fig_args,
    save_figure,
)


def main() -> None:
    args = parse_fig_args("Generate Figure 2: QII_S4 and omega^2_key across all 12 models.")
    apply_pub_style()

    csv_path = args.results_dir / "pilot_m1_m2_in1k" / "summary_by_model_layer.csv"
    fix3_path = args.results_dir / "pilot_m1_m2_in1k" / "fix3_and_g1.json"

    rows_by_ml = {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows_by_ml[(r["model_id"], int(r["layer"]))] = r

    fix3 = json.loads(fix3_path.read_text())
    l_star_map = fix3["l_star_by_model"]

    # Collect l* rows for all 12 models (using Guide rule: argmax omega2 among H_norm <= 0.90, fallback 11)
    records = []
    for mid in ZOO12_ORDER:
        if mid in l_star_map:
            entry = l_star_map[mid]
            ls = int(entry["layer"] if isinstance(entry, dict) else entry)
        else:
            # Compute deterministically from CSV
            cands = [rows_by_ml[(mid, l)] for l in (8, 9, 10, 11) if (mid, l) in rows_by_ml]
            elig = [c for c in cands if float(c["entropy_norm"]) <= 0.90]
            chosen = max(elig, key=lambda c: float(c["omega2_key"])) if elig else rows_by_ml[(mid, 11)]
            ls = int(chosen["layer"])
        r = rows_by_ml[(mid, ls)]
        records.append({
            "model_id": mid,
            "label": f"{MODEL_SHORT[mid]} (l*={ls})",
            "short": MODEL_SHORT[mid],
            "family": MODEL_FAMILY[mid],
            "color": FAMILY_COLORS[MODEL_FAMILY[mid]],
            "l_star": ls,
            "qii": float(r["qii_s4"]),
            "qii_lo": float(r["qii_s4_ci95_lo"]),
            "qii_hi": float(r["qii_s4_ci95_hi"]),
            "ent": float(r["entropy_norm"]),
            "auc": float(r["box_auc_attn"]),
            "auc_lo": float(r["box_auc_attn_ci95_lo"]),
            "auc_hi": float(r["box_auc_attn_ci95_hi"]),
            "w2": float(r["omega2_key"]),
            "w2_lo": float(r["omega2_key_ci95_lo"]),
            "w2_hi": float(r["omega2_key_ci95_hi"]),
            "degen": r["status_s4"] != "eligible",
        })

    # Compute exact Spearman correlations for Fix 3 annotation
    rho_all, p_all = spearmanr([r["qii"] for r in records], [r["w2"] for r in records])
    non_deg = [r for r in records if not r["degen"]]
    rho_nd, p_nd = spearmanr([r["qii"] for r in non_deg], [r["w2"] for r in non_deg])

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.4), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.18, 1.0]})

    # --- Panel A: Paired QII_S4 vs Map Quality (box-AUC) at l* ---
    ax = axes[0]
    y = np.arange(len(records))
    h = 0.36

    for i, r in enumerate(records):
        hatch = "///" if r["degen"] else None
        # QII_S4 bar (solid family color)
        ax.barh(i - h / 2, r["qii"], height=h, color=r["color"], alpha=0.92,
                edgecolor="#1E293B", lw=0.6, hatch=hatch)
        # box-AUC bar (lighter tint)
        ax.barh(i + h / 2, r["auc"], height=h, color=r["color"], alpha=0.32,
                edgecolor=r["color"], lw=1.0, hatch=hatch)
        ax.errorbar(r["auc"], i + h / 2,
                    xerr=[[r["auc"] - r["auc_lo"]], [r["auc_hi"] - r["auc"]]],
                    fmt="none", ecolor="#1E293B", elinewidth=0.9, capsize=2)
        ax.text(r["qii"] + 0.008, i - h / 2, f"{r['qii']:.3f}", va="center", fontsize=7.5, fontweight="bold", color="#0F172A")
        ax.text(r["auc"] + 0.012, i + h / 2, f"{r['auc']:.3f}", va="center", fontsize=7.3, color="#334155")

    ax.axvline(0.50, color="#64748B", ls=":", lw=1.1)
    ax.text(0.503, -0.62, "RALR HIGH (0.50)", fontsize=7.0, color="#475569", va="bottom", ha="left")

    # Subtle family dividers (SSL 7 | Label+Text 4 | MAE 1)
    ax.axhline(6.5, color="#475569", ls="--", lw=0.9, alpha=0.65)
    ax.axhline(10.5, color="#94A3B8", ls=":", lw=0.8, alpha=0.65)

    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in records], fontsize=8.3)
    ax.invert_yaxis()
    ax.set_ylim(11.65, -0.85)
    ax.set_xlim(0.45, 1.08)
    ax.set_xlabel("Score at Unlabelled $l^*$ (Dark bar = QII$_{S4}$, Light bar = True-Query box-AUC)")
    ax.set_title(
        "A. Universal Query-Blindness vs. Divergent Map Quality\n"
        "SSL (7) vs. Label+Text (4): QII$_{S4}$ $U=6$ ($p=0.164$, n.s.)  |  box-AUC $U=28/28$ ($p=0.0061$, zero overlap)",
        fontsize=8.6,
    )

    # Shared figure-level legend at bottom so Panel A bars are 100% unobstructed
    handles = [
        Patch(facecolor=FAMILY_COLORS[k], edgecolor="#1E293B", label=FAMILY_LABELS[k])
        for k in ("SSL", "label", "text", "recon")
    ]
    handles.append(Patch(facecolor="#CBD5E0", edgecolor="#475569", hatch="///", label="Degenerate ($H_{\\mathrm{norm}}>0.90$)"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.065), ncol=5, fontsize=7.7)

    # --- Panel B: Fix 3 Dissociation Scatter (QII_S4 vs omega^2_key at l*) ---
    ax = axes[1]
    for r in records:
        marker = "X" if r["degen"] else "o"
        sz = 95 if r["degen"] else 75
        ax.errorbar(
            r["qii"], r["w2"],
            xerr=[[r["qii"] - r["qii_lo"]], [r["qii_hi"] - r["qii"]]],
            yerr=[[r["w2"] - r["w2_lo"]], [r["w2_hi"] - r["w2"]]],
            fmt="none", ecolor=r["color"], alpha=0.55, elinewidth=0.9, capsize=2,
        )
        ax.scatter(r["qii"], r["w2"], s=sz, color=r["color"], marker=marker,
                   edgecolors="#0F172A", linewidths=0.8, zorder=4)

    # Hand-tuned non-overlapping label offsets
    offsets = {
        "dino_vits16": (7, -3),
        "ibot_vits16": (-56, -13),
        "dinov2_vits14": (-74, -14),
        "dinov2_vits14_reg": (7, 5),
        "mocov3_vitb16": (-64, 9),
        "dinov2_vitb14": (7, -12),
        "dinov2_vitb14_reg": (6, -16),
        "deit3_small_patch16_224": (7, 4),
        "deit3_small_patch16_224_in21k": (-110, 8),
        "openclip_vitb16": (7, 8),
        "metaclip_vitb16": (-82, -16),
        "mae_vitb16": (-82, 9),
    }
    for r in records:
        dx, dy = offsets.get(r["model_id"], (6, 4))
        ax.annotate(
            r["short"], (r["qii"], r["w2"]),
            textcoords="offset points", xytext=(dx, dy),
            fontsize=7.5, color="#1E293B",
            arrowprops=dict(arrowstyle="-", color="#94A3B8", lw=0.6) if abs(dx) > 20 else None,
        )

    ax.text(
        0.03, 0.96,
        f"Fix 3 Non-Redundancy ($l^*$):\n"
        f"All 12: $\\rho = {rho_all:+.3f}$ ($p = {p_all:.3f}$)\n"
        f"Excl. MAE ($n=11$): $\\rho = {rho_nd:+.3f}$ ($p = {p_nd:.3f}$)\n"
        f"Decision: $|\\rho| < 0.80 \\Rightarrow$ keep both\n"
        f"H1 (DINO$-$DeiT-III): $+0.0052 < 0.02$ min\n"
        f"$\\Rightarrow$ H1 falsified",
        transform=ax.transAxes, va="top", ha="left", fontsize=7.6,
        bbox=dict(boxstyle="round,pad=0.38", fc="#F8FAFC", ec="#94A3B8", lw=0.9),
    )

    ax.set_xlim(0.910, 1.012)
    ax.set_ylim(0.12, 0.96)
    ax.set_xlabel("In-Manifold Query-Invariance Index QII$_{S4}$ at $l^*$")
    ax.set_ylabel("Unbiased Key Dominance $\\omega^2_{\\mathrm{key}}$ at $l^*$")
    ax.set_title("B. Fix 3 Dissociation: QII$_{S4}$ vs. $\\omega^2_{\\mathrm{key}}$ ($n = 12$)")

    fig.suptitle(
        "Measure M1 (QII$_{S4}$) & Measure M2 ($\\omega^2_{\\mathrm{key}}$) Across the 12-Model Zoo on ImageNet-1K (N = 1,000; * = New-4)",
        fontsize=11.0, fontweight="bold", y=1.03,
    )
    save_figure(fig, args.out_dir, "fig2_qii_omega2_12models")


if __name__ == "__main__":
    main()
