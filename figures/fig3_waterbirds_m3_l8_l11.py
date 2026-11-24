#!/usr/bin/env python3
"""Figure 3 (H2b M3 Leg): Waterbirds balanced 2x2 Background Main Effect at l=8 and l=11 (with in21k contrast).

Reads verbatim from:
  - <results_dir>/pilot_waterbirds_m3/waterbirds_congruence.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import (
    FAMILY_COLORS,
    FAMILY_LABELS,
    MODEL_FAMILY,
    MODEL_SHORT,
    apply_pub_style,
    parse_fig_args,
    save_figure,
)

# Order grouped by size (S-scale first, then B-scale, then MAE) so size-matched SSL comparison is immediate
WB_ORDER = [
    # S-scale
    "dino_vits16",
    "ibot_vits16",
    "dinov2_vits14",
    "dinov2_vits14_reg",
    "deit3_small_patch16_224",
    "deit3_small_patch16_224_in21k",
    # B-scale
    "mocov3_vitb16",
    "dinov2_vitb14",
    "dinov2_vitb14_reg",
    "openclip_vitb16",
    "metaclip_vitb16",
    # Excluded reconstruction
    "mae_vitb16",
]

S_SSL = {"dino_vits16", "ibot_vits16", "dinov2_vits14", "dinov2_vits14_reg"}
B_SSL = {"mocov3_vitb16", "dinov2_vitb14", "dinov2_vitb14_reg"}


def main() -> None:
    args = parse_fig_args("Generate Figure 3: Waterbirds M3 background main effect at l=8 and l=11.")
    apply_pub_style()

    wb_path = args.results_dir / "pilot_waterbirds_m3" / "waterbirds_congruence.json"
    wb = json.loads(wb_path.read_text())

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.6), sharey=True, constrained_layout=True)

    y = np.arange(len(WB_ORDER))
    # Add visual group gap between S-scale (0..5), B-scale (6..10), MAE (11)
    y_pos = np.array([0, 1, 2, 3, 4, 5, 6.7, 7.7, 8.7, 9.7, 10.7, 12.1], dtype=float)

    for ax, layer, title_suffix in [
        (axes[0], "l8", "A. Layer l = 8 (Layer Frac 0.75) — Split Unseen Evidence"),
        (axes[1], "l11", "B. Layer l = 11 (Layer Frac 1.00) — All 4/4 Label+Text Exceed SSL"),
    ]:
        vals = []
        los = []
        his = []
        for mid in WB_ORDER:
            bg = wb[mid][layer]["balanced_2x2"]["bg_main_effect_water_minus_land"]
            vals.append(bg["mean"])
            los.append(bg["mean"] - bg["ci95_low"])
            his.append(bg["ci95_high"] - bg["mean"])

        max_s_ssl = max(wb[m][layer]["balanced_2x2"]["bg_main_effect_water_minus_land"]["mean"] for m in S_SSL)
        max_b_ssl = max(wb[m][layer]["balanced_2x2"]["bg_main_effect_water_minus_land"]["mean"] for m in B_SSL)

        # Size-matched max SSL reference lines within each scale block
        ax.plot([max_s_ssl, max_s_ssl], [-0.55, 5.55], ls="--", color="#1B5E8A", lw=1.3, alpha=0.85)
        ax.plot([max_b_ssl, max_b_ssl], [6.15, 11.25], ls="--", color="#1B5E8A", lw=1.3, alpha=0.85)
        ax.axvline(0.0, color="#334155", lw=0.9)

        # Subtle scale dividers
        ax.axhline(5.85, color="#CBD5E0", lw=1.0, ls="-")
        ax.axhline(11.45, color="#CBD5E0", lw=1.0, ls="-")

        for i, mid in enumerate(WB_ORDER):
            fam = MODEL_FAMILY[mid]
            col = FAMILY_COLORS[fam]
            hatch = "///" if mid == "mae_vitb16" else None
            ax.barh(y_pos[i], vals[i], height=0.66, color=col, alpha=0.88,
                    edgecolor="#1E293B", lw=0.65, hatch=hatch)
            ax.errorbar(vals[i], y_pos[i], xerr=[[los[i]], [his[i]]],
                        fmt="none", ecolor="#0F172A", elinewidth=1.1, capsize=2.5)
            # Annotate exact value + ratio for label+text models
            txt_x = vals[i] + his[i] + 0.006 if vals[i] >= 0 else vals[i] - los[i] - 0.006
            ha = "left" if vals[i] >= 0 else "right"
            if fam in ("label", "text"):
                ref = max_s_ssl if mid.startswith("deit3") else max_b_ssl
                ratio = vals[i] / ref
                lbl = f"{vals[i]:+.3f} ({ratio:.2f}×)"
                fw = "bold"
            else:
                lbl = f"{vals[i]:+.3f}"
                fw = "normal"
            ax.text(txt_x, y_pos[i], lbl, va="center", ha=ha, fontsize=7.6, fontweight=fw, color="#0F172A")

        # Annotate max SSL reference lines cleanly in whitespace
        ax.text(max_s_ssl + 0.004, 5.58, f"max S-SSL ({max_s_ssl:+.3f})", fontsize=7.0, color="#1B5E8A", va="bottom")
        ax.text(max_b_ssl + 0.004, 11.28, f"max B-SSL ({max_b_ssl:+.3f})", fontsize=7.0, color="#1B5E8A", va="bottom")

        ax.set_xlabel("Balanced 2×2 Background Main Effect (Water − Land $\\Delta\\mathrm{BSS}_{\\mathrm{norm}}$, 95% CI)")
        ax.set_title(title_suffix, pad=8)

    axes[0].set_xlim(-0.105, 0.295)
    axes[1].set_xlim(-0.025, 0.210)

    # Highlight the in21k l=8 vs l=11 contrast explicitly
    axes[0].annotate(
        "Unseen in21k drops below\nall S-SSL (0.38×, 0/3 terciles)",
        xy=(0.088, 5.0), xytext=(0.132, 1.8),
        fontsize=7.6, color="#7F1D1D",
        bbox=dict(boxstyle="round,pad=0.32", fc="#FEF2F2", ec="#FCA5A5", lw=0.9),
        arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.1),
    )
    axes[1].annotate(
        "At l=11, unseen in21k\nrebounds to 2.46× S-SSL",
        xy=(0.146, 5.0), xytext=(0.105, 1.8),
        fontsize=7.6, color="#7F1D1D",
        bbox=dict(boxstyle="round,pad=0.32", fc="#FEF2F2", ec="#FCA5A5", lw=0.9),
        arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.1),
    )

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels([MODEL_SHORT[m] for m in WB_ORDER], fontsize=8.5)
    axes[0].invert_yaxis()

    # Scale labels on right margin of Panel A
    axes[0].text(-0.100, 2.5, "ViT-S", rotation=90, va="center", ha="left", fontsize=8.0, fontweight="bold", color="#475569")
    axes[0].text(-0.100, 8.7, "ViT-B", rotation=90, va="center", ha="left", fontsize=8.0, fontweight="bold", color="#475569")

    handles = [
        Patch(facecolor=FAMILY_COLORS[k], edgecolor="#1E293B", label=FAMILY_LABELS[k])
        for k in ("SSL", "label", "text", "recon")
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=4, fontsize=7.8)

    fig.suptitle(
        "Figure 3. Waterbirds Balanced 2×2 Background Effect (M3 Leg; N = 5,794; H2b Conjunction Not Supported, p = 0.788)",
        fontsize=10.8, fontweight="bold", y=1.03,
    )
    save_figure(fig, args.out_dir, "fig3_waterbirds_m3_l8_l11")


if __name__ == "__main__":
    main()
