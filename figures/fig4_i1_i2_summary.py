#!/usr/bin/env python3
"""Figure 4 (H2 & H2b I2 Leg): Interventions I1 (query replacement) and I2 (equal-count FG vs BG_eq_FG).

Reads verbatim from:
  - <results_dir>/confirmatory_interventions_i1_i2_in1k/i1_i2_report.json
  - <results_dir>/confirmatory_stats_zoo12_final.json
"""
from __future__ import annotations

import json
import math
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
    ZOO12_ORDER,
    apply_pub_style,
    parse_fig_args,
    save_figure,
)


def _find_i1(rows: list[dict], l_int: int, surr: str) -> dict:
    for r in rows:
        if int(r["L_int"]) == l_int and r["surrogate"] == surr:
            return r
    raise KeyError(f"Missing I1 row L_int={l_int}, surr={surr}")


def _find_i2_contrast(rows: list[dict], layer: int, mode: str) -> dict:
    for r in rows:
        if int(r["layer"]) == layer and r["mode"] == mode:
            return r
    raise KeyError(f"Missing I2 contrast layer={layer}, mode={mode}")


def _log_ratio(c: dict, eps: float = 1e-6) -> float:
    return math.log((float(c["fg_cos_drop"]) + eps) / (float(c["bg_eq_fg_cos_drop"]) + eps))


def main() -> None:
    args = parse_fig_args("Generate Figure 4: I1 query replacement and I2 FG vs BG_eq_FG log-ratio summary.")
    apply_pub_style()

    i12 = json.loads((args.results_dir / "confirmatory_interventions_i1_i2_in1k" / "i1_i2_report.json").read_text())
    stats_path = args.results_dir / "confirmatory_stats_zoo12_final.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.5), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.15, 1.0]})

    # --- Panel A: I1 L_int=1 delta box-AUC (In-manifold S4 vs Off-manifold S1) ---
    ax = axes[0]
    y = np.arange(len(ZOO12_ORDER))
    h = 0.36

    for i, mid in enumerate(ZOO12_ORDER):
        fam = MODEL_FAMILY[mid]
        col = FAMILY_COLORS[fam]
        hatch = "///" if mid == "mae_vitb16" else None
        r_s4 = _find_i1(i12[mid]["i1_results"], 1, "S4")
        r_s1 = _find_i1(i12[mid]["i1_results"], 1, "S1")

        d4 = float(r_s4["delta_box_auc"])
        lo4, hi4 = r_s4["delta_box_auc_ci95"]
        d1 = float(r_s1["delta_box_auc"])
        lo1, hi1 = r_s1["delta_box_auc_ci95"]

        # S4 bar (solid family color)
        ax.barh(i - h / 2, d4, height=h, color=col, alpha=0.92, edgecolor="#0F172A", lw=0.65, hatch=hatch)
        ax.errorbar(d4, i - h / 2, xerr=[[d4 - lo4], [hi4 - d4]],
                    fmt="none", ecolor="#0F172A", elinewidth=1.0, capsize=2)

        # S1 bar (muted light bar showing off-manifold collapse)
        ax.barh(i + h / 2, d1, height=h, color=col, alpha=0.25, edgecolor=col, lw=0.9, hatch=hatch)
        ax.errorbar(d1, i + h / 2, xerr=[[d1 - lo1], [hi1 - d1]],
                    fmt="none", ecolor="#475569", elinewidth=0.8, capsize=2)

        # Annotate S4 value
        tx = (hi4 + 0.008) if d4 >= 0 else (lo4 - 0.008)
        ha = "left" if d4 >= 0 else "right"
        if d4 < 0 and d4 > -0.04:
            # Place cleanly to the right of zero so it doesn't collide with S1 bar
            tx, ha = 0.010, "left"
        ax.text(tx, i - h / 2, f"{d4:+.3f}", va="center", ha=ha,
                fontsize=7.6, fontweight="bold" if abs(d4) > 0.02 else "normal", color="#0F172A")

    ax.axvline(0.0, color="#0F172A", lw=1.0)
    ax.axvspan(-0.02, 0.02, color="#E2E8F0", alpha=0.45, zorder=0)

    ax.text(
        0.02, 0.42,
        "Exploratory (single recipe):\nBoth DeiT-III gain (+0.070, +0.036);\nOpenCLIP/MetaCLIP flat (−0.002)\nH2 OLS fallback (n=11, df=8):\nβ = −0.0027, p = 0.888 (Not supported)",
        transform=ax.transAxes, fontsize=7.2, color="#1E293B", va="center",
        bbox=dict(boxstyle="round,pad=0.32", fc="#F8FAFC", ec="#94A3B8", lw=0.9, alpha=0.95),
    )

    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_SHORT[m] for m in ZOO12_ORDER], fontsize=8.3)
    ax.invert_yaxis()
    ax.set_xlim(-0.40, 0.145)
    ax.set_xlabel("$\\Delta$box-AUC under I1 ($L_{\\mathrm{int}} = 1$; Dark = In-Manifold S4, Light = Off-Manifold S1)")
    ax.set_title("A. Intervention I1: In-Manifold S4 vs. Off-Manifold S1 Collapse")

    # --- Panel B: I2 Equal-Count Log-Ratio R_m across all 4 (layer x mode) settings ---
    ax = axes[1]
    settings = [
        (11, "mean_v", "l=11 mean_v\n(Registered\nPrimary)"),
        (11, "zero", "l=11 zero\n(Sensitivity)"),
        (10, "mean_v", "l=10 mean_v\n(Sensitivity)"),
        (10, "zero", "l=10 zero\n(Sensitivity)"),
    ]
    non_mae = [m for m in ZOO12_ORDER if m != "mae_vitb16"]
    ssl_ids = [m for m in non_mae if MODEL_FAMILY[m] == "SSL"]
    lt_ids = [m for m in non_mae if MODEL_FAMILY[m] in ("label", "text")]

    # Compute exact permutation p-values for each of the 4 settings (330 exact combos)
    from itertools import combinations
    rng = np.random.default_rng(0)

    x_centers = np.arange(len(settings))
    offsets_x = {"SSL": -0.17, "label": 0.17, "text": 0.17}

    for s_idx, (layer, mode, _) in enumerate(settings):
        r_ssl = [_log_ratio(_find_i2_contrast(i12[m]["i2_fg_bg_contrasts"], layer, mode)) for m in ssl_ids]
        r_lt = [_log_ratio(_find_i2_contrast(i12[m]["i2_fg_bg_contrasts"], layer, mode)) for m in lt_ids]
        diff = float(np.mean(r_ssl) - np.mean(r_lt))

        all_vals = np.array(r_ssl + r_lt, dtype=float)
        n_ssl = len(r_ssl)
        n_tot = len(all_vals)
        perm_diffs = []
        for idx_s in combinations(range(n_tot), n_ssl):
            mask = np.zeros(n_tot, dtype=bool)
            mask[list(idx_s)] = True
            perm_diffs.append(all_vals[mask].mean() - all_vals[~mask].mean())
        perm_diffs = np.array(perm_diffs)
        p_exact = float(np.mean(np.abs(perm_diffs) >= abs(diff) - 1e-12))

        # Plot group means +/- SEM bars + individual model dots
        m_ssl, m_lt = float(np.mean(r_ssl)), float(np.mean(r_lt))
        ax.hlines(m_ssl, s_idx - 0.30, s_idx - 0.04, color=FAMILY_COLORS["SSL"], lw=2.5, zorder=4)
        ax.hlines(m_lt, s_idx + 0.04, s_idx + 0.30, color="#B91C1C", lw=2.5, zorder=4)

        # Jittered dots per model
        for j, m in enumerate(ssl_ids):
            val = r_ssl[j]
            jx = s_idx - 0.17 + (j - 3) * 0.025
            ax.scatter(jx, val, s=44, color=FAMILY_COLORS["SSL"], edgecolors="#0F172A", lw=0.7, zorder=5)
        for j, m in enumerate(lt_ids):
            val = r_lt[j]
            fam = MODEL_FAMILY[m]
            jx = s_idx + 0.17 + (j - 1.5) * 0.04
            marker = "s" if fam == "label" else "D"
            ax.scatter(jx, val, s=48, color=FAMILY_COLORS[fam], marker=marker, edgecolors="#0F172A", lw=0.7, zorder=5)

        # Highlight primary panel background
        if s_idx == 0:
            ax.axvspan(s_idx - 0.42, s_idx + 0.42, color="#FEF9C3", alpha=0.35, zorder=0)

        ax.text(
            s_idx, 4.68,
            f"$\\Delta R = {diff:+.3f}$\n$p = {p_exact:.3f}$",
            ha="center", va="top", fontsize=7.6,
            fontweight="bold" if s_idx == 0 else "normal",
            color="#991B1B" if p_exact < 0.05 else "#1E293B",
        )

    ax.set_xticks(x_centers)
    ax.set_xticklabels([s[2] for s in settings], fontsize=8.1)
    ax.set_ylim(0.85, 4.82)
    ax.set_ylabel("Scale-Invariant Log-Ratio $R_m = \\log\\frac{\\overline{1-\\cos}_{\\mathrm{FG}}}{\\overline{1-\\cos}_{\\mathrm{BG\\_eq}}}$")
    ax.set_title("B. Intervention I2: Equal-Count FG / BG Ratio ($n_{\\mathrm{eq}} = 439\\text{–}465/1000$)")

    handles = [
        Patch(facecolor=FAMILY_COLORS[k], edgecolor="#0F172A", label=FAMILY_LABELS[k])
        for k in ("SSL", "label", "text", "recon")
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.065), fontsize=7.7, ncol=4)

    fig.suptitle(
        "Forward-Pass Causal Interventions I1 & I2 on ImageNet-1K (12 Models; I2 Subset = Smaller-Object Images <50% Area; * = New-4)",
        fontsize=10.6, fontweight="bold", y=1.03,
    )
    save_figure(fig, args.out_dir, "fig4_i1_i2_summary")


if __name__ == "__main__":
    main()
