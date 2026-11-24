"""Shared publication styling & CLI helpers for reproducible thesis figures (Guide §11.3)."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "zoo12_summaries"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent.parent / "thesis_draft" / "figures"

# Canonical 12-model ordering grouped by objective family and scale
ZOO12_ORDER = [
    "dino_vits16",
    "ibot_vits16",
    "dinov2_vits14",
    "dinov2_vits14_reg",
    "mocov3_vitb16",
    "dinov2_vitb14",
    "dinov2_vitb14_reg",
    "deit3_small_patch16_224",
    "deit3_small_patch16_224_in21k",
    "openclip_vitb16",
    "metaclip_vitb16",
    "mae_vitb16",
]

NEW4_MODELS = {
    "dinov2_vits14",
    "dinov2_vits14_reg",
    "deit3_small_patch16_224_in21k",
    "metaclip_vitb16",
}

MODEL_SHORT = {
    "dino_vits16": "DINO S/16",
    "ibot_vits16": "iBOT S/16",
    "mocov3_vitb16": "MoCo v3 B/16",
    "dinov2_vitb14": "DINOv2 B/14",
    "dinov2_vitb14_reg": "DINOv2 B/14+reg",
    "dinov2_vits14": "DINOv2 S/14*",
    "dinov2_vits14_reg": "DINOv2 S/14+reg*",
    "deit3_small_patch16_224": "DeiT-III S/16 (1K)",
    "deit3_small_patch16_224_in21k": "DeiT-III S/16 (21K)*",
    "openclip_vitb16": "OpenCLIP B/16",
    "metaclip_vitb16": "MetaCLIP B/16*",
    "mae_vitb16": "MAE B/16 (degen.)",
}

MODEL_FAMILY = {
    "dino_vits16": "SSL",
    "ibot_vits16": "SSL",
    "mocov3_vitb16": "SSL",
    "dinov2_vitb14": "SSL",
    "dinov2_vitb14_reg": "SSL",
    "dinov2_vits14": "SSL",
    "dinov2_vits14_reg": "SSL",
    "deit3_small_patch16_224": "label",
    "deit3_small_patch16_224_in21k": "label",
    "openclip_vitb16": "text",
    "metaclip_vitb16": "text",
    "mae_vitb16": "recon",
}

FAMILY_COLORS = {
    "SSL": "#1B5E8A",       # deep academic blue
    "label": "#C0392B",     # crimson red
    "text": "#D97706",      # warm amber
    "recon": "#64748B",     # slate gray
}

FAMILY_LABELS = {
    "SSL": "SSL (n = 7)",
    "label": "Label-Supervised (n = 2)",
    "text": "Image-Text Contrastive (n = 2)",
    "recon": "Reconstruction / MAE (excl.)",
}


def apply_pub_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.labelweight": "bold",
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.2,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#CBD5E0",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#334155",
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "grid.color": "#E2E8F0",
        "grid.linewidth": 0.7,
        "grid.linestyle": "--",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def parse_fig_args(description: str) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument(
        "--results_dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Path to unpacked zoo12_summaries / results directory.",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory to write PNG (300 DPI) and vector PDF outputs.",
    )
    return ap.parse_args()


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png_path} and {pdf_path}")
