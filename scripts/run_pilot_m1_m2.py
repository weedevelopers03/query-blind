"""Pilot runner for Measures M1 (QII) and M2 (Key Dominance) on CUB-200-2011 (Guide §§2-3, §8.4, §11.1-11.2).

Runs all 8 pilot models (or any --models subset) under the primary 14x14 matched-grid protocol:
  * 224x224 for /16 models; 196x196 for /14 models (dinov2_vitb14, dinov2_vitb14_reg).
  * Per-model normalization constants read from checkpoint config (never hardcoded, §8.2).
  * Ground-truth boxes / masks transformed with identical Resize + CenterCrop (§8.2).
  * Saves 20-image visual box overlay PNGs at 224px and 196px (Guide §8.2 & Review (3)).
  * Fixed CUB-200-2011 test pilot split (500 images, seed 0) saved once to configs/splits/ (§8.1, §8.4).
  * Per-model outputs saved atomically so --resume skips completed models cleanly.
  * Computes Fix 3 QII-KD Spearman correlation (§3.6), G1 paired bootstrap gate (§11.2),
    shuffled-key / spatial-shuffle / permutation-null sanity checks, and throughput (§11.1).

Usage on Kaggle:
    PYTHONPATH=. python3 scripts/run_pilot_m1_m2.py \\
        --cub_dir /kaggle/input/cub2002011/CUB_200_2011 \\
        --out_dir results/pilot_m1_m2_cub
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.base import attn_probs
from adapters.open_clip_vit import OpenCLIPViTAdapter
from adapters.timm_vit import TimmViTAdapter
from eval.preprocess import (
    ImageNetSample,
    build_image_transform,
    get_model_norm_stats,
    get_transformed_gt_mask,
    get_transformed_imagenet_gt_mask,
    load_cub_pilot_split,
    load_imagenet_val_pilot_split,
    save_box_overlays,
    save_imagenet_box_overlays,
    target_size_for_patch_size,
)
from extract.cache import last_third_blocks
from measures import keydom, qii
from measures._common import (
    LOG2,
    box_auc_torch,
    compute_head_weights,
    get_output_proj_and_layerscale,
    jsd_nats_torch,
    normalized_entropy_torch,
    patch_attention,
    write_records,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PILOT8_SPECS = {
    "dino_vits16": dict(
        family="timm_vit", arch="vit_small_patch16_224.dino", patch_size=16, loader="timm",
    ),
    "deit3_small_patch16_224": dict(
        family="timm_vit", arch="deit3_small_patch16_224.fb_in1k", patch_size=16, loader="timm",
    ),
    "ibot_vits16": dict(
        family="timm_vit",
        arch="vit_small_patch16_224",
        patch_size=16,
        loader="ibot_url",
        url="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vits_16/checkpoint_teacher.pth",
        filename="ibot_vits16_teacher.pth",
        # Official ByteDance iBOT pretraining uses standard ImageNet normalization (not timm's bare Google-ViT 0.5/0.5 default)
        norm_override=(IMAGENET_MEAN, IMAGENET_STD),
    ),
    "mocov3_vitb16": dict(
        family="timm_vit",
        arch="vit_base_patch16_224",
        patch_size=16,
        loader="mocov3_url",
        url="https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar",
        filename="mocov3_vitb16_300ep.pth.tar",
        # Official Meta MoCo v3 pretraining uses standard ImageNet normalization (not timm's bare Google-ViT 0.5/0.5 default)
        norm_override=(IMAGENET_MEAN, IMAGENET_STD),
    ),
    "mae_vitb16": dict(
        family="timm_vit", arch="vit_base_patch16_224.mae", patch_size=16, loader="timm",
    ),
    "dinov2_vitb14": dict(
        family="timm_vit", arch="vit_base_patch14_dinov2.lvd142m", patch_size=14, loader="timm",
    ),
    "dinov2_vitb14_reg": dict(
        family="timm_vit", arch="vit_base_patch14_reg4_dinov2.lvd142m", patch_size=14, loader="timm",
    ),
    "openclip_vitb16": dict(
        family="open_clip_vit", arch="ViT-B-16", pretrained_tag="laion2b_s34b_b88k", patch_size=16, loader="open_clip",
    ),
}

# Pre-registered 4 unseen expansion checkpoints (OSF §1, configs/checkpoints.yaml commit 8bfc63b)
EXPANSION4_SPECS = {
    "dinov2_vits14": dict(
        family="timm_vit", arch="vit_small_patch14_dinov2.lvd142m", patch_size=14, loader="timm",
    ),
    "dinov2_vits14_reg": dict(
        family="timm_vit", arch="vit_small_patch14_reg4_dinov2.lvd142m", patch_size=14, loader="timm",
    ),
    "deit3_small_patch16_224_in21k": dict(
        family="timm_vit", arch="deit3_small_patch16_224.fb_in22k_ft_in1k", patch_size=16, loader="timm",
    ),
    "metaclip_vitb16": dict(
        family="timm_vit", arch="vit_base_patch16_clip_224.metaclip_2pt5b", patch_size=16, loader="timm",
    ),
}

ZOO12_SPECS = {**PILOT8_SPECS, **EXPANSION4_SPECS}
CLASSIFIER_MODEL_IDS = frozenset({"deit3_small_patch16_224", "deit3_small_patch16_224_in21k"})


def resolve_model_ids(models_arg: str) -> list[str]:
    """Resolve --models selector ('all' -> Pilot-8, 'expansion4' -> 4 unseen, 'zoo12' -> all 12, or comma list)."""
    key = models_arg.strip()
    if key == "all":
        return list(PILOT8_SPECS.keys())
    if key == "expansion4":
        return list(EXPANSION4_SPECS.keys())
    if key == "zoo12":
        return list(ZOO12_SPECS.keys())
    mids = [m.strip() for m in key.split(",") if m.strip()]
    unknown = [m for m in mids if m not in ZOO12_SPECS]
    if unknown:
        raise ValueError(f"Unknown model_id(s): {unknown}. Available: {list(ZOO12_SPECS.keys())}")
    return mids


# --------------------------------------------------------------------------- Checkpoint loading
def _download_if_missing(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url} -> {dest}", flush=True)
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)
    return dest


def clean_ibot_state_dict(raw_ckpt: dict) -> dict[str, torch.Tensor]:
    """Extract ViT backbone keys from an official ByteDance iBOT checkpoint."""
    sd = raw_ckpt.get("state_dict", raw_ckpt.get("teacher", raw_ckpt))
    out = {}
    for k, v in sd.items():
        name = k
        for prefix in ("module.", "backbone."):
            if name.startswith(prefix):
                name = name[len(prefix):]
        if name.startswith("head.") or name == "masked_embed":
            continue
        out[name] = v
    return out


def clean_mocov3_state_dict(raw_ckpt: dict) -> dict[str, torch.Tensor]:
    """Extract online base_encoder ViT backbone keys from an official Meta MoCo v3 checkpoint."""
    sd = raw_ckpt.get("state_dict", raw_ckpt)
    out = {}
    for k, v in sd.items():
        if k.startswith("module.base_encoder."):
            name = k[len("module.base_encoder."):]
        elif k.startswith("base_encoder."):
            name = k[len("base_encoder."):]
        else:
            continue
        if name.startswith(("head.", "fc.")):
            continue
        out[name] = v
    return out


def load_pilot_model(
    model_id: str,
    device: torch.device,
    weights_cache_dir: Path,
    smoke_random: bool = False,
    keep_head: bool = False,
):
    """Load one pilot-8 model + adapter + exact per-model normalization (mean, std) + target_size."""
    import timm

    spec = ZOO12_SPECS[model_id]
    target_size = target_size_for_patch_size(spec["patch_size"])
    loader = spec["loader"]

    if loader == "open_clip":
        import open_clip

        tag = None if smoke_random else spec["pretrained_tag"]
        full_model, _, preprocess = open_clip.create_model_and_transforms(spec["arch"], pretrained=tag)
        model = full_model.visual
        adapter = OpenCLIPViTAdapter()
        adapter.prepare(model)
        mean, std = get_model_norm_stats(model, "open_clip_vit", openclip_preprocess=preprocess)
        return adapter, model.to(device), mean, std, target_size

    extra_kw = {} if keep_head else {"num_classes": 0}
    if spec["patch_size"] == 14:
        extra_kw.update(img_size=target_size, dynamic_img_size=True)

    if loader == "timm":
        model = timm.create_model(spec["arch"], pretrained=not smoke_random, **extra_kw)
    elif loader in ("ibot_url", "mocov3_url"):
        model = timm.create_model(spec["arch"], pretrained=False, **extra_kw)
        if not smoke_random:
            ckpt_path = _download_if_missing(spec["url"], weights_cache_dir / spec["filename"])
            raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sd = clean_ibot_state_dict(raw) if loader == "ibot_url" else clean_mocov3_state_dict(raw)
            msg = model.load_state_dict(sd, strict=False)
            if msg.missing_keys:
                raise RuntimeError(f"{model_id}: missing backbone keys after load: {msg.missing_keys}")
    else:
        raise ValueError(f"Unknown loader: {loader}")

    adapter = TimmViTAdapter()
    adapter.prepare(model)
    if "norm_override" in spec:
        mean, std = spec["norm_override"]
    else:
        mean, std = get_model_norm_stats(model, "timm_vit")
    return adapter, model.to(device), mean, std, target_size


# --------------------------------------------------------------------------- Fast single-pass batch extractor
@torch.no_grad()
def extract_last_third_batch(adapter, model, x: torch.Tensor, layers: list[int]) -> dict[int, dict[str, torch.Tensor]]:
    """Single forward pass extracting (q_g, k_patch, q_patches, head_weights) at every l in layers.

    Mathematically identical to calling qii.extract_m1_inputs_from_adapter() at each l separately,
    but runs blocks 0..max(layers) only ONCE per batch.
    """
    layer_set = set(layers)
    max_l = max(layers)
    out: dict[int, dict[str, torch.Tensor]] = {}

    if adapter.family == "timm_vit":
        layout = adapter.token_layout(model, x)
        cls_idx = layout["cls_idx"]
        patch_idx = layout["patch_idx"]
        z = model.patch_embed(x)
        z = model._pos_embed(z)
        z = model.patch_drop(z)
        z = model.norm_pre(z)

        for l, blk in enumerate(model.blocks[: max_l + 1]):
            if l in layer_set:
                a = blk.attn
                B, T, _ = z.shape
                qkv = a.qkv(blk.norm1(z)).reshape(B, T, 3, a.num_heads, a.head_dim).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                q, k = a.q_norm(q), a.k_norm(k)
                scale = float(a.scale)
                attn_row = attn_probs(q[:, :, cls_idx : cls_idx + 1], k, scale).squeeze(2)  # (B, H, T)
                W_O, _, gamma = get_output_proj_and_layerscale(adapter, model, l)
                w_h = compute_head_weights(attn_row, v, W_O, gamma=gamma)
                out[l] = {
                    "q_g": q[:, :, cls_idx, :].float().cpu(),
                    "k_patch": k[:, :, patch_idx, :].float().cpu(),
                    "q_patches": q[:, :, patch_idx, :].float().cpu(),
                    "head_weights": w_h.float().cpu(),
                    "scale": scale,
                    "grid": layout["grid"],
                }
            if l < max_l:
                z = blk(z)
        return out

    if adapter.family == "open_clip_vit":
        vt = adapter._visual(model)
        layout = adapter.token_layout(model)
        cls_idx = layout["cls_idx"]
        patch_idx = layout["patch_idx"]

        z = vt.conv1(x).reshape(x.shape[0], vt.conv1.out_channels, -1).permute(0, 2, 1)
        cls = vt.class_embedding.to(z.dtype).expand(z.shape[0], 1, -1)
        z = torch.cat([cls, z], dim=1) + vt.positional_embedding.to(z.dtype)
        z = vt.patch_dropout(z)
        z = vt.ln_pre(z)

        for l, blk in enumerate(vt.transformer.resblocks[: max_l + 1]):
            if l in layer_set:
                a = blk.attn
                B, T, _ = z.shape
                H, d = a.num_heads, a.head_dim
                zn = blk.ln_1(z)
                Wq, Wk, Wv = a.in_proj_weight.chunk(3, dim=0)
                bq, bk, bv = a.in_proj_bias.chunk(3, dim=0)
                q = (zn @ Wq.T + bq).reshape(B, T, H, d).transpose(1, 2)
                k = (zn @ Wk.T + bk).reshape(B, T, H, d).transpose(1, 2)
                v = (zn @ Wv.T + bv).reshape(B, T, H, d).transpose(1, 2)
                scale = float(d ** -0.5)
                attn_row = attn_probs(q[:, :, cls_idx : cls_idx + 1], k, scale).squeeze(2)
                W_O, _, gamma = get_output_proj_and_layerscale(adapter, model, l)
                w_h = compute_head_weights(attn_row, v, W_O, gamma=gamma)
                out[l] = {
                    "q_g": q[:, :, cls_idx, :].float().cpu(),
                    "k_patch": k[:, :, patch_idx, :].float().cpu(),
                    "q_patches": q[:, :, patch_idx, :].float().cpu(),
                    "head_weights": w_h.float().cpu(),
                    "scale": scale,
                    "grid": layout["grid"],
                }
            if l < max_l:
                z = blk(z)
        return out

    raise NotImplementedError(f"Batched extractor not implemented for family={adapter.family!r}")


# --------------------------------------------------------------------------- Bootstrap & Provenance
def bootstrap_ci(values: np.ndarray, n_resamples: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Image-level bootstrap mean and 95% CI (Guide §10.2)."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    if arr.size < 2:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return mean, float(lo), float(hi)


def paired_bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, n_resamples: int = 1000, seed: int = 0) -> dict:
    """Guide §10.2 & §11.2: paired cluster bootstrap of (a - b) over identical images."""
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(a64) & np.isfinite(b64)
    diff = a64[valid] - b64[valid]
    mean, lo, hi = bootstrap_ci(diff, n_resamples=n_resamples, seed=seed)
    return {
        "n_images": int(diff.size),
        "mean_diff": mean,
        "ci95_low": lo,
        "ci95_high": hi,
        "ci_excludes_zero": bool((lo > 0.0) or (hi < 0.0)) if np.isfinite(lo) else False,
    }


def _file_sha256_16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else "missing"


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        sha_file = REPO_ROOT / "GIT_SHA"
        return sha_file.read_text().strip() if sha_file.exists() else None


def _load_registry_meta() -> dict[str, dict]:
    p = REPO_ROOT / "configs" / "checkpoints.yaml"
    if not p.exists():
        return {}
    return {r["id"]: r for r in yaml.safe_load(p.read_text())}


# --------------------------------------------------------------------------- GPU-vectorized M1 + M2 layer scorer
@torch.no_grad()
def score_layer_gpu(
    qg_all: np.ndarray,
    kp_all: np.ndarray,
    qp_all: np.ndarray,
    wh_all: np.ndarray,
    gt_masks_np: np.ndarray,
    scale: float,
    grid: tuple[int, int],
    pca: qii.QueryPCA,
    device: torch.device,
    *,
    seeds: tuple[int, ...] = qii.DEFAULT_SEEDS,
    chunk_size: int = 64,
) -> dict:
    """Vectorized GPU/CPU computation of M1 (QII S1..S4 + entropy + box_auc) and M2 (omega^2_key + keymaps).

    Produces numerically identical results (<1e-5) to calling qii.compute_qii_image() and
    keydom.compute_key_dominance_image() per image, while keeping GPU utilization high (~30-100x faster).
    """
    I, H, d_h = qg_all.shape
    N = kp_all.shape[2]
    gh, gw = grid
    target_hw = gt_masks_np.shape[1:]
    n_seeds = len(seeds)

    # Normalize head weights per image: (I, H)
    w_np = wh_all.astype(np.float64)
    w_sum = w_np.sum(axis=-1, keepdims=True)
    w_np = np.where(w_sum > 0, w_np / np.where(w_sum > 0, w_sum, 1.0), 1.0 / H)
    w_t = torch.from_numpy(w_np).to(device=device, dtype=torch.float64)  # (I, H)

    # Pre-generate surrogates on CPU with the exact per-(seed, image_idx) RNGs of qii.py
    surr_q: dict[str, np.ndarray] = {}
    for S in qii.SURROGATES:
        arr = np.empty((I, H, n_seeds, d_h), dtype=np.float32)
        for i in range(I):
            for s_pos, seed in enumerate(seeds):
                if S == "S1":
                    q_s = qii.make_surrogate_s1(qg_all[i], qii._seed_rng(seed, i, 101))
                elif S == "S2":
                    q_s = qii.make_surrogate_s2(i, qg_all, seed, n_seeds=n_seeds)
                elif S == "S3":
                    q_s = qii.make_surrogate_s3(qp_all[i], i, seed, n_seeds=n_seeds)
                elif S == "S4":
                    q_s = qii.make_surrogate_s4(qg_all[i], pca, qii._seed_rng(seed, i, 404))
                arr[i, :, s_pos, :] = q_s
        surr_q[S] = arr

    # Pre-generate fixed |Q|=64 query sets with exact RNGs of keydom.py
    Q64_all = np.empty((I, H, keydom.NUM_QUERIES, d_h), dtype=np.float32)
    for i in range(I):
        Q64_all[i] = keydom.build_query_set_64(qg_all[i], qp_all[i], qg_all, image_idx=i, seed=0)

    # Output buffers on CPU
    ent_per_head = np.empty((I, H), dtype=np.float64)
    auc_attn_per_head = np.empty((I, H), dtype=np.float64)
    auc_attn_weighted = np.empty(I, dtype=np.float64)
    jsd_by_s = {S: np.empty((I, n_seeds, H), dtype=np.float64) for S in qii.SURROGATES}
    qii_by_s = {S: np.empty((I, n_seeds, H), dtype=np.float64) for S in qii.SURROGATES}
    qii_weighted_by_s = {S: np.empty(I, dtype=np.float64) for S in qii.SURROGATES}

    omega2_per_head = np.empty((I, H), dtype=np.float64)
    omega2_weighted = np.empty(I, dtype=np.float64)
    keymaps_fp16 = np.empty((I, H, gh, gw), dtype=np.float16)
    auc_km_weighted = np.empty(I, dtype=np.float64)

    for start in range(0, I, chunk_size):
        end = min(I, start + chunk_size)
        B = end - start
        qg_c = torch.from_numpy(qg_all[start:end]).to(device=device, dtype=torch.float32)   # (B, H, d_h)
        kp_c = torch.from_numpy(kp_all[start:end]).to(device=device, dtype=torch.float32)   # (B, H, N, d_h)
        w_c = w_t[start:end]                                                                 # (B, H) fp64
        m_c = torch.from_numpy(gt_masks_np[start:end]).to(device=device)                    # (B, H_img, W_img)

        # --- M1: true patch attention, entropy, bilinear upsample + exact GPU box-AUC ---
        a_true = patch_attention(qg_c, kp_c, scale)  # (B, H, N) fp32
        ent_c = normalized_entropy_torch(a_true, dim=-1)  # (B, H) fp64
        ent_per_head[start:end] = ent_c.cpu().numpy()

        a_grids = a_true.reshape(B, H, gh, gw)
        a_up = torch.nn.functional.interpolate(a_grids, size=target_hw, mode="bilinear", align_corners=False)
        auc_attn_per_head[start:end] = box_auc_torch(a_up, m_c.unsqueeze(1)).cpu().numpy()

        agg_grid = torch.einsum("bh,bhxy->bxy", w_c.float(), a_grids)
        agg_up = torch.nn.functional.interpolate(agg_grid.unsqueeze(1), size=target_hw, mode="bilinear", align_corners=False).squeeze(1)
        auc_attn_weighted[start:end] = box_auc_torch(agg_up, m_c).cpu().numpy()

        # --- M1: surrogates S1..S4 on GPU ---
        for S in qii.SURROGATES:
            qs_c = torch.from_numpy(surr_q[S][start:end]).to(device=device, dtype=torch.float32)  # (B, H, S_n, d_h)
            a_surr = patch_attention(qs_c, kp_c, scale)  # (B, H, S_n, N) fp32
            jsd_c = jsd_nats_torch(a_true.unsqueeze(2).expand_as(a_surr), a_surr, dim=-1)  # (B, H, S_n) fp64
            qii_c = 1.0 - jsd_c / LOG2
            # Store as (B, S_n, H)
            jsd_by_s[S][start:end] = jsd_c.transpose(1, 2).cpu().numpy()
            qii_by_s[S][start:end] = qii_c.transpose(1, 2).cpu().numpy()
            qii_h_mean = qii_c.mean(dim=-1)  # (B, H)
            qii_weighted_by_s[S][start:end] = (w_c * qii_h_mean).sum(dim=-1).cpu().numpy()

        # --- M2: two-way ANOVA omega^2_key on GPU in fp64 ---
        Q64_c = torch.from_numpy(Q64_all[start:end]).to(device=device, dtype=torch.float32)  # (B, H, 64, d_h)
        logits = ((Q64_c @ kp_c.transpose(-2, -1)) * float(scale)).to(torch.float64)         # (B, H, 64, N) fp64
        n_q, n_k = logits.shape[-2], logits.shape[-1]
        mu = logits.mean(dim=(-2, -1), keepdim=True)
        alpha_bar = logits.mean(dim=-1, keepdim=True) - mu
        beta_bar = logits.mean(dim=-2, keepdim=True) - mu
        ss_total = ((logits - mu) ** 2).sum(dim=(-2, -1))
        ss_query = float(n_k) * (alpha_bar ** 2).sum(dim=(-2, -1))
        ss_key = float(n_q) * (beta_bar ** 2).sum(dim=(-2, -1))
        ss_resid = (ss_total - ss_query - ss_key).clamp_min(0.0)
        ms_resid = ss_resid / float((n_q - 1) * (n_k - 1))
        denom = ss_total + ms_resid
        om2 = torch.where(denom > 0.0, (ss_key - float(n_k - 1) * ms_resid) / denom, torch.zeros_like(denom)).clamp(0.0, 1.0)

        omega2_per_head[start:end] = om2.cpu().numpy()
        omega2_weighted[start:end] = (w_c * om2).sum(dim=-1).cpu().numpy()

        km = beta_bar.squeeze(-2).reshape(B, H, gh, gw).float()  # (B, H, gh, gw)
        keymaps_fp16[start:end] = km.cpu().numpy().astype(np.float16)

        km_agg = torch.einsum("bh,bhxy->bxy", w_c.float(), km)
        flat = km_agg.reshape(B, -1)
        lo = flat.min(dim=-1).values.reshape(B, 1, 1)
        hi = flat.max(dim=-1).values.reshape(B, 1, 1)
        km_agg = torch.where(hi > lo, (km_agg - lo) / (hi - lo).clamp_min(1e-12), torch.zeros_like(km_agg))
        km_up = torch.nn.functional.interpolate(km_agg.unsqueeze(1), size=target_hw, mode="bilinear", align_corners=False).squeeze(1)
        auc_km_weighted[start:end] = box_auc_torch(km_up, m_c).cpu().numpy()

    ent_weighted = (w_np * ent_per_head).sum(axis=-1)
    return {
        "w_norm": w_np,
        "ent_per_head": ent_per_head,
        "ent_weighted": ent_weighted,
        "auc_attn_per_head": auc_attn_per_head,
        "auc_attn_weighted": auc_attn_weighted,
        "jsd_by_s": jsd_by_s,
        "qii_by_s": qii_by_s,
        "qii_weighted_by_s": qii_weighted_by_s,
        "omega2_per_head": omega2_per_head,
        "omega2_weighted": omega2_weighted,
        "keymaps_fp16": keymaps_fp16,
        "auc_km_weighted": auc_km_weighted,
    }


def _write_qii_columns(col_chunks: list[dict[str, np.ndarray]], out_dir: Path, model_id: str) -> Path:
    """Fast columnar Parquet writer (avoids constructing ~500k Python dicts); falls back to CSV."""
    cols = {k: np.concatenate([c[k] for c in col_chunks], axis=0) for k in qii.QII_SCHEMA_COLUMNS}
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
        import pyarrow  # noqa: F401

        dest = out_dir / f"{model_id}.parquet"
        tmp = dest.with_suffix(".tmp.parquet")
        pd.DataFrame(cols).to_parquet(tmp, index=False)
        os.replace(tmp, dest)
        return dest
    except Exception:
        dest = out_dir / f"{model_id}.csv"
        tmp = dest.with_suffix(".tmp.csv")
        n = len(cols["qii"])
        with tmp.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(qii.QII_SCHEMA_COLUMNS)
            for idx in range(n):
                w.writerow([cols[k][idx] for k in qii.QII_SCHEMA_COLUMNS])
        os.replace(tmp, dest)
        return dest


# --------------------------------------------------------------------------- Per-model runner
def run_one_model(
    model_id: str,
    samples,
    device: torch.device,
    out_dir: Path,
    weights_cache_dir: Path,
    reg_meta: dict,
    *,
    batch_size: int = 32,
    seeds: tuple[int, ...] = qii.DEFAULT_SEEDS,
    pca_k: int = qii.DEFAULT_PCA_K,
    prefer_seg: bool = True,
    smoke_random: bool = False,
) -> dict:
    t0 = time.perf_counter()
    adapter, model, mean, std, target_size = load_pilot_model(
        model_id, device, weights_cache_dir, smoke_random=smoke_random
    )
    n_blocks = adapter.n_blocks(model)
    layers = last_third_blocks(n_blocks)
    tf = build_image_transform(target_size, mean, std)

    n_img = len(samples)
    print(f"\n[{model_id}] device={device}, res={target_size}, layers={layers}, norm=({[round(m,3) for m in mean]}, {[round(s,3) for s in std]})", flush=True)

    # 1. Single-pass batched GPU extraction across last third of blocks
    t_ext0 = time.perf_counter()
    collected: dict[int, dict[str, list[torch.Tensor]]] = {
        l: {"q_g": [], "k_patch": [], "q_patches": [], "head_weights": []} for l in layers
    }
    scales: dict[int, float] = {}
    grid = (14, 14)
    gt_masks: list[np.ndarray] = []

    for start in range(0, n_img, batch_size):
        batch_samples = samples[start : start + batch_size]
        imgs_t = []
        for s in batch_samples:
            pil = Image.open(s.image_path).convert("RGB")
            ow, oh = pil.size
            imgs_t.append(tf(pil))
            if isinstance(s, ImageNetSample):
                mask, _ = get_transformed_imagenet_gt_mask(s, ow, oh, target_size)
            else:
                mask, _ = get_transformed_gt_mask(s, ow, oh, target_size, prefer_seg=prefer_seg)
            gt_masks.append(mask)
        xb = torch.stack(imgs_t, dim=0).to(device)
        batch_out = extract_last_third_batch(adapter, model, xb, layers)
        for l in layers:
            for key in ("q_g", "k_patch", "q_patches", "head_weights"):
                collected[l][key].append(batch_out[l][key])
            scales[l] = batch_out[l]["scale"]
            grid = batch_out[l]["grid"]

    extract_sec = time.perf_counter() - t_ext0
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    gt_masks_np = np.stack(gt_masks, axis=0)  # (I, S, S) uint8
    data_by_layer: dict[int, dict[str, np.ndarray]] = {
        l: {k: torch.cat(v, dim=0).numpy() for k, v in collected[l].items()} for l in layers
    }
    del collected

    # 2. Vectorized GPU M1 (QII) + M2 (Key Dominance) per layer
    row_meta = reg_meta.get(model_id, {})
    objective = row_meta.get("objective", "unknown")
    set_name = row_meta.get("set", "pilot")
    image_ids_arr = (
        np.array([s.image_id for s in samples], dtype=object)
        if samples and isinstance(samples[0], ImageNetSample)
        else np.array([s.cub_id for s in samples], dtype=np.int64)
    )
    grid_str = f"{grid[0]}x{grid[1]}"

    qii_col_chunks: list[dict[str, np.ndarray]] = []
    layer_summaries: list[dict] = []
    per_image_arrays: dict[str, dict[str, list[float]]] = {}
    sanity_checks: dict[str, dict] = {}
    H = data_by_layer[layers[0]]["k_patch"].shape[1]
    keymaps_all = np.zeros((n_img, len(layers), H, grid[0], grid[1]), dtype=np.float16)
    n_seeds = len(seeds)
    seeds_arr = np.array(seeds, dtype=np.int64)

    for li, l in enumerate(layers):
        qg_all = data_by_layer[l]["q_g"]
        kp_all = data_by_layer[l]["k_patch"]
        qp_all = data_by_layer[l]["q_patches"]
        wh_all = data_by_layer[l]["head_weights"]
        scale = scales[l]

        # Fit & cache empirical PCA for S4 (§2.2) + query-concentration diagnostic (||q_g - mu|| / ||mu||)
        pca = qii.fit_query_pca(qg_all, k=pca_k)
        pca.save(out_dir / "pca" / f"{model_id}_l{l}.npz")
        qg64 = qg_all.astype(np.float64)
        mu_q = qg64.mean(axis=0, keepdims=True)  # (1, H, d_h)
        mu_norm = np.linalg.norm(mu_q, axis=-1)  # (1, H)
        qg_norm = np.linalg.norm(qg64, axis=-1)  # (I, H)
        cos_mu = np.sum(qg64 * mu_q, axis=-1) / np.maximum(qg_norm * mu_norm, 1e-12)
        rel_spread = np.linalg.norm(qg64 - mu_q, axis=-1) / np.maximum(mu_norm, 1e-12)
        q_cos_mu = float(np.mean(cos_mu))
        q_rel_spread = float(np.mean(rel_spread))

        # Sanity checks on image 0 (Guide §2.3 & §3.5)
        shuf_ok, shuf_diff = qii.shuffled_key_null_check(
            qg_all[0], kp_all[0], scale, grid,
            q_global_pool=qg_all, image_idx=0, q_patches=qp_all[0], pca=pca,
            head_weights=wh_all[0], seeds=seeds,
        )
        Q64_0 = keydom.build_query_set_64(qg_all[0], qp_all[0], qg_all, image_idx=0, seed=0)
        kd_shuf = keydom.spatial_shuffle_check(Q64_0, kp_all[0], scale, grid, head_weights=wh_all[0], gt_mask=gt_masks_np[0], seed=0)
        logits_0 = (torch.from_numpy(Q64_0) @ torch.from_numpy(kp_all[0]).transpose(-2, -1)) * scale
        kd_perm = keydom.permutation_null_check(logits_0, seed=0)
        sanity_checks[f"l{l}"] = {
            "qii_shuffled_key_ok": bool(shuf_ok),
            "qii_shuffled_key_max_diff": float(shuf_diff),
            "kd_spatial_shuffle_ok": bool(kd_shuf["ok"]),
            "kd_spatial_shuffle_max_diff": float(kd_shuf["max_omega2_diff"]),
            "kd_perm_null_omega2_mean": float(kd_perm["omega2_mean"]),
            "kd_perm_null_eta2_mean": float(kd_perm["eta2_mean"]),
        }

        scored = score_layer_gpu(
            qg_all, kp_all, qp_all, wh_all, gt_masks_np, scale, grid, pca, device, seeds=seeds,
        )
        keymaps_all[:, li] = scored["keymaps_fp16"]

        # Build columnar §2.6 schema arrays for this layer (shape per S: I * n_seeds * H)
        l_frac = float(l + 1) / float(n_blocks)
        n_rows_s = n_img * n_seeds * H
        img_rep = np.repeat(image_ids_arr, n_seeds * H)
        seed_rep = np.tile(np.repeat(seeds_arr, H), n_img)
        head_rep = np.tile(np.arange(H, dtype=np.int64), n_img * n_seeds)
        ent_rep = np.repeat(scored["ent_per_head"][:, None, :], n_seeds, axis=1).ravel()
        auc_rep = np.repeat(scored["auc_attn_per_head"][:, None, :], n_seeds, axis=1).ravel()
        w_rep = np.repeat(scored["w_norm"][:, None, :], n_seeds, axis=1).ravel()
        for S in qii.SURROGATES:
            qii_col_chunks.append({
                "model_id": np.full(n_rows_s, model_id, dtype=object),
                "objective": np.full(n_rows_s, objective, dtype=object),
                "set": np.full(n_rows_s, set_name, dtype=object),
                "layer": np.full(n_rows_s, l, dtype=np.int64),
                "layer_frac": np.full(n_rows_s, l_frac, dtype=np.float64),
                "head": head_rep,
                "image_id": img_rep,
                "surrogate": np.full(n_rows_s, S, dtype=object),
                "seed": seed_rep,
                "jsd": scored["jsd_by_s"][S].ravel(),
                "qii": scored["qii_by_s"][S].ravel(),
                "entropy_norm": ent_rep,
                "box_auc": auc_rep,
                "head_weight": w_rep,
                "resolution": np.full(n_rows_s, target_size, dtype=np.int64),
                "grid": np.full(n_rows_s, grid_str, dtype=object),
            })

        ent_vals = scored["ent_weighted"]
        auc_attn_vals = scored["auc_attn_weighted"]
        omega2_vals = scored["omega2_weighted"]
        auc_km_vals = scored["auc_km_weighted"]
        qii_s_vals = scored["qii_weighted_by_s"]

        ent_m, ent_lo, ent_hi = bootstrap_ci(ent_vals)
        s4_m, s4_lo, s4_hi = bootstrap_ci(qii_s_vals["S4"])
        om_m, om_lo, om_hi = bootstrap_ci(omega2_vals)
        aa_m, aa_lo, aa_hi = bootstrap_ci(auc_attn_vals)
        ak_m, ak_lo, ak_hi = bootstrap_ci(auc_km_vals)
        s1_m = float(np.nanmean(qii_s_vals["S1"]))
        s2_m = float(np.nanmean(qii_s_vals["S2"]))
        s3_m = float(np.nanmean(qii_s_vals["S3"]))
        status_s4 = qii.classify_qii_outcome(s4_m, ent_m)

        layer_summaries.append(
            {
                "model_id": model_id,
                "objective": objective,
                "layer": l,
                "layer_frac": round((l + 1) / float(n_blocks), 4),
                "resolution": target_size,
                "n_images": n_img,
                "qii_s1": round(s1_m, 4),
                "qii_s2": round(s2_m, 4),
                "qii_s3": round(s3_m, 4),
                "qii_s4": round(s4_m, 4),
                "qii_s4_ci95_lo": round(s4_lo, 4),
                "qii_s4_ci95_hi": round(s4_hi, 4),
                "entropy_norm": round(ent_m, 4),
                "entropy_ci95_lo": round(ent_lo, 4),
                "entropy_ci95_hi": round(ent_hi, 4),
                "box_auc_attn": round(aa_m, 4),
                "box_auc_attn_ci95_lo": round(aa_lo, 4),
                "box_auc_attn_ci95_hi": round(aa_hi, 4),
                "omega2_key": round(om_m, 4),
                "omega2_key_ci95_lo": round(om_lo, 4),
                "omega2_key_ci95_hi": round(om_hi, 4),
                "box_auc_keymap": round(ak_m, 4),
                "box_auc_keymap_ci95_lo": round(ak_lo, 4),
                "box_auc_keymap_ci95_hi": round(ak_hi, 4),
                "q_cos_mu": round(q_cos_mu, 4),
                "q_rel_spread": round(q_rel_spread, 4),
                "status_s4": status_s4,
            }
        )
        per_image_arrays[f"l{l}"] = {
            "qii_s4": qii_s_vals["S4"].tolist(),
            "entropy_norm": ent_vals.tolist(),
            "box_auc_attn": auc_attn_vals.tolist(),
            "omega2_key": omega2_vals.tolist(),
            "box_auc_keymap": auc_km_vals.tolist(),
        }
        print(
            f"  l={l:2d} ({(l+1)/n_blocks:.2f}) | QII S1={s1_m:.3f} S2={s2_m:.3f} S3={s3_m:.3f} S4={s4_m:.3f} "
            f"| H={ent_m:.3f} ({status_s4}) | boxAUC(attn)={aa_m:.3f} | w2_key={om_m:.3f} | boxAUC(km)={ak_m:.3f} "
            f"| cos(q,mu)={q_cos_mu:.3f} spread={q_rel_spread:.3f}",
            flush=True,
        )

    # Save per-model outputs atomically
    keydom.save_keymaps(keymaps_all, out_dir / "keymaps" / f"{model_id}.npy")
    _write_qii_columns(qii_col_chunks, out_dir / "qii", model_id)

    total_sec = time.perf_counter() - t0
    model_result = {
        "model_id": model_id,
        "objective": objective,
        "n_blocks": n_blocks,
        "layers": layers,
        "resolution": target_size,
        "norm_mean": list(mean),
        "norm_std": list(std),
        "n_images": n_img,
        "extract_sec": round(extract_sec, 2),
        "extract_img_per_sec": round(n_img / max(extract_sec, 1e-6), 2),
        "total_sec": round(total_sec, 2),
        "sanity_checks": sanity_checks,
        "layer_summaries": layer_summaries,
        "per_image": per_image_arrays,
    }
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    tmp_json = out_dir / "models" / f"{model_id}.json.tmp"
    tmp_json.write_text(json.dumps(model_result, indent=2))
    os.replace(tmp_json, out_dir / "models" / f"{model_id}.json")
    return model_result


# --------------------------------------------------------------------------- Cross-model summary, Fix 3 & G1
def _pick_l_star(layer_summaries: list[dict]) -> dict:
    """Guide §7.1: l* = argmax_{l : E(l) <= 0.9} omega2_key(l); fallback to max omega2_key if none eligible."""
    eligible = [r for r in layer_summaries if r["entropy_norm"] <= qii.ENTROPY_ELIGIBILITY_THRESHOLD]
    pool = eligible if eligible else layer_summaries
    return max(pool, key=lambda r: r["omega2_key"])


def write_cross_model_report(model_results: dict[str, dict], out_dir: Path) -> dict:
    rows: list[dict] = []
    for mid, mr in model_results.items():
        rows.extend(mr["layer_summaries"])

    summary_cols = list(rows[0].keys()) if rows else []
    write_records(rows, out_dir / "summary_by_model_layer.csv", columns=summary_cols)

    # Fix 3 across-model Spearman at final layer AND at l*
    qii_last, kd_last = {}, {}
    qii_lstar, kd_lstar, lstar_by_model = {}, {}, {}
    for mid, mr in model_results.items():
        ls = mr["layer_summaries"]
        last_r = max(ls, key=lambda r: r["layer"])
        star_r = _pick_l_star(ls)
        qii_last[mid], kd_last[mid] = last_r["qii_s4"], last_r["omega2_key"]
        qii_lstar[mid], kd_lstar[mid] = star_r["qii_s4"], star_r["omega2_key"]
        lstar_by_model[mid] = {"layer": star_r["layer"], "layer_frac": star_r["layer_frac"], "status_s4": star_r["status_s4"]}

    fix3_last = keydom.qii_kd_spearman(qii_last, kd_last)
    fix3_lstar = keydom.qii_kd_spearman(qii_lstar, kd_lstar)

    # G1 gate check (Guide §11.2): paired image bootstrap DINO S/16 vs DeiT-III S/16
    g1 = None
    if "dino_vits16" in model_results and "deit3_small_patch16_224" in model_results:
        dino_r = model_results["dino_vits16"]
        deit_r = model_results["deit3_small_patch16_224"]
        dino_lstar = f"l{lstar_by_model['dino_vits16']['layer']}"
        deit_lstar = f"l{lstar_by_model['deit3_small_patch16_224']['layer']}"
        dino_last = f"l{max(dino_r['layers'])}"
        deit_last = f"l{max(deit_r['layers'])}"
        g1 = {
            "at_l_star": {
                "dino_layer": dino_lstar,
                "deit3_layer": deit_lstar,
                "qii_s4_diff_dino_minus_deit3": paired_bootstrap_diff_ci(
                    dino_r["per_image"][dino_lstar]["qii_s4"],
                    deit_r["per_image"][deit_lstar]["qii_s4"],
                ),
                "box_auc_diff_dino_minus_deit3": paired_bootstrap_diff_ci(
                    dino_r["per_image"][dino_lstar]["box_auc_attn"],
                    deit_r["per_image"][deit_lstar]["box_auc_attn"],
                ),
            },
            "at_final_layer": {
                "qii_s4_diff_dino_minus_deit3": paired_bootstrap_diff_ci(
                    dino_r["per_image"][dino_last]["qii_s4"],
                    deit_r["per_image"][deit_last]["qii_s4"],
                ),
                "box_auc_diff_dino_minus_deit3": paired_bootstrap_diff_ci(
                    dino_r["per_image"][dino_last]["box_auc_attn"],
                    deit_r["per_image"][deit_last]["box_auc_attn"],
                ),
            },
        }

    report_obj = {
        "l_star_by_model": lstar_by_model,
        "fix3_spearman_final_layer": fix3_last,
        "fix3_spearman_l_star": fix3_lstar,
        "g1_dino_vs_deit3": g1,
    }
    (out_dir / "fix3_and_g1.json").write_text(json.dumps(report_obj, indent=2))

    # Readable Markdown table
    lines = [
        "# Pilot M1 (QII) + M2 (Key Dominance) Summary\n",
        "| Model | Obj | Layer (frac) | QII S1 | QII S2 | QII S3 | QII S4 [95% CI] | H_norm | Status | box-AUC(attn) | w2_key [95% CI] | box-AUC(km) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model_id']} | {r['objective']} | {r['layer']} ({r['layer_frac']:.2f}) | "
            f"{r['qii_s1']:.3f} | {r['qii_s2']:.3f} | {r['qii_s3']:.3f} | "
            f"{r['qii_s4']:.3f} [{r['qii_s4_ci95_lo']:.3f}, {r['qii_s4_ci95_hi']:.3f}] | "
            f"{r['entropy_norm']:.3f} | {r['status_s4']} | {r['box_auc_attn']:.3f} | "
            f"{r['omega2_key']:.3f} [{r['omega2_key_ci95_lo']:.3f}, {r['omega2_key_ci95_hi']:.3f}] | "
            f"{r['box_auc_keymap']:.3f} |"
        )
    lines.append("")
    lines.append(f"**Fix 3 Spearman (final layer):** rho = {fix3_last['rho']:.3f}, p = {fix3_last['p_value']:.4g} -> **{fix3_last['decision']}**")
    lines.append(f"**Fix 3 Spearman (l*):** rho = {fix3_lstar['rho']:.3f}, p = {fix3_lstar['p_value']:.4g} -> **{fix3_lstar['decision']}**")
    if g1 is not None:
        d_star = g1["at_l_star"]["qii_s4_diff_dino_minus_deit3"]
        a_star = g1["at_l_star"]["box_auc_diff_dino_minus_deit3"]
        lines.append(
            f"**G1 (DINO - DeiT-III at l*):** dQII_S4 = {d_star['mean_diff']:+.3f} "
            f"[{d_star['ci95_low']:+.3f}, {d_star['ci95_high']:+.3f}] (excl0={d_star['ci_excludes_zero']}); "
            f"dBoxAUC = {a_star['mean_diff']:+.3f} [{a_star['ci95_low']:+.3f}, {a_star['ci95_high']:+.3f}]"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    return report_obj


# --------------------------------------------------------------------------- CLI
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Run pilot Measures M1 (QII) + M2 (omega^2_key) on CUB-200-2011 or ImageNet-1K val.")
    ap.add_argument("--dataset", choices=["cub", "imagenet"], default="cub")
    ap.add_argument("--cub_dir", default=None, help="Path to CUB_200_2011 directory (containing images.txt).")
    ap.add_argument("--images_dir", default=None, help="Path to ImageNet-1K val images directory (for --dataset imagenet).")
    ap.add_argument("--imagenet_xml_dir", default=None, help="Path to ImageNet-1K val XML annotations directory.")
    ap.add_argument("--seg_dir", default=None, help="Optional path to CUB segmentations directory.")
    ap.add_argument("--out_dir", default="results/pilot_m1_m2_cub", help="Output directory.")
    ap.add_argument("--split_file", default=None, help="Fixed split ID file (defaults to configs/splits/{cub_pilot_500|imagenet_pilot_1000}_seed0.txt).")
    ap.add_argument("--n_images", type=int, default=None, help="Number of images (defaults: 500 CUB, 1000 ImageNet).")
    ap.add_argument("--seed", type=int, default=0, help="Split RNG seed (Guide §8.4: 0).")
    ap.add_argument("--models", default="all", help="'all' for all 8 pilot models, or comma-separated IDs.")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--weights_cache_dir", default=str(Path.home() / ".cache" / "queryblind_weights"))
    ap.add_argument("--no_seg", action="store_true", help="Force boxes even if CUB segmentations/ exist.")
    ap.add_argument("--resume", action="store_true", help="Skip models that already have out_dir/models/<id>.json.")
    ap.add_argument("--smoke_random", action="store_true", help="Random-init weights (offline smoke test only).")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    default_split = "configs/splits/cub_pilot_500_seed0.txt" if args.dataset == "cub" else "configs/splits/imagenet_pilot_1000_seed0.txt"
    split_file = Path(args.split_file or default_split)
    if not split_file.is_absolute():
        split_file = REPO_ROOT / split_file
    n_images = args.n_images if args.n_images is not None else (500 if args.dataset == "cub" else 1000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_ids = resolve_model_ids(args.models)

    if args.dataset == "cub":
        if not args.cub_dir:
            raise ValueError("--cub_dir is required for --dataset cub")
        samples = load_cub_pilot_split(
            args.cub_dir, n_images=n_images, seed=args.seed, split_file=split_file, seg_dir=args.seg_dir,
        )
        print(f"Loaded {len(samples)} CUB test images (split file: {split_file})", flush=True)
        save_box_overlays(samples, 224, out_dir / "box_overlays_20_224px.png", n_images=20)
        save_box_overlays(samples, 196, out_dir / "box_overlays_20_196px.png", n_images=20)
        ds_label = "CUB-200-2011"
        n_seg = sum(s.seg_path is not None for s in samples)
    else:
        if not args.images_dir or not args.imagenet_xml_dir:
            raise ValueError("--images_dir and --imagenet_xml_dir are required for --dataset imagenet")
        samples = load_imagenet_val_pilot_split(
            args.images_dir, args.imagenet_xml_dir, n_images=n_images, seed=args.seed, split_file=split_file,
        )
        print(f"Loaded {len(samples)} ImageNet-1K val images (split file: {split_file})", flush=True)
        save_imagenet_box_overlays(samples, 224, out_dir / "box_overlays_20_224px.png", n_images=20)
        save_imagenet_box_overlays(samples, 196, out_dir / "box_overlays_20_196px.png", n_images=20)
        ds_label = "ImageNet-1K-val"
        n_seg = 0

    reg_meta = _load_registry_meta()
    weights_cache = Path(args.weights_cache_dir)
    results: dict[str, dict] = {}
    t_all = time.perf_counter()

    for mid in model_ids:
        done_file = out_dir / "models" / f"{mid}.json"
        if args.resume and done_file.exists():
            print(f"[{mid}] --resume: loaded existing {done_file}", flush=True)
            results[mid] = json.loads(done_file.read_text())
            continue
        results[mid] = run_one_model(
            mid, samples, device, out_dir, weights_cache, reg_meta,
            batch_size=args.batch_size, prefer_seg=not args.no_seg, smoke_random=args.smoke_random,
        )

    cross = write_cross_model_report(results, out_dir)
    cfg_payload = json.dumps({"models": model_ids, "n_images": len(samples), "seed": args.seed, "smoke_random": args.smoke_random}, sort_keys=True)
    meta = {
        "git_sha": _git_sha(),
        "config_hash": hashlib.sha256(cfg_payload.encode()).hexdigest()[:16],
        "file_sha256_16": {
            rel: _file_sha256_16(REPO_ROOT / rel)
            for rel in (
                "measures/_common.py", "measures/qii.py", "measures/keydom.py",
                "eval/preprocess.py", "scripts/run_pilot_m1_m2.py", "configs/checkpoints.yaml",
            )
        },
        "device": str(device),
        "dataset": ds_label,
        "split_file": str(split_file),
        "n_images": len(samples),
        "n_with_seg_mask": n_seg,
        "smoke_random": bool(args.smoke_random),
        "total_seconds": round(time.perf_counter() - t_all, 2),
        "throughput_by_model": {
            m: {"extract_sec": r["extract_sec"], "img_per_sec": r["extract_img_per_sec"], "total_sec": r["total_sec"]}
            for m, r in results.items()
        },
        "sanity_checks_all_passed": all(
            c["qii_shuffled_key_ok"] and c["kd_spatial_shuffle_ok"]
            for r in results.values() for c in r["sanity_checks"].values()
        ),
        "unix_time": int(time.time()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print("\n" + (out_dir / "summary.md").read_text(), flush=True)
    return cross


if __name__ == "__main__":
    main()
