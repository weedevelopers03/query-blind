"""Exact block-by-block intervened forward passes for timm_vit and open_clip_vit (Guide §6).

Guarantees:
1. With no edits, reproduces native model.forward_features(x) / model(x) / model.encode_image(x) to <1e-5.
2. Under I1 (query replacement) and I2 (token-group knockout, zero-ablation or mean-value ablation):
   - ONLY the global [CLS] token's attention read-out row is modified.
   - ALL patch queries, keys, values, and patch token outputs remain identical to the clean forward pass.
3. Supports caching the hidden state z_start at block `start_layer` (e.g. block 8) so repeated sweeps
   across L_int in {1,2,3,4}, surrogate families, token groups, and 5 seeds execute only the last 1-4 blocks.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from adapters.base import attn_probs
from measures._common import compute_head_weights, get_output_proj_and_layerscale, patch_attention


# --------------------------------------------------------------------------- Stem & Head helpers
@torch.no_grad()
def forward_to_block_input(adapter, model, x: torch.Tensor, target_block: int) -> tuple[torch.Tensor, dict]:
    """Run stem + blocks 0..(target_block - 1) and return (z_in_target_block, layout)."""
    if adapter.family == "timm_vit":
        layout = adapter.token_layout(model, x)
        z = model.patch_embed(x)
        z = model._pos_embed(z)
        z = model.patch_drop(z)
        z = model.norm_pre(z)
        for blk in model.blocks[:target_block]:
            z = blk(z)
        return z, layout

    if adapter.family == "open_clip_vit":
        vt = adapter._visual(model)
        layout = adapter.token_layout(model)
        z = vt.conv1(x).reshape(x.shape[0], vt.conv1.out_channels, -1).permute(0, 2, 1)
        cls = vt.class_embedding.to(z.dtype).expand(z.shape[0], 1, -1)
        z = torch.cat([cls, z], dim=1) + vt.positional_embedding.to(z.dtype)
        z = vt.patch_dropout(z)
        z = vt.ln_pre(z)
        for blk in vt.transformer.resblocks[:target_block]:
            z = blk(z)
        return z, layout

    raise NotImplementedError(f"forward_to_block_input not implemented for family={adapter.family!r}")


@torch.no_grad()
def read_global_embedding_and_logits(
    adapter, model, z_final: torch.Tensor, layout: dict
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply final norm + projection/head to block-output sequence z_final (B, T, D).

    Returns:
        emb:    (B, D_out) global representation after final norm (and OpenCLIP visual proj if present).
        logits: (B, C) classification logits if the model has a real classifier head (e.g. DeiT-III), else None.
    """
    cls_idx = layout["cls_idx"]
    if adapter.family == "timm_vit":
        x_norm = model.norm(z_final)
        emb = x_norm[:, cls_idx, :]
        # Detect real classification head (num_classes > 0 and non-Identity head)
        head = getattr(model, "head", None)
        num_classes = int(getattr(model, "num_classes", 0))
        if num_classes > 0 and head is not None and not isinstance(head, torch.nn.Identity):
            logits = model.forward_head(x_norm)
        else:
            logits = None
        return emb.float(), (logits.float() if logits is not None else None)

    if adapter.family == "open_clip_vit":
        vt = adapter._visual(model)
        pooled = vt.ln_post(z_final[:, cls_idx, :])
        if getattr(vt, "proj", None) is not None:
            pooled = pooled @ vt.proj
        return pooled.float(), None

    raise NotImplementedError(f"read_global_embedding_and_logits not implemented for family={adapter.family!r}")


@torch.no_grad()
def read_patch_mean_embedding(
    adapter, model, z_final: torch.Tensor, layout: dict
) -> torch.Tensor:
    """Mean-pooled patch representation after final norm (standard read-out for MAE / patch-pooled ViTs)."""
    patch_idx = layout["patch_idx"]
    if adapter.family == "timm_vit":
        # Note: timm MAE uses fc_norm on the mean-pooled patch tokens when global_pool='avg';
        # if fc_norm is present and not Identity, apply fc_norm(patch_mean), else norm(z)[:, patch_idx].mean(1).
        fc_norm = getattr(model, "fc_norm", None)
        if fc_norm is not None and not isinstance(fc_norm, torch.nn.Identity):
            pooled = fc_norm(z_final[:, patch_idx, :].mean(dim=1))
        else:
            x_norm = model.norm(z_final)
            pooled = x_norm[:, patch_idx, :].mean(dim=1)
        return pooled.float()

    if adapter.family == "open_clip_vit":
        vt = adapter._visual(model)
        pooled = vt.ln_post(z_final[:, patch_idx, :]).mean(dim=1)
        if getattr(vt, "proj", None) is not None:
            pooled = pooled @ vt.proj
        return pooled.float()

    raise NotImplementedError(f"read_patch_mean_embedding not implemented for family={adapter.family!r}")


# --------------------------------------------------------------------------- Intervened single-block attention
def _apply_cls_only_attention(
    q: torch.Tensor,         # (B, H, T, d_h)
    k: torch.Tensor,         # (B, H, T, d_h)
    v: torch.Tensor,         # (B, H, T, d_h)
    scale: float,
    cls_idx: int,
    *,
    q_cls_override: torch.Tensor | None = None,    # (B, H, d_h) for I1
    ko_mask: torch.Tensor | None = None,           # (B, T) bool True = knock out for I2
    ko_mode: str = "zero",                         # "zero" or "mean_v"
    v_mean: torch.Tensor | None = None,            # (H, d_h) dataset-mean value vector for I2 mean_v
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (attn_out_cat (B, T, H*d_h), cls_attn_row (B, H, T)) modifying ONLY the [CLS] row.

    Patch rows (t != cls_idx) use the exact clean q, k, v so the patch stream is bit-for-bit invariant.
    """
    B, H, T, d_h = q.shape
    # 1. Clean attention for ALL tokens (preserves patch stream identically)
    attn_clean = attn_probs(q, k, scale)  # (B, H, T, T) fp32
    out_all = torch.matmul(attn_clean, v.float()).to(v.dtype)  # (B, H, T, d_h)
    cls_attn_row = attn_clean[:, :, cls_idx, :]  # (B, H, T) fp32

    # 2. Apply I1 (query replacement) on [CLS] row ONLY
    if q_cls_override is not None:
        q_rep = q_cls_override.to(device=q.device, dtype=q.dtype).unsqueeze(2)  # (B, H, 1, d_h)
        cls_attn_row = attn_probs(q_rep, k, scale).squeeze(2)  # (B, H, T) fp32
        cls_out = torch.matmul(cls_attn_row.unsqueeze(2), v.float()).squeeze(2).to(v.dtype)
        out_all = out_all.clone()
        out_all[:, :, cls_idx, :] = cls_out

    # 3. Apply I2 (token-group knockout) on [CLS] row ONLY
    if ko_mask is not None:
        m = ko_mask.to(device=q.device, dtype=torch.bool)  # (B, T)
        if ko_mode == "zero":
            # Zero [CLS] attention weights to knocked-out tokens and renormalize remaining to sum to 1
            m_exp = m.unsqueeze(1)  # (B, 1, T)
            a_mod = cls_attn_row.masked_fill(m_exp, 0.0)
            denom = a_mod.sum(dim=-1, keepdim=True)
            # Fallback to unmodified row only if all tokens were masked
            cls_attn_row = torch.where(denom > 1e-12, a_mod / denom.clamp_min(1e-12), cls_attn_row)
            cls_out = torch.matmul(cls_attn_row.unsqueeze(2), v.float()).squeeze(2).to(v.dtype)
            out_all = out_all.clone()
            out_all[:, :, cls_idx, :] = cls_out
        elif ko_mode == "mean_v":
            if v_mean is None:
                raise ValueError("I2 mean_v mode requires dataset-mean value vector v_mean (H, d_h)")
            # Keep attention weights intact; replace v_i for i in group with dataset-mean value vector
            v_rep = v_mean.to(device=v.device, dtype=torch.float32).view(1, H, 1, d_h).expand(B, H, T, d_h)
            v_mod = torch.where(m.view(B, 1, T, 1), v_rep, v.float())
            cls_out = torch.matmul(cls_attn_row.unsqueeze(2), v_mod).squeeze(2).to(v.dtype)
            out_all = out_all.clone()
            out_all[:, :, cls_idx, :] = cls_out
        else:
            raise ValueError(f"Unknown ko_mode {ko_mode!r}; expected 'zero' or 'mean_v'")

    # Transpose & merge heads: (B, H, T, d_h) -> (B, T, H*d_h)
    attn_cat = out_all.transpose(1, 2).reshape(B, T, H * d_h)
    return attn_cat, cls_attn_row


# --------------------------------------------------------------------------- Block-by-block forward from start_block
@torch.no_grad()
def forward_from_block_with_interventions(
    adapter,
    model,
    z_start: torch.Tensor,
    layout: dict,
    start_block: int,
    *,
    q_cls_overrides: dict[int, torch.Tensor] | None = None,  # {layer_idx: (B, H, d_h)}
    ko_masks: dict[int, torch.Tensor] | None = None,         # {layer_idx: (B, T) bool}
    ko_mode: str = "zero",
    v_means: dict[int, torch.Tensor] | None = None,          # {layer_idx: (H, d_h)}
    capture_clean_stats: bool = False,
) -> dict[str, Any]:
    """Run blocks [start_block .. L-1] with optional I1/I2 edits on the [CLS] read-out row.

    Returns dict with:
        z_final:           (B, T, D) final block output
        emb:               (B, D_out) global representation
        logits:            (B, C) or None
        last_attn_patch:   (B, H, N) [CLS] attention over patches at final block (renormalized over patches)
        last_head_weights: (B, H) Kobayashi head weights at final block
        last_weighted_map: (B, N) head-weighted [CLS] patch attention map at final block
        (optional if capture_clean_stats=True):
        q_g_by_layer:      {l: (B, H, d_h)}
        v_sum_by_layer:    {l: (H, d_h) fp64 sum over B and T}
        v_count_by_layer:  {l: int (B * T)}
        patch_norms_by_l:  {l: (B, N)}
    """
    q_ov = q_cls_overrides or {}
    ko_m = ko_masks or {}
    vm = v_means or {}

    cls_idx = layout["cls_idx"]
    patch_idx = layout["patch_idx"]
    n_blocks = adapter.n_blocks(model)
    z = z_start

    q_g_by_layer = {}
    v_sum_by_layer = {}
    v_count_by_layer = {}
    patch_norms_by_l = {}
    last_attn_patch = None
    last_head_weights = None
    last_weighted_map = None

    if adapter.family == "timm_vit":
        for l in range(start_block, n_blocks):
            blk = model.blocks[l]
            has_edit = (l in q_ov) or (l in ko_m)
            need_detail = has_edit or capture_clean_stats or (l == n_blocks - 1)
            if not need_detail:
                z = blk(z)
                continue

            a = blk.attn
            B, T, _ = z.shape
            zn = blk.norm1(z)
            qkv = a.qkv(zn).reshape(B, T, 3, a.num_heads, a.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = a.q_norm(q), a.k_norm(k)
            scale = float(a.scale)

            if capture_clean_stats:
                q_g_by_layer[l] = q[:, :, cls_idx, :].float().cpu()
                v_sum_by_layer[l] = v.to(torch.float64).sum(dim=(0, 2)).cpu()
                v_count_by_layer[l] = int(B * T)

            attn_cat, cls_attn_row = _apply_cls_only_attention(
                q, k, v, scale, cls_idx,
                q_cls_override=q_ov.get(l),
                ko_mask=ko_m.get(l),
                ko_mode=ko_mode,
                v_mean=vm.get(l),
            )
            attn_out = a.proj_drop(a.proj(attn_cat))
            attn_out = blk.drop_path1(blk.ls1(attn_out))
            z = z + attn_out
            z = z + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(z))))

            if capture_clean_stats:
                patch_norms_by_l[l] = z[:, patch_idx, :].float().norm(dim=-1).cpu()

            if l == n_blocks - 1:
                W_O, _, gamma = get_output_proj_and_layerscale(adapter, model, l)
                w_h = compute_head_weights(cls_attn_row, v, W_O, gamma=gamma)  # (B, H)
                w_norm = w_h / w_h.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                ap = cls_attn_row[:, :, patch_idx]
                ap = ap / ap.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                last_attn_patch = ap.float()
                last_head_weights = w_norm.float()
                last_weighted_map = (ap * w_norm.unsqueeze(-1)).sum(dim=1).float()

    elif adapter.family == "open_clip_vit":
        vt = adapter._visual(model)
        for l in range(start_block, n_blocks):
            blk = vt.transformer.resblocks[l]
            has_edit = (l in q_ov) or (l in ko_m)
            need_detail = has_edit or capture_clean_stats or (l == n_blocks - 1)
            if not need_detail:
                z = blk(z)
                continue

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

            if capture_clean_stats:
                q_g_by_layer[l] = q[:, :, cls_idx, :].float().cpu()
                v_sum_by_layer[l] = v.to(torch.float64).sum(dim=(0, 2)).cpu()
                v_count_by_layer[l] = int(B * T)

            attn_cat, cls_attn_row = _apply_cls_only_attention(
                q, k, v, scale, cls_idx,
                q_cls_override=q_ov.get(l),
                ko_mask=ko_m.get(l),
                ko_mode=ko_mode,
                v_mean=vm.get(l),
            )
            attn_out = a.out_proj(attn_cat)
            ls_1 = getattr(blk, "ls_1", None)
            if ls_1 is not None:
                attn_out = ls_1(attn_out)
            z = z + attn_out
            mlp_out = blk.mlp(blk.ln_2(z))
            ls_2 = getattr(blk, "ls_2", None)
            if ls_2 is not None:
                mlp_out = ls_2(mlp_out)
            z = z + mlp_out

            if capture_clean_stats:
                patch_norms_by_l[l] = z[:, patch_idx, :].float().norm(dim=-1).cpu()

            if l == n_blocks - 1:
                W_O, _, gamma = get_output_proj_and_layerscale(adapter, model, l)
                w_h = compute_head_weights(cls_attn_row, v, W_O, gamma=gamma)
                w_norm = w_h / w_h.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                ap = cls_attn_row[:, :, patch_idx]
                ap = ap / ap.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                last_attn_patch = ap.float()
                last_head_weights = w_norm.float()
                last_weighted_map = (ap * w_norm.unsqueeze(-1)).sum(dim=1).float()
    else:
        raise NotImplementedError(f"forward_from_block_with_interventions not implemented for family={adapter.family!r}")

    emb, logits = read_global_embedding_and_logits(adapter, model, z, layout)
    emb_patch_mean = read_patch_mean_embedding(adapter, model, z, layout)
    out = {
        "z_final": z,
        "emb": emb,
        "emb_patch_mean": emb_patch_mean,
        "logits": logits,
        "last_attn_patch": last_attn_patch,
        "last_head_weights": last_head_weights,
        "last_weighted_map": last_weighted_map,
    }
    if capture_clean_stats:
        out["q_g_by_layer"] = q_g_by_layer
        out["v_sum_by_layer"] = v_sum_by_layer
        out["v_count_by_layer"] = v_count_by_layer
        out["patch_norms_by_l"] = patch_norms_by_l
    return out
