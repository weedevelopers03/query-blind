"""Guide §1.6, "Caching layout" (page 6). Directory spec, verbatim from the guide:

    cache/{model_id}/{dataset}/{split}/
        meta.json    # layout, grid, resolution, code git sha
        q.npy    (I, L', H, d_h)   # global query only
        k.npy    (I, L', H, T, d_h)
        v.npy    (I, L', H, T, d_h)
        attn.npy (I, L', H, T)     # global-row attention only
        norms.npy(I, L', T)       # patch token L2 norms
        cls.npy  (I, L', d)
        pooled.npy (I, L', d)

L' = last_third_blocks() below. T is every token the model has (cls + registers + patches, per
token_layout()) - metric-specific slicing (e.g. M1's patch-only keys, §2.1) happens when READING
the cache, not writing it, so one cache serves M1-M4 instead of committing to one measure's subset
up front. fp16 throughout, per the guide.

Two fields the guide's one-line shapes don't fully disambiguate - flagging both, same as the
MAP-head adapter's own flagged extension:
  - norms.npy: "patch token L2 norms" but shaped (I, L', T) not (I, L', N_patches). Stored here as
    ALL T tokens' norms (cls+registers+patches), since register-vs-patch norm separation (the
    property test 4 checks) needs both groups; slice to patch_idx for a patches-only read.
  - cls.npy vs pooled.npy: for CLS-token families these are the same value (the residual-stream
    global token leaving block l) - there's no second projection in what these adapters expose, so
    "pooled" isn't doing extra work there. For MAP-head models there IS no cls token at all
    (stored as NaN), so pooled is the only real field: adapter.global_update(), i.e. what the
    pooling head would output if it read out at that depth (see map_head.py's own flagged
    extension - this reuses it, doesn't add a new one).
"""
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from adapters.map_head import MapHeadAdapter
from adapters.timm_vit import TimmViTAdapter


def last_third_blocks(n_blocks: int) -> list:
    """The guide's phrase made precise: the final ~1/3 of blocks, e.g. 8,9,10,11 of 12."""
    return list(range(n_blocks - n_blocks // 3, n_blocks))


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


@torch.no_grad()
def _extract_layer(adapter, model, x, l):
    """One image (B=1), one block index l. -> q (H,d_h), k (H,T,d_h), v (H,T,d_h), attn_row (H,T)."""
    if isinstance(adapter, MapHeadAdapter):
        q = adapter.get_global_query(model, x, l)[0]
        k, v = adapter.get_keys_values(model, x, l)
        attn_row = adapter.attention_probs(model, x, l)[0]  # already just the probe's row
    else:
        q, k, v = adapter.get_qkv(model, x, l)
        g = adapter.token_layout(model)["cls_idx"]
        attn_row = adapter.attention_probs(model, x, l)[0, :, g, :]
        q = q[0, :, g, :]
    return q, k[0], v[0], attn_row


@torch.no_grad()
def _residual_out(adapter, model, x, l):
    """Residual stream leaving block l, i.e. residual_in(l+1) - dispatched because MapHeadAdapter
    doesn't expose residual_in directly (see map_head.py: it wraps TimmViTAdapter internally and
    the final-norm handling only kicks in at l == n_blocks, via _patch_stream)."""
    if isinstance(adapter, MapHeadAdapter):
        return adapter._patch_stream(model, x, l + 1)[0]
    return adapter.residual_in(model, x, l + 1)[0]


@torch.no_grad()
def _cls_and_pooled(adapter, model, x, l, d_model):
    """See module docstring for what these two fields mean per family."""
    if isinstance(adapter, MapHeadAdapter):
        cls = torch.full((d_model,), float("nan"))
        pooled = adapter.global_update(model, x, l)[0]
    else:
        z = _residual_out(adapter, model, x, l)
        cls = z[adapter.token_layout(model)["cls_idx"]]
        pooled = cls
    return cls, pooled


@torch.no_grad()
def extract_shard(adapter, model, images: torch.Tensor, model_id: str, dataset: str, split: str,
                   out_root: str, d_model: int, image_ids=None):
    """Extract a tensor batch or a list of differently sized, patch-padded images."""
    n_blocks = adapter.n_blocks(model)
    layers = last_third_blocks(n_blocks)
    image_list = [images[i:i + 1] for i in range(len(images))] if isinstance(images, torch.Tensor) else [
        image.unsqueeze(0) if image.ndim == 3 else image for image in images
    ]
    I = len(image_list)

    Q, K, V, ATTN, NORMS, CLS, POOLED = [], [], [], [], [], [], []
    layouts, resolutions, token_counts = [], [], []
    for i, x in enumerate(image_list):
        layout = adapter.token_layout(model, x) if isinstance(adapter, TimmViTAdapter) else adapter.token_layout(model)
        layouts.append(layout)
        resolutions.append(list(x.shape[-2:]))
        token_counts.append(len(layout["patch_idx"]) + len(layout["register_idx"])
                            + (layout["cls_idx"] is not None) + len(layout["dist_idx"]))
        q_l, k_l, v_l, a_l, n_l, c_l, p_l = [], [], [], [], [], [], []
        for l in layers:
            q, k, v, a = _extract_layer(adapter, model, x, l)
            cls, pooled = _cls_and_pooled(adapter, model, x, l, d_model)
            norms = _residual_out(adapter, model, x, l).norm(dim=-1)  # (T,) all tokens, see docstring
            q_l.append(q.half().numpy()); k_l.append(k.half().numpy()); v_l.append(v.half().numpy())
            a_l.append(a.half().numpy()); n_l.append(norms.half().numpy())
            c_l.append(cls.half().numpy()); p_l.append(pooled.half().numpy())
        Q.append(np.stack(q_l)); K.append(np.stack(k_l)); V.append(np.stack(v_l))
        ATTN.append(np.stack(a_l)); NORMS.append(np.stack(n_l))
        CLS.append(np.stack(c_l)); POOLED.append(np.stack(p_l))

    shard_dir = Path(out_root) / model_id / dataset / split
    shard_dir.mkdir(parents=True, exist_ok=True)
    max_t = max(token_counts)
    def pad_tokens(a, axis, fill=np.nan):
        widths = [(0, 0)] * a.ndim
        widths[axis] = (0, max_t - a.shape[axis])
        return np.pad(a, widths, mode="constant", constant_values=fill)
    np.save(shard_dir / "q.npy", np.stack(Q))
    np.save(shard_dir / "k.npy", np.stack([np.stack([pad_tokens(z, 1) for z in image]) for image in K]))
    np.save(shard_dir / "v.npy", np.stack([np.stack([pad_tokens(z, 1) for z in image]) for image in V]))
    np.save(shard_dir / "attn.npy", np.stack([np.stack([pad_tokens(z, 1) for z in image]) for image in ATTN]))
    np.save(shard_dir / "norms.npy", np.stack([np.stack([pad_tokens(z, 0) for z in image]) for image in NORMS]))
    np.save(shard_dir / "cls.npy", np.stack(CLS))
    np.save(shard_dir / "pooled.npy", np.stack(POOLED))
    meta = dict(
        model_id=model_id, dataset=dataset, split=split,
        n_images=I, n_blocks=n_blocks, layers=layers,
        image_ids=list(image_ids) if image_ids is not None else None,
        grid=list(layouts[0]["grid"]) if len(set(tuple(v["grid"]) for v in layouts)) == 1 else None,
        grids=[list(v["grid"]) for v in layouts], resolutions=resolutions, token_counts=token_counts,
        code_git_sha=_git_sha(), written_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    (shard_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return shard_dir


def load_shard(shard_dir):
    shard_dir = Path(shard_dir)
    meta = json.loads((shard_dir / "meta.json").read_text())
    arrays = {name: np.load(shard_dir / f"{name}.npy") for name in
              ["q", "k", "v", "attn", "norms", "cls", "pooled"]}
    return meta, arrays
