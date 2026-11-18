"""Pilot-8 correctness checks: Guide §1.3 (five tests) x §11.1 (eight pilot models).

Front-ends: `python scripts/pilot8_table.py` writes the 5 x 8 table; `pytest tests/test_pilot8.py`
fails if any cell fails.  This module holds the checks.

WHAT RUNS WITHOUT TRAINED WEIGHTS ("arch-level") AND WHAT DOES NOT
  t1 attention reconstruction, t2 residual reconstruction, t3 token layout are forward-computation
  identities, so they run on a randomly-initialised model of each pilot model's exact timm/open_clip
  architecture.  They do NOT show that the real checkpoint file loads into that architecture; that is a
  separate check (strict state-dict load) that needs the real files.
  At the libraries' default init, attention is near-uniform, biases are zero and LayerScale gamma is
  ~1e-5, which hides whole classes of bug.  So every check runs at default init AND on a "boosted" copy
  (peaked attention, non-zero biases, gamma ~ U(0.5, 1.5), non-trivial LayerNorm affine).
  t4 register indexing: 4a (layer-0 sentinel perturbation) runs here; the Guide's norm test (4b) needs the
  real DINOv2-reg weights and is skipped unless PILOT8_REAL_WEIGHTS=1.
  t5 known-result: 5b recomputes CorLoc from a per-image CSV of a real run (PILOT8_CORLOC_CSV); 5a (the DINO
  paper's two attention figures) needs real weights plus the two images and is always a skip here.

SEEDED BUGS ("mutants")
  Every check is also run against a deliberately broken adapter; the check must FAIL there.  A PASS for a
  check whose seeded bug went undetected means nothing, and is reported as a failure of the table.
"""
import contextlib
import csv
import gc
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from adapters.open_clip_vit import OpenCLIPViTAdapter
from adapters.timm_vit import TimmViTAdapter

TOL = 1e-4  # Guide §1.3 tests 1-2
SEED = 0
BOOST = dict(qk_gain=4.0, ls_range=(0.5, 1.5), bias_std=0.1, ln_gain_range=(0.5, 1.5), ln_bias_std=0.1)

# id -> (family, architecture name, constructor kwargs).  /14 models run at 196 px (14x14 grid, Guide §1.5).
PILOT8 = {
    "dino_vits16": ("timm", "vit_small_patch16_224", {}),
    "deit3_vits16": ("timm", "deit3_small_patch16_224", {}),
    "ibot_vits16": ("timm", "vit_small_patch16_224", {}),
    "mocov3_vitb16": ("timm", "vit_base_patch16_224", {}),
    "mae_vitb16": ("timm", "vit_base_patch16_224", {}),
    "dinov2_vitb14": ("timm", "vit_base_patch14_dinov2", {"img_size": 196}),
    "dinov2_vitb14_reg": ("timm", "vit_base_patch14_reg4_dinov2", {"img_size": 196}),
    "openclip_vitb16": ("open_clip", "ViT-B-16", {}),
}
HAS_REGISTERS = {"dinov2_vitb14_reg"}  # test 4 applies only here
KNOWN_RESULT = {"dino_vits16"}         # test 5 applies only here
PUBLISHED_CORLOC, CORLOC_TOL, VOC07_TRAINVAL = 61.9, 1.0, 5011


def arch_key(name):
    fam, arch, kw = PILOT8[name]
    return (fam, arch, tuple(sorted(kw.items())))


# ----------------------------------------------------------------------------------- construction
def build(name, pretrained=False):
    fam, arch, kw = PILOT8[name]
    torch.manual_seed(SEED)
    if fam == "timm":
        model, adapter = timm.create_model(arch, pretrained=pretrained, **kw), TimmViTAdapter()
    else:
        import open_clip
        model, adapter = open_clip.create_model(arch, pretrained=None).visual, OpenCLIPViTAdapter()
    adapter.prepare(model)
    return adapter, model


@torch.no_grad()
def boost(model, seed=SEED, cfg=BOOST):
    g = torch.Generator().manual_seed(seed)
    u = lambda shape, lo, hi: torch.rand(shape, generator=g) * (hi - lo) + lo
    nrm = lambda shape, std: torch.randn(shape, generator=g) * std
    for _, m in model.named_modules():
        if isinstance(m, nn.LayerNorm):
            m.weight.copy_(u(m.weight.shape, *cfg["ln_gain_range"]))
            if m.bias is not None:
                m.bias.copy_(nrm(m.bias.shape, cfg["ln_bias_std"]))
        elif isinstance(m, nn.Linear) and m.bias is not None:
            m.bias.copy_(nrm(m.bias.shape, cfg["bias_std"]))
        elif getattr(m, "in_proj_weight", None) is not None and getattr(m, "in_proj_bias", None) is not None:
            # fused qkv, rows [q; k; v]: torch nn.MultiheadAttention (open_clip <= 3.x) AND open_clip >= 4's own
            # Attention class (not an nn.MultiheadAttention any more, same parameter names)
            m.in_proj_bias.copy_(nrm(m.in_proj_bias.shape, cfg["bias_std"]))
            m.in_proj_weight[: 2 * m.in_proj_weight.shape[1]].mul_(cfg["qk_gain"])
        elif type(m).__name__ == "LayerScale":
            m.gamma.copy_(u(m.gamma.shape, *cfg["ls_range"]))
    for blk in getattr(model, "blocks", []):  # timm: fused qkv, rows [q; k; v]
        e = blk.attn.qkv.weight.shape[1]
        blk.attn.qkv.weight[: 2 * e].mul_(cfg["qk_gain"])
    return model


def _patch_embed(model):
    return model.patch_embed if hasattr(model, "patch_embed") else model.conv1


def _patch_size(model):
    pe = _patch_embed(model)
    return pe.patch_size[0] if hasattr(pe, "patch_size") else pe.kernel_size[0]


def image(adapter, model, batch=2, seed=0):
    gh, gw = adapter.token_layout(model)["grid"]
    ps = _patch_size(model)
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, 3, gh * ps, gw * ps, generator=g)


def layers(adapter, model):
    n = adapter.n_blocks(model)
    return sorted({n // 2, n - 1})


# ---------------------------------------------------------------------------------- seeded bugs
def mutate(adapter, kind):
    """A copy of the adapter with one realistic bug (guide §12 'silent killers'), or None if it cannot apply."""
    base = type(adapter)
    if kind == "wrong_scale":  # 1/d_h instead of 1/sqrt(d_h)
        class M(base):
            def attn_scale(self, model, l):
                return base.attn_scale(self, model, l) ** 2
    elif kind == "head_reshape":  # reshape instead of transpose when splitting heads
        class M(base):
            def get_qkv(self, model, x, l):
                q, k, v = base.get_qkv(self, model, x, l)
                b, h, t, d = q.shape
                f = lambda a: a.transpose(1, 2).reshape(b, h, t, d)
                return f(q), f(k), f(v)
    elif kind == "wrong_global_token":  # global query read from the wrong token
        class M(base):
            def get_qkv(self, model, x, l):
                q, k, v = base.get_qkv(self, model, x, l)
                return q.roll(1, dims=2), k, v
    elif kind == "colmajor_patch_idx":  # patch tokens listed column-major
        class M(base):
            def token_layout(self, model):
                lay = dict(base.token_layout(self, model))
                g = lay["grid"][0]
                first = min(lay["patch_idx"])
                lay["patch_idx"] = [first + r * g + c for c in range(g) for r in range(g)]
                return lay
    elif kind == "patch_idx_minus1":  # off by one
        class M(base):
            def token_layout(self, model):
                lay = dict(base.token_layout(self, model))
                lay["patch_idx"] = [i - 1 for i in lay["patch_idx"]]
                return lay
    elif kind == "patch_idx_ignores_registers":  # patches start at 1 instead of 1+R
        class M(base):
            def token_layout(self, model):
                lay = dict(base.token_layout(self, model))
                lay["patch_idx"] = [i - len(lay["register_idx"]) for i in lay["patch_idx"]]
                return lay
    elif kind == "register_idx_minus1":
        class M(base):
            def token_layout(self, model):
                lay = dict(base.token_layout(self, model))
                lay["register_idx"] = [i - 1 for i in lay["register_idx"]]
                return lay
    else:
        raise KeyError(kind)
    return M()


# ------------------------------------------------------------------------------------------ test 1
def openclip_attn_probs(blk, zn):
    """An open_clip block's own post-softmax attention (B, H, T, T), read from the layer, not from our adapter.
    open_clip <= 3.x: blk.attn is torch's nn.MultiheadAttention, which returns its weights on request.
    open_clip >= 4  : blk.attn is open_clip's own Attention: no `need_weights` argument, and it fuses softmax into
                      scaled_dot_product_attention by default.  So run its explicit-softmax branch (use_sdpa=False)
                      and tap the probabilities where that branch hands them to attn_drop (as for the timm models)."""
    attn = blk.attn
    if isinstance(attn, nn.MultiheadAttention):
        return attn(zn, zn, zn, need_weights=True, average_attn_weights=False)[1]
    if hasattr(attn, "use_sdpa") and hasattr(attn, "attn_drop"):
        cap, old = {}, attn.use_sdpa
        h = attn.attn_drop.register_forward_pre_hook(lambda m, inp: cap.setdefault("p", inp[0]))
        attn.use_sdpa = False
        try:
            blk.attention(q_x=zn)
        finally:
            h.remove()
            attn.use_sdpa = old
        if "p" not in cap:
            raise RuntimeError("open_clip attention probabilities were not captured (the explicit-softmax branch did not run)")
        return cap["p"]
    raise NotImplementedError(f"cannot read post-softmax attention from {type(attn).__module__}.{type(attn).__name__}: "
                              "update openclip_attn_probs() for this open_clip version")


def attn_real(adapter, model, x, l):
    """The model's own post-softmax attention at block l, captured independently of the adapter's q/k split."""
    if isinstance(adapter, OpenCLIPViTAdapter):
        z = adapter.residual_in(model, x, l)
        blk = model.transformer.resblocks[l]
        return openclip_attn_probs(blk, blk.ln_1(z))
    cap = {}
    h = model.blocks[l].attn.attn_drop.register_forward_pre_hook(lambda m, inp: cap.setdefault("p", inp[0]))
    adapter.residual_in(model, x, l + 1)
    h.remove()
    return cap["p"]


def t1_diff(adapter, model, x, l):
    return (adapter.attention_probs(model, x, l).float() - attn_real(adapter, model, x, l).float()).abs().max().item()


# ------------------------------------------------------------------------------------------ test 2
def block_parts(adapter, model, l):
    if isinstance(adapter, OpenCLIPViTAdapter):
        b = model.transformer.resblocks[l]
        return dict(blk=b, ls_name="ls_1", attn=lambda z: b.ls_1(b.attention(q_x=b.ln_1(z))),
                    mlp=lambda s: b.ls_2(b.mlp(b.ln_2(s))))
    b = model.blocks[l]
    return dict(blk=b, ls_name="ls1", attn=lambda z: b.ls1(b.attn(b.norm1(z))), mlp=lambda s: b.ls2(b.mlp(b.norm2(s))))


def has_layerscale(adapter, model, l):
    P = block_parts(adapter, model, l)
    return type(getattr(P["blk"], P["ls_name"])).__name__ == "LayerScale"


@contextlib.contextmanager
def _no_layerscale(adapter, model, l):
    P = block_parts(adapter, model, l)
    old = getattr(P["blk"], P["ls_name"])
    setattr(P["blk"], P["ls_name"], nn.Identity())
    try:
        yield
    finally:
        setattr(P["blk"], P["ls_name"], old)


def t2_diffs(adapter, model, x, l, mlp_on="mid", drop_ls=False):
    """(a) branch-level: global_update() vs the model's own attn submodule + LayerScale at the global token;
    (b) Guide test 2: x_out = x_in + attn_branch + mlp_branch at the global token, with OUR attn branch and the
    model's own block forward as ground truth."""
    c = adapter.token_layout(model)["cls_idx"]
    P = block_parts(adapter, model, l)
    z = adapter.residual_in(model, x, l)
    ctx = _no_layerscale(adapter, model, l) if drop_ls else contextlib.nullcontext()
    with ctx:
        upd = adapter.global_update(model, x, l)
    d_branch = (upd - P["attn"](z)[:, c].float()).abs().max().item()
    mid = z[:, c] + upd.to(z.dtype)
    mlp_in = mid if mlp_on == "mid" else z[:, c]
    ours = mid + P["mlp"](mlp_in.unsqueeze(1))[:, 0]
    d_resid = (ours.float() - P["blk"](z)[:, c].float()).abs().max().item()
    return d_branch, d_resid


# ------------------------------------------------------------------------------------------ test 3
def t3a_patch_embed_argmax(adapter, model):
    """The repo's original test 3: bright patch -> argmax of per-patch embedding norm (patch_embed only)."""
    gh, gw = adapter.token_layout(model)["grid"]
    row, col, ps = gh // 3, gw // 2, _patch_size(model)
    img = torch.zeros(1, 3, gh * ps, gw * ps)
    img[:, :, row * ps:(row + 1) * ps, col * ps:(col + 1) * ps] = 1.0
    pe = _patch_embed(model)(img)
    pe = pe.flatten(2).transpose(1, 2) if pe.dim() == 4 else pe
    return divmod(int(pe[0].norm(dim=-1).reshape(gh, gw).argmax()), gw) == (row, col)


def t3b_full_stream(adapter, model):
    """Guide test 3 through the whole token stream: paint a square, take layer-0 tokens, select patches with
    token_layout()['patch_idx'], reshape to (G,G), upsample, and require the change to sit exactly on the square.
    (Uses the difference to an all-black image, so it is independent of LayerNorm / init scale.)"""
    lay = adapter.token_layout(model)
    G, ps = lay["grid"][0], _patch_size(model)
    row, col, side = G // 3, G // 2, G * ps
    black = torch.zeros(1, 3, side, side)
    sq = black.clone()
    sq[:, :, row * ps:(row + 1) * ps, col * ps:(col + 1) * ps] = 1.0
    d = (adapter.residual_in(model, sq, 0) - adapter.residual_in(model, black, 0)).norm(dim=-1)[0]  # (T,)
    idx = lay["patch_idx"]
    if min(idx) < 0 or max(idx) >= d.shape[0]:
        return False
    m = d[idx].reshape(1, 1, G, G)
    prefix = [i for i in range(d.shape[0]) if i not in set(idx)]
    only_square = bool(((m[0, 0] > 1e-6).nonzero().tolist()) == [[row, col]]) and float(d[prefix].abs().max()) <= 1e-6
    up = F.interpolate(m, size=(side, side), mode="bilinear", align_corners=False)
    y, xx = divmod(int(up.flatten().argmax()), side)
    return only_square and (row * ps <= y < (row + 1) * ps) and (col * ps <= xx < (col + 1) * ps)


# ------------------------------------------------------------------------------------------ test 4
def t4a_prefix_index_map(adapter, model):
    """Structural register-indexing check: perturbing reg_token must change layer-0 tokens at exactly
    layout['register_idx'] (and cls_token at exactly cls_idx).  Weight-independent."""
    lay = adapter.token_layout(model)
    gh, gw = lay["grid"]
    x = torch.zeros(1, 3, gh * _patch_size(model), gw * _patch_size(model))
    ok = True
    for attr, expected in (("cls_token", [lay["cls_idx"]]), ("reg_token", list(lay["register_idx"]))):
        p = getattr(model, attr, None)
        if p is None:
            ok &= expected == []
            continue
        z0, old = adapter.residual_in(model, x, 0), p.detach().clone()
        with torch.no_grad():
            p.add_(1.0)
        z1 = adapter.residual_in(model, x, 0)
        with torch.no_grad():
            p.copy_(old)
        got = ((z1 - z0).abs().amax(dim=-1)[0] > 1e-6).nonzero().flatten().tolist()
        ok &= got == expected
    return ok


def t4b_register_norms_real(name):
    """The Guide's test 4 on REAL DINOv2-reg weights: mean register-token L2 norm vs the patch-token median."""
    fam, arch, kw = PILOT8[name]
    model = timm.create_model(arch + ".lvd142m", pretrained=True, **kw)
    adapter = TimmViTAdapter()
    adapter.prepare(model)
    d = os.environ.get("PILOT8_IMAGES_DIR")
    if d:
        from PIL import Image
        import torchvision.transforms as T
        cfg = timm.data.resolve_data_config({}, model=model)
        tf = T.Compose([T.Resize(196), T.CenterCrop(196), T.ToTensor(), T.Normalize(cfg["mean"], cfg["std"])])
        files = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png")))[:8]
        x, src = torch.stack([tf(Image.open(os.path.join(d, f)).convert("RGB")) for f in files]), f"{len(files)} images"
    else:
        x, src = image(adapter, model, batch=4, seed=3), "Gaussian noise (set PILOT8_IMAGES_DIR for real photos)"
    lay = adapter.token_layout(model)
    with torch.no_grad():
        norms = adapter.residual_in(model, x, adapter.n_blocks(model)).norm(dim=-1)
    reg, pat = float(norms[:, lay["register_idx"]].mean()), float(norms[:, lay["patch_idx"]].median())
    return reg > pat, dict(register_mean=round(reg, 3), patch_median=round(pat, 3), ratio=round(reg / pat, 3), input=src)


# ------------------------------------------------------------------------------------------ test 5
def t5b_corloc_from_csv(path):
    rows = list(csv.DictReader(open(path)))
    hits = sum(str(r["correct"]).strip() in ("1", "1.0", "True", "true") for r in rows)
    corloc = 100.0 * hits / len(rows)
    ok = len(rows) == VOC07_TRAINVAL and abs(corloc - PUBLISHED_CORLOC) <= CORLOC_TOL
    return ok, dict(n=len(rows), hits=hits, corloc=round(corloc, 2), published=PUBLISHED_CORLOC,
                    band=[PUBLISHED_CORLOC - CORLOC_TOL, PUBLISHED_CORLOC + CORLOC_TOL], csv=os.path.basename(path))


# ------------------------------------------------------------------------------------------- driver
def _mut(results, name, detected, **info):
    results[name] = dict(detected=detected, **info)


def run_arch(name):
    """Run all applicable checks for one pilot model's architecture.  Returns {test: cell}."""
    with torch.no_grad():
        adapter, base_model = build(name)
        boosted = boost(__import__("copy").deepcopy(base_model))
        cells = {}
        variants = (("default", base_model), ("boosted", boosted))

        # ---- test 1
        diffs, status = {}, "PASS"
        for tag, m in variants:
            for l in layers(adapter, m):
                d = t1_diff(adapter, m, image(adapter, m, seed=0), l)
                diffs[f"{tag}@{l}"] = d
                status = "FAIL" if not d < TOL else status
        muts, l0 = {}, layers(adapter, boosted)[0]
        x = image(adapter, boosted, seed=0)
        for kind in ("wrong_scale", "head_reshape"):
            d = t1_diff(mutate(adapter, kind), boosted, x, l0)
            _mut(muts, kind, d >= TOL, diff=d)
        cells["t1"] = dict(status=status, diffs=diffs, mutants=muts)

        # ---- test 2
        diffs, status = {}, "PASS"
        for tag, m in variants:
            for l in layers(adapter, m):
                db, dr = t2_diffs(adapter, m, image(adapter, m, seed=1), l)
                diffs[f"{tag}@{l}:branch"], diffs[f"{tag}@{l}:residual"] = db, dr
                status = "FAIL" if not (db < TOL and dr < TOL) else status
        muts, x = {}, image(adapter, boosted, seed=1)
        db, dr = t2_diffs(adapter, boosted, x, l0, mlp_on="x_in")
        _mut(muts, "mlp_on_pre_attn_stream", dr >= TOL, diff=dr)
        db, dr = t2_diffs(mutate(adapter, "wrong_global_token"), boosted, x, l0)
        _mut(muts, "wrong_global_token", max(db, dr) >= TOL, diff=max(db, dr))
        if has_layerscale(adapter, boosted, l0):
            db, dr = t2_diffs(adapter, boosted, x, l0, drop_ls=True)
            _mut(muts, "layerscale_omitted", max(db, dr) >= TOL, diff=max(db, dr))
        else:
            _mut(muts, "layerscale_omitted", None, note="n/a: no LayerScale in this architecture")
        cells["t2"] = dict(status=status, diffs=diffs, mutants=muts)

        # ---- test 3
        ok_a = all(t3a_patch_embed_argmax(adapter, m) for _, m in variants)
        ok_b = all(t3b_full_stream(adapter, m) for _, m in variants)
        muts = {}
        for kind in ("colmajor_patch_idx", "patch_idx_minus1"):
            _mut(muts, kind, not t3b_full_stream(mutate(adapter, kind), boosted))
        if name in HAS_REGISTERS:
            _mut(muts, "patch_idx_ignores_registers", not t3b_full_stream(mutate(adapter, "patch_idx_ignores_registers"), boosted))
        cells["t3"] = dict(status="PASS" if (ok_a and ok_b) else "FAIL", checks=dict(patch_embed_argmax=ok_a, full_stream=ok_b), mutants=muts)

        # ---- test 4 (only models with registers)
        if name in HAS_REGISTERS:
            ok = all(t4a_prefix_index_map(adapter, m) for _, m in variants)
            muts = {}
            _mut(muts, "register_idx_minus1", not t4a_prefix_index_map(mutate(adapter, "register_idx_minus1"), boosted))
            cells["t4"] = dict(status="PASS" if ok else "FAIL", structural_4a=ok, mutants=muts)
        del boosted, base_model
        gc.collect()
    return cells


def run_real_weight_checks(name):
    """Tests 4b and 5b: only when the environment provides what they need.  Returns {test: cell-patch}."""
    out = {}
    if name in HAS_REGISTERS:
        if os.environ.get("PILOT8_REAL_WEIGHTS") == "1":
            try:
                ok, info = t4b_register_norms_real(name)
                out["t4b"] = dict(status="PASS" if ok else "FAIL", **info)
            except Exception as e:  # real weights were asked for but 4b could not run: never a pass, and never lose the table
                out["t4b"] = dict(status="FAIL", reason=f"PILOT8_REAL_WEIGHTS=1 but 4b could not run: {type(e).__name__}: {e}")
        else:
            out["t4b"] = dict(status="SKIP", reason="needs real DINOv2-reg weights (set PILOT8_REAL_WEIGHTS=1 with Hugging Face access)")
    if name in KNOWN_RESULT:
        path = os.environ.get("PILOT8_CORLOC_CSV")
        if not path:
            out["t5b"] = dict(status="SKIP", reason="no per-image CorLoc CSV given (set PILOT8_CORLOC_CSV to the closeout run's CSV)")
        elif not os.path.exists(path):  # a path that was given but is wrong must not look like "nothing given"
            out["t5b"] = dict(status="FAIL", reason=f"PILOT8_CORLOC_CSV is set but there is no such file: {path}")
        else:
            try:
                ok, info = t5b_corloc_from_csv(path)
                out["t5b"] = dict(status="PASS" if ok else "FAIL", **info)
            except Exception as e:
                out["t5b"] = dict(status="FAIL", reason=f"could not read {path} as a per-image CorLoc CSV: {type(e).__name__}: {e}")
        out["t5a"] = dict(status="SKIP", reason="DINO-paper attention figure on its two images: needs real weights + the two images")
    return out
