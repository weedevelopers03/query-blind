"""Which positional-embedding interpolation ran - and DINO's own function to check it against.

Supervisor, Oct 21: "a test running the adapter at a non-native size against DINO's interpolate_pos_encoding,
and a meta.json flag recording which interpolation path ran".

  dino_reference(path=None)          DINO's VisionTransformer.interpolate_pos_encoding as a callable
                                     (path to DINO's vision_transformer.py -> that file's own function, via AST;
                                      otherwise a transcription, labelled as such everywhere it is reported)
  interpolation_path(model)          fingerprint of the model's LIVE _pos_embed: probes it at a non-native grid and
                                     matches the output against DINO's function and timm's stock _pos_embed
  write_run_meta(path, model, ...)   meta.json: git SHA + dirty flag, config hash, versions, the interpolation path
"""
import ast
import hashlib
import json
import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
LABEL_DINO = "dino_interpolate_pos_encoding"   # bicubic, scale_factor=(h0/sqrt(N), w0/sqrt(N)) with +0.1, no antialias
LABEL_TIMM = "timm_resample_abs_pos_embed"     # bicubic, exact target size, antialias=True
TOL_POS = 1e-6


def _dino_transcribed(pos_embed, npatch, w, h, patch_size):
    """facebookresearch/dino vision_transformer.py, VisionTransformer.interpolate_pos_encoding, TRANSCRIBED by hand.
    `w`, `h` are DINO's names for dims 2 and 3 of the input image tensor (height and width in pixels)."""
    N = pos_embed.shape[1] - 1
    if npatch == N and w == h:
        return pos_embed
    class_pos_embed = pos_embed[:, 0]
    patch_pos_embed = pos_embed[:, 1:]
    dim = pos_embed.shape[-1]
    w0 = w // patch_size
    h0 = h // patch_size
    # DINO: "we add a small number to avoid floating point error in the interpolation" (dino issue #8)
    w0, h0 = w0 + 0.1, h0 + 0.1
    patch_pos_embed = F.interpolate(
        patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
        scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
        mode="bicubic",
    )
    assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


def _dino_from_file(path):
    """DINO's own interpolate_pos_encoding, lifted out of its vision_transformer.py by AST (no import of DINO's
    package, so its `from utils import ...` is not needed). Returns fn(pos_embed, npatch, w, h, patch_size)."""
    text = Path(path).read_text()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.FunctionDef) and node.name == "interpolate_pos_encoding":
            src = textwrap.dedent(" " * node.col_offset + ast.get_source_segment(text, node))
            ns = {"math": math, "torch": torch, "nn": torch.nn, "F": F}
            exec(compile(src, str(path), "exec"), ns)
            fn = ns["interpolate_pos_encoding"]

            def call(pos_embed, npatch, w, h, patch_size):
                self_ = SimpleNamespace(pos_embed=pos_embed, patch_embed=SimpleNamespace(patch_size=patch_size))
                x = torch.empty(1, 1 + npatch, pos_embed.shape[-1])   # DINO reads only x.shape from it
                return fn(self_, x, w, h)
            return call
    raise ValueError(f"no interpolate_pos_encoding in {path}")


def dino_reference(path=None):
    """-> (fn(pos_embed, grid_hw, patch_size) -> (1, 1+Hg*Wg, C), kind). `path` or $DINO_VIT_PY selects DINO's file."""
    path = path or os.environ.get("DINO_VIT_PY")
    raw, kind = (_dino_from_file(path), f"dino_file:{path}") if path else (None, "transcribed")
    raw = raw or (lambda pe, npatch, w, h, p: _dino_transcribed(pe, npatch, w, h, p))

    def fn(pos_embed, grid_hw, patch_size=16):
        gh, gw = grid_hw
        return raw(pos_embed, gh * gw, gh * patch_size, gw * patch_size, patch_size)
    return fn, kind


@torch.no_grad()
def interpolation_path(model, probe_grids=None, dino_vit_py=None, tol=TOL_POS):
    """Which interpolation does `model._pos_embed` actually run? Probe it (zero patch tokens, so output = cls + pos)
    at the native grid and at non-native grids (default: one axis smaller and one larger than native; and a
    non-square grid with as many patches as native) and match every probe against DINO's function and against
    timm's stock method. -> dict(path=LABEL_DINO | LABEL_TIMM | 'unknown' | 'ambiguous' | 'not_applicable', ...)."""
    ref, kind = dino_reference(dino_vit_py)
    out = dict(path="not_applicable", reference=kind, probe_grids=None, max_abs_diff_vs_dino=None, max_abs_diff_vs_timm_stock=None)
    if not getattr(model, "dynamic_img_size", False) or not hasattr(model, "_pos_embed"):
        out["reason"] = "no dynamic_img_size _pos_embed: fixed-size model, nothing is interpolated"
        return out
    if getattr(model, "num_prefix_tokens", 1) != 1 or getattr(model, "no_embed_class", False):
        out["reason"] = "probe assumes exactly one [CLS] prefix token and an embedded class position"
        return out
    gh, gw = model.patch_embed.grid_size
    probes = [tuple(g) for g in probe_grids] if probe_grids else [(gh - 1, gw + 5), (gh, gw), (gh // 2, 2 * gw)]
    ps = model.patch_embed.patch_size
    ps = ps[0] if isinstance(ps, (tuple, list)) else ps
    C = model.pos_embed.shape[-1]
    out["probe_grids"] = [list(g) for g in probes]
    d_dino = d_timm = 0.0
    try:
        for ph, pw in probes:
            probe = torch.zeros(1, ph, pw, C, device=model.pos_embed.device)
            live = model._pos_embed(probe.clone())
            stock = type(model)._pos_embed(model, probe.clone())
            dino = torch.cat((model.cls_token.expand(1, -1, -1), probe.view(1, -1, C)), 1) + ref(model.pos_embed, (ph, pw), ps)
            if live.shape != dino.shape:
                raise ValueError(f"live output {tuple(live.shape)} vs DINO {tuple(dino.shape)} at grid {ph}x{pw}")
            d_dino = max(d_dino, (live - dino).abs().max().item())
            d_timm = max(d_timm, (live - stock).abs().max().item())
    except Exception as e:  # a path that cannot even run at a probe size is not DINO's
        out.update(path="unknown", error=f"{type(e).__name__}: {e}")
        return out
    out.update(max_abs_diff_vs_dino=d_dino, max_abs_diff_vs_timm_stock=d_timm)
    is_d, is_t = d_dino <= tol, d_timm <= tol
    out["path"] = "ambiguous" if is_d and is_t else LABEL_DINO if is_d else LABEL_TIMM if is_t else "unknown"
    return out


def _git_state(ignore=()):
    """(HEAD sha, uncommitted files) or (None, None). Run outputs (results/, caches, `ignore`) are not code under test."""
    try:
        g = ["git", "-C", str(ROOT)]
        sha = subprocess.check_output(g + ["rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        st = subprocess.check_output(g + ["status", "--porcelain", "--", "."], stderr=subprocess.DEVNULL).decode().splitlines()
        skip = {Path(p).resolve() for p in ignore if p}
        keep = []
        for s in st:
            p = Path(s[3:].split(" -> ")[-1].strip('"'))
            if {"results", "__pycache__"} & set(p.parts) or (ROOT / p).resolve() in skip:
                continue
            keep.append(s)
        return sha, keep
    except Exception:
        return None, None


def write_run_meta(path, model=None, config=None, run=None, ignore=()):
    """meta.json for an extraction run. `config` = the settings (hashed), `run` = results; if `model` is given the
    interpolation path of its live _pos_embed is recorded under pos_embed_interpolation."""
    import timm
    sha, dirty = _git_state(list(ignore) + [path])
    cfg = config or {}
    meta = dict(
        git_sha=sha, git_dirty=(None if dirty is None else bool(dirty)), git_uncommitted=dirty,
        config=cfg, config_hash=hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16],
        versions=dict(python=sys.version.split()[0], torch=torch.__version__, timm=timm.__version__),
        run=run or {},
    )
    if model is not None:
        meta["pos_embed_interpolation"] = interpolation_path(model)
    Path(path).write_text(json.dumps(meta, indent=1, default=str))
    return meta
