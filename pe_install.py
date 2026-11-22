"""One-cell installer: positional-embedding interpolation test + run flag (supervisor, Oct 21).

Run from the repo root (the folder that holds adapters/, extract/, tests/):      !python pe_install.py
It
  - writes two NEW files: adapters/pos_embed_interp.py and tests/test_pe_interpolation.py
  - patches extract/run_lost_native_res.py in place: 4 edits anchored on exact lines (each must occur once, otherwise
    NOTHING is written) + a dated note in its docstring. Idempotent; the original goes to the temp dir, git has it too.
Nothing else is touched. Then:   !python -m tests.test_pe_interpolation
"""
import py_compile
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd()
RUNNER = ROOT / "extract" / "run_lost_native_res.py"
NEW_FILES = {
    "adapters/pos_embed_interp.py": r'''"""Which positional-embedding interpolation ran - and DINO's own function to check it against.

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
''',
    "tests/test_pe_interpolation.py": r'''"""Positional-embedding interpolation at a non-native size (supervisor, Oct 21).

    pytest tests/test_pe_interpolation.py
    python -m tests.test_pe_interpolation                       # prints the report, exit 1 on any FAIL
    DINO_VIT_PY=/path/to/dino/vision_transformer.py python -m tests.test_pe_interpolation
                                                                # reference = DINO's own function from that file

Checks, and what makes each one fail:
  PE-1  extract/run_lost_native_res.py::dino_pos_embed (the function that ships) gives the same tokens as DINO's
        interpolate_pos_encoding, at the native 14x14 grid and at non-native grids (VOC07 trainval grids from
        corloc_per_image.csv + two synthetic ones). Any deviation > 1e-6 fails.
  PE-2  through the adapter: adapter.get_qkv at a non-native size on the patched model == on a model whose _pos_embed
        is DINO's. Control: the same comparison on timm's stock model must DIFFER, otherwise a pass proves nothing.
  PE-3  seeded bugs in the shipped function (drop +0.1, antialias, size= for scale_factor, H/W swap, ...) are each
        caught by PE-1. A bug that is not caught fails the test.
  PE-4  the meta.json flag: the fingerprint of the live model says "dino_interpolate_pos_encoding" for the patched
        model, "timm_resample_abs_pos_embed" for the stock one, and neither for a seeded bug.
  PE-5  main() of the runner binds the patch (types.MethodType(dino_pos_embed, model)), checks the path and writes it.

Random weights, pos_embed ~ N(0,1) (so differences are not hidden by a small init), no downloads, CPU, seconds.
The reference is a hand transcription of DINO's function unless DINO_VIT_PY points at DINO's vision_transformer.py;
the report and meta.json say which one was used.
"""
import ast
import json
import math
import sys
import tempfile
import types
from functools import lru_cache
from pathlib import Path

import timm
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from adapters.pos_embed_interp import LABEL_DINO, LABEL_TIMM, dino_reference, interpolation_path, write_run_meta  # noqa: E402
from adapters.timm_vit import TimmViTAdapter  # noqa: E402

RUNNER = REPO / "extract" / "run_lost_native_res.py"
PATCH = 16
TOL_POS = 1e-6      # same F.interpolate call on the same input: expect bit-identical
TOL_KEYS = 1e-5     # last-block keys through the adapter
MIN_POWER = 1e-3    # control: stock timm must move the keys by at least this much, or the comparison is vacuous
GRIDS = [  # (grid_h, grid_w), note. VOC counts: 5011-image trainval run
    ((14, 14), "native 224x224: identity"),
    ((24, 32), "VOC 384x512, 2302 images"),
    ((32, 24), "VOC 512x384, 431 images"),
    ((21, 32), "VOC 336x512, 1161 images"),
    ((32, 32), "VOC 512x512, 32 images"),
    ((6, 32), "VOC smallest height, 1 image"),
    ((13, 19), "synthetic: one axis down, one up"),
    ((7, 28), "synthetic: 7*28 == 196 but not square"),
]
E2E_GRIDS = [(24, 32), (32, 24), (13, 19)]
MUTANTS = [  # (what is broken, text in dino_pos_embed, replacement)
    ("drop DINO's +0.1 epsilon", "w0, h0 = W + 0.1, H + 0.1", "w0, h0 = W, H"),
    ("exact size= instead of scale_factor", "scale_factor=(h0 / math.sqrt(N), w0 / math.sqrt(N)),", "size=(H, W),"),
    ("antialias=True (timm's setting)", 'mode="bicubic",', 'mode="bicubic", antialias=True,'),
    ("align_corners=True", 'mode="bicubic",', 'mode="bicubic", align_corners=True,'),
    ("bilinear instead of bicubic", 'mode="bicubic",', 'mode="bilinear",'),
    ("H and W swapped in scale_factor", "scale_factor=(h0 / math.sqrt(N), w0 / math.sqrt(N)),",
     "scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),"),
    ("no native-size shortcut", "if H * W == N", "if False"),
    ("pos_embed not added", "dim=1) + pos_embed", "dim=1) + 0 * pos_embed"),
    ("patch positions sliced off by one", "self.pos_embed[:, 1:]", "self.pos_embed[:, :-1]"),
]


# ---------------------------------------------------------------- helpers
def shipped_source():
    text = RUNNER.read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "dino_pos_embed")
    return ast.get_source_segment(text, node)


def compile_fn(src):
    ns = {"math": math, "torch": torch, "F": F}
    exec(compile(src, str(RUNNER), "exec"), ns)
    return ns["dino_pos_embed"]


def make_model(dynamic=True, seed=0):
    torch.manual_seed(seed)
    m = timm.create_model("vit_small_patch16_224", pretrained=False, dynamic_img_size=dynamic).eval()
    with torch.no_grad():
        m.pos_embed.copy_(torch.randn_like(m.pos_embed))
    return m


def image(grid, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 3, grid[0] * PATCH, grid[1] * PATCH, generator=g)


def ref_embed(model, pe, ref):
    """DINO's prepare_tokens: [CLS] + patch tokens + interpolate_pos_encoding. pe is NHWC (B, H, W, C)."""
    B, H, W, C = pe.shape
    return torch.cat((model.cls_token.expand(B, -1, -1), pe.reshape(B, H * W, C)), 1) + ref(model.pos_embed, (H, W), PATCH)


@torch.no_grad()
def pos_diff(fn, model, grid, ref):
    """max |tokens(fn) - tokens(DINO)| for fn(model, NHWC tokens); inf + reason if fn raises or the shape is wrong."""
    pe = model.patch_embed(image(grid))
    want = ref_embed(model, pe, ref)
    try:
        got = fn(pe.clone())
    except Exception as e:
        return float("inf"), f"raised {type(e).__name__}"
    if got.shape != want.shape:
        return float("inf"), f"shape {tuple(got.shape)} != {tuple(want.shape)}"
    return (got - want).abs().max().item(), ""


@torch.no_grad()
def adapter_keys(model, grid):
    ad = TimmViTAdapter()
    ad.prepare(model)
    _, k, _ = ad.get_qkv(model, image(grid), ad.n_blocks(model) - 1)
    return k


def patched_model(fn):
    m = make_model()
    m._pos_embed = types.MethodType(fn, m)
    return m


# ---------------------------------------------------------------- the checks
@lru_cache(maxsize=1)
def run_all():
    ref, ref_kind = dino_reference()
    src = shipped_source()
    shipped = compile_fn(src)
    base = make_model()
    R = dict(reference=ref_kind, pe1=[], pe2=[], pe3=[], pe4={}, pe5={})

    # PE-1: shipped function vs DINO, token tensor
    for grid, note in GRIDS:
        d, why = pos_diff(types.MethodType(shipped, base), base, grid, ref)
        stock, _ = pos_diff(lambda pe: type(base)._pos_embed(base, pe), base, grid, ref)
        R["pe1"].append(dict(grid=grid, px=(grid[0] * PATCH, grid[1] * PATCH), note=note, diff=d, why=why,
                             timm_stock_diff=stock, ok=d <= TOL_POS))

    # PE-2: through the adapter at a non-native size, with a control
    refm, shipm, stockm = make_model(), patched_model(shipped), make_model()
    refm._pos_embed = lambda t, m=refm: ref_embed(m, t, ref)
    for grid in E2E_GRIDS:
        try:
            k_ref, k_ship, k_stock = adapter_keys(refm, grid), adapter_keys(shipm, grid), adapter_keys(stockm, grid)
            d = (k_ship - k_ref).abs().max().item()
            power = (k_stock - k_ref).abs().max().item()
            R["pe2"].append(dict(grid=grid, diff=d, control_diff=power, ok=d <= TOL_KEYS and power >= MIN_POWER))
        except Exception as e:
            R["pe2"].append(dict(grid=grid, diff=float("inf"), control_diff=None, ok=False, why=f"{type(e).__name__}: {e}"))

    # PE-3: seeded bugs in the shipped function must be caught by the PE-1 comparison
    for what, old, new in MUTANTS:
        if old not in src:
            R["pe3"].append(dict(what=what, applicable=False, caught=None))
            continue
        bad = compile_fn(src.replace(old, new, 1))
        worst, where = 0.0, None
        for grid, _ in GRIDS:
            d, why = pos_diff(types.MethodType(bad, base), base, grid, ref)
            if d > worst:
                worst, where = d, (grid, why)
        R["pe3"].append(dict(what=what, applicable=True, caught=worst > TOL_POS, worst=worst, where=where))

    # PE-4: the flag
    fp_ship = interpolation_path(patched_model(shipped))
    fp_stock = interpolation_path(make_model())
    fp_fixed = interpolation_path(make_model(dynamic=False))
    fp_mut = {w: interpolation_path(patched_model(compile_fn(src.replace(o, n, 1)))) for w, o, n in MUTANTS if o in src}
    R["pe4"] = dict(
        shipped=fp_ship, stock=fp_stock, fixed_size=fp_fixed, mutants={w: v["path"] for w, v in fp_mut.items()},
        ok=(fp_ship["path"] == LABEL_DINO and fp_stock["path"] == LABEL_TIMM and fp_fixed["path"] == "not_applicable"
            and all(v["path"] != LABEL_DINO for v in fp_mut.values())),
    )
    with tempfile.TemporaryDirectory() as d:
        meta = write_run_meta(Path(d) / "meta.json", patched_model(shipped), config={"probe": "pe-test"}, run={})
        back = json.loads((Path(d) / "meta.json").read_text())
    R["pe4"]["meta_roundtrip"] = back["pos_embed_interpolation"]["path"] == LABEL_DINO and all(
        k in back for k in ("git_sha", "git_dirty", "config_hash", "versions"))

    # PE-5: the runner binds the patch, checks the path, writes the flag
    main = next((n for n in ast.parse(RUNNER.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "main"), None)
    called = {c.func.id for c in ast.walk(main) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)} if main else set()
    binds = main is not None and any(
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == "_pos_embed" for t in n.targets)
        and isinstance(n.value, ast.Call) and getattr(n.value.func, "attr", "") == "MethodType"
        and n.value.args and getattr(n.value.args[0], "id", "") == "dino_pos_embed" for n in ast.walk(main))
    R["pe5"] = dict(binds_patch=binds, fingerprints_path="interpolation_path" in called, writes_meta="write_run_meta" in called)
    R["pe5"]["ok"] = all(R["pe5"].values())
    return R


# ---------------------------------------------------------------- pytest
def test_pe1_shipped_matches_dino_at_non_native_sizes():
    bad = [(r["grid"], r["diff"], r["why"]) for r in run_all()["pe1"] if not r["ok"]]
    assert not bad, bad


def test_pe2_adapter_at_non_native_size_matches_dino_and_has_power():
    bad = [r for r in run_all()["pe2"] if not r["ok"]]
    assert not bad, bad


def test_pe3_every_seeded_bug_is_caught():
    r = run_all()["pe3"]
    assert any(x["applicable"] for x in r), "no mutant applied: the shipped function text changed, update MUTANTS"
    missed = [x["what"] for x in r if x["applicable"] and not x["caught"]]
    assert not missed, missed


def test_pe4_flag_identifies_the_path_that_ran():
    assert run_all()["pe4"]["ok"], run_all()["pe4"]
    assert run_all()["pe4"]["meta_roundtrip"]


def test_pe5_runner_binds_checks_and_records_the_path():
    assert run_all()["pe5"]["ok"], run_all()["pe5"]


# ---------------------------------------------------------------- report
def report():
    R = run_all()
    ok_all = True
    print(f"Positional-embedding interpolation at non-native size (random weights, pos_embed ~ N(0,1))")
    print(f"reference: {R['reference']}\n")
    print("PE-1  shipped dino_pos_embed vs DINO interpolate_pos_encoding: max |token diff| (tol %.0e)" % TOL_POS)
    print(f"  {'grid':7s}{'px':10s}{'shipped':>10s}  {'timm stock':>10s}   note")
    for r in R["pe1"]:
        ok_all &= r["ok"]
        print(f"  {r['grid'][0]}x{r['grid'][1]:<4}{r['px'][0]}x{r['px'][1]:<6}{r['diff']:>10.1e}  {r['timm_stock_diff']:>10.1e}   "
              f"{'ok  ' if r['ok'] else 'FAIL'} {r['note']} {r['why']}")
    print("\nPE-2  through the adapter, last-block keys (tol %.0e); control = stock timm must move them >= %.0e" % (TOL_KEYS, MIN_POWER))
    for r in R["pe2"]:
        ok_all &= r["ok"]
        cd = "n/a" if r["control_diff"] is None else f"{r['control_diff']:.1e}"
        print(f"  {r['grid'][0]}x{r['grid'][1]:<4} shipped-vs-DINO {r['diff']:.1e}   control (stock-vs-DINO) {cd}   "
              f"{'ok' if r['ok'] else 'FAIL'} {r.get('why', '')}")
    app = [x for x in R["pe3"] if x["applicable"]]
    caught = sum(bool(x["caught"]) for x in app)
    ok_all &= bool(app) and caught == len(app)
    print(f"\nPE-3  seeded bugs caught {caught}/{len(app)}" + ("" if len(app) == len(R['pe3']) else f"  ({len(R['pe3']) - len(app)} not applicable: text not found)"))
    for x in R["pe3"]:
        tag = "n/a " if not x["applicable"] else "ok  " if x["caught"] else "MISS"
        print(f"  {tag} {x['what']}")
    p4 = R["pe4"]
    ok_all &= p4["ok"] and p4["meta_roundtrip"]
    print(f"\nPE-4  flag: shipped -> {p4['shipped']['path']} | stock timm -> {p4['stock']['path']} | fixed-size -> {p4['fixed_size']['path']}"
          f" | seeded bugs -> {sorted(set(p4['mutants'].values()))} | meta.json round-trip {'ok' if p4['meta_roundtrip'] else 'FAIL'}"
          f"  => {'ok' if p4['ok'] else 'FAIL'}")
    p5 = R["pe5"]
    ok_all &= p5["ok"]
    print(f"\nPE-5  runner main(): binds patch={p5['binds_patch']}, fingerprints path={p5['fingerprints_path']}, "
          f"writes meta={p5['writes_meta']}  => {'ok' if p5['ok'] else 'FAIL'}")
    print("\nRESULT:", "PASS" if ok_all else "FAIL")
    return ok_all


if __name__ == "__main__":
    sys.exit(0 if report() else 1)
''',
}

EDITS = [
    ("import",
     "from adapters.timm_vit import TimmViTAdapter\n",
     "from adapters.timm_vit import TimmViTAdapter\n"
     "from adapters.pos_embed_interp import LABEL_DINO, interpolation_path, write_run_meta\n"),
    ("shortcut",
     "    if H * W == N:\n        pos_embed = self.pos_embed\n",
     "    if H * W == N and H == W:  # DINO: `npatch == N and w == h`; a non-square grid with N patches is still interpolated\n"
     "        pos_embed = self.pos_embed\n"),
    ("fingerprint",
     '    print("--> Patched _pos_embed: reproducing DINO\'s exact interpolate_pos_encoding (bicubic, no antialias, +0.1 epsilon)")\n',
     '    print("--> Patched _pos_embed: reproducing DINO\'s exact interpolate_pos_encoding (bicubic, no antialias, +0.1 epsilon)")\n'
     "    interp = interpolation_path(model)  # probe the live model: which interpolation will actually run\n"
     "    print(f\"--> Interpolation path that will run: {interp['path']} (reference: {interp['reference']})\")\n"
     "    assert interp[\"path\"] == LABEL_DINO, f\"positional-embedding interpolation is not DINO's: {interp}\"\n"),
    ("meta",
     '          + ("INSIDE" if 60.9 <= corloc <= 62.9 else "OUTSIDE"))\n',
     '          + ("INSIDE" if 60.9 <= corloc <= 62.9 else "OUTSIDE"))\n'
     '    meta_path = Path(csv_out or preds_out or "corloc_per_image.csv").with_name("corloc_meta.json")\n'
     "    write_run_meta(meta_path, model, ignore=[csv_out, preds_out],\n"
     "                   config=dict(image_dir=str(img_dir), split=split, limit=limit, pretrained=pretrained),\n"
     "                   run=dict(n_images=total, correct=correct, corloc_percent=round(corloc, 4)))\n"
     '    print(f"--> Run metadata (git SHA, config hash, interpolation path) written to {meta_path}")\n'),
    ("docstring",
     "Usage: PYTHONPATH=.:/path/to/LOST python3 extract/run_lost_native_res.py \\\\\n",
     "2026-11-19: (1) dino_pos_embed's native-size shortcut now also requires a square grid, as DINO's\n"
     "`npatch == N and w == h` does: before, a non-square grid with exactly N patches (7x28 = 196) skipped the\n"
     "interpolation. No VOC07 trainval image has such a grid (0 of 5011 in corloc_per_image.csv), so the 61.42% is\n"
     "unaffected. (2) main() fingerprints the live model's _pos_embed (adapters/pos_embed_interp.py), refuses to run\n"
     "unless it is DINO's path, and writes corloc_meta.json (git SHA, config hash, versions, interpolation path).\n"
     "Test: tests/test_pe_interpolation.py.\n\n"
     "Usage: PYTHONPATH=.:/path/to/LOST python3 extract/run_lost_native_res.py \\\\\n"),
]



def patch_runner():
    text = RUNNER.read_text()
    todo = []
    for name, old, new in EDITS:
        if new in text:
            print(f"  runner edit {name}: already applied")
        elif text.count(old) != 1:
            sys.exit(f"ABORT, nothing written: anchor for '{name}' occurs {text.count(old)}x (need exactly 1) in {RUNNER}")
        else:
            todo.append((name, old, new))
    if not todo:
        return
    bak = Path(tempfile.gettempdir()) / (RUNNER.name + ".bak")
    shutil.copy(RUNNER, bak)
    for name, old, new in todo:
        text = text.replace(old, new, 1)
        print(f"  runner edit {name}: applied")
    RUNNER.write_text(text)
    py_compile.compile(str(RUNNER), doraise=True)
    print(f"  {RUNNER.relative_to(ROOT)} patched (original saved to {bak}); compiles")


def main():
    for need in ("adapters/timm_vit.py", "extract/run_lost_native_res.py", "tests"):
        if not (ROOT / need).exists():
            sys.exit(f"ABORT: run this from the repo root; {need} not found under {ROOT}")
    patch_runner()  # first: it may abort, and then no new file is written either
    for rel, src in NEW_FILES.items():
        p = ROOT / rel
        p.write_text(src)
        py_compile.compile(str(p), doraise=True)
        print(f"  wrote {rel}")
    print("\ninstalled. Next:  !python -m tests.test_pe_interpolation")


main()