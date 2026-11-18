"""pytest front-end for the pilot-8 table (checks live in tests/pilot8_checks.py; the table is written by
scripts/pilot8_table.py).  One test per pilot model: fails if any check fails OR any seeded bug goes undetected.
Skips (real weights / CorLoc CSV not supplied) are reported, never counted as passes.

    python -m pytest tests/test_pilot8.py -v
"""
import importlib.util
import sys
from pathlib import Path

import pytest

# load the checks by file path: a repo's own `tests` package would otherwise shadow tests/pilot8_checks.py when
# these files sit in a folder next to the repo instead of inside it.  (adapters/ must be importable: run from the repo root.)
_spec = importlib.util.spec_from_file_location("pilot8_checks", Path(__file__).with_name("pilot8_checks.py"))
C = importlib.util.module_from_spec(_spec)
sys.modules["pilot8_checks"] = C
_spec.loader.exec_module(C)

_CACHE = {}


def _run(name):
    k = C.arch_key(name)
    if k not in _CACHE:
        _CACHE.clear()  # one architecture in memory at a time
        _CACHE[k] = C.run_arch(name)
    return _CACHE[k]


@pytest.mark.parametrize("name", list(C.PILOT8))
def test_arch_level_checks_1_to_4a(name):
    bad = {}
    for t, c in _run(name).items():
        if c["status"] == "FAIL":
            bad[t] = "check failed"
        missed = [m for m, v in c["mutants"].items() if v["detected"] is False]
        if missed:
            bad[t + "_power"] = f"seeded bug(s) not detected: {missed}"
    assert not bad, bad


@pytest.mark.parametrize("name", sorted(C.HAS_REGISTERS | C.KNOWN_RESULT))  # only models the test applies to,
def test_real_weight_checks_4b_5(name):                                    # so every skip below is a real skip
    res = C.run_real_weight_checks(name)
    if any(c["status"] == "FAIL" for c in res.values()):
        pytest.fail(str(res))
    skipped = {t: c["reason"] for t, c in res.items() if c["status"] == "SKIP"}
    if skipped:
        pytest.skip(f"{skipped}")


def test_openclip_reference_attention_is_the_layers_own():
    """Test 1's reference on open_clip must (a) work with the installed open_clip, (b) be the layer's real attention:
    rows sum to 1 and, applied to v and out_proj, they reproduce the block's own output, and (c) leave the block as it
    was.  (open_clip 4.x replaced torch's nn.MultiheadAttention, so the old `need_weights=True` call crashed.)"""
    pytest.importorskip("open_clip")
    import torch
    adapter, model = C.build("openclip_vitb16")
    model = C.boost(model)
    x = C.image(adapter, model, seed=0)
    l = adapter.n_blocks(model) // 2
    blk = model.transformer.resblocks[l]
    with torch.no_grad():
        zn = blk.ln_1(adapter.residual_in(model, x, l))
        before = blk.attention(q_x=zn)
        p = C.openclip_attn_probs(blk, zn)
        after = blk.attention(q_x=zn)
        B, T, E = zn.shape
        H = blk.attn.num_heads
        assert p.shape == (B, H, T, T)
        assert (p.sum(-1) - 1).abs().max() < C.TOL                      # a probability distribution
        v = torch.nn.functional.linear(zn, blk.attn.in_proj_weight, blk.attn.in_proj_bias)[..., 2 * E:]
        v = v.reshape(B, T, H, E // H).transpose(1, 2)
        out = blk.attn.out_proj((p @ v).transpose(1, 2).reshape(B, T, E))
        assert (out - before).abs().max() < C.TOL                       # it IS this block's attention
        assert torch.equal(before, after)                               # capture left the block as it was
    assert not getattr(getattr(blk.attn, "attn_drop", None), "_forward_pre_hooks", {})  # no hook left behind
