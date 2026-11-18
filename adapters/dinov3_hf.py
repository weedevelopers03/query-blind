"""Adapter for DINOv3 via HuggingFace `transformers`' DINOv3ViTModel - not timm. timm does have
DINOv3 (`vit_small_patch16_dinov3` etc.), but it's built on timm's Eva-family class for RoPE
support, whose `_pos_embed` returns `(tokens, rope_freqs)` - exactly the case TimmViTAdapter
already refuses ("RoPE-style ViTs return (tokens, rope); not supported here", see timm_vit.py).
transformers' own class applies rotation internally and just hands back clean (B,H,T,T) attention
and per-layer hidden states via `output_attentions=True, output_hidden_states=True` - no need to
reimplement RoPE by hand to get those two.

Needs `attn_implementation="eager"`: the default (`sdpa`) refuses `output_attentions=True`
outright - the same fused-kernel problem the guide's §1.2 warns about, in a different library.
Setting it on an already-instantiated model's `.config` is sufficient; it doesn't need to be
passed at construction time.

--- RoPE and [CLS]/registers (supervisor review, 2026-10-09; confirmed against transformers
5.19.0's modeling_dinov3_vit.py, both by reading the source and by testing a random-init
DINOv3ViTModel end-to-end - see tests/test_correctness.py's test_dinov3_attention_reconstruction
and the appended verification notes below) ---

CONFIRMED: `apply_rotary_pos_emb()` rotates ONLY the patch tokens. [CLS] (index 0) and the
register tokens are split off first ("prefix tokens"), passed through completely unrotated, and
re-concatenated afterwards. Quote from the library source: "Applies Rotary Position Embedding to
the query and key tensors, but only to the patch tokens, ignoring the prefix tokens (cls token and
register tokens)."

Two consequences, previously wrong in this file:

1. The GLOBAL QUERY (q_g = [CLS]'s query) is therefore IDENTICAL whether read pre- or post-RoPE -
   there is no rotation applied to it either way, so the earlier version of this docstring's
   rationale ("both images get the same rotation applied anyway") was reaching the right practical
   conclusion (pre/post doesn't matter for q_g) by the wrong argument (CLS isn't rotated at all,
   not "rotated identically"). This part was harmless.

2. The PATCH KEYS used by M1 (QII) and M2 (key dominance) are NOT harmless to get wrong. Real
   attention is softmax(q_g . ROTATED(k_i)^T / sqrt(d_h)) - each patch key is rotated by an amount
   that depends on ITS OWN spatial position, so pre-RoPE patch keys do not preserve the same
   relative geometry as the keys the model actually attends over. Computing QII or the key-effect
   map from pre-RoPE keys (what get_keys_values() used to return, via get_qkv()'s hook on the raw
   k_proj output) would silently measure a geometrically different, wrong quantity for every
   DINOv3 checkpoint (Set D) - the guide's "patch-key geometry" claim specifically depends on using
   the keys the network itself computes over. Fixed below: get_qkv() / get_keys_values() /
   get_global_query() now source q,k from AFTER rotation (v is unaffected by RoPE in any case -
   only q and k pass through apply_rotary_pos_emb).

How the post-RoPE capture works, since `apply_rotary_pos_emb` returns new tensors from inside
`DINOv3ViTAttention.forward` rather than exposing them as a submodule a forward hook could attach
to: the module-level function reference is monkeypatched for the duration of one forward call
(`_dinov3_mod.apply_rotary_pos_emb = wrapped`, restored in a `finally`), counting calls to isolate
the one at layer l (the function runs once per layer per forward pass, in layer order). v is
captured the same way as before, via a plain `register_forward_hook` on `v_proj`. This was verified
end-to-end against a small random-init `DINOv3ViTModel` (no pretrained weights needed for this
specific check, since it is a structural/correctness property of the rotation code, not a trained
one): the independent reconstruction softmax(q_post . k_post^T * scaling), built entirely from
these hooked tensors, matches `out.attentions[l]` to within float rounding (~1e-7, i.e. exact to
fp32 precision) at every layer tried; repeating the same computation with the OLD pre-RoPE q,k
gives a real, order-1e-2 to 1e-3 discrepancy (at this tiny untrained scale) - confirming the
post-RoPE check is both correct and non-vacuous (it would catch a real rotation bug, unlike the
previous version of this test - see the module docstring note in test_correctness.py).

Previously (supervisor-flagged circularity): `test_dinov3_branch_reconstruction` checked
`global_update()` against calling `layer.attention(...)` directly - but `global_update()` reads
its attention weights from `out.attentions[l]`, which is itself produced by the model calling
`layer.attention(...)` internally. Both sides of that comparison therefore shared the same
HF-internal attention-probability computation, so a bug in how those probabilities are computed
(wrong scale, a missed RoPE detail, etc.) could never have been caught by it - only the downstream
v_proj/o_proj/layer_scale wiring was actually being tested. That test is kept below (it still
checks something real), but Test 1 is now the independent, from-hooked-q/k reconstruction
described above, matching what the guide's §1.3 Test 1 asks for in the other three families.

`get_qkv()` / `get_global_query()` / `get_keys_values()` all now return POST-RoPE tensors (q is
numerically identical to the old pre-RoPE version, since CLS is never rotated; k differs for
patches, which is the fix). The legacy pre-rotation hook is kept as `_get_qkv_prerope()`, private
and unused by any guide-required function, purely so a past cache built on the old (incorrect)
patch keys can be identified/reproduced if needed - new extraction should not call it.

Token layout matches DINOv2-reg's convention: [CLS] at 0, then registers, then patches.

Performance note: each call below re-runs the full forward pass (transformers doesn't expose an
"only up to layer l" mode the way the timm/open_clip adapters' hand-rolled residual_in() does), so
this adapter does more redundant compute per call than the other three. Fine for the correctness
suite and one-off checks; worth batching/caching properly before any at-scale extraction run.
"""
import torch

from .base import make_layout

import transformers.models.dinov3_vit.modeling_dinov3_vit as _dinov3_mod


class Dinov3HFAdapter:
    family = "dinov3_hf"

    # ------------------------------------------------------------------ setup
    def prepare(self, model):
        model.eval()
        model.config._attn_implementation = "eager"  # see module docstring
        return model

    def n_blocks(self, model):
        return model.config.num_hidden_layers

    # ---------------------------------------------------------------- layout
    def token_layout(self, model):
        img = model.config.image_size
        img = img if isinstance(img, int) else img[0]
        gh = gw = img // model.config.patch_size
        n_reg = model.config.num_register_tokens
        return make_layout(
            cls_idx=0, register_idx=range(1, 1 + n_reg),
            patch_idx=range(1 + n_reg, 1 + n_reg + gh * gw), grid=(gh, gw),
        )

    # ------------------------------------------------------------ extraction
    @torch.no_grad()
    def _forward(self, model, x):
        return model(x, output_attentions=True, output_hidden_states=True)

    @torch.no_grad()
    def attention_probs(self, model, x, l):
        return self._forward(model, x).attentions[l]

    @torch.no_grad()
    def residual_in(self, model, x, l):
        """Input to block l - hidden_states[0] is the embedding output (input to block 0)."""
        return self._forward(model, x).hidden_states[l]

    def _reshape_heads(self, model, t):
        H = model.config.num_attention_heads
        B, T, _ = t.shape
        d_h = t.shape[-1] // H
        return t.reshape(B, T, H, d_h).transpose(1, 2)

    @torch.no_grad()
    def _get_qkv_prerope(self, model, x, l):
        """Legacy pre-rotation q,k,v (kept only to reproduce/identify a cache built before the
        2026-10-09 fix - see module docstring). Do not use for new extraction."""
        attn = model.model.layer[l].attention
        captured = {}
        handles = [
            attn.q_proj.register_forward_hook(lambda m, i, o: captured.setdefault("q", o)),
            attn.k_proj.register_forward_hook(lambda m, i, o: captured.setdefault("k", o)),
            attn.v_proj.register_forward_hook(lambda m, i, o: captured.setdefault("v", o)),
        ]
        try:
            self._forward(model, x)
        finally:
            for h in handles:
                h.remove()
        return (self._reshape_heads(model, captured["q"]),
                self._reshape_heads(model, captured["k"]),
                self._reshape_heads(model, captured["v"]))

    @torch.no_grad()
    def get_qkv(self, model, x, l):
        """POST-RoPE q, k (v is unaffected by RoPE either way), each (B, H, T, d_h). See module
        docstring: patch tokens are rotated, [CLS]/registers are not, and M1/M2 need the keys the
        model actually attends over, which are these post-rotation ones."""
        orig_fn = _dinov3_mod.apply_rotary_pos_emb
        call_count = {"n": 0}
        captured = {}

        def wrapped(q, k, cos, sin, **kwargs):
            q_out, k_out = orig_fn(q, k, cos, sin, **kwargs)
            if call_count["n"] == l:
                captured["q"] = q_out
                captured["k"] = k_out
            call_count["n"] += 1
            return q_out, k_out

        v_captured = {}
        v_handle = model.model.layer[l].attention.v_proj.register_forward_hook(
            lambda m, i, o: v_captured.setdefault("v", o)
        )
        _dinov3_mod.apply_rotary_pos_emb = wrapped
        try:
            self._forward(model, x)
        finally:
            _dinov3_mod.apply_rotary_pos_emb = orig_fn
            v_handle.remove()
        return captured["q"], captured["k"], self._reshape_heads(model, v_captured["v"])

    # ------------------------------------------------- the four guide functions
    def get_global_query(self, model, x, l):
        q, _, _ = self.get_qkv(model, x, l)
        return q[:, :, 0, :]

    def get_keys_values(self, model, x, l):
        _, k, v = self.get_qkv(model, x, l)
        return k, v

    @torch.no_grad()
    def global_update(self, model, x, l):
        """Attn-branch contribution to [CLS]: layer_scale1(o_proj(concat_h sum_i a_i^h v_i^h)),
        using the model's own real (post-rotation) attention - not a hand-rolled softmax. This is
        the branch-level / Test 2 check (wiring of v_proj -> a@v -> o_proj -> layer_scale); Test 1
        (are the attention weights themselves right) is test_dinov3_attention_reconstruction in
        test_correctness.py, built on get_qkv() above."""
        out = self._forward(model, x)
        a = out.attentions[l][:, :, 0:1, :]  # (B,H,1,T), the model's real attention
        layer = model.model.layer[l]
        v = layer.attention.v_proj(layer.norm1(out.hidden_states[l]))
        B, T, _ = v.shape
        H = model.config.num_attention_heads
        v = v.reshape(B, T, H, -1).transpose(1, 2)
        o = (a.float() @ v.float()).transpose(1, 2).reshape(B, 1, -1)
        o = layer.attention.o_proj(o.to(v.dtype))
        return layer.layer_scale1(o)[:, 0].float()


_A = Dinov3HFAdapter()
prepare = _A.prepare
get_global_query = _A.get_global_query
get_keys_values = _A.get_keys_values
token_layout = _A.token_layout
global_update = _A.global_update