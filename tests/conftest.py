"""Checkpoints used by the §1.3 suite. Random-init, architecture-only (see tests/README.md for
why pretrained weights aren't used here) - one per adapter to exercise its distinct code path,
plus a DINOv2-reg model to exercise the register-indexing branch of TimmViTAdapter.
"""
import torch
import timm
import open_clip
import pytest
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from adapters.timm_vit import TimmViTAdapter
from adapters.open_clip_vit import OpenCLIPViTAdapter
from adapters.map_head import MapHeadAdapter
from adapters.dinov3_hf import Dinov3HFAdapter


def _timm(name, **kw):
    return TimmViTAdapter(), timm.create_model(name, pretrained=False, **kw)


def _open_clip(name):
    model = open_clip.create_model(name, pretrained=None)
    return OpenCLIPViTAdapter(), model.visual


def _map_head(name):
    return MapHeadAdapter(), timm.create_model(name, pretrained=False)


def _dinov3(hf_id):
    # real run: DINOv3ViTModel.from_pretrained(hf_id) - needs internet/HF access, so this test
    # suite builds the SAME architecture from a bare config instead (random weights); see the
    # smoke-test note in tests/README.md, same reasoning as the other three families
    return Dinov3HFAdapter(), DINOv3ViTModel.from_pretrained(hf_id, dtype=torch.float32)


CHECKPOINTS = {
    "deit3_small": lambda: _timm("deit3_small_patch16_224"),          # row 1, plain [CLS]
    # guide §1.5 primary protocol: /14 models run at 196px for a 14x14 grid (this one defaults
    # to 518px, which a plain torch.randn(...,224,224) in a test would silently NOT catch as
    # wrong-resolution - it just asserts and fails, which is how this default was found).
    "dinov2_reg_small": lambda: _timm("vit_small_patch14_reg4_dinov2", img_size=196),  # row 2, [CLS]+registers
    "open_clip_vitb32": lambda: _open_clip("ViT-B-32"),                # nn.MultiheadAttention path
    "siglip_vitb16": lambda: _map_head("vit_base_patch16_siglip_224"), # MAP head, no [CLS]
    "dinov3_vits16": lambda: _dinov3("facebook/dinov3-vits16-pretrain-lvd1689m"),  # RoPE, via transformers not timm
}


def _patch_embed_module(model):
    """The conv/projection layer that turns pixels into patch tokens, across all 4 families'
    different attribute names."""
    if hasattr(model, "patch_embed"):
        return model.patch_embed                       # timm
    if hasattr(model, "embeddings"):
        return model.embeddings.patch_embeddings        # transformers (DINOv3)
    return model.conv1                                  # open_clip


def random_image(adapter, model, batch=2, seed=0):
    """A random image sized from the checkpoint's OWN patch size and grid (§1.5), not a hardcoded
    224 - the wrong constant for any /14 model or any non-224-native checkpoint."""
    gh, gw = adapter.token_layout(model)["grid"]
    pe = _patch_embed_module(model)
    ps = pe.patch_size[0] if hasattr(pe, "patch_size") else pe.kernel_size[0]
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, 3, gh * ps, gw * ps, generator=g)


@pytest.fixture(params=list(CHECKPOINTS), ids=list(CHECKPOINTS))
def checkpoint(request):
    adapter, model = CHECKPOINTS[request.param]()
    adapter.prepare(model)
    return request.param, adapter, model