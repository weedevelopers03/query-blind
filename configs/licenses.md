# Checkpoint & Dataset Licensing Audit (Methodology Guide §8.5)

> Every checkpoint and dataset used in the audit is logged here with its source, license, and usage restrictions.
> All Pilot-8 checkpoints are verified for academic research use.

---

## 1. Pilot-8 Checkpoints (Verified)

| Checkpoint ID | Set | Objective | Source | License | Academic Use / Notes |
|---|---|---|---|---|---|
| `dino_vits16` | A | Self-distillation | `timm/vit_small_patch16_224.dino` | Apache-2.0 | Open; verified |
| `deit3_small_patch16_224` | A | Label-supervised | `timm/deit3_small_patch16_224.fb_in1k` | Apache-2.0 | Open; verified |
| `ibot_vits16` | A | Self-distillation + MIM | `github.com/bytedance/ibot` (`vits_16/checkpoint_teacher.pth`) | Apache-2.0 | Open; ImageNet norm verified |
| `mocov3_vitb16` | A | Contrastive | `github.com/facebookresearch/moco-v3` (`vit-b-300ep.pth.tar`) | CC-BY-NC-4.0 | Non-commercial academic use permitted |
| `mae_vitb16` | A | Masked reconstruction | `timm/vit_base_patch16_224.mae` | CC-BY-NC-4.0 | Non-commercial academic use permitted |
| `dinov2_vitb14` | B | Self-distillation | `timm/vit_base_patch14_dinov2.lvd142m` | Apache-2.0 | Open; verified |
| `dinov2_vitb14_reg` | B | Self-distillation + registers | `timm/vit_base_patch14_reg4_dinov2.lvd142m` | Apache-2.0 | Open; verified |
| `openclip_vitb16` | F | Image-text contrastive | `open_clip: ViT-B-16 / laion2b_s34b_b88k` | MIT | Open; LAION-2B-en (2.32B imgs) |

---

## 2. Extended Zoo Checkpoints (Sets B–F, Medical & Controls)

| Set | Checkpoint IDs | License | Access / Action Required Before Full Sweep (§8.5) |
|---|---|---|---|
| **A** | `mocov3_vits`, `tscam_deits` | CC-BY-NC-4.0 / Apache-2.0 | Academic use OK |
| **B** | `dinov2_vit{s,l,g}14{,_reg}` | Apache-2.0 | Open via `timm` |
| **C** | `webssl_dino_vit1b`, `metaclip_vitb16` | CC-BY-NC | Verify exact MetaCLIP-2.5B vs Web-SSL subset match; if mismatched, relabel Set C as generality set (§8.5) |
| **D** | `dinov3_vit{s,b,l,hplus}16` | Meta DINOv3 License | Gated on Hugging Face; requires approved `HF_TOKEN` |
| **E** | `augreg_vit{s,b,l}16_in21k`, `deit3_small_patch16_224_in21k` | Apache-2.0 (verify before use) | Deprioritized per Guide §13.1 (appendix only if run) |
| **F** | `siglip2_vit{b,l}16`, `pe_core_{b16,l14}`, `open_clip_vit{b32,l14}`, `siglip_vitb16` | Apache-2.0 / MIT | Open via `timm` / `open_clip` / Meta `perception_models` |
| **Medical** | `rad_dino` (MSRLA), `medsiglip_448` (HAI-DEF), `biomedclip_vitb16` (MIT) | MSRLA / HAI-DEF / MIT | Separate MICCAI track (§13.1); review HAI-DEF & MSRLA terms before clinical evaluation |
| **Controls** | `lastvit_{dino,clip,labelsup}`, `dinov2_vitl14_ttreg`, `open_clip_ttreg` | Verify upstream repo license | Check `ChengShiest/LAST-ViT` and `nickjiang2378/test-time-registers` license files prior to redistribution |

---

## 3. Datasets

| Dataset | Split | License / Access | Notes |
|---|---|---|---|
| **CUB-200-2011** | Test (5,794; 500-img pilot `seed 0`) | Caltech academic research license | Pixel segmentations + 1-based boxes (`eval/preprocess.py`) |
| **Waterbirds (`wilds` v1.0)** | Test (5,794: 2,897 cong / 2,897 incong) | MIT / CUB + Places365 academic terms | Official WILDS mirror (`pangwei/wilds-waterbirds`) + CUB masks |
| **PASCAL VOC 2007** | `trainval` (5,011) | VOC academic research terms | Discovery CorLoc convention (§8.1) |
| **ImageNet-1K (ILSVRC2012)** | Val (50,000; 1,000-img pilot `seed 0`) | ImageNet non-commercial research terms | Requires ILSVRC2012 val images + XML boxes |
