# Week-3 Pilot Report & Gate G1 Decision (§11.1)

- **Student:** Isma Anwar (`SP25-RCS-011`, MSCS, COMSATS University Islamabad)
- **Supervisor:** Dr. Tehseen Zia
- **Date:** 2026-11-23
- **Repository:** `https://github.com/weedevelopers03/query-blind` (frozen split & protocol commit `9cf7490`; pilot runs logged under `4e42dc56`, `5a83128`, `789a6b0`, `644228d`)
- **Archived Results Backup:** `IsmaAnwar/queryblind-results` (`results_backup.zip`, Hugging Face private dataset)
- **Pre-Registration:** [osf_preregistration.md](osf_preregistration.md)

---

## Executive Summary & Gate G1 Verdict

I have completed all seven deliverables required by Methodology Guide §11.1 across the 8 pilot checkpoints on `CUB-200-2011` (`N = 500`, seed 0), `ImageNet-1K val` (`N = 1,000`, seed 0), and `Waterbirds` (`N = 5,794` full test for M3; `N = 2,000` for M4).

- **Gate G1 Decision (§11.2): PASS via M4 (H3).** I am explicitly reporting **"G1 passes via M4"**, not "both criteria":
  - *Criterion 1 (`QII_S4` gap, `dino_vits16 − deit3_small_patch16_224` at $l^*$):* **Failed on CUB** (`-0.0006 [-0.0016, +0.0005]`, CI crosses zero) and **negligible on ImageNet-1K** (`+0.005 [+0.004, +0.006]` on a compressed `0.97–1.00` ceiling, even though `dino_vits16` leads on map `box-AUC` by `+0.190` on CUB and `+0.223` on IN-1K).
  - *Criterion 2 (M4 artifact-controlled similarity `τ(C0, C1) < τ_noise`):* **Decisive pass on all three datasets** (`CUB-500`, `Waterbirds-2,000`, `ImageNet-1,000`). Masking high-norm outlier tokens (`OUT`, $> \text{median} + 5\,\text{MAD}$) drops patch-level (`R3`) debiased CKA rank correlation from split-half noise `τ_noise = 0.939–0.983` down to `τ(C0, C1) = 0.090–0.476` (`Δτ = -0.505` to `-0.864`, both same-half and cross-half 95% CIs far below zero), whereas a 5-seed matched-count random patch drop stays at `τ(C0, C1_rand) = 0.967–0.997` (`sd ≤ 0.007`).
- **Hypothesis Status After Pilot:**
  - **H1 (Query-Blind Localization as pre-registered): Not supported.** All 8 ViTs exhibit high in-cone invariance (`QII_S4 = 0.92–1.00`) because `[CLS]` queries concentrate tightly around their empirical mean ($\cos(\mathbf{q}, \boldsymbol{\mu}) = 0.85–0.999$). This matches the proposal's own *"architectural, not objective-driven"* fallback outcome.
  - **H2 / H2b (Routing Differences):** **Supported on M3 Waterbirds background main effect** against size-matched SSL ranges at matched layers `l=8` (`deit3` `1.9×–3.1×` `S/16` SSL; `openclip` `2.7×–3.9×` `B`-scale SSL) and `l=11` (`deit3` `2.7×–3.3×` `S/16` SSL; `openclip` `1.6×–1.8×` `B`-scale SSL). Exploratory CUB I1 query replacement is **not supported** (`deit3_small_patch16_224` improves `+0.021–0.024`). ImageNet-1K I1/I2 and I5 outputs are held in a separate review packet pending inspection before integration into this report.
  - **H3 (Artifact-Distorted Similarity):** **Strongly supported** for `C1` masking across all 3 datasets; `C2` (test-time registers via I3) remains to be tested.
  - **H4 (Registers):** `QII_S4` survives registers (`0.972 → 0.983` CUB; `0.919 → 0.938` IN-1K) and `share_OUT` drops `93%` (`0.15 → 0.01`) as mass shifts to `REG` (`0.15`). The `+4.3 pt` CUB / `+5.7 pt` IN-1K map `box-AUC` gain is a map-quality effect rather than H4 itself.
  - **H5 (RALR):** **Mixed on primary `box-AUC`** (CUB `6/8`, two-sided `p = 0.289`; IN-1K `4/8`, two-sided `p = 1.000`), **strong on zero-tuning `Otsu CorLoc@0.5`** (CUB `7/8`, `0.440 → 0.693`; IN-1K `8/8`, `0.329 → 0.599`, two-sided `p = 0.0078`, beating top-2% winsorized and OUT-masked `[CLS]@l*` controls on `8/8` IN-1K models).
- **Causal vs. Observational Scope (Fix 1):** All forward-pass interventions (`I1–I5`) measure causal effects of inference-time routing paths within a frozen checkpoint; cross-objective differences are controlled observational comparisons.

---

## 1. Correctness Suite, Registry & Security Closeout (§11.1 #1)

### 1.1 Per-Checkpoint 5-Test Table (`PILOT8_REAL_WEIGHTS=1`, `config_hash = f3a7322f2cecdf15`)
All 8 pilot architectures passed mutation-tested correctness checks (`0` failures, `0` uncaught seeded mutants; max diffs `t1 ≤ 2.3e-6`, `t2 ≤ 1.7e-6`, well below `1e-4`):

| Test | `dino_vits16` | `deit3_small_patch16_224` | `ibot_vits16` | `mocov3_vitb16` | `mae_vitb16` | `dinov2_vitb14` | `dinov2_vitb14_reg` | `openclip_vitb16` |
|---|---|---|---|---|---|---|---|---|
| **1. Attn recon** | PASS 2/2 | PASS 2/2 | PASS 2/2 | PASS 2/2 | PASS 2/2 | PASS 2/2 | PASS 2/2 | PASS 2/2 |
| **2. Residual recon** | PASS 2/2 | PASS 3/3 | PASS 2/2 | PASS 2/2 | PASS 2/2 | PASS 3/3 | PASS 3/3 | PASS 2/2 |
| **3. Token layout** | PASS 2/2 | PASS 2/2 | PASS 2/2 | PASS 2/2 | PASS 2/2 | PASS 2/2 | PASS 3/3 | PASS 2/2 |
| **4. Register idx** | n/a | n/a | n/a | n/a | n/a | n/a | 4a PASS 1/1; **4b PASS** (`ratio=1.261`, 8 VOC imgs) | n/a |
| **5. Known result** | 5a SKIP; **5b PASS** (`61.42%`, `3078/5011`) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| **Exact block fwd (I1/I2)** | PASS (`<1e-4`) | PASS (`<1e-4` incl. logits) | PASS (`<1e-4`) | PASS (`<1e-4`) | PASS (`<1e-4`) | PASS (`<1e-4`) | PASS (`<1e-4`) | PASS (`<1e-4`) |

- **PE-interpolation test (`tests/test_pe_interpolation.py`):** Verified directly against official `LOST/dino/vision_transformer.py`: `0.0e+00` max diff across all 8 grids (vs `0.56–3.00` stock `timm`), `0.0e+00` last-block key diff, `9/9` seeded mutants caught.
- **CorLoc closeout (`VOC07_trainval`):** `61.42%` (`3078/5011`) vs official LOST `61.48%` (`3081/5011`, McNemar `p = 0.648`), inside `[60.9, 62.9]`.
- **Registry (`configs/checkpoints.yaml`):** 8/8 pilot rows complete and passing `tests/test_registry.py` (`dinov2_vitb14_reg` `epochs = 9.01` is marked **complete** rather than confirmed because it rests on the published DINOv2 schedule).
- **Security & Split Hygiene:** Prior exposed HF token revoked (`401`), rotated into Kaggle Secrets (`HF_TOKEN`, verified `200`), zero plaintext tokens in `recent.ipynb`. All 4 seed-0 split & random-20 eye-check ID lists are SHA-verified and committed in `configs/splits/` (`9cf7490`).
- **Remaining Phase 0 Caveats (disclosed honestly):** `t5a` (DINO paper 2-image qualitative figure) remains `SKIP`; Tests 1–3 ran on random-init instances rather than trained checkpoints (compensated by `t4b`, `t5b`, and the real-weights IN-1K sanity check in §8); `verify_cache.py` was bypassed during pilot because M1–M4 ran directly from live GPU forward passes without fp16 disk caching (needed only if we cache 10k-image features).

---

## 2. Measure M1 — QII Across Depth with Entropy & `box-AUC` (§11.1 #2)

### 2.1 ImageNet-1K Val (`N = 1,000`, seed 0; `14×14` matched grid: `224px /16`, `196px /14`)
*(Selected key layers `l=8` and `l=11` plus unlabelled $l^*$; full 32-row table in `results/pilot_m1_m2_in1k`)*

| Model | Objective | $l^*$ | `QII_S1` ($l^*$) | `QII_S2` ($l^*$) | `QII_S3` ($l^*$) | `QII_S4` ($l^*$) [95% CI] | $H_{\text{norm}}$ ($l^*$) | Status (Fix 2) | `box-AUC(attn)` | $\omega^2_{\text{key}}$ [95% CI] | `box-AUC(km)` |
|---|---|---:|---:|---:|---:|---|---:|---|---:|---|---:|
| `dino_vits16` | self_distillation | 10 | 0.759 | 0.975 | 0.908 | 0.979 [0.979, 0.980] | 0.869 | eligible | 0.840 | 0.815 [0.812, 0.818] | 0.838 |
| `deit3_small_patch16_224` | label_supervised | 11 | 0.586 | 0.959 | 0.888 | 0.974 [0.973, 0.975] | 0.777 | eligible | 0.617 | 0.862 [0.860, 0.865] | 0.692 |
| `ibot_vits16` | self_distillation | 11 | 0.747 | 0.958 | 0.779 | 0.967 [0.966, 0.968] | 0.875 | eligible | 0.825 | 0.389 [0.386, 0.392] | 0.803 |
| `mocov3_vitb16` | contrastive | 10 | 0.730 | 0.946 | 0.757 | 0.950 [0.949, 0.951] | 0.849 | eligible | 0.796 | 0.540 [0.536, 0.544] | 0.803 |
| `mae_vitb16` | reconstruction | 11* | 0.821 | 0.993 | 0.653 | 0.994 [0.993, 0.994] | 0.948 | **degenerate_invariance** | 0.747 | 0.199 [0.196, 0.202] | 0.727 |
| `dinov2_vitb14` | self_distillation | 11 | 0.578 | 0.908 | 0.714 | 0.919 [0.917, 0.920] | 0.753 | eligible | 0.755 | 0.415 [0.411, 0.418] | 0.755 |
| `dinov2_vitb14_reg` | self_distillation | 11 | 0.609 | 0.927 | 0.697 | 0.938 [0.937, 0.939] | 0.789 | eligible | 0.812 | 0.490 [0.487, 0.493] | 0.798 |
| `openclip_vitb16` | image_text_contrastive | 11 | 0.597 | 0.950 | 0.769 | 0.957 [0.956, 0.959] | 0.757 | eligible | 0.729 | 0.525 [0.521, 0.529] | 0.763 |

*\*`mae_vitb16` has zero eligible layers ($H_{\text{norm}} = 0.948–0.958 > 0.90$ across `l=8..11` on IN-1K; `0.925–0.956` on CUB) and falls back to `l=11` with `degenerate_invariance` flagged — directly validating Fix 2.*

### 2.2 CUB-200-2011 (`N = 500`, seed 0; `500/500` pixel masks) — Summary at $l^*$
- `dino_vits16` ($l^*=10$): `QII_S4 = 0.991`, `H = 0.754`, `box-AUC(attn) = 0.980`, $\omega^2_{\text{key}} = 0.841$, `box-AUC(km) = 0.985`.
- `deit3_small_patch16_224` ($l^*=11$): `QII_S4 = 0.992`, `H = 0.789`, `box-AUC(attn) = 0.746` (`0.790` unmasked eval), $\omega^2_{\text{key}} = 0.876$, `box-AUC(km) = 0.825`.
- `dinov2_vitb14` → `dinov2_vitb14_reg` ($l^*=11$): `QII_S4 = 0.972 → 0.983`, `box-AUC(attn) = 0.910 → 0.963` (`+5.3 pt` on pixel masks / `+4.3 pt` on box eval).
- **Gate G1 Criterion 1 (`dino_vits16 − deit3_small_patch16_224` at $l^*$):** CUB `ΔQII_S4 = -0.0006 [-0.0016, +0.0005]` (`excl0 = False`, `Δbox-AUC = +0.1904 [+0.1822, +0.1997]`); IN-1K `ΔQII_S4 = +0.005 [+0.004, +0.006]` (`excl0 = True`, `Δbox-AUC = +0.223 [+0.212, +0.235]`). **Conclusion:** H1 as pre-registered is **not supported**; in-cone query invariance is a near-universal ViT architectural property, whereas map localization quality (`box-AUC`) differs by `19–22 pt`.

---

## 3. Measure M2 — Key Dominance ($\omega^2_{\text{key}}$) & Fix 3 Spearman Decision (§11.1 #3)

Across depth (`l=8 → l=11`), $\omega^2_{\text{key}}$ rises toward the final blocks (`dino_vits16`: `0.468 → 0.815` at `l=10`; `deit3_small_patch16_224`: `0.229 → 0.862` at `l=11`), while `ibot_vits16` (confirmed post-ImageNet-norm fix) stays lower at intermediate layers (`0.101–0.143` at `l=8..10`, rising to `0.389` at `l=11` on IN-1K / `0.443` on CUB) despite matching `dino_vits16` on `box-AUC`.

| Dataset | Scope | $N$ | Spearman $\rho(\text{QII}_{S4}, \omega^2_{\text{key}})$ | $p$-value | Fix 3 Decision (`ρ ≥ 0.9` demotes QII) |
|---|---|---:|---:|---:|---|
| CUB-500 | Final layer (all models) | 8 | `+0.048` | `0.9108` | **keep_both** |
| CUB-500 | Final layer (excl. degenerate MAE) | 7 | `+0.571` | `0.1802` | **keep_both** |
| CUB-500 | $l^*$ (all models, MAE fallback `l=11`) | 8 | `-0.190` | `0.6514` | **keep_both** |
| CUB-500 | $l^*$ (excl. degenerate MAE) | 7 | `+0.214` | `0.6445` | **keep_both** |
| ImageNet-1K | Final layer (all models) | 8 | `+0.143` | `0.7358` | **keep_both** |
| ImageNet-1K | Final layer (excl. degenerate MAE) | 7 | `+0.714` | `0.0713` | **keep_both** |
| ImageNet-1K | $l^*$ (all models, MAE fallback `l=11`) | 8 | `+0.048` | `0.9108` | **keep_both** |
| ImageNet-1K | $l^*$ (excl. degenerate MAE) | 7 | `+0.571` | `0.1802` | **keep_both** |

**Fix 3 verdict:** `QII_S4` and $\omega^2_{\text{key}}$ are empirically non-redundant (`max ρ = 0.714 < 0.90` across all 8 checks) → **keep both**.

---

## 4. Measure M3 — Waterbirds Routing Decomposition & Balanced 2×2 CIs (§11.1 #4)

Evaluated on the **full `5,794`-image Waterbirds test set** (`2,897` congruent / `2,897` incongruent). Because Waterbirds test is ~78% landbirds / 22% waterbirds, the pooled `cong − incong` difference conflates a **Water-vs-Land background main effect** with the **true balanced 2×2 congruence interaction** $\frac{1}{2}(\text{L/L}+\text{W/W}) - \frac{1}{2}(\text{L/W}+\text{W/L})$. Below I report both at **matched layers `l=8` and `l=11`** (no oracle layer picking):

| Model | Layer | `BSS_norm` | Bg Main Effect (`W − L`) [95% CI] | Excl 0? | Balanced Interaction (`C − I`) [95% CI] | Excl 0? |
|---|---:|---:|---|---:|---|---:|
| `dino_vits16` | `l=8` | 0.192 | `+0.0254 [+0.0189, +0.0316]` | Yes | `-0.0034 [-0.0100, +0.0026]` | No |
| `deit3_small_patch16_224` | `l=8` | 0.485 | **`+0.0778 [+0.0659, +0.0895]`** (`1.9×–3.1×` S/16 SSL) | Yes | **`-0.0141 [-0.0262, -0.0025]`** | **Yes** |
| `ibot_vits16` | `l=8` | 0.216 | `+0.0410 [+0.0347, +0.0477]` | Yes | `-0.0041 [-0.0108, +0.0019]` | No |
| `mocov3_vitb16` | `l=8` | 0.297 | `+0.0473 [+0.0382, +0.0565]` | Yes | `-0.0083 [-0.0178, +0.0004]` | No |
| `mae_vitb16` | `l=8` | 0.678 | `-0.0647 [-0.0754, -0.0535]` | Yes | `-0.0060 [-0.0166, +0.0040]` | No |
| `dinov2_vitb14` | `l=8` | 0.459 | `+0.0685 [+0.0569, +0.0801]` | Yes | `-0.0079 [-0.0207, +0.0033]` | No |
| `dinov2_vitb14_reg` | `l=8` | 0.347 | `+0.0568 [+0.0470, +0.0660]` | Yes | `-0.0049 [-0.0150, +0.0041]` | No |
| `openclip_vitb16` | `l=8` | 1.560 | **`+0.1854 [+0.1396, +0.2326]`** (`2.7×–3.9×` B-scale SSL) | Yes | `-0.0188 [-0.0717, +0.0277]` | No |
| `dino_vits16` | `l=11` | 0.206 | `+0.0374 [+0.0309, +0.0437]` | Yes | `-0.0021 [-0.0088, +0.0041]` | No |
| `deit3_small_patch16_224` | `l=11` | 0.314 | **`+0.1015 [+0.0459, +0.1387]`** (`2.7×–3.3×` S/16 SSL) | Yes | `-0.0235 [-0.0815, +0.0131]` | No |
| `ibot_vits16` | `l=11` | 0.209 | `+0.0303 [+0.0231, +0.0375]` | Yes | `-0.0020 [-0.0095, +0.0051]` | No |
| `mocov3_vitb16` | `l=11` | 0.315 | `+0.0394 [+0.0279, +0.0510]` | Yes | `-0.0086 [-0.0208, +0.0033]` | No |
| `mae_vitb16` | `l=11` | 0.415 | `+0.0239 [+0.0132, +0.0342]` | Yes | `-0.0191 [-0.0299, -0.0083]` | Yes |
| `dinov2_vitb14` | `l=11` | 0.226 | `+0.0360 [+0.0249, +0.0475]` | Yes | `+0.0043 [-0.0067, +0.0152]` | No |
| `dinov2_vitb14_reg` | `l=11` | 0.153 | `+0.0412 [+0.0348, +0.0475]` | Yes | `-0.0003 [-0.0068, +0.0058]` | No |
| `openclip_vitb16` | `l=11` | 0.261 | **`+0.0653 [+0.0562, +0.0747]`** (`1.6×–1.8×` B-scale SSL) | Yes | `-0.0006 [-0.0099, +0.0084]` | No |

- **Area-tercile stratification (§4.4) passes:** `deit3_small_patch16_224 > dino_vits16`, `openclip_vitb16 l=8 > dino_vits16`, and `dinov2_vitb14_reg < dinov2_vitb14` hold inside all three object-area terciles.
- **Size-matched SSL range summary:**
  - At **`l=8`**: `deit3_small_patch16_224` (`S/16`, `+0.0778`) is **`1.9×` `ibot_vits16` (`+0.0410`)** and **`3.1×` `dino_vits16` (`+0.0254`)**; `openclip_vitb16` (`B/16`, `+0.1854`) is **`2.7×` `dinov2_vitb14` (`+0.0685`)**, **`3.3×` `dinov2_vitb14_reg` (`+0.0568`)**, and **`3.9×` `mocov3_vitb16` (`+0.0473`)** — all with non-overlapping 95% CIs.
  - At **`l=11`**: `deit3_small_patch16_224` (`S/16`, `+0.1015`) is **`2.7×` `dino_vits16` (`+0.0374`)** and **`3.3×` `ibot_vits16` (`+0.0303`)**; `openclip_vitb16` (`B/16`, `+0.0653`) is **`1.6×` `dinov2_vitb14_reg` (`+0.0412`)**, **`1.7×` `mocov3_vitb16` (`+0.0394`)**, and **`1.8×` `dinov2_vitb14` (`+0.0360`)** — all with non-overlapping 95% CIs.
- **Honest CI qualification on Balanced Interaction:** Only `deit3_small_patch16_224` at `l=8` (`-0.0141 [-0.0262, -0.0025]`) and degenerate `mae_vitb16` at `l=11` exclude zero; `deit3 l=11` (`-0.0235 [-0.0815, +0.0131]`) and `openclip l=8` (`-0.0188 [-0.0717, +0.0277]`) cross zero and are not claimed as significant interactions.

---

## 5. Measure M4 — Artifact-Controlled Similarity & Matched Random-Drop Control (§11.1 #5 — Gate G1 PASS)

Across all three datasets (`CUB-500`, `Waterbirds-2,000`, `ImageNet-1,000`), masking `OUT` tokens (`C1`) severely reorders inter-model similarity rankings for patch-level (`R3`) debiased linear CKA — whereas dropping the exact same count of random patches (`C1_rand`, 5 seeds) barely moves CKA at all:

| Dataset | Layer | `R3` CKA `τ(C0, C1_OUT)` | `τ_noise` [95% CI] | `τ(C0, C1_rand)` (5 seeds) | `Δτ (OUT − rand)` | Max `\|Δsim\|` (`OUT` vs `rand`) | `R3` mKNN `τ(C0, C1)` |
|---|---:|---:|---|---|---:|---|---:|
| `cub_500` | `l=8` | **0.148** | 0.946 [0.905, 0.979] | 0.990 ± 0.006 | **-0.842** | 0.819 vs 0.005 | 0.862 |
| `cub_500` | `l=9` | **0.196** | 0.936 [0.894, 0.968] | 0.981 ± 0.007 | **-0.785** | 0.790 vs 0.010 | 0.915 |
| `cub_500` | `l=10` | **0.286** | 0.935 [0.889, 0.974] | 0.977 ± 0.007 | **-0.691** | 0.724 vs 0.006 | 0.937 |
| `cub_500` | `l=11` | **0.423** | 0.983 [0.968, 0.995] | 0.993 ± 0.003 | **-0.569** | 0.546 vs 0.034 | 0.989 |
| `waterbirds_2k` | `l=8` | **0.127** | 0.959 [0.937, 0.979] | 0.992 ± 0.004 | **-0.865** | 0.794 vs 0.007 | 0.884 |
| `waterbirds_2k` | `l=9` | **0.090** | 0.958 [0.937, 0.979] | 0.979 ± 0.005 | **-0.889** | 0.761 vs 0.005 | 0.910 |
| `waterbirds_2k` | `l=10` | **0.238** | 0.968 [0.947, 0.984] | 0.967 ± 0.006 | **-0.729** | 0.704 vs 0.004 | 0.937 |
| `waterbirds_2k` | `l=11` | **0.418** | 0.978 [0.963, 0.989] | 0.992 ± 0.003 | **-0.574** | 0.518 vs 0.035 | 0.984 |
| `imagenet_1k` | `l=8` | **0.159** | 0.948 [0.904, 0.979] | 0.990 ± 0.006 | **-0.832** | 0.681 vs 0.005 | 0.915 |
| `imagenet_1k` | `l=9` | **0.190** | 0.939 [0.899, 0.968] | 0.984 ± 0.003 | **-0.794** | 0.621 vs 0.003 | 0.931 |
| `imagenet_1k` | `l=10` | **0.365** | 0.954 [0.910, 0.979] | 0.993 ± 0.003 | **-0.628** | 0.531 vs 0.005 | 0.926 |
| `imagenet_1k` | `l=11` | **0.476** | 0.976 [0.952, 0.989] | 0.997 ± 0.003 | **-0.521** | 0.339 vs 0.019 | 0.989 |

- **Additional M4 findings:** Mean-pooled (`R2`) CKA also breaks at `l=8..9` (`τ = 0.270–0.783` vs `τ_noise = 0.927–0.959`). Raw `R3` CKA's apparent alignment with localization map IoU collapses under `C1` (`l=9`: `+0.796 → +0.244` IN-1K; `+0.810 → +0.066` WB), showing that raw patch CKA reflects shared artifact spikes. **Mutual $k$-NN (`k=10`, cosine)** is robust (`τ = 0.862–0.989`, `max|Δsim| ≤ 0.050`). Shuffled-feature null floors match theory (`CKA ≈ -0.0015`, `mKNN ≈ 10/(N-1)`).

---

## 6. RALR Readout vs. `[CLS]@l*` & Spike-Clipped Controls (§11.1 #6)

All methods below use **zero box tuning** except `MaxBoxAccV2` (explicitly marked `tunes=YES`). Because `QII_S4 > 0.5` for all 8 models, adaptive `RALR` routes identically to `ralr_high` (query-free key-effect map).

| Benchmark | Method | Tunes on Boxes? | Mean `box-AUC` | Mean `PxAP` | Mean `Pointing` | Mean `Otsu CorLoc@0.5` | Mean `Cal-τ* CorLoc` | Mean `MaxBoxAccV2` *(tunes=YES)* |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **CUB-500** | `cls_last` | NO | 0.912 | 0.669 | 0.605 | 0.433 | 0.421 | 0.657 |
| **CUB-500** | `cls_l_star` (`[CLS]@l*`) | NO | 0.917 | 0.680 | 0.613 | 0.440 | 0.465 | 0.665 |
| **CUB-500** | `cls_l_star_clip98` (top-2% clip) | NO | 0.921 | 0.709 | 0.675 | 0.564 | 0.585 | **0.775** |
| **CUB-500** | `cls_l_star_no_out` (OUT masked) | NO | 0.917 | 0.737 | 0.790 | 0.484 | 0.491 | 0.732 |
| **CUB-500** | `ralr_low` (fixed `bg_high`) | NO | 0.921 | 0.755 | **0.852** | 0.438 | 0.461 | 0.741 |
| **CUB-500** | **`ralr_adaptive` (`ralr_high`)** | **NO** | **0.936** | **0.759** | 0.728 | **0.693** | **0.649** | 0.739 |
| **IN-1K (1,000)** | `cls_last` | NO | 0.759 | 0.740 | 0.664 | 0.334 | 0.305 | 0.521 |
| **IN-1K (1,000)** | `cls_l_star` (`[CLS]@l*`) | NO | 0.759 | 0.739 | 0.661 | 0.329 | 0.309 | 0.518 |
| **IN-1K (1,000)** | `cls_l_star_clip98` (top-2% clip) | NO | 0.761 | 0.744 | 0.644 | 0.416 | 0.411 | **0.604** |
| **IN-1K (1,000)** | `cls_l_star_no_out` (OUT masked) | NO | 0.772 | 0.768 | 0.775 | 0.402 | 0.382 | 0.579 |
| **IN-1K (1,000)** | `ralr_low` (fixed `bg_high`) | NO | 0.772 | **0.769** | **0.793** | 0.394 | 0.407 | 0.587 |
| **IN-1K (1,000)** | **`ralr_adaptive` (`ralr_high`)** | **NO** | **0.773** | 0.749 | 0.664 | **0.599** | **0.562** | 0.575 |

- **Honest H5 Sign-Test Summary (vs `[CLS]@l*`):**
  - Primary `box-AUC`: CUB **`6/8`** (one-sided `p = 0.1445`, two-sided `0.2891`; `mae_vitb16` and `ibot_vits16` lose); IN-1K **`4/8`** (one-sided `p = 0.6367`, two-sided `1.000`; wins on `deit3 +8.7 pt`, `openclip +6.3 pt`, `dinov2 +1.2 pt`, `mocov3 +0.7 pt`; slight losses `-0.2` to `-2.2 pt` on clean self-distilled/MAE).
  - Secondary zero-tuning `Otsu CorLoc@0.5`: CUB **`7/8`** (one-sided `p = 0.0352`, two-sided `0.0703`); IN-1K **`8/8`** (one-sided `p = 0.0039`, two-sided `0.0078`). RALR also beats both `cls_l_star_clip98` (`+18.3 pt`) and `cls_l_star_no_out` (`+19.7 pt`) on **`8/8` IN-1K backbones** (`p = 0.0039`), showing that the gain is not merely top-2% spike clipping — though the largest Otsu jumps (`openclip` `0.006 → 0.864` CUB / `0.020 → 0.615` IN-1K; `dinov2` `0.006 → 0.782` CUB / `0.079 → 0.597` IN-1K) are driven by replacing spiky artifact-dominated `[CLS]` maps with smooth key-effect maps.
  - Missing for Gate G3: `LOST`, `TokenCut`, and a third benchmark (`OpenImages30k`).

---

## 7. Measured Pilot Throughput & Revised Compute Estimate (§11.1 #7)

Measured on a single Kaggle Tesla T4 GPU (`16 GB`):

| Pilot Run | Dataset | $N$ | Total Wall Time (s) | Mean Throughput (img/s) | Min Throughput (img/s) | `git_sha` |
|---|---|---:|---:|---:|---:|---|
| `pilot_m1_m2_cub` | CUB-200-2011 | 500 | 72.9 | 62.0 | 46.9 | `4e42dc56` |
| `pilot_m1_m2_in1k` | ImageNet-1K-val | 1,000 | 606.3 | 53.8 | 42.5 | `4e42dc56` |
| `pilot_waterbirds_m3` | Waterbirds-test | 5,794 | 753.1 | 74.2 | 51.9 | `4e42dc56` |
| `pilot_similarity_m4_cub` | CUB-200-2011 | 500 | 483.8 | 66.8 | 47.9 | `4e42dc56` |
| `pilot_similarity_m4_wb2k` | Waterbirds-2k | 2,000 | 1,329.7 | 72.5 | 51.7 | `4e42dc56` |
| `pilot_similarity_m4_in1k` | ImageNet-1K-val | 1,000 | 655.1 | 59.6 | 44.3 | `4e42dc56` |
| `pilot_ralr_cub` | CUB-200-2011 | 500 | 967.2 | 68.1 | 47.6 | `4e42dc56` |
| `pilot_ralr_in1k` | ImageNet-1K-val | 1,000 | 1,973.9 | 59.1 | 42.3 | `4e42dc56` |

- **Revised Compute Estimate & Coverage Statement (vs. Guide's `~90 GPU-hours` MVP budget):**
  - **Covered in the `7.82 T4 GPU-hours` figure:** 12 ViT-S/B checkpoints (Sets A+B only), `14×14` primary matched grid (`224px /16`, `196px /14`), 10k-image M1/M2 (`2.53h`), full `5,794`-image Waterbirds M3 (`0.31h`), 10k-image M4 `C0/C1` (`1.14h`), 1k-image cached-`z_start` I1/I2/I5 (`2.43h`), and 3×1k-image RALR (`1.41h`).
  - **Explicitly OMITTED from `7.82h` (accounting for the gap to the Guide's `~90 GPU-hours` MVP / `~190 GPU-hours` full budget):**
    1. Full-scale localization sweeps (`50,000`-image ImageNet-1K val WSOL and `30,000`-image OpenImages30k; scaling RALR from `1k` to `50k + 30k + 5.8k` adds ~`35–45 T4-h`),
    2. Larger backbones in Sets C–F (`ViT-L/14`, `ViT-H/14`, `ViT-g`, 7B Web-SSL, DINOv3-L/7B),
    3. Secondary high-resolution (`448/518px`) extraction protocol (§1.5),
    4. Intervention **I3** (test-time register sweeps) + M4 Condition `C2`,
    5. Intervention **I4** (linear stitching), `LOST`/`TokenCut` baselines on full splits, disk-cache extraction/verification, VOC12/COCO20k, medical extension, and debugging/rerun contingency.

---

## 8. Exploratory Addendum: IN-1K Feature Sanity Check & Exploratory CUB I1/I2

- **ImageNet-1K Feature Sanity Check (`N = 1,000` pilot, `disjoint_val_5000` gallery, `scripts/check_in1k_feature_sanity.py`):**
  - `deit3_small_patch16_224`: IN-1K **top-1 accuracy (own logits) = `82.10%`**, IN-1K `kNN@20 = 76.20%`.
  - `mocov3_vitb16`: IN-1K `kNN@20 = 48.90%` (vs `dino_vits16` `53.50%` and `dinov2_vitb14_reg` `68.90%`).
  - *Conclusion:* Post-norm `[CLS]` features and classifier logits are verified healthy; lower CUB-200 200-way kNN for `deit3_small_patch16_224` (`0.400`) and `mocov3_vitb16` (`0.428`) reflects fine-grained OOD transfer rather than a feature-extraction bug.
- **Exploratory CUB-500 I1/I2 Summary (explicitly non-confirmatory; ImageNet-1K I1/I2 & I5 held separately for review before inclusion):**
  - **I1:** Under in-cone surrogates (`S2, S3, S4`, `L_int = 1..4`), all 8 models remain within `±0.024` `Δbox-AUC` and `|ΔkNN| ≤ 0.029` (`deit3_small_patch16_224` improves `+0.021–0.024`), whereas off-manifold `S1` collapses every model (`ΔAUC = -0.283` to `-0.440`, `ΔkNN` down `-0.34` to `-0.71`, DeiT-III `KL = 1.70 nats`). Thus H2's query-swap leg is **not supported; DeiT-III improves**.
  - **I2:** Map quality at `l=10` (unablated `l=11` map) survives token knockout within `±0.006` for 6/8 models while `[CLS]` function moves substantially (`l=11` FG `1−cos`: `0.259` `dinov2_vitb14_reg`, `0.179` `dinov2_vitb14`, `0.121` `deit3_small_patch16_224`, `0.078` `openclip_vitb16`). On `dinov2_vitb14_reg`, knocking out the 4 `REG` tokens shifts `[CLS]` (`net 1−cos = +0.054`) while leaving the map unchanged (`ΔAUC = -0.0008`).

---

## 9. Immediate Next Steps (Toward 4 Dec First Draft)

1. Export the remaining 2 split ID lists (`cub_gallery_train_5994.txt` and `waterbirds_m4_2000_seed0.txt`), commit SHAs, and **post [osf_preregistration.md](osf_preregistration.md) to OSF tonight**.
2. Share the standalone ImageNet-1K I1/I2 and I5 outputs + scripts for supervisor/reviewer sign-off before folding them into this report.
3. Run unseen confirmatory items (**12-model Sets A+B expansion**, **I3 / C2**, **`LOST` / `TokenCut` RALR baselines**, and **`stats/`** Holm + mixed-effects + descriptive confound regression), applying the 16 Dec fallback rule if the 12-model run cannot finish in time.
