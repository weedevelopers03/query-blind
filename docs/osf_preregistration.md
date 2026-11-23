# OSF Pre-Registration: Query-Blind Localization Audit

- **Title:** Query-Blind Localization: A Training-Free Causal Audit of How Supervision Routes Object Information in Vision Foundation Models
- **Author:** Isma Anwar (`SP25-RCS-011`, MS Computer Science, COMSATS University Islamabad)
- **Supervisor:** Dr. Tehseen Zia
- **OSF Registration Timestamp:** *[recorded automatically by OSF upon freezing]*
- **Code Repository:** `https://github.com/weedevelopers03/query-blind`
- **Relevant Code Commits:**
  - `8bfc63b` — `configs/checkpoints.yaml` defining all candidate checkpoints including the 4 unseen expansion models listed in §1.
  - `9cf7490` — committed frozen seed-0 pilot & random-20 eye-check split files (`configs/splits/`).
  - `a0cfbe5` — frozen I1/I2 & I5 protocol code that produced the ImageNet-1K I1/I2 and ImageNet-1K/CUB I5 model runs.
  - `e28e35b` — I5 prediction-table parser fix (`layer_summaries`) used to render I5 summaries via `--resume` (zero model recomputation).
  - Pre-registration freeze commit: stamped in `GIT_SHA` on push immediately prior to OSF upload.

---

## 1. Timing, Unseen Model List & Fallback Rule

This pre-registration is posted retroactively relative to Dr. Tehseen Zia's 6 November schedule. To prevent any ambiguity about what was observed prior to OSF posting:

- **Observed prior to OSF posting (NOT pre-registered confirmatory):**
  1. Pilot-8 observational measures **M1 (`QII`)**, **M2 ($\omega^2_{\text{key}}$)**, **M3 (`Waterbirds` `BSS_norm`)**, **M4 (`C0` vs `C1` + `C1_rand`)**, and **`RALR`** on `CUB-500` and `ImageNet-1K-1000`.
  2. Pilot-8 Interventions **I1 and I2 on `CUB-500`** (exploratory read-out debugging runs).
  3. Pilot-8 **ImageNet-1K I1/I2** and **ImageNet-1K / CUB-500 I5** — **run under the committed protocol (`a0cfbe5`, summary parser `e28e35b`) before OSF posting, results seen.** These are classified as *run under committed protocol before posting*, **not** pre-registered confirmatory.
  4. ImageNet-1K feature sanity check (`deit3_small_patch16_224` top-1 = `82.10%`, `mocov3_vitb16` `kNN@20 = 48.90%`).
- **Unseen at OSF posting (genuinely pre-registered confirmatory):**
  1. **4 new frozen checkpoints** (all pre-existing in `configs/checkpoints.yaml`, commit `8bfc63b`, all `14×14` matched grid) expanding Pilot-8 to **12 models**:
     - `dinov2_vits14` (`Set B`, `self_distillation`, `timm/vit_small_patch14_dinov2.lvd142m`, 196px)
     - `dinov2_vits14_reg` (`Set B`, `self_distillation`, `timm/vit_small_patch14_reg4_dinov2.lvd142m`, 196px)
     - `deit3_small_patch16_224_in21k` (`label_supervised`, `timm/deit3_small_patch16_224.fb_in22k_ft_in1k`, 224px)
     - `metaclip_vitb16` (`image_text_contrastive`, `timm/vit_base_patch16_clip_224.metaclip_2pt5b`, 224px)
     - Resulting 12-model family counts: **`SSL` = 7** (`dino_vits16`, `ibot_vits16`, `mocov3_vitb16`, `dinov2_vitb14`, `dinov2_vitb14_reg`, `dinov2_vits14`, `dinov2_vits14_reg`), **`label+text` = 4** (`deit3_small_patch16_224`, `deit3_small_patch16_224_in21k`, `openclip_vitb16`, `metaclip_vitb16`; 2 label + 2 text), **`reconstruction` = 1** (`mae_vitb16`, excluded from H2, H2b, and H5 primary tests per §5).
  2. **Intervention I3** (test-time registers) and **M4 Condition `C2`** (artifact-controlled similarity after I3).
  3. **RALR external baselines (`LOST` and `TokenCut`)** under the identical zero-box-tuning protocol, plus a **third localization benchmark (`OpenImages30k`)** for Gate G3.
- **Deadline Fallback Rule (16 December 2026):** If the 12-model run is not completed by 16 December 2026, the MS thesis will report **Pilot-8 only, with no confirmatory claims** (all Pilot-8 results explicitly labelled exploratory / post-hoc).

---

## 2. Causal vs. Observational Scope Statement (Fix 1)

All forward-pass interventions (`I1`, `I2`, `I3`, `I5`) identify **causal effects of inference-time routing paths within a frozen checkpoint**. All comparisons across pretraining objectives are **controlled observational comparisons** adjusted for recorded architecture and training covariates (§6); I do not claim a causal effect of the training objective itself.

---

## 3. Primary Hypotheses, Holm-Bonferroni Family & Exact $p$-Value Rules

Five primary hypothesis tests form the **Holm-Bonferroni family** ($\alpha = 0.05$, two-sided; step thresholds $\alpha/5 = 0.0100$, $\alpha/4 = 0.0125$, $\alpha/3 = 0.0167$, $\alpha/2 = 0.0250$, $\alpha/1 = 0.0500$).

**How CI-based rules (`H1`, `H3`, `H2b-M3`) enter Holm:** Every bootstrap-based hypothesis produces a two-sided bootstrap $p$-value
$$p_{\text{boot}} = \frac{2 \min\!\big(\#\{\theta^* \le 0\},\; \#\{\theta^* \ge 0\}\big) + 1}{B + 1}$$
($B = 2,000$ image-cluster resamples for `H1` and `H2b-M3`; $B = 2,000$ split-half resamples / exact paired bootstrap for `H3`) alongside the corresponding **Holm-adjusted confidence interval** at level $1 - \alpha_{\text{step}}$ (e.g., `99.0%` CI at Step 1). Every test outside this 5-test family — including **Intervention I5** — is explicitly labelled **exploratory**.

| ID | Hypothesis | Primary Statistic & Layer Rule | Statistical Test & Named Contrast | Minimum Effect / Pass Rule |
|---|---|---|---|---|
| **H1** | *(Original proposal test — retained for transparent falsification)* Self-distilled ViTs exhibit higher head-weighted `QII_S4` than label-supervised ViTs at non-degenerate layers. | Head-weighted `QII_S4` at unlabelled $l^* = \arg\max_{l \in \text{last third},\, H_{\text{norm}}(l) \le 0.90} \omega^2_{\text{key}}(l)$. | Paired image-level bootstrap ($B = 2,000$, $p_{\text{boot}}$) within matched `S/16` pairs (`dino_vits16 − deit3_small_patch16_224`). | **Falsified if $p_{\text{boot}}$ fails Holm or `\|ΔQII_S4\| < 0.02`.** *(Already not supported on Pilot-8 CUB `-0.001` and negligible on IN-1K `+0.005`; kept in the Holm family so falsification is preserved.)* |
| **H2** | Inference-time query replacement (`I1`) impacts localization (`Δbox-AUC`) differently for `SSL` vs. `label+text` supervision. | Single-layer `Δbox-AUC` at **`L_int = 1`** under `I1-S4` (`I1-S1` off-manifold control reported alongside), with clean `box_auc_clean` as a baseline covariate. `L_int ∈ {1..4}` functional sweep (`1 − cos`, `ΔkNN`, `KL`) is secondary. | **Named contrast:** binary `group ∈ {SSL (n=7), label+text (n=4)}`. **`mae_vitb16` (`reconstruction`) is EXCLUDED from H2** (neither SSL nor label+text, and Fix 2 degenerate at all layers). Linear mixed model: `Δbox_AUC_{m,i} ~ I(label+text) + box_auc_clean_{m,i} + (1 \| model_id) + (1 \| image_id)`.<br>**Identifiability caveat:** If a family has only 1 checkpoint (Pilot-8), the family fixed effect and `(1 \| model_id)` cannot be separated; mixed-effects inference applies only once the 4 new models give ≥2 checkpoints per family. | `\|β_{label+text vs SSL}\| ≥ 0.02` (`2.0 pt` `box-AUC`) with Holm-adjusted two-sided $p < 0.05$. *(Disclosure: `0.02` sits just below `deit3_small_patch16_224`'s exploratory CUB S4 shift `+0.023` and was set after seeing that result.)* |
| **H2b** | `label+text` ViTs route more global computation through background tokens and exhibit weaker foreground-over-background selectivity than `SSL` ViTs. | **Two legs (both evaluated; `mae_vitb16` excluded):**<br>1. **M3 leg (Waterbirds):** Balanced 2×2 Background Main Effect `(Water − Land)` at **matched layers `l=8` and `l=11`** is **larger** for `label+text` than for size-matched `SSL`.<br>2. **I2 leg (scale-invariant log-ratio selectivity):** Per-model log ratio $R_m = \log\!\big((\overline{1-\cos}_{\text{FG},m} + \varepsilon) / (\overline{1-\cos}_{\text{BG\_eq},m} + \varepsilon)\big)$ ($\varepsilon = 10^{-6}$, `l=11` `mean_v`, exact equal count `m_FG` over `5` random subsets of `BG \ OUT`) is **larger** for `SSL` than for `label+text` (`mean_SSL(R_m) − mean_{label+text}(R_m) > 0`). *(Using the log ratio removes per-model `[CLS]` sensitivity scale.)* | **Combination rule (conjunction via intersection-union test):** **BOTH** the M3 leg and the I2 between-family log-ratio leg must hold for full H2b support, yielding combined Holm $p$-value $p_{\text{H2b}} = \max(p_{\text{M3}}, p_{\text{I2}})$ (where $p_{\text{M3}}$ is the image-cluster bootstrap $p_{\text{boot}}$ comparing `label+text` vs the max size-matched SSL baseline across area terciles, and $p_{\text{I2}}$ is the two-sided permutation/bootstrap test of `mean_SSL(R_m) − mean_{label+text}(R_m)` across checkpoints). If only one leg clears Holm, report partial support honestly (M3-only or I2-only). | Both legs in the predicted direction with $p_{\text{H2b}} = \max(p_{\text{M3}}, p_{\text{I2}})$ clearing Holm ($\alpha = 0.05$) and M3 holding in ≥2 of 3 area terciles. |
| **H3** | High-norm outlier tokens (`OUT`, $\|\mathbf{z}_i\|_2 > \text{median} + 5\,\text{MAD}$) distort inter-model similarity rankings (`M4`), and masking (`C1`) or test-time registers (`C2`) lowers Kendall's $\tau$ below the split-half noise floor. | `Δτ = τ(C0, C1) − τ_noise` (and `τ(C0, C2) − τ_noise` for models passing I3 verification) on `R2` and `R3` debiased linear CKA at **matched layers `l ∈ {8, 9, 10, 11}`**, plus 5-seed matched random-drop `τ(C0, C1_rand)`. | Paired bootstrap $p_{\text{boot}}$ across split-half resamples (`same-half` and `cross-half`) + Holm-adjusted CI strictly `< 0`, plus `τ(C0, C1_rand) − τ(C0, C1) ≥ 0.10`. | Both same-half and cross-half $p_{\text{boot}}$ clear Holm and `τ(C0, C1_rand) − τ(C0, C1) ≥ 0.10`. |
| **H5** | Training-free, zero-box-tuning `RALR` improves object localization over layer-matched `[CLS]@l*` across non-degenerate ViT backbones. | **Primary read-out (MUST pass for H5 to pass):** per-backbone paired `Δbox-AUC = box_AUC(RALR) − box_AUC([CLS]@l*)`. **Secondary (cannot rescue a failed `box-AUC` test):** zero-tuning `Otsu CorLoc@0.5` vs `[CLS]@l*`, `cls_l_star_clip98`, `cls_l_star_no_out`, `LOST`, and `TokenCut`. | **Two-sided exact binomial sign test** across backbones on primary `box-AUC` (Gate G3: ≥2 of 3 benchmarks).<br>**Exact denominator `n` & Holm Step-1 ($\alpha/5 = 0.01$) requirements:**<br>• **Primary (`mae_vitb16` excluded as Fix-2 degenerate):** on the 12-model zoo, **$n = 11$** → requires **`11/11`** wins ($p = 0.0010 < 0.01$; `10/11` yields $p = 0.0117$, which clears Step 2 $\alpha/4 = 0.0125$). On Pilot-8, **$n = 7$** → even `7/7` yields $p = 0.0156$ (cannot clear Step 1; clears Step 3 $\alpha/3 = 0.0167$).<br>• **Sensitivity (including MAE):** **$n = 12$** → requires **`≥ 11/12`** ($p = 0.0063$); **$n = 8$** → requires **`8/8`** ($p = 0.0078$). | **Primary `box-AUC` sign test (`n = 11` non-degenerate models) must clear Holm on ≥2 of 3 benchmarks for H5 to pass.** |

*Exploratory / secondary analyses outside the 5-test Holm family:*
- **H4 (Registers):** Matched register pairs (`dinov2_vitb14/reg`, `dinov2_vits14/reg`): `QII_S4` survives (`|ΔQII_S4| ≤ 0.05`), `share_OUT` drops ≥50% into `REG`, and `REG` knockout moves `[CLS]` (`1 − cos`) without moving `l=10` `box-AUC` beyond `±0.01`.
- **I5 (`[CLS]` Randomization Prediction Test):** Explicitly **exploratory** (outside Holm; across-model Spearman $\rho$ and Pearson $r$ reported with 95% bootstrap CIs, explicitly noting multiple `L_r × mode` looks and narrow `QII_S4` range).

---

## 4. Surrogate Families & Documented Deviations from Guide §2.2

Guide §2.2 specifies four query surrogates (`S1` isotropic Gaussian, `S2` cross-image global query, `S3` same-image random patch query, `S4` random Haar rotation of $\mathbf{q}_g$ inside the top-$k=32$ PCA subspace fit on `5,000` images; 5 seeds `0..4`). Exact codebase specifications and deviations:

| Module | `S1` | `S2` | `S3` | `S4` | Deviations from Guide §2.2 |
|---|---|---|---|---|---|
| **M1 (`measures/qii.py`)** | Isotropic Gaussian rescaled to $\|\mathbf{q}_g\|_2$ (5 seeds) | Cross-image global query $j \neq i$ (5 seeds; `None` for MAP-head) | Random patch query from same image (5 seeds) | **Haar random rotation** of $\mathbf{q}_g$ inside **top-$k=32$ PCA subspace**, norm-matched (5 seeds) | PCA fit over pilot split (`N = 500` CUB / `N = 1,000` IN-1K) instead of `5,000` images. |
| **I1 (`interventions/i1_i2.py`)** | Isotropic Gaussian rescaled to $\|\mathbf{q}_g\|_2$ (5 seeds) | Empirical mean query $\boldsymbol{\mu}_h$ (deterministic, **1 seed**, image bootstrap CI) | Cross-image `[CLS]` query derangement $j \neq i$ (5 seeds) | **Top-$k=32$ (`qii.DEFAULT_PCA_K`) PCA-subspace Gaussian** $\boldsymbol{\mu}_h + \mathbf{U}_h(\boldsymbol{\sigma}_h \odot \boldsymbol{\epsilon})$, norm-matched (5 seeds) | (1) `I1-S4` uses PCA-subspace Gaussian rather than Haar rotation of $\mathbf{q}_g$; (2) `I1-S2` is empirical mean $\boldsymbol{\mu}$ and `I1-S3` is cross-image query (re-indexed vs M1); (3) PCA fit on pilot split (`N = 500/1,000`). |
| **I5 (`interventions/i5_cls_rand.py`)** | `iso_norm`: isotropic Gaussian on $\mathbf{z}_{\text{cls}}$ rescaled to $\|\mathbf{z}_{\text{cls}}\|_2$ (5 seeds) | `cross_img`: cross-image real $\mathbf{z}_{\text{cls}}$ derangement (5 seeds) | — | `cov_matched`: **top-$k=16$ PCA-subspace Gaussian** on $\mathbf{z}_{\text{cls}}$, norm-matched (5 seeds) | Hidden-state randomization entering block $L_r \in \{8, 9, 10, 11\}$ with $k=16$ PCA Gaussian. |

### 4.1 Disjoint Token-Group Definition & Frozen Read-Outs (§6–§7)
Let $M_{\text{fg}}$ denote the ground-truth foreground patch set (**pixel-segmentation mask patches** on `CUB-200-2011`, with `--cub_fg_source box` sensitivity check; **union-of-XML-boxes patches** on `ImageNet-1K`) and $\text{OUT}$ denote high-norm outlier patches ($\|\mathbf{z}_i\|_2 > \text{median} + 5\,\text{MAD}$). Disjoint groups: `OUT`, `FG` $= M_{\text{fg}} \setminus \text{OUT}$, `BG` $= M_{\text{fg}}^c \setminus \text{OUT}$, and `REG`.
- **I1:** Localization (`Δbox-AUC`, `ΔOtsu`) at **`L_int = 1`** only; function (`1 − cos`, `ΔkNN@20`, `KL`) swept across `L_int ∈ {1, 2, 3, 4}`.
- **I2:** Localization (`Δbox-AUC`) scored **strictly at `l=10`** over all $N$ images using the unablated `l=11` map (`0` ablated tokens); function (`1 − cos`, `ΔkNN@20`, `KL`) scored at `l=10` & `l=11`. Controls: 5-seed matched random patches for `FG`/`OUT`/`REG`; raw `BG` with 95% CI; plus 5-seed equal-count `BG_eq_FG` (`BG \ OUT` of exact size `m_FG`, evaluated on `n_exact` images where `|BG_i| ≥ |FG_i|`) yielding scale-invariant log ratio $R_m = \log(\text{FG} / \text{BG\_eq})$.
- **I3 (Test-Time Registers):** Sweep shifted neuron count $n$; exclude checkpoints failing $\ge 50\%$ `OUT` reduction from `C2`.
- **RALR (§7):** Unlabelled $l^*$, `QII_S4 ≥ 0.50` HIGH branch, Step-7 shortcut rule frozen at `bg_high` (`keyaff > q70`, `c > q90`; `guide_literal` as ablation), `tunes_on_boxes ∈ {NO, YES}` labelled on every metric.

---

## 5. Exclusion Rules

1. **Degenerate-Invariance Layers (Fix 2):** Layers with $H_{\text{norm}} > 0.90$ are excluded from eligible $l^*$ selection (`mae_vitb16` has 0 eligible layers and falls back to `l=11` with `degen = True`).
2. **MAE Exclusions:** `mae_vitb16` (`reconstruction`) is **excluded from H2, H2b, H5 primary sign-test denominator ($n = 11$), and all function-based tests (`1 − cos`, `ΔkNN`, `KL`)**; it is retained for Fix 2 degeneracy auditing, Fix 3 Spearman sensitivity (`n=8` vs `n=7`), and H5 sensitivity (`n=12`).
3. **I3 Verification Failure:** Checkpoints failing $\ge 50\%$ `OUT` token reduction are excluded from `M4` Condition `C2` and logged explicitly.
4. **Equal-Count I2 Contrast:** Images with $|\text{BG}_i| < |\text{FG}_i|$ or $|\text{FG}_i| = 0$ are excluded from paired `BG_eq_FG` comparisons (`n_exact / N` always reported).

---

## 6. Covariate List for §10.3 Confound Regression (Descriptive)

All covariates are recorded in `configs/checkpoints.yaml` (validated by `tests/test_registry.py`):

```text
Y ~ objective_family + log10(params) + layer_frac + patch_size
    + log10(pretrain_images_m) + log10(epochs) + has_registers + resolution
    + (1 | model_family)
```

With 8–12 models, this multi-covariate regression is underpowered for formal inference and is reported strictly as **descriptive**.

---

## 7. Frozen Split Definitions, Seeds & SHA-256 Digests (§8.4)

All 7 split files reside in `configs/splits/` (loaded verbatim, never regenerated):

| Split File | Dataset & Pool | $N$ | Sampling Rule | `sha256_16` |
|---|---|---:|---|---|
| `configs/splits/cub_pilot_500_seed0.txt` | CUB-200-2011 official test (`5,794`) | 500 | `default_rng(0).choice(test_ids, 500, replace=False)`, sorted | `59103db7759a4c3c` |
| `configs/splits/cub_eyecheck_20_seed0.txt` | Random eye-check from CUB-500 | 20 | `default_rng(0).choice(cub_500, 20, replace=False)` | `611810061fb3faa8` |
| `configs/splits/cub_gallery_train_5994.txt` | CUB-200-2011 official train (`is_train == 1`) | 5,994 | All official train IDs sorted (`train_test_split.txt`) | `0bc6a172a24a9155` |
| `configs/splits/imagenet_pilot_1000_seed0.txt` | ILSVRC2012 val (`50,000` w/ XML) | 1,000 | `default_rng(0).choice(val_stems, 1000, replace=False)`, sorted | `78ed7d48307ff451` |
| `configs/splits/imagenet_eyecheck_20_seed0.txt` | Random eye-check from IN-1K-1000 | 20 | `default_rng(0).choice(in1k_1000, 20, replace=False)` | `35e8507680fbd3e3` |
| `configs/splits/imagenet_gallery_5000_seed1.txt` | Disjoint ILSVRC2012 val (`49,000` remaining) | 5,000 | `default_rng(1).choice(disjoint_val, 5000, replace=False)`, sorted | `0b9f00b2293bdb24` |
| `configs/splits/waterbirds_m4_2000_seed0.txt` | Waterbirds official test (`split == 2`, `5,794`) | 2,000 | `default_rng(0).choice(test_rows, 2000, replace=False)`, sorted | `5daa49dceb07dc2b` |

*(Waterbirds M3 evaluates the full `5,794`-image official test split: `2,897` congruent / `2,897` incongruent.)*

---

## 8. Decision Gates & MS Thesis Scope Cuts (§11.2, §13.1)

- **Gate G1 (Week 3 — Evaluated):** **Passes via M4 (H3)** (`τ(C0, C1) ≪ τ_noise` across CUB, Waterbirds, and ImageNet-1K, confirmed against 5-seed matched random-drop controls).
- **Gate G2 (Week 8 — Framing Decision):** Determines whether the thesis frames around routing + artifact-controlled similarity + RALR (H2b/H3/H5) or solely around the H3 artifact-controlled measurement audit.
- **Gate G3 (Week 11 — RALR Main-Result Entry):** RALR enters the main results only if primary `box-AUC` (`n = 11` non-degenerate models) clears Holm on ≥2 of 3 benchmarks.
- **Pre-registered Scope Cuts (4 Dec first draft, 16 Dec final):** Cut Set E, Universal SAEs, VOC12/COCO20k, medical extension, and I4 linear stitching; retain correctness suite, Waterbirds balanced 2×2 congruence, `[CLS]@l*` + spike-clipped/`LOST`/`TokenCut` baselines, the 4 pre-registered expansion models (§1, subject to the 16 Dec fallback rule), and descriptive confound regression.
