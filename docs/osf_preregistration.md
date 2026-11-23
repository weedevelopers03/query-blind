# OSF Pre-Registration: Query-Blind Localization Audit

- **Title:** Query-Blind Localization: A Training-Free Causal Audit of How Supervision Routes Object Information in Vision Foundation Models
- **Author:** Isma Anwar (`SP25-RCS-011`, MS Computer Science, COMSATS University Islamabad)
- **Supervisor:** Dr. Tehseen Zia
- **Date Frozen:** 2026-11-23
- **Code Repository:** `https://github.com/weedevelopers03/query-blind`
- **Relevant Code Commits:**
  - `9cf7490` — committed frozen seed-0 pilot & eye-check split files (`configs/splits/`).
  - `a0cfbe5` — frozen I1/I2 & I5 protocol code that produced the ImageNet-1K I1/I2 and ImageNet-1K/CUB I5 model runs.
  - `e28e35b` — I5 prediction-table schema parser fix (`layer_summaries`) used to render I5 summaries via `--resume` (zero model recomputation).
  - Pre-registration freeze commit: stamped in `GIT_SHA` on push immediately prior to OSF upload.

---

## 1. Timing & Exploratory vs. Confirmatory Disclosure

This pre-registration is posted retroactively relative to Dr. Tehseen Zia's 6 November schedule. To prevent any ambiguity about what was observed before OSF posting:

- **Observed prior to OSF posting (NOT pre-registered confirmatory):**
  1. Pilot-8 observational measures **M1 (`QII`)**, **M2 ($\omega^2_{\text{key}}$)**, **M3 (`Waterbirds` `BSS_norm`)**, **M4 (`C0` vs `C1` + `C1_rand`)**, and **`RALR`** on `CUB-500` and `ImageNet-1K-1000`.
  2. Pilot-8 Interventions **I1 and I2 on `CUB-500`** (exploratory read-out debugging runs).
  3. Pilot-8 **ImageNet-1K I1/I2** and **ImageNet-1K / CUB-500 I5** — **run under the committed protocol (`a0cfbe5`, summary parser `e28e35b`) before OSF posting, results seen.** Because these results were inspected prior to OSF registration, they are classified as *post-hoc / run under committed protocol prior to posting*, **not** pre-registered confirmatory tests.
  4. ImageNet-1K feature sanity check (`deit3_small_patch16_224` top-1 = `82.10%`, `mocov3_vitb16` `kNN@20 = 48.90%`).
- **Unseen at OSF posting (genuinely pre-registered confirmatory):**
  1. **12-model Sets A+B expansion** (adding 4 unseen matched checkpoints beyond Pilot-8 across M1–M4, I1/I2, and `RALR`).
  2. **Intervention I3** (test-time registers) and **M4 Condition `C2`** (artifact-controlled similarity after I3).
  3. **RALR external baselines (`LOST` and `TokenCut`)** under the identical zero-box-tuning protocol, plus a **third localization benchmark (`OpenImages30k`)** for Gate G3.
- **Deadline Fallback Rule (16 December 2026):** If the 12-model Sets A+B expansion is not completed by 16 December 2026, the MS thesis will report **Pilot-8 only, with no confirmatory pre-registered claims** (all Pilot-8 analyses explicitly labelled exploratory / protocol-frozen post-hoc).

---

## 2. Causal vs. Observational Scope Statement (Fix 1)

All forward-pass interventions (`I1`, `I2`, `I3`, `I5`) identify **causal effects of inference-time routing paths within a frozen checkpoint**. All comparisons across pretraining objectives (`self_distillation`, `label_supervised`, `contrastive`, `image_text_contrastive`, `reconstruction`) are **controlled observational comparisons** adjusted for recorded architecture and training covariates (§6); I do not claim a causal effect of the training objective itself.

---

## 3. Primary Hypotheses, Frozen Statistics, Effect Thresholds & Holm Family

Five primary hypothesis tests form the **Holm-Bonferroni family** ($\alpha = 0.05$, two-sided). Every other test — including **Intervention I5** (which is outside the Holm family), sub-contrasts, per-layer curves, and secondary metrics — is explicitly labelled **exploratory**.

| ID | Hypothesis | Primary Statistic & Layer Rule | Statistical Test & Named Contrast | Minimum Effect / Decision Rule |
|---|---|---|---|---|
| **H1** | *(Original proposal test — retained for transparent falsification)* Self-distilled ViTs exhibit higher head-weighted `QII_S4` than label-/text-supervised ViTs at non-degenerate layers. | Head-weighted `QII_S4` at unlabelled $l^* = \arg\max_{l \in \text{last third},\, H_{\text{norm}}(l) \le 0.90} \omega^2_{\text{key}}(l)$ (fallback `l_last` if no layer is eligible). | Paired image-level bootstrap difference (`1,000` resamples) within matched Sets A–B pairs (`self_distillation − label_supervised`, e.g. `dino_vits16 − deit3_small_patch16_224`). | **Falsified if 95% CI overlaps 0 or `\|ΔQII_S4\| < 0.02`.** *(Already not supported on Pilot-8 CUB `-0.001` and negligible on IN-1K `+0.005`; kept in the Holm family so falsification is preserved.)* |
| **H2** | Inference-time query replacement (`I1`) impacts localization (`Δbox-AUC`) differently for SSL vs. label+text supervision. | Single-layer `Δbox-AUC` at **`L_int = 1`** under `I1-S4` (primary in-cone; `I1-S1` off-manifold control reported alongside), with clean `box_auc_clean` as a baseline covariate. Functional sweep `L_int ∈ {1, 2, 3, 4}` (`1 − cos`, `ΔkNN@20`, `KL`) is secondary. | **Named primary contrast:** binary `group ∈ {SSL (self_distillation + contrastive), label+text (label_supervised + image_text_contrastive)}` (MAE excluded from function tests). Linear mixed model: `Δbox_AUC_{m,i} ~ I(label+text) + box_auc_clean_{m,i} + (1 \| model_id) + (1 \| image_id)`.<br>**Identifiability caveat:** Whenever a family contains only **1 checkpoint** (as in Pilot-8 for `label_supervised` and `image_text_contrastive`), the family fixed effect and `(1 \| model_id)` random intercept are **not separable**; mixed-effects inference requires ≥2 checkpoints per group (Sets A+B 12-model run), and single-model families are reported only as descriptive paired contrasts. | `\|β_{label+text vs SSL}\| ≥ 0.02` (`2.0 pt` `box-AUC`) with Holm-adjusted two-sided $p < 0.05$. *(Disclosure: the `0.02` threshold sits just below `deit3_small_patch16_224`'s observed exploratory CUB `S4` shift `+0.023` and was chosen after seeing that CUB result.)* |
| **H2b** | Label- and image-text-supervised ViTs route more global computation through background tokens (and show weaker foreground-over-background selectivity) than SSL ViTs. | **Predicted direction:** (i) M3 Waterbirds balanced 2×2 Background Main Effect `(Water − Land)` at **matched layers `l=8` and `l=11`** is **larger** for `label+text` than for size-matched `SSL`; and (ii) I2 equal-count foreground selectivity `S_m = (1 − cos)_{FG} − (1 − cos)_{BG_eq_FG}` (at `l=10` & `l=11`) is **smaller** for `label+text` than for `SSL` (i.e. `mean_SSL(S_m) − mean_{label+text}(S_m) > 0`). | **Across-family contrast:** M3 compares `label+text` against the **size-matched SSL range** (stratified by object-area terciles, `1,000` image-cluster bootstrap resamples). I2 computes `S_m = FG − BG_eq_FG` per checkpoint (`5` random subsets of `BG \ OUT` of exact size `m_FG`), then compares `S_m` **between objective families** (`SSL` vs `label+text`). *(Note: `FG ≫ BG_eq` within each individual model does NOT test H2b; only the between-family contrast tests H2b.)* | M3: `label+text` exceeds the full size-matched SSL range with non-overlapping 95% CIs in ≥2 of 3 area terciles. I2: between-family contrast `mean_SSL(S_m) − mean_{label+text}(S_m) > 0` with Holm-adjusted $p < 0.05$ (requiring ≥2 models/family in the 12-model run). |
| **H3** | High-norm outlier tokens (`OUT`, $\|\mathbf{z}_i\|_2 > \text{median} + 5\,\text{MAD}$) distort inter-model representation similarity rankings (`M4`), and masking (`C1`) or test-time registers (`C2`) lowers Kendall's $\tau$ well below the split-half noise floor. | `Δτ = τ(C0, C1) − τ_noise` (and `τ(C0, C2) − τ_noise` for models passing I3 verification) on `R2` and `R3` debiased linear CKA at **matched layers `l ∈ {8, 9, 10, 11}`**, alongside 5-seed matched random-drop `τ(C0, C1_rand)`. | Paired bootstrap across 40 split-half resamples (`same-half` and `cross-half` 95% CIs strictly `< 0`), plus `τ(C0, C1) < τ(C0, C1_rand)`. | Both same-half and cross-half 95% CIs of `τ(C0, C1) − τ_noise` (and `C2` when available) strictly `< 0`, with `τ(C0, C1_rand) − τ(C0, C1) ≥ 0.10`. |
| **H5** | Training-free, zero-box-tuning `RALR` improves object localization over layer-matched `[CLS]@l*` across ViT backbones. | **Primary read-out (required for H5 to pass):** per-backbone paired `Δbox-AUC = box_AUC(RALR) − box_AUC([CLS]@l*)` (unlabelled $l^*$, zero box tuning). **Secondary read-out (cannot rescue a failed `box-AUC` test):** zero-tuning `Otsu CorLoc@0.5` vs `[CLS]@l*`, `cls_l_star_clip98`, `cls_l_star_no_out`, `LOST`, and `TokenCut`. | **Two-sided exact binomial sign test** across backbones on primary `box-AUC` (Gate G3 requires majority wins on ≥2 of 3 benchmarks).<br>**Exact counts needed to clear Holm Step 1 ($\alpha/5 = 0.01$, two-sided):** at **$n = 12$ models**, **`≥ 11/12`** wins ($p = 0.0063$; $10/12$ yields $p = 0.0386$); at **$n = 8$ models**, **`8/8`** wins ($p = 0.0078$; $7/8$ yields $p = 0.0703$). | **Primary `box-AUC` must clear Holm-adjusted two-sided sign-test $p < 0.05$ on ≥2 of 3 benchmarks for H5 to pass.** |

*Secondary / Exploratory items outside the 5-test Holm family:*
- **H4 (Registers):** On matched register pairs (`dinov2_vitb14` vs `dinov2_vitb14_reg` and Set B extensions), `QII_S4` survives (`|ΔQII_S4| ≤ 0.05`), `share_OUT` drops ≥50% into `REG`, and `REG` knockout moves `[CLS]` (`1 − cos`) without moving `l=10` `box-AUC` beyond `±0.01`.
- **I5 (`[CLS]` Randomization Prediction Test):** Explicitly **exploratory** (outside Holm family; reported with Spearman $\rho$ + 95% bootstrap CI and Pearson $r$ + 95% CI across models, noting multiple `L_r × mode` unadjusted looks and narrow `QII_S4` range at `n = 8`).

---

## 4. Surrogate Families & Documented Deviations from Guide §2.2

Guide §2.2 specifies four query surrogates (`S1` isotropic Gaussian, `S2` cross-image global query, `S3` same-image random patch query, `S4` random Haar rotation of $\mathbf{q}_g$ inside the top-$k=32$ PCA subspace fit on `5,000` images; 5 seeds `0..4`). Our codebase implements the following exact specifications (deviations from Guide §2.2 stated explicitly):

| Module | `S1` | `S2` | `S3` | `S4` | Deviations from Guide §2.2 |
|---|---|---|---|---|---|
| **M1 (`measures/qii.py`)** | Isotropic Gaussian rescaled to $\|\mathbf{q}_g\|_2$ (5 seeds) | Cross-image global query $j \neq i$ (5 seeds; `None` for MAP-head) | Random patch query from same image (5 seeds) | **Haar random rotation** of $\mathbf{q}_g$ inside **top-$k=32$ PCA subspace**, norm-matched (5 seeds) | PCA fit over pilot split (`N = 500` CUB / `N = 1,000` IN-1K) instead of `5,000` images. |
| **I1 (`interventions/i1_i2.py`)** | Isotropic Gaussian rescaled to $\|\mathbf{q}_g\|_2$ (5 seeds) | Empirical mean query $\boldsymbol{\mu}_h$ (deterministic, **1 seed**, image bootstrap CI) | Cross-image `[CLS]` query derangement $j \neq i$ (5 seeds) | **Top-$k=32$ (`qii.DEFAULT_PCA_K`) PCA-subspace Gaussian** $\boldsymbol{\mu}_h + \mathbf{U}_h(\boldsymbol{\sigma}_h \odot \boldsymbol{\epsilon})$, norm-matched (5 seeds) | (1) `I1-S4` uses PCA-subspace Gaussian sampling rather than Haar rotation of $\mathbf{q}_g$; (2) `I1-S2` is empirical mean $\boldsymbol{\mu}$ and `I1-S3` is cross-image query (re-indexed vs M1 `S2`/`S3`); (3) PCA fit on pilot split (`N = 500/1,000`). |
| **I5 (`interventions/i5_cls_rand.py`)** | `iso_norm`: isotropic Gaussian on `[CLS]` hidden state $\mathbf{z}_{\text{cls}}$ rescaled to $\|\mathbf{z}_{\text{cls}}\|_2$ (5 seeds) | `cross_img`: cross-image real $\mathbf{z}_{\text{cls}}$ derangement (5 seeds) | — | `cov_matched`: **top-$k=16$ PCA-subspace Gaussian** on $\mathbf{z}_{\text{cls}}$, norm-matched (5 seeds) | Operates on hidden state $\mathbf{z}_{\text{cls}}$ entering block $L_r \in \{8, 9, 10, 11\}$ using $k=16$ PCA Gaussian. |

### 4.1 Disjoint Token-Group Definition & Frozen Intervention Read-Outs (§6)
For every dataset, let $M_{\text{fg}}$ denote the ground-truth foreground patch set (**pixel-segmentation mask patches** on `CUB-200-2011`, with `--cub_fg_source box` reported as a cross-dataset sensitivity check; **union-of-XML-bounding-box patches** on `ImageNet-1K`), and let $\text{OUT}$ denote high-norm outlier patches ($\|\mathbf{z}_i\|_2 > \text{median} + 5\,\text{MAD}$ at the intervened layer). All tokens partition disjointly into `OUT`, `FG` $= M_{\text{fg}} \setminus \text{OUT}$, `BG` $= M_{\text{fg}}^c \setminus \text{OUT}$, and `REG`.
- **I1:** Localization (`Δbox-AUC`, `ΔOtsu`) evaluated at **`L_int = 1`** only; functional read-outs (`1 − cos`, `ΔkNN@20`, `KL`) swept across `L_int ∈ {1, 2, 3, 4}`.
- **I2:** Localization (`Δbox-AUC`) scored **strictly at `l=10`** over all $N$ images using the unablated `l=11` attention map (`0` ablated tokens at `l=11`); functional read-outs (`1 − cos`, `ΔkNN@20`, `KL`) scored at `l=10` & `l=11`. Controls: 5-seed matched random patches (`all_patches`) for `FG`/`OUT`/`REG`; **raw BG effects with 95% CIs** for `BG`; plus **5-seed equal-count `FG vs BG_eq_FG` contrast** (subsets of `BG = M_fg^c \ OUT` of exact size `m_FG`, evaluated on `n_exact` images where `|BG_i| ≥ |FG_i|`) alongside image-bootstrapped per-token `FG:BG` ratio.
- **I3 (Test-Time Registers):** Sweep shifted high-norm neurons $n$; exclude checkpoints failing $\ge 50\%$ `OUT` reduction from `C2`.
- **RALR (§7):** Unlabelled $l^*$ selection, `QII_S4 ≥ 0.50` HIGH branch, Step-7 shortcut rule frozen at `bg_high` (`keyaff > q70`, `c > q90`; `guide_literal` retained as ablation), every metric explicitly flagged `tunes_on_boxes ∈ {NO, YES}`.

---

## 5. Exclusion Rules

1. **Degenerate-Invariance Layers (Fix 2):** Layers with $H_{\text{norm}} > 0.90$ are flagged `degenerate_invariance` and excluded from eligible $l^*$ selection (`mae_vitb16` falls back to `l=11` with `degen = True` and is reported both included and excluded in Fix 3 Spearman).
2. **MAE Function Exclusion:** `mae_vitb16` is **excluded from all function-based tests (`1 − cos`, `ΔkNN`, `KL`)** and retained solely for localization (`box-AUC`) and Fix 2 degeneracy auditing.
3. **I3 Verification Failure:** Checkpoints where I3 fails to cut `OUT` count by $\ge 50\%$ are excluded from `M4` Condition `C2` and logged explicitly.
4. **Equal-Count I2 Contrast:** Images where $|\text{BG}_i| < |\text{FG}_i|$ or $|\text{FG}_i| = 0$ are excluded from paired exact-count `FG − BG_eq_FG` comparisons (`n_exact / N` always reported).

---

## 6. Covariate List for §10.3 Confound Regression

All checkpoint covariates are stored in `configs/checkpoints.yaml` (validated by `tests/test_registry.py`):

```text
Y ~ objective_family + log10(params) + layer_frac + patch_size
    + log10(pretrain_images_m) + log10(epochs) + has_registers + resolution
    + (1 | model_family)
```

With 8–12 checkpoints, this multi-covariate regression is underpowered for formal inference and is reported strictly as **descriptive**.

---

## 7. Frozen Split Definitions, Seeds & SHA-256 Digests (§8.4)

All committed split files reside in `configs/splits/` (loaded verbatim, never regenerated):

| Split File | Dataset & Pool | $N$ | Sampling Rule | `sha256_16` |
|---|---|---:|---|---|
| `configs/splits/cub_pilot_500_seed0.txt` | CUB-200-2011 official test (`5,794`) | 500 | `default_rng(0).choice(test_ids, 500, replace=False)`, sorted | `59103db7759a4c3c` |
| `configs/splits/cub_eyecheck_20_seed0.txt` | Random eye-check from CUB-500 | 20 | `default_rng(0).choice(cub_500, 20, replace=False)` | `611810061fb3faa8` |
| `configs/splits/imagenet_pilot_1000_seed0.txt` | ILSVRC2012 val (`50,000` w/ XML) | 1,000 | `default_rng(0).choice(val_stems, 1000, replace=False)`, sorted | `78ed7d48307ff451` |
| `configs/splits/imagenet_eyecheck_20_seed0.txt` | Random eye-check from IN-1K-1000 | 20 | `default_rng(0).choice(in1k_1000, 20, replace=False)` | `35e8507680fbd3e3` |
| `configs/splits/imagenet_gallery_5000_seed1.txt` | Disjoint ILSVRC2012 val (`49,000` remaining) | 5,000 | `default_rng(1).choice(disjoint_val, 5000, replace=False)`, sorted | `0b9f00b2293bdb24` |
| `configs/splits/cub_gallery_train_5994.txt` | CUB-200-2011 official train (`is_train == 1`) | 5,994 | All official train IDs sorted (`train_test_split.txt`) | *[export from Kaggle before posting]* |
| `configs/splits/waterbirds_m4_2000_seed0.txt` | Waterbirds official test (`split == 2`, `5,794`) | 2,000 | `default_rng(0).choice(test_rows, 2000, replace=False)`, sorted | *[export from Kaggle before posting]* |

*(Waterbirds M3 evaluates the full `5,794`-image official test split: `2,897` congruent / `2,897` incongruent.)*

---

## 8. Decision Gates & MS Thesis Scope Cuts (§11.2, §13.1)

- **Gate G1 (Week 3 — Evaluated):** **Passes via M4 (H3)** (`τ(C0, C1) ≪ τ_noise` across CUB, Waterbirds, and ImageNet-1K, confirmed against 5-seed matched random-drop controls). Criterion 1 (`QII_S4` gap DINO vs DeiT-III) failed on CUB (`-0.001`) and was negligible on IN-1K (`+0.005` on a `0.97–1.00` ceiling).
- **Gate G2 (Week 8 — Framing Decision):** Determines whether the thesis frames primarily around routing + artifact-controlled similarity + RALR (H2b/H3/H5) or solely around the H3 artifact-controlled measurement audit.
- **Gate G3 (Week 11 — RALR Main-Result Entry):** RALR enters the main results only if primary `box-AUC` beats `[CLS]@l*` on ≥2 of 3 benchmarks for a majority of backbones (with two-sided sign test clearing Holm).
- **Pre-registered Scope Cuts for MS Thesis Deadline (4 Dec first draft, 16 Dec final):** Cut Set E, Universal SAEs, VOC12/COCO20k, medical extension, and I4 linear stitching; retain correctness suite, Waterbirds balanced 2×2 congruence, `[CLS]@l*` + spike-clipped/LOST/TokenCut baselines, Sets A+B expansion (12 models, subject to the 16 Dec fallback rule in §1), and descriptive confound regression.
