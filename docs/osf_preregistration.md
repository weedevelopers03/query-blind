# OSF Pre-Registration: Query-Blind Localization Audit

- **Title:** Query-Blind Localization: A Training-Free Causal Audit of How Supervision Routes Object Information in Vision Foundation Models
- **Author:** Isma Anwar (`SP25-RCS-011`, MS Computer Science, COMSATS University Islamabad)
- **Supervisor:** Dr. Tehseen Zia
- **Date Frozen:** 2026-11-23
- **Code Repository:** `https://github.com/weedevelopers03/query-blind`
- **Frozen Analysis Code Git SHA:** `9cf7490` (prior to confirmatory ImageNet-1K I1/I2, I5, I3, and 12-model Sets A+B runs)

---

## 1. Timing & Exploratory vs. Confirmatory Disclosure

Per Dr. Tehseen Zia's recovery schedule, this pre-registration was scheduled for 6 November and is posted retroactively after completing the Week-3 Pilot-8 audit and exploratory CUB-500 I1/I2 runs. To preserve full scientific transparency:

- **Already observed (explicitly classified as exploratory):**
  - Pilot-8 observational measures M1 (`QII`), M2 ($\omega^2_{\text{key}}$), M3 (`Waterbirds` `BSS_norm`), M4 (`CKA`/`Procrustes`/`mKNN` `C0` vs `C1`), and `RALR` on `CUB-500` and `ImageNet-1K-1000`.
  - Pilot-8 Interventions **I1 and I2 on `CUB-500`** (v1 and v2 read-out debugging runs). **All CUB-500 I1/I2 results remain strictly exploratory** and will never be reported as confirmatory tests.
  - ImageNet-1K feature sanity check (`deit3_small_patch16_224` logits top-1 = `82.10%`, `mocov3_vitb16` IN-1K `kNN@20 = 48.90%`).
- **Unseen at registration time (confirmatory):**
  - Frozen-protocol Interventions **I1 and I2 on `ImageNet-1K-1000`** (and CUB box-mask sensitivity check).
  - **Intervention I5** (`[CLS]` randomization across Pilot-8 and Sets A+B).
  - **Intervention I3** (test-time registers) and **M4 Condition `C2`**.
  - **12-model expansion (Sets A+B)** across M1–M4, I1/I2/I5, and `RALR` (including `LOST`, `TokenCut`, and a third localization benchmark).

---

## 2. Causal vs. Observational Scope Statement (Fix 1)

All forward-pass interventions (`I1`, `I2`, `I3`, `I5`) identify **causal effects of inference-time routing paths within a frozen checkpoint**. All comparisons across pretraining objectives (`self_distillation`, `label_supervised`, `contrastive`, `image_text_contrastive`, `reconstruction`) are **controlled observational comparisons** adjusted for recorded architecture and training covariates (§7); I will not claim a causal effect of the training objective itself.

---

## 3. Primary Hypotheses, Frozen Statistics, Effect Thresholds & Holm Family

Five primary hypothesis tests form the **Holm-Bonferroni family** ($\alpha = 0.05$, two-sided unless explicitly noted). Every other test (sub-analyses, exploratory surrogates, per-layer curves, CUB I1/I2) is explicitly labelled **exploratory**.

| ID | Hypothesis | Primary Statistic & Layer Rule | Statistical Test | Minimum Meaningful Effect / Falsification Rule |
|---|---|---|---|---|
| **H1** | *(Original proposal test — retained for transparent falsification)* Self-distilled ViTs exhibit higher head-weighted `QII_S4` than label-/text-supervised ViTs at non-degenerate layers. | Head-weighted `QII_S4` at unlabelled $l^* = \arg\max_{l \in \text{last third},\, H_{\text{norm}}(l) \le 0.90} \omega^2_{\text{key}}(l)$ (fallback `l_last` if no layer is eligible). | Paired image-level bootstrap difference (`1,000` resamples) within matched Sets A–B pairs (e.g. `dino_vits16 − deit3_small_patch16_224`). | **Falsified if 95% CI overlaps 0 or `|ΔQII_S4| < 0.02`.** *(Note: Already not supported on Pilot-8 CUB `-0.001 [-0.002, +0.000]` and negligible on IN-1K `+0.005 [+0.004, +0.006]`; kept in Holm family so the architectural query-cone null is preserved without HARKing.)* |
| **H2** | Inference-time query replacement (`I1`) impacts localization and global function differently across supervision objectives. | **Localization:** single-layer `Δbox-AUC` at **`L_int = 1`** (since `L_int = 1` already replaces `q_CLS` at the `l=11` readout layer) under `S4` (primary in-cone) and `S1` (off-manifold control), with clean baseline `box_auc_clean` as a covariate. **Function:** `1 − cos(z_clean, z_int)`, `ΔkNN@20`, and `KL` swept across `L_int ∈ {1, 2, 3, 4}`. | Linear mixed-effects model: `Δbox_AUC_{m,i} ~ objective_family + box_auc_clean_{m,i} + (1 \| model_id) + (1 \| image_id)` (and functional analog at `L_int = 4`). | `|β_obj| ≥ 0.02` (`2.0 pt` `box-AUC`) or `|β_obj| ≥ 0.05` on `1 − cos` with Holm-adjusted $p < 0.05$. |
| **H2b** | Label- and image-text-supervised ViTs route more global computation through background tokens than self-distilled ViTs (`M3` Waterbirds & `I2` token-group knockout). | **M3 Waterbirds:** Area-normalised `BSS_norm` at **matched layers `l=8` and `l=11`** (no oracle layer selection), decomposed into (i) balanced 2×2 **Background Main Effect** `(Water − Land)` and (ii) balanced 2×2 **Congruence Interaction** `½(L/L + W/W) − ½(L/W + W/L)`, stratified by object-area terciles.<br>**I2 Knockout:** Equal-count paired contrast **`FG − BG_eq_FG`** (`1 − cos`, `ΔkNN@20`, `KL` at `l=10` & `l=11`; unablated `Δbox-AUC` strictly at **`l=10`**). | Image-cluster bootstrap 95% CIs (`1,000` resamples) on paired differences across matched Sets A–B checkpoints; tercile-stratified consistency check. | Non-overlapping 95% CIs across matched pairs in ≥2 of 3 area terciles (`M3`), or `|Δ(FG − BG_eq_FG)| ≥ 0.02` on `1 − cos` (`I2`). |
| **H3** | High-norm outlier tokens (`OUT`, $\|\mathbf{z}_i\|_2 > \text{median} + 5\,\text{MAD}$) distort inter-model representation similarity rankings (`M4`), and masking (`C1`) or test-time registers (`C2`) shifts Kendall's $\tau$ well below the split-half noise floor. | `Δτ = τ(C0, C1) − τ_noise` (and `τ(C0, C2) − τ_noise` when I3 passes verification) on `R2` (mean-pooled) and `R3` (patch-level) debiased linear CKA at **matched layers `l ∈ {8, 9, 10, 11}`**, plus 5-seed matched random-drop control `τ(C0, C1_rand)`. | Paired bootstrap across 40 split-half resamples (`same-half` and `cross-half` 95% CIs strictly `< 0`), plus `τ(C0, C1) < τ(C0, C1_rand)`. | Both same-half and cross-half 95% CIs of `τ(C0, C1) − τ_noise` strictly below `0`, with `τ(C0, C1_rand) − τ(C0, C1) ≥ 0.10`. |
| **H5** | Training-free, zero-box-tuning `RALR` improves object localization over layer-matched `[CLS]@l*` across diverse ViT backbones. | Primary: per-backbone paired `Δbox-AUC = box_AUC(RALR) − box_AUC([CLS]@l*)` (unlabelled $l^*$ selection, zero box tuning). Secondary: zero-tuning `Otsu CorLoc@0.5` vs `[CLS]@l*`, `cls_l_star_clip98` (top-2% winsorized), `cls_l_star_no_out` (OUT-masked), `LOST`, and `TokenCut`. | Paired image-level bootstrap per backbone + **two-sided exact binomial sign test** across backbones; Gate G3 requires majority wins on ≥2 of 3 benchmarks. | Majority of non-degenerate backbones improve on `box-AUC` (and secondary `Otsu CorLoc@0.5`) on ≥2/3 benchmarks with Holm-adjusted two-sided sign-test $p < 0.05$. |

*Secondary hypothesis **H4 (Registers):** Evaluated on matched register pairs (`dinov2_vitb14` vs `dinov2_vitb14_reg` and Set B extensions): (i) `QII_S4` survives registers (`|ΔQII_S4| ≤ 0.05`), (ii) `share_OUT` decreases by ≥50% as mass shifts to `REG`, and (iii) under `I2`, knocking out `REG` shifts `[CLS]` (`1 − cos`) while leaving `l=10` localization `Δbox-AUC` within `±0.01`.*

---

## 4. Frozen Intervention Protocols, Surrogates & Read-Outs (§6)

### 4.1 Surrogate Query Families (`M1` & `I1`)
All surrogates operate on the `[CLS]` query row only while keeping patch queries, keys, values, and patch stream intact at the intervened layer:
- **`S1` (Isotropic off-manifold control):** Isotropic Gaussian $\mathcal{N}(0, I_{d_h})$ rescaled to each image's exact per-head query norm $\|\mathbf{q}_{g,h}\|_2$. **5 seeds (`0..4`)**; seed SD reported separately from image bootstrap CI.
- **`S2` (Empirical mean query):** Dataset mean query $\boldsymbol{\mu}_h$ broadcast to every image. **Deterministic (`1` seed, `seed_sd = 0.0000`)**; uncertainty quantified solely via image-level bootstrap CI.
- **`S3` (Cross-image `[CLS]` derangement):** True `[CLS]` query from a random different image ($j \neq i$). **5 seeds (`0..4`)**.
- **`S4` (PCA-subspace Gaussian, primary in-cone surrogate):** Sampled from the empirical top-$k$ ($k=16$) query PCA subspace $\boldsymbol{\mu}_h + \mathbf{U}_h \mathbf{z}$ and norm-matched to $\|\mathbf{q}_{g,h}\|_2$. **5 seeds (`0..4`)**.

### 4.2 Disjoint Token-Group Definition & Cross-Dataset Comparability (`M3` & `I2`)
For every dataset, let $M_{\text{fg}}$ denote the ground-truth foreground patch set (**pixel-segmentation mask patches** on `CUB-200-2011`, with a `--cub_fg_source box` bounding-box sensitivity check reported alongside; **union-of-bounding-box patches** on `ImageNet-1K`), and let $\text{OUT}$ denote high-norm outlier patches ($\|\mathbf{z}_i\|_2 > \text{median} + 5\,\text{MAD}$ at the intervened layer). All tokens partition disjointly into:
- `OUT`: outlier patch tokens (`OUT` takes precedence over `FG`/`BG`),
- `FG` $= M_{\text{fg}} \setminus \text{OUT}$,
- `BG` $= M_{\text{fg}}^c \setminus \text{OUT}$,
- `REG`: register tokens (empty if model has no registers).

### 4.3 Frozen Primary Read-Out Per Intervention
- **I1 (Query Replacement):**
  - Localization (`Δbox-AUC`, `ΔOtsu`): evaluated at **`L_int = 1`** only (replacing `q_CLS` at `l=11` already determines the `l=11` attention map; multi-layer edits `L_int = 2..4` do not alter `l=11` map beyond `L_int = 1`).
  - Function (`1 − cos`, `ΔkNN@20` against external gallery, classifier `KL` + top-1 agreement): swept across **`L_int ∈ {1, 2, 3, 4}`**.
- **I2 (Token-Group Knockout — `zero` & `mean_v` ablation):**
  - **Localization (`Δbox-AUC`):** scored **strictly at penultimate layer `l=10`** over all $N$ images using the downstream `l=11` attention map (which has **0 ablated tokens**, eliminating both the `l=11` selected-subset artifact where `FG` knockout removes all non-OUT positives and the structural `0.0000` of `mean_v` at `l=11`).
  - **Function (`1 − cos`, `ΔkNN@20` on external gallery, `KL`):** scored at **both `l=10` and `l=11`**.
  - **Controls:**
    1. `FG`, `OUT`, `REG`: 5-seed matched-count random patch control (`all_patches`, never selecting `[CLS]` or `REG`).
    2. `BG`: **raw BG effects with 95% CIs** (no confounded `non_fg` random subtraction).
    3. **Clean equal-count `FG vs BG_eq_FG` contrast:** 5 random subsets of `BG` ($= M_{\text{fg}}^c \setminus \text{OUT}$, zero FG or OUT leakage) of exact size $m_{\text{FG},i} = |\text{FG}_i|$, reporting paired `FG − BG_eq_FG` (95% CI) next to the image-bootstrapped per-token `FG:BG` ratio.
- **I5 (`[CLS]` Randomization):** Norm-matched randomization of `[CLS]` at layer $l$, testing whether unlabelled `QII_S4` and $\omega^2_{\text{key}}$ predict map survival (`box-AUC` retention).
- **I3 (Test-Time Registers, Jiang et al.):** Sweep high-norm neuron shift count $n$; accept into `C2` only if `OUT` count drops by $\ge 50\%$.

### 4.4 Frozen RALR Specification (§7)
- Unlabelled layer selection: $l^* = \arg\max_{l \in \text{last third},\, H_{\text{norm}}(l) \le 0.90} \omega^2_{\text{key}}(l)$.
- Branch threshold: `QII_S4(l*) ≥ 0.50` → `HIGH` (query-free key-effect map $\bar{\beta}_i$); `< 0.50` → `LOW`.
- Step-7 shortcut rule frozen at `bg_high` (`keyaff > q70` and `c > q90`, since the largest key cluster is background on 8/8 pilot models); the literal `keyaff < q30` rule is retained as a documented ablation.
- Every evaluation table explicitly flags `tunes_on_boxes ∈ {NO, YES}` (`MaxBoxAccV2` = `YES`; `box-AUC`, `PxAP`, `Pointing`, `Otsu CorLoc`, `Cal-τ* CorLoc` = `NO`).

---

## 5. Exclusion Rules

1. **Degenerate-Invariance Layers (Fix 2):** Layers with normalized attention entropy $H_{\text{norm}} > 0.90$ are flagged `degenerate_invariance` and excluded from eligible $l^*$ selection. If a model has zero eligible layers (e.g. `mae_vitb16`), it falls back to `l_last` with `degen = True` and is reported both included (`n=8`) and excluded (`n=7`) in cross-model correlations (Fix 3 Spearman).
2. **MAE Function Exclusion:** `mae_vitb16` (pure pixel-reconstruction pretraining without supervised fine-tuning; CUB `[CLS]` `kNN@20 = 0.058`, patch-mean `kNN@20 = 0.028`, patch-mean `1−cos ≈ 0` under `[CLS]`-row edits) is **excluded from all function-based hypothesis tests (`1−cos`, `ΔkNN`, `KL`)** and retained solely for localization (`box-AUC`) and Fix 2 degeneracy auditing.
3. **I3 Verification Failure:** Any checkpoint for which test-time registers fail to reduce `OUT` token count by $\ge 50\%$ is excluded from `M4` Condition `C2` and recorded in an explicit failure log.
4. **Equal-Count I2 Contrast (`FG vs BG_eq_FG`):** Images where $|\text{BG}_i| < |\text{FG}_i|$ or $|\text{FG}_i| = 0$ are excluded from the paired exact-count contrast; `n_exact / N` is reported in every table row.

---

## 6. Covariate List for §10.3 Confound Regression

All checkpoint covariates are frozen in `configs/checkpoints.yaml` (verified via `tests/test_registry.py`):

```text
Y ~ objective_family + log10(params) + layer_frac + patch_size
    + log10(pretrain_images_m) + log10(epochs) + has_registers + resolution
    + (1 | model_family)
```

*Note:* With 8–12 checkpoints (Sets A+B MVP), this multi-covariate regression is underpowered for formal inference and will be explicitly labelled **descriptive / exploratory** alongside matched-pair contrasts.

---

## 7. Frozen Split Definitions, Seeds & SHA-256 Digests (§8.4)

All pilot and eye-check splits are committed under `configs/splits/` (commit `9cf7490`) and loaded verbatim (`seed = 0`, never regenerated):

| Split File | Dataset & Pool | $N$ | Sampling Rule | `sha256_16` |
|---|---|---:|---|---|
| `configs/splits/cub_pilot_500_seed0.txt` | CUB-200-2011 official test (`5,794` imgs) | 500 | `default_rng(0).choice(..., replace=False)`, sorted | `59103db7759a4c3c` |
| `configs/splits/cub_eyecheck_20_seed0.txt` | Random eye-check from CUB-500 | 20 | `default_rng(0).choice(cub_500, 20, replace=False)` | `611810061fb3faa8` |
| `configs/splits/imagenet_pilot_1000_seed0.txt` | ILSVRC2012 val (`50,000` imgs w/ XML) | 1,000 | `default_rng(0).choice(..., replace=False)`, sorted | `78ed7d48307ff451` |
| `configs/splits/imagenet_eyecheck_20_seed0.txt` | Random eye-check from IN-1K-1000 | 20 | `default_rng(0).choice(in1k_1000, 20, replace=False)` | `35e8507680fbd3e3` |
| Waterbirds test | Official Waterbirds test (`2,897` cong / `2,897` incong) | 5,794 | Full test set (M3); `2,000` seed-0 subset (M4) | — |
| kNN Galleries | CUB official train (`5,994` imgs); IN-1K disjoint val (`5,000` imgs, `seed=1`) | 5,994 / 5,000 | Disjoint from all pilot queries | — |

---

## 8. Decision Gates & MVP Scope Cuts (§11.2, §13.1)

- **Gate G1 (Week 3 — Evaluated):** **Passes via M4 (H3)** (`τ(C0, C1) ≪ τ_noise` across CUB, Waterbirds, and ImageNet-1K, confirmed against 5-seed matched random-drop controls). Criterion 1 (`QII_S4` gap DINO vs DeiT-III) failed on CUB (`-0.001`) and was negligible on ImageNet (`+0.005` on a `0.97–1.00` ceiling).
- **Gate G2 (Week 8 — Venue Framing):** If confirmatory ImageNet-1K I1/I2 + Sets A+B support H2b/H3 + RALR, target ICCV/thesis full audit; if H2 remains weak outside M3 Waterbirds, frame around the **H3 measurement audit + routing decomposition + RALR** (ICML measurement track + arXiv).
- **Gate G3 (Week 11 — RALR Main-Paper Entry):** RALR enters the main paper only if it beats `[CLS]@l*` on ≥2/3 benchmarks for a majority of backbones (reporting `box-AUC` primary and `Otsu CorLoc@0.5` secondary honestly).
- **Pre-registered Scope Cuts for MS Thesis Deadline (4 Dec first draft, 16 Dec final):** Drop Set E, Universal SAEs, VOC12/COCO20k, medical extension, and I4 stitching; retain correctness suite, Waterbirds balanced 2×2 congruence, `[CLS]@l*` + spike-clipped controls, matched-pair Sets A+B (12 models), and descriptive confound regression.
