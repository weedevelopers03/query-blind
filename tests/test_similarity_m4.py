"""Unit, seeded-bug, and end-to-end smoke tests for Measure M4 (Guide §5 & §12)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import kendalltau

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from measures import similarity as S
from scripts import run_pilot_similarity_m4 as runner_m4


def _biased_linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Deliberately biased (uncorrected) linear CKA for testing against debiased CKA."""
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    K = Xc @ Xc.T
    L = Yc @ Yc.T
    num = np.sum(K * L)
    den = np.sqrt(np.sum(K * K) * np.sum(L * L))
    return float(num / den)


def test_debiased_cka_invariance_and_null_vs_biased():
    rng = np.random.default_rng(0)
    N, D = 64, 256
    X = rng.standard_normal((N, D))
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    Y_rot = 7.5 * (X @ Q) + rng.standard_normal((1, D)) * 50.0

    # Identity + orthogonal/scale/translation invariance
    assert abs(S.debiased_linear_cka(X, X, batch_size=512) - 1.0) < 1e-10
    assert abs(S.debiased_linear_cka(X, Y_rot, batch_size=512) - 1.0) < 1e-10

    # High-dim independent Gaussian features: biased CKA is heavily inflated; debiased CKA is near 0
    Y_ind = rng.standard_normal((N, D))
    biased = _biased_linear_cka(X, Y_ind)
    debiased = S.debiased_linear_cka(X, Y_ind, batch_size=512)
    assert biased > 0.65, f"Expected biased CKA inflation for D>>N, got {biased}"
    assert abs(debiased) < 0.12, f"Expected debiased CKA near 0, got {debiased}"


def test_seeded_bug_per_batch_division_bias():
    """Guide §5.2 & §12: averaging HSIC before dividing vs the buggy divide-per-batch-then-average.

    With heavy-tailed / heterogeneous sample norms (exactly like ViT token representations with outliers),
    E[HSIC(K,L) / sqrt(HSIC(K,K)*HSIC(L,L))] != E[HSIC(K,L)] / sqrt(E[HSIC(K,K)]*E[HSIC(L,L)]).
    """
    rng = np.random.default_rng(7)
    N, D = 240, 64
    # 90% low-norm uncorrelated samples + 10% high-norm strongly correlated samples (heterogeneous batch variance)
    X = rng.standard_normal((N, D))
    Y = rng.standard_normal((N, D))
    hi = np.arange(0, N, 10)  # 24 high-norm correlated samples
    shared = 12.0 * rng.standard_normal((len(hi), D))
    X[hi] = shared + 0.5 * rng.standard_normal((len(hi), D))
    Y[hi] = shared + 0.5 * rng.standard_normal((len(hi), D))

    exact_full = S.debiased_linear_cka(X, Y, batch_size=1024)
    minibatch_ok = S.debiased_linear_cka(X, Y, batch_size=12, min_batches=300, seed=1)

    # Seeded bug: divide per minibatch and average ratios
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    bug_ratios = []
    rng_b = np.random.default_rng(1)
    for _ in range(15):
        perm = rng_b.permutation(N)
        for s in range(0, N - 12 + 1, 12):
            idx = perm[s : s + 12]
            K = torch.as_tensor(Xc[idx] @ Xc[idx].T)
            L = torch.as_tensor(Yc[idx] @ Yc[idx].T)
            num = float(S.hsic_unbiased_gram_torch(K, L))
            dx = float(S.hsic_unbiased_gram_torch(K, K))
            dy = float(S.hsic_unbiased_gram_torch(L, L))
            if dx > 0 and dy > 0:
                bug_ratios.append(num / np.sqrt(dx * dy))
    buggy_mean = float(np.mean(bug_ratios))

    assert abs(minibatch_ok - exact_full) < 0.05, f"Minibatch CKA drifted: {minibatch_ok} vs {exact_full}"
    assert abs(buggy_mean - exact_full) > 0.15, f"Expected large per-batch division bias, got bug={buggy_mean} vs true={exact_full}"
    assert abs(minibatch_ok - exact_full) < abs(buggy_mean - exact_full)


def test_debiased_cka_matrix_matches_pairwise():
    rng = np.random.default_rng(3)
    feats = {
        "m0": rng.standard_normal((90, 48)),
        "m1": rng.standard_normal((90, 64)),
        "m2": rng.standard_normal((90, 32)),
    }
    feats["m1"][:, :24] += 1.5 * feats["m0"][:, :24]

    # Case 1: full-batch (N <= batch_size)
    mat_full = S.debiased_cka_matrix(feats, ["m0", "m1", "m2"], batch_size=512, seed=0)
    for i, mi in enumerate(["m0", "m1", "m2"]):
        for j, mj in enumerate(["m0", "m1", "m2"]):
            if i == j:
                assert abs(mat_full[i, j] - 1.0) < 1e-12
            else:
                val = S.debiased_linear_cka(feats[mi], feats[mj], batch_size=512, seed=0)
                assert abs(mat_full[i, j] - val) < 1e-10

    # Case 2: minibatch (N > batch_size)
    mat_mb = S.debiased_cka_matrix(feats, ["m0", "m1", "m2"], batch_size=30, min_batches=15, seed=5)
    for i, mi in enumerate(["m0", "m1", "m2"]):
        for j, mj in enumerate(["m0", "m1", "m2"]):
            if i < j:
                val = S.debiased_linear_cka(feats[mi], feats[mj], batch_size=30, min_batches=15, seed=5)
                assert abs(mat_mb[i, j] - val) < 1e-10


def test_procrustes_invariance_unequal_dims_and_evr():
    rng = np.random.default_rng(11)
    N = 120
    # Intrinsic rank 20 embedded in unequal dims 48 and 80
    Z = rng.standard_normal((N, 20))
    Qa, _ = np.linalg.qr(rng.standard_normal((48, 48)))
    Qb, _ = np.linalg.qr(rng.standard_normal((80, 80)))
    X = Z @ Qa[:20, :] + 100.0
    Y = 4.2 * (Z @ Qb[:20, :]) - 35.0

    res = S.orthogonal_procrustes_similarity(X, Y, d_star=32)
    assert abs(res["similarity"] - 1.0) < 1e-8, f"Expected 1.0 for identical subspace, got {res}"
    assert abs(res["evr_a"] - 1.0) < 1e-8
    assert abs(res["evr_b"] - 1.0) < 1e-8
    assert res["d_eff"] == 32

    # Truncated PCA retains < 100% variance
    X_full = rng.standard_normal((N, 60))
    res_trunc = S.orthogonal_procrustes_similarity(X_full, X_full, d_star=16)
    assert 0.1 < res_trunc["evr_a"] < 0.95
    assert abs(res_trunc["similarity"] - 1.0) < 1e-8

    # Matrix API matches pairwise
    mat, evrs = S.procrustes_matrix({"a": X, "b": Y}, ["a", "b"], d_star=32)
    assert abs(mat[0, 1] - 1.0) < 1e-8
    assert abs(evrs["a"] - 1.0) < 1e-8


def test_mutual_knn_exact_self_exclusion_and_floor():
    rng = np.random.default_rng(19)
    N, D = 150, 32
    X = rng.standard_normal((N, D))
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    Y_rot = 3.0 * (X @ Q) + 25.0

    # Self-exclusion check
    nbrs = S.cosine_topk_neighbors(X, k=10)
    for i in range(N):
        assert i not in nbrs[i], f"Sample {i} appeared in its own top-k neighbors!"

    assert abs(S.mutual_knn_overlap(X, X, k=10) - 1.0) < 1e-12
    assert abs(S.mutual_knn_overlap(X, Y_rot, k=10) - 1.0) < 1e-12

    # Shuffled calibration floor should sit near k / (N - 1) ≈ 0.067
    sf = S.shuffled_floor_calibration(X, X, batch_size=512, d_star=16, knn_k=10, seed=42)
    expected_knn_floor = 10.0 / (N - 1)
    assert abs(sf["mknn"] - expected_knn_floor) < 0.04, f"Unexpected mKNN floor {sf['mknn']}"
    assert abs(sf["cka"]) < 0.10, f"Unexpected CKA floor {sf['cka']}"


def test_c0_vs_c1_artifact_reordering_and_g1_gate():
    """H3 synthetic benchmark: high-norm OUT artifacts scramble raw C0 CKA ranking; C1 restores truth & triggers G1."""
    rng = np.random.default_rng(2026)
    I, N_patch, D = 200, 64, 32
    models = ["m0", "m1", "m2", "m3", "m4", "m5"]

    # Ground-truth patch structure: low-rank image signal + per-model progressive mix so CKA pairs are well-separated
    img_sig_a = rng.standard_normal((I, 1, D))
    img_sig_b = rng.standard_normal((I, 1, D))
    clean_patches = {}
    for idx, m in enumerate(models):
        alpha = idx / float(len(models) - 1)
        clean_patches[m] = (
            (1.0 - alpha) * img_sig_a
            + alpha * img_sig_b
            + 0.35 * rng.standard_normal((I, N_patch, D))
        ).astype(np.float32)

    # First check: with ZERO outliers, C0 == C1, tau == 1.0, and Gate G1 must NOT trigger
    clean_r2_c0, clean_r2_c1 = {}, {}
    for m in models:
        c0, c1, _ = S.compute_r2_representations(clean_patches[m])
        clean_r2_c0[m], clean_r2_c1[m] = c0, c1
    clean_gate = S.split_half_tau_test(clean_r2_c0, clean_r2_c1, models, metric="cka", n_splits=25, batch_size=512, seed=0)
    assert clean_gate["tau_c0_c1_full"] > 0.99
    assert not clean_gate["diff_same_minus_noise"]["below_noise_ci_excl0"]

    # Inject 3% high-norm (50x) SHARED artifact tokens into opposite ends (m0 & m5) and (m1 & m4)
    art_dir_a = rng.standard_normal((I, 1, D))
    art_dir_b = rng.standard_normal((I, 1, D))
    corrupt_patches = {m: clean_patches[m].copy() for m in models}
    for m, direction in [("m0", art_dir_a), ("m5", art_dir_a), ("m1", art_dir_b), ("m4", art_dir_b)]:
        corrupt_patches[m][:, :2, :] = (50.0 * direction + 0.5 * rng.standard_normal((I, 2, D))).astype(np.float32)

    r2_c0, r2_c1, r3_tok, r3_out = {}, {}, {}, {}
    p_idx = S.sample_r3_patch_indices(I, N_patch, patches_per_image=8, seed=0)
    for m in models:
        c0, c1, om = S.compute_r2_representations(corrupt_patches[m])
        r2_c0[m], r2_c1[m] = c0, c1
        t3, o3 = S.extract_r3_subsampled(corrupt_patches[m], om, p_idx)
        r3_tok[m], r3_out[m] = t3, o3
        if m in ("m0", "m5", "m1", "m4"):
            assert om[:, :2].mean() > 0.98, f"detect_outlier_patches missed injected OUT tokens in {m}"

    mat_true = S.debiased_cka_matrix(clean_r2_c0, models, batch_size=512)
    st_r2 = S.split_half_tau_test(r2_c0, r2_c1, models, metric="cka", n_splits=40, batch_size=512, seed=0)

    tau_c0_true, _ = S.kendall_tau_b_matrices(st_r2["mat_c0_full"], mat_true)
    tau_c1_true, _ = S.kendall_tau_b_matrices(st_r2["mat_c1_full"], mat_true)

    assert tau_c0_true < 0.70, f"Injected artifacts should scramble C0 ranking, got tau(C0,true)={tau_c0_true}"
    assert tau_c1_true > 0.95, f"C1 masking should recover true ranking, got tau(C1,true)={tau_c1_true}"
    assert st_r2["tau_c0_c1_full"] < 0.70
    assert st_r2["diff_same_minus_noise"]["below_noise_ci_excl0"], (
        f"Gate G1 should trigger on artifact-scrambled R2 CKA: {st_r2['diff_same_minus_noise']}"
    )

    # Also verify fast shared-Gram R3 C1 matches exact brute-force pairwise debiased_linear_cka (N <= batch_size)
    small_tok = {m: r3_tok[m][:30] for m in models[:3]}  # 30*8 = 240 <= 512
    small_out = {m: r3_out[m][:30] for m in models[:3]}
    fast_c1 = S.r3_pairwise_matrices(small_tok, small_out, models[:3], condition="C1", batch_size=512, metrics=("cka",))["cka"]
    for i, mi in enumerate(models[:3]):
        for j, mj in enumerate(models[:3]):
            if i < j:
                ti = small_tok[mi].reshape(-1, D)
                tj = small_tok[mj].reshape(-1, D)
                valid = ~(small_out[mi].reshape(-1) | small_out[mj].reshape(-1))
                ref = S.debiased_linear_cka(ti[valid], tj[valid], batch_size=512)
                assert abs(fast_c1[i, j] - ref) < 1e-10, f"Fast shared-Gram R3 C1 mismatch: {fast_c1[i, j]} vs {ref}"

    # Also verify R3 C1 recovers the true pair structure cleanly AND causes far larger shift than matched random drop
    ctrl = S.r3_matched_random_drop_control(r3_tok, r3_out, models, seeds=(0, 1, 2), batch_size=256, min_batches=10)
    assert ctrl["tau_c0_c1_out"] < 0.85, f"Expected R3 CKA reordering under C1, got {ctrl}"
    assert ctrl["delta_tau_out_minus_rand"] < -0.25, f"OUT drop must reorder much more than random drop: {ctrl}"
    assert ctrl["max_abs_shift_out"] > 3.0 * ctrl["max_abs_shift_rand_mean"], f"OUT shift must dwarf random drop: {ctrl}"


def test_otsu_map_iou_and_spearman_grounding():
    # Bimodal 14x14 maps: model A & B highlight top-left quadrant, model C highlights bottom-right
    I = 10
    m_tl = np.zeros((I, 14, 14), dtype=np.float64)
    m_tl[:, :7, :7] = 1.0
    m_br = np.zeros((I, 14, 14), dtype=np.float64)
    m_br[:, 7:, 7:] = 1.0
    m_mid = np.zeros((I, 14, 14), dtype=np.float64)
    m_mid[:, 3:10, 3:10] = 1.0

    iou = S.map_iou_matrix({"A": m_tl, "B": m_tl + 0.01, "C": m_br, "D": m_mid}, ["A", "B", "C", "D"])
    assert abs(iou[0, 1] - 1.0) < 1e-12
    assert abs(iou[0, 2] - 0.0) < 1e-12
    assert 0.1 < iou[0, 3] < 0.5

    corr = S.correlate_similarity_with_map_iou(iou, iou)
    assert abs(corr["spearman_rho"] - 1.0) < 1e-12


def test_m4_runner_end_to_end_smoke(tmp_path: Path):
    """End-to-end smoke test of scripts/run_pilot_similarity_m4.py with random weights on 12 synthetic images."""
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(12):
        arr = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / f"img_{i:02d}.jpg")

    out_dir = tmp_path / "m4_out"
    report = runner_m4.main([
        "--dataset", "folder",
        "--images_dir", str(img_dir),
        "--out_dir", str(out_dir),
        "--models", "dino_vits16,deit3_small_patch16_224,dinov2_vitb14_reg",
        "--layers", "11",
        "--batch_size", "6",
        "--cka_batch_size", "512",
        "--d_star", "8",
        "--knn_k", "4",
        "--r3_patches", "4",
        "--n_splits_r2", "6",
        "--n_splits_r3", "4",
        "--smoke_random",
    ])
    assert (out_dir / "summary_similarity_m4.md").exists()
    assert (out_dir / "similarity_m4_report.json").exists()
    assert (out_dir / "meta.json").exists()

    meta = json.loads((out_dir / "meta.json").read_text())
    assert meta["git_sha"] is not None and len(meta["git_sha"]) >= 7
    assert meta["n_images"] == 12
    assert "l11" in report["layers"]
    for lvl in ("R1", "R2", "R3"):
        assert lvl in report["layers"]["l11"]["levels"]
    for metric in ("cka", "procrustes", "mknn"):
        assert np.isfinite(report["layers"]["l11"]["levels"]["R2"]["metrics"][metric]["tau_c0_c1_full"])
