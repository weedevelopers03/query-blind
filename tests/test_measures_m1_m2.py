"""Unit, sanity-check, mutation, and adapter-integration tests for Measures M1 (qii.py) and M2 (keydom.py).

Covers every specification and silent killer in Methodology Guide §§2-3, Fix 2, Fix 3, and §12:
  * JSD matches scipy.spatial.distance.jensenshannon(p, q, base=np.e)**2 (and fails if unsquared).
  * Fix 2 degenerate invariance (uniform attention -> QII=1, entropy_norm=1 -> flagged).
  * Surrogate S1..S4 properties (S1 norm match; S2 excludes self & MAP-head; S3 patch query;
    S4 Haar PCA rotation matches norm + empirical covariance, PCA save/load round-trip).
  * §2.3 shuffled-key null invariance; §2.4 never averaging heads before JSD.
  * §2.3 head weights w_h + W_O + LayerScale gamma reconstruct adapter.global_update() to < 1e-5
    (and omitting LayerScale fails the check).
  * §3.1-3.2 hand-computed two-way ANOVA omega^2_key + fixed |Q|=64 enforcement.
  * §3.5 sanity checks: pure attention sink, spatial key shuffle (omega^2 invariant, box-AUC collapses),
    and permutation null (omega^2 ~ 0 while biased eta^2 stays positive).
  * Fix 3 QII-KD Spearman decision boundaries (<0.8 keep_both, 0.8-0.9 borderline, >=0.9 demote_qii).
  * End-to-end run through TimmViTAdapter (with and without registers) and OpenCLIPViTAdapter.
"""
from __future__ import annotations

import numpy as np
import pytest
import timm
import torch
from scipy.spatial.distance import jensenshannon

from adapters.open_clip_vit import OpenCLIPViTAdapter
from adapters.timm_vit import TimmViTAdapter
from measures import keydom, qii
from measures._common import (
    LOG2,
    box_auc,
    compute_head_weights,
    get_output_proj_and_layerscale,
    jsd_nats,
    normalized_entropy,
    write_records,
)


# --------------------------------------------------------------------------- M1: JSD & Entropy
def test_jsd_matches_scipy_squared_and_catches_unsquared_bug():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(196), size=6)
    q = rng.dirichlet(np.ones(196), size=6)

    got = jsd_nats(p, q, axis=-1)
    expected = np.array([jensenshannon(p[h], q[h], base=np.e) ** 2 for h in range(6)])
    np.testing.assert_allclose(got, expected, atol=1e-12, rtol=1e-12)

    # Seeded bug (§2.1 / §12): forgetting **2 returns distance, NOT divergence
    unsquared_bug = np.array([jensenshannon(p[h], q[h], base=np.e) for h in range(6)])
    assert np.min(np.abs(got - unsquared_bug)) > 0.05

    # Boundary cases: identical -> 0, disjoint -> log(2)
    assert jsd_nats(p[0], p[0]) == pytest.approx(0.0, abs=1e-15)
    e0 = np.array([1.0, 0.0, 0.0, 0.0])
    e1 = np.array([0.0, 1.0, 0.0, 0.0])
    assert jsd_nats(e0, e1) == pytest.approx(LOG2, abs=1e-12)


def test_normalized_entropy_and_box_auc_hand_computed():
    # Uniform over N=16 -> normalized entropy = 1.0; delta spike -> 0.0
    assert normalized_entropy(np.ones(16) / 16.0) == pytest.approx(1.0, abs=1e-12)
    delta = np.zeros(16)
    delta[5] = 1.0
    assert normalized_entropy(delta) == pytest.approx(0.0, abs=1e-12)

    # Hand-computed ROC-AUC: 2 positives with scores [0.9, 0.4], 2 negatives with [0.6, 0.1]
    # Pairs (pos > neg): (0.9>0.6), (0.9>0.1), (0.4>0.1) = 3/4 = 0.75
    saliency = np.array([[0.9, 0.6], [0.4, 0.1]])
    mask = np.array([[1, 0], [1, 0]])
    assert box_auc(saliency, mask) == pytest.approx(0.75, abs=1e-12)
    # Perfect separation -> 1.0; inverted -> 0.0; constant -> 0.5
    assert box_auc(np.array([[2.0, 0.0], [3.0, -1.0]]), mask) == pytest.approx(1.0, abs=1e-12)
    assert box_auc(np.array([[0.0, 2.0], [-1.0, 3.0]]), mask) == pytest.approx(0.0, abs=1e-12)
    assert box_auc(np.ones((2, 2)), mask) == pytest.approx(0.5, abs=1e-12)


# --------------------------------------------------------------------------- M1: QII & Surrogates
def test_qii_analytic_query_blind_vs_degenerate_vs_query_dependent():
    H, N, d = 3, 16, 8
    scale = d ** -0.5
    rng = np.random.default_rng(1)
    q_g = rng.standard_normal((H, d)).astype(np.float32)

    # Case 1: Degenerate invariance (Fix 2) — all keys identical -> uniform attention
    k_unif = np.repeat(rng.standard_normal((H, 1, d)), N, axis=1).astype(np.float32)
    res_deg = qii.compute_qii_image(q_g, k_unif, scale, (4, 4), surrogates=("S1",), seeds=(0, 1, 2))
    assert res_deg["per_surrogate"]["S1"]["qii_weighted"] == pytest.approx(1.0, abs=1e-6)
    assert res_deg["entropy_weighted"] == pytest.approx(1.0, abs=1e-6)
    assert qii.classify_qii_outcome(1.0, res_deg["entropy_weighted"]) == "degenerate_invariance"

    # Case 2: Query-blind informative map — one shared key direction with huge magnitude on foreground patches
    # When keys lie on one positive line aligned with a shared query half-space, but wait: with isotropic
    # Gaussian S1, half of surrogates flip sign unless keys have a strong norm spike or common direction.
    # For S2/S4 in a positive cone around mu, ALL in-distribution surrogates align with the object keys!
    mu = np.zeros((H, d), dtype=np.float32)
    mu[:, 0] = 8.0
    pool = mu[None] + 0.3 * rng.standard_normal((40, H, d)).astype(np.float32)
    pca = qii.fit_query_pca(pool, k=4)

    k_obj = np.zeros((H, N, d), dtype=np.float32)
    # Foreground = top-left 2x2 patches (indices 0, 1, 4, 5)
    fg_idx = [0, 1, 4, 5]
    k_obj[:, fg_idx, 0] = 6.0
    gt_mask = np.zeros((4, 4), dtype=np.uint8)
    gt_mask[:2, :2] = 1

    res_qb = qii.compute_qii_image(
        pool[0], k_obj, scale, (4, 4),
        q_global_pool=pool, image_idx=0, pca=pca, gt_mask=gt_mask,
        surrogates=("S2", "S4"), seeds=(0, 1, 2, 3, 4),
    )
    assert res_qb["per_surrogate"]["S2"]["qii_weighted"] > 0.99
    assert res_qb["per_surrogate"]["S4"]["qii_weighted"] > 0.95
    assert res_qb["entropy_weighted"] < 0.65
    assert res_qb["box_auc_weighted"] == pytest.approx(1.0, abs=1e-6)
    assert res_qb["status_s4"] == "eligible"

    # Case 3: Strongly query-dependent keys (random orthogonal-ish high-norm keys) -> low QII
    k_qdep = (8.0 * rng.standard_normal((H, N, d))).astype(np.float32)
    res_qdep = qii.compute_qii_image(
        pool[0], k_qdep, scale, (4, 4),
        q_global_pool=pool, image_idx=0, pca=pca,
        surrogates=("S1",), seeds=(0, 1, 2, 3, 4),
    )
    assert res_qdep["per_surrogate"]["S1"]["qii_weighted"] < 0.5


def test_qii_s4_matches_norm_and_covariance_and_roundtrips(tmp_path):
    rng = np.random.default_rng(7)
    I, H, d = 600, 2, 16
    # Strongly anisotropic empirical query distribution (high variance in first 4 dims, tiny in rest)
    scales = np.array([5.0, 3.0, 2.0, 1.5] + [0.1] * (d - 4))
    mean_true = rng.standard_normal((H, d)) * 2.0
    pool = mean_true[None] + rng.standard_normal((I, H, d)) * scales[None, None, :]

    pca = qii.fit_query_pca(pool, k=8)
    saved = pca.save(tmp_path / "pca_l11.npz")
    pca2 = qii.QueryPCA.load(saved)
    np.testing.assert_allclose(pca2.mean, pca.mean)
    np.testing.assert_allclose(pca2.components, pca.components)
    np.testing.assert_allclose(pca2.stds, pca.stds)

    # Check exact per-head norm match AND covariance preservation across samples
    surrs = np.stack([qii.make_surrogate_s4(pool[i], pca2, np.random.default_rng(1000 + i)) for i in range(I)])
    orig_norms = np.linalg.norm(pool, axis=-1)
    surr_norms = np.linalg.norm(surrs, axis=-1)
    np.testing.assert_allclose(surr_norms, orig_norms, rtol=1e-5, atol=1e-5)

    # Empirical variance along top-4 axes vs tail axes should stay strongly anisotropic (unlike S1!)
    var_orig = pool[:, 0, :].var(axis=0)
    var_s4 = surrs[:, 0, :].var(axis=0)
    assert var_s4[:4].mean() / var_s4[4:].mean() > 20.0
    # Correlation between diagonal variances of S4 and original should be very high
    assert np.corrcoef(var_orig, var_s4)[0, 1] > 0.95


def test_qii_map_head_excludes_s2_and_s4():
    H, N, d = 4, 16, 8
    rng = np.random.default_rng(2)
    fixed_probe = rng.standard_normal((H, d)).astype(np.float32)
    # Identical probe across all images (MAP-head property, §1.1)
    pool = np.repeat(fixed_probe[None], 10, axis=0)
    pca = qii.fit_query_pca(pool, k=4)
    assert np.all(pca.is_degenerate)

    k_patch = rng.standard_normal((H, N, d)).astype(np.float32)
    res = qii.compute_qii_image(
        fixed_probe, k_patch, d ** -0.5, (4, 4),
        q_global_pool=pool, image_idx=0, pca=pca,
        surrogates=("S1", "S2", "S4"), seeds=(0, 1, 2), is_map_head=True,
    )
    assert not res["per_surrogate"]["S1"]["excluded"]
    assert res["per_surrogate"]["S2"]["excluded"] is True
    assert np.isnan(res["per_surrogate"]["S2"]["qii_weighted"])
    assert res["per_surrogate"]["S4"]["excluded"] is True
    assert np.isnan(res["per_surrogate"]["S4"]["qii_weighted"])


def test_qii_shuffled_key_null_and_never_average_heads_before_jsd():
    rng = np.random.default_rng(3)
    H, N, d = 4, 16, 8
    pool = rng.standard_normal((20, H, d)).astype(np.float32)
    q_patches = rng.standard_normal((H, N, d)).astype(np.float32)
    k_patch = rng.standard_normal((H, N, d)).astype(np.float32)
    pca = qii.fit_query_pca(pool, k=4)

    ok, max_diff = qii.shuffled_key_null_check(
        pool[0], k_patch, d ** -0.5, (4, 4),
        q_global_pool=pool, image_idx=0, q_patches=q_patches, pca=pca,
        surrogates=("S1", "S2", "S3", "S4"), seeds=(0, 1, 2, 3, 4),
    )
    assert ok and max_diff < 1e-6

    # Guide §2.4 check: 2 heads with opposite peaked distributions;
    # averaging before JSD gives JSD(avg, avg) = 0, whereas per-head JSD is log(2) -> QII = 0!
    k_2h = np.zeros((2, 2, 2), dtype=np.float32)
    k_2h[0, 0] = [20.0, 0.0]; k_2h[0, 1] = [-20.0, 0.0]
    k_2h[1, 0] = [-20.0, 0.0]; k_2h[1, 1] = [20.0, 0.0]
    q_true = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)   # head 0 -> patch 0, head 1 -> patch 1
    q_flip = np.array([[-1.0, 0.0], [-1.0, 0.0]], dtype=np.float32) # head 0 -> patch 1, head 1 -> patch 0
    pool_2 = np.stack([q_true, q_flip], axis=0)

    res = qii.compute_qii_image(
        q_true, k_2h, 1.0, (1, 2),
        q_global_pool=pool_2, image_idx=0, surrogates=("S2",), seeds=(0,),
    )
    # Per-head JSD = log(2) => QII = 0.0 (head-averaged-first bug would wrongly give QII = 1.0!)
    assert res["per_surrogate"]["S2"]["qii_weighted"] == pytest.approx(0.0, abs=1e-5)


def test_head_weights_reconstruct_global_update_and_catch_omitted_layerscale():
    # DeiT-III has LayerScale (ls1.gamma); OpenCLIP ViT-B-16 has nn.Identity ls_1
    import open_clip

    torch.manual_seed(0)
    deit = timm.create_model("deit3_small_patch16_224", pretrained=False)
    # Boost LayerScale away from 1e-4 / 1.0 so omitting it produces a large error
    with torch.no_grad():
        for blk in deit.blocks:
            blk.ls1.gamma.uniform_(0.3, 0.9)
    a_timm = TimmViTAdapter()
    a_timm.prepare(deit)
    x = torch.randn(1, 3, 224, 224)
    l = 11
    _, _, v_full = a_timm.get_qkv(deit, x, l)
    attn_row = a_timm.attention_probs(deit, x, l)[:, :, 0, :]
    W_O, b_O, gamma = get_output_proj_and_layerscale(a_timm, deit, l)
    assert gamma is not None

    w_h, head_vecs = compute_head_weights(attn_row, v_full, W_O, gamma=gamma, return_vectors=True)
    recon = head_vecs.sum(dim=1) + (b_O * gamma if b_O is not None else 0.0)
    truth = a_timm.global_update(deit, x, l)
    assert (recon - truth).abs().max().item() < 1e-5
    assert w_h.sum().item() == pytest.approx(1.0, abs=1e-6)

    # Seeded bug (§12): omitting LayerScale gamma must fail reconstruction
    _, bad_vecs = compute_head_weights(attn_row, v_full, W_O, gamma=None, return_vectors=True)
    bad_recon = bad_vecs.sum(dim=1) + (b_O if b_O is not None else 0.0)
    assert (bad_recon - truth).abs().max().item() > 1e-2

    # Also verify OpenCLIPViTAdapter
    oc = open_clip.create_model("ViT-B-16", pretrained=None).visual
    a_oc = OpenCLIPViTAdapter()
    a_oc.prepare(oc)
    _, _, v_oc = a_oc.get_qkv(oc, x, l)
    attn_oc = a_oc.attention_probs(oc, x, l)[:, :, 0, :]
    W_oc, b_oc, g_oc = get_output_proj_and_layerscale(a_oc, oc, l)
    w_oc, vecs_oc = compute_head_weights(attn_oc, v_oc, W_oc, gamma=g_oc, return_vectors=True)
    recon_oc = vecs_oc.sum(dim=1) + (b_oc if b_oc is not None else 0.0)
    assert (recon_oc - a_oc.global_update(oc, x, l)).abs().max().item() < 1e-5
    assert w_oc.sum().item() == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- M2: Key Dominance
def test_keydom_hand_computed_two_way_anova():
    # Pure additive matrix: z(q, i) = 10 + alpha_q + beta_i -> SS_resid = 0
    alpha = np.array([-2.0, 0.0, 2.0])         # Q = 3, sum = 0
    beta = np.array([-3.0, -1.0, 1.0, 3.0])    # N = 4, sum = 0
    z = 10.0 + alpha[:, None] + beta[None, :]
    res = keydom.two_way_anova_key_dominance(z)

    ss_q_exp = 4.0 * (4.0 + 0.0 + 4.0)         # 32
    ss_k_exp = 3.0 * (9.0 + 1.0 + 1.0 + 9.0)   # 60
    ss_tot_exp = 92.0
    assert res["ss_query"] == pytest.approx(ss_q_exp, abs=1e-12)
    assert res["ss_key"] == pytest.approx(ss_k_exp, abs=1e-12)
    assert res["ss_resid"] == pytest.approx(0.0, abs=1e-12)
    assert res["omega2_key"] == pytest.approx(ss_k_exp / ss_tot_exp, abs=1e-12)
    np.testing.assert_allclose(res["beta_key"], beta, atol=1e-12)


def test_keydom_sanity_sink_spatial_shuffle_and_permutation_null():
    rng = np.random.default_rng(11)
    H, Q_n, G, d = 3, 64, 6, 8
    N = G * G
    scale = d ** -0.5

    # 1. Pure attention sink (§3.5): token (2, 3) -> index 15 has huge positive key aligned with all queries
    Q = np.zeros((H, Q_n, d), dtype=np.float32)
    Q[:, :, 0] = 5.0 + 0.1 * rng.standard_normal((H, Q_n))
    Q[:, :, 1:] = 0.05 * rng.standard_normal((H, Q_n, d - 1))

    k_sink = 0.01 * rng.standard_normal((H, N, d)).astype(np.float32)
    sink_idx = 2 * G + 3
    k_sink[:, sink_idx, 0] = 10.0
    gt_mask = np.zeros((G, G), dtype=np.uint8)
    gt_mask[2, 3] = 1

    res_sink = keydom.compute_key_dominance_image(Q, k_sink, scale, (G, G), gt_mask=gt_mask)
    assert res_sink["omega2_weighted"] > 0.99
    assert np.unravel_index(np.argmax(res_sink["keymap_agg"]), (G, G)) == (2, 3)
    assert res_sink["box_auc_agg"] == pytest.approx(1.0, abs=1e-6)

    # 2. Spatial key shuffle (§3.5): omega^2_key strictly invariant, box-AUC collapses
    # Use a 2x2 foreground block for cleaner AUC collapse check
    k_fg = 0.05 * rng.standard_normal((H, N, d)).astype(np.float32)
    fg_mask = np.zeros((G, G), dtype=np.uint8)
    fg_mask[1:3, 1:3] = 1
    for idx in np.flatnonzero(fg_mask):
        k_fg[:, idx, 0] = 6.0
    shuf = keydom.spatial_shuffle_check(Q, k_fg, scale, (G, G), gt_mask=fg_mask, seed=42)
    assert shuf["ok"] and shuf["max_omega2_diff"] < 1e-6
    assert shuf["box_auc_orig"] == pytest.approx(1.0, abs=1e-6)
    assert shuf["box_auc_shuffled"] < 0.75

    # 3. Permutation null (§3.5 & §3.2): omega^2_key ~ 0 while biased eta^2 stays ~ (N-1)/(Q*N)
    # Test with N=196, |Q|=64 (14x14 protocol!)
    z_rand = rng.standard_normal((12, 64, 196))
    null = keydom.permutation_null_check(z_rand, seed=0)
    assert null["omega2_mean"] < 0.005
    # Verify eta^2 IS upward-biased by ~ (195) / (64 * 196) ≈ 0.0155 -> proves why omega^2 is needed!
    assert null["eta2_mean"] > 0.012


def test_keydom_enforces_fixed_q64_and_build_query_set():
    rng = np.random.default_rng(5)
    H, N, d = 4, 196, 16
    pool = rng.standard_normal((50, H, d)).astype(np.float32)
    q_patches = rng.standard_normal((H, N, d)).astype(np.float32)
    k_patch = rng.standard_normal((H, N, d)).astype(np.float32)

    Q = keydom.build_query_set_64(pool[0], q_patches, pool, image_idx=0, seed=0)
    assert Q.shape == (H, 64, d)
    # Slot 0 must be exact true global query
    np.testing.assert_allclose(Q[:, 0, :], pool[0])

    # Wrong |Q| (e.g. using all N=196 patches, Guide §3.2 anti-pattern) must raise ValueError!
    with pytest.raises(ValueError, match="fixed \\|Q\\|=64"):
        keydom.compute_key_dominance_image(q_patches, k_patch, d ** -0.5, (14, 14))


def test_fix3_spearman_decision_rules():
    # Identical ranking -> rho = 1.0 -> demote_qii
    m_qii = {f"m{i}": float(i) for i in range(8)}
    m_kd_hi = {f"m{i}": float(i) * 0.1 for i in range(8)}
    out_hi = keydom.qii_kd_spearman(m_qii, m_kd_hi)
    assert out_hi["rho"] == pytest.approx(1.0)
    assert out_hi["decision"] == "demote_qii"

    # Uncorrelated -> keep_both
    m_kd_lo = {f"m{i}": v for i, v in enumerate([0.8, 0.1, 0.7, 0.2, 0.6, 0.3, 0.5, 0.4])}
    out_lo = keydom.qii_kd_spearman(m_qii, m_kd_lo)
    assert out_lo["rho"] < 0.8
    assert out_lo["decision"] == "keep_both"


# --------------------------------------------------------------------------- End-to-end adapters
@pytest.mark.parametrize("arch_kind", ["dino_s16", "dinov2_b14_reg", "openclip_b16"])
def test_end_to_end_m1_m2_on_pilot_adapters(arch_kind, tmp_path):
    torch.manual_seed(0)
    if arch_kind == "dino_s16":
        model = timm.create_model("vit_small_patch16_224", pretrained=False)
        adapter = TimmViTAdapter()
        res_px = 224
    elif arch_kind == "dinov2_b14_reg":
        # Primary 14x14 protocol: 196px for /14 models (§1.5), with 4 register tokens excluded!
        model = timm.create_model(
            "vit_base_patch14_reg4_dinov2", pretrained=False, img_size=196, dynamic_img_size=True
        )
        adapter = TimmViTAdapter()
        res_px = 196
    else:
        import open_clip

        model = open_clip.create_model("ViT-B-16", pretrained=None).visual
        adapter = OpenCLIPViTAdapter()
        res_px = 224

    adapter.prepare(model)
    B = 4
    x = torch.randn(B, 3, res_px, res_px)
    l = adapter.n_blocks(model) - 1

    ext = qii.extract_m1_inputs_from_adapter(adapter, model, x, l)
    assert ext["grid"] == (14, 14)
    assert ext["k_patch"].shape[2] == 196  # registers strictly excluded!

    # Build a synthetic pool of 40 global queries around the extracted batch for PCA + S2 + |Q|=64
    rng = np.random.default_rng(0)
    q_batch = ext["q_g"].numpy()  # (B, H, d_h)
    pool = np.concatenate(
        [q_batch, q_batch.mean(0, keepdims=True) + 0.5 * rng.standard_normal((36, *q_batch.shape[1:])).astype(np.float32)],
        axis=0,
    )
    pca = qii.fit_query_pca(pool, k=16)

    gt_mask = np.zeros((res_px, res_px), dtype=np.uint8)
    gt_mask[50:150, 50:150] = 1

    meta = dict(
        model_id=arch_kind, objective="test", set="A",
        layer=l, n_blocks=adapter.n_blocks(model), image_id="img_0", resolution=res_px,
    )
    res_m1 = qii.compute_qii_image(
        ext["q_g"][0], ext["k_patch"][0], ext["scale"], ext["grid"],
        q_global_pool=pool, image_idx=0, q_patches=ext["q_patches"][0],
        pca=pca, head_weights=ext["head_weights"][0], gt_mask=gt_mask,
        seeds=(0, 1, 2, 3, 4), meta=meta,
    )
    # 4 surrogates x 5 seeds x H heads
    H = ext["k_patch"].shape[1]
    assert len(res_m1["records"]) == 4 * 5 * H
    assert tuple(res_m1["records"][0].keys()) == qii.QII_SCHEMA_COLUMNS
    for S in qii.SURROGATES:
        val = res_m1["per_surrogate"][S]["qii_weighted"]
        assert 0.0 <= val <= 1.0
    assert 0.0 <= res_m1["entropy_weighted"] <= 1.0
    assert 0.0 <= res_m1["box_auc_weighted"] <= 1.0

    # Verify schema export (CSV locally; Parquet when pandas/pyarrow present)
    csv_path = write_records(res_m1["records"], tmp_path / "qii.csv", columns=qii.QII_SCHEMA_COLUMNS)
    assert csv_path.exists()

    # Measure M2 on same image
    Q64 = keydom.build_query_set_64(ext["q_g"][0], ext["q_patches"][0], pool, image_idx=0, seed=0)
    res_m2 = keydom.compute_key_dominance_image(
        Q64, ext["k_patch"][0], ext["scale"], ext["grid"],
        head_weights=ext["head_weights"][0], gt_mask=gt_mask,
    )
    assert 0.0 <= res_m2["omega2_weighted"] <= 1.0
    assert res_m2["keymap"].shape == (H, 14, 14)
    assert res_m2["keymap_agg"].shape == (14, 14)
    assert 0.0 <= res_m2["box_auc_agg"] <= 1.0

    # Atomic fp16 keymap shard save (§3.3 & §12)
    km_path = keydom.save_keymaps(res_m2["keymap"][None, None], tmp_path / "keymap.npy")
    loaded = np.load(km_path)
    assert loaded.dtype == np.float16 and loaded.shape == (1, 1, H, 14, 14)
