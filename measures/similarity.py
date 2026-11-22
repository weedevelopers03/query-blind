"""Measure M4 — Artifact-Controlled Representation Similarity (Methodology Guide §5).

Implements:
- Three representation levels (§5.1):
    * R1: global token ([CLS] / pooled vector), shape (I, d)
    * R2: mean-pooled patch tokens, shape (I, d) — C0 all patches, C1 excluding OUT patches
    * R3: subsampled individual patch tokens, shape (I * N', d) with N' = 8 fixed patch positions per image;
          under C1, rows where either model flags OUT are dropped so row pairing is exact with zero resampling noise.
- Three estimators (§5.2), all with mandatory feature centering across the sample:
    1. Debiased linear CKA using Song et al. (2007) unbiased HSIC_1 and Nguyen, Raghu & Kornblith (ICLR 2021)
       minibatch accumulation (averaging numerator and denominators across batches BEFORE dividing).
    2. Orthogonal Procrustes similarity after PCA reduction to d* = 256 (equal dimensionality),
       Frobenius-normalized for scale invariance, reporting explained variance retained (EVR).
    3. Mutual k-NN overlap (k = 10, cosine metric on centered features).
- Rank stability & noise floor (§5.4 & Gate G1 Criterion 2):
    * Kendall's tau-b between C0 and C1 model-pair rankings.
    * Split-half noise floor tau_noise across repeated disjoint 50/50 image splits (no duplicate rows),
      with paired CIs for both same-half (tau(C0_H1, C1_H1) - tau_noise) and cross-half
      (tau(C0_H1, C1_H2) - tau_noise).
- Grounding via Otsu-binarized localization map IoU (§5.5) and shuffled-feature calibration floor (§5.6).

Performance notes (Tesla T4 GPU):
- T4 fp64 matmul is 32x slower than fp32. Therefore, Gram matrices (X @ X^T) and covariance matrices
  (X^T @ X) are formed in fp32 on CUDA (fp64 on CPU) after global centering, while all HSIC_1 scalar
  accumulations and (d, d) eigendecompositions run in fp64 for full numerical accuracy.
- For R3 C1 CKA, each minibatch forms only M=8 Gram matrices once and slices valid=(~out_i & ~out_j)
  per pair (exploiting the exact translation-invariance of Song et al.'s unbiased HSIC_1).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr

from .routing import detect_outlier_tokens

DEFAULT_MINIBATCH_SIZE = 512
DEFAULT_MIN_BATCHES = 40
DEFAULT_PROCRUSTES_DIM = 256
DEFAULT_KNN_K = 10
DEFAULT_R3_PATCHES_PER_IMAGE = 8


def detect_outlier_patches(patch_norms: np.ndarray | torch.Tensor, k_mad: float = 5.0) -> np.ndarray:
    """Thin wrapper around M3's detect_outlier_tokens() when input already contains patch-only norms (..., N)."""
    arr = patch_norms.detach().cpu().numpy() if isinstance(patch_norms, torch.Tensor) else np.asarray(patch_norms)
    n_patches = arr.shape[-1]
    out_mask, _ = detect_outlier_tokens(arr, np.arange(n_patches), k_mad=k_mad)
    return out_mask


def _matmul_dtype(dev: torch.device) -> torch.dtype:
    """Use fp32 for large GEMMs on CUDA (32x faster on T4), fp64 on CPU."""
    return torch.float32 if dev.type == "cuda" else torch.float64


# --------------------------------------------------------------------------- Feature prep & C0/C1 builders
def center_features(X: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Subtract sample mean across rows (Guide §5.2 & §12)."""
    if isinstance(X, torch.Tensor):
        return X - X.mean(dim=0, keepdim=True)
    arr = np.asarray(X, dtype=np.float64)
    return arr - arr.mean(axis=0, keepdims=True)


def sample_r3_patch_indices(n_images: int, n_patches: int, patches_per_image: int = DEFAULT_R3_PATCHES_PER_IMAGE, seed: int = 0) -> np.ndarray:
    """Deterministically sample N' = 8 distinct patch indices per image (shared across all models)."""
    k = min(patches_per_image, n_patches)
    rng = np.random.default_rng(seed)
    idx = np.empty((n_images, k), dtype=np.int64)
    for i in range(n_images):
        idx[i] = rng.choice(n_patches, size=k, replace=False)
    return idx


def compute_r2_representations(
    patch_tokens: np.ndarray,
    out_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute R2 (mean-pooled patches) under C0 (raw) and C1 (OUT-masked)."""
    pts = np.asarray(patch_tokens, dtype=np.float32)
    if out_mask is None:
        norms = np.linalg.norm(pts.astype(np.float64), axis=-1)
        out_mask = detect_outlier_patches(norms)
    else:
        out_mask = np.asarray(out_mask, dtype=bool)

    r2_c0 = pts.mean(axis=1, dtype=np.float64).astype(np.float32)

    valid = ~out_mask
    counts = valid.sum(axis=1)
    empty = counts == 0
    if np.any(empty):
        valid = valid.copy()
        valid[empty] = True
        counts = valid.sum(axis=1)
    r2_c1 = (pts.astype(np.float64) * valid[:, :, None]).sum(axis=1) / counts[:, None]
    return r2_c0, r2_c1.astype(np.float32), out_mask


def extract_r3_subsampled(
    patch_tokens: np.ndarray,
    out_mask: np.ndarray,
    patch_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract subsampled R3 patch tokens and matching OUT flags using shared patch_indices."""
    I = patch_tokens.shape[0]
    rows = np.arange(I)[:, None]
    r3_tokens = np.asarray(patch_tokens[rows, patch_indices], dtype=np.float32)
    r3_out = np.asarray(out_mask[rows, patch_indices], dtype=bool)
    return r3_tokens, r3_out


# --------------------------------------------------------------------------- Estimator 1: Debiased Linear CKA
def hsic_unbiased_gram_torch(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Song et al. (2007) unbiased HSIC_1 estimator from Gram matrices K, L of shape (n, n), n >= 4.

    Computed in fp64 in O(n^2) time without forming K @ L:
        tr(K~ L~) = sum(K~ * L~)
        1^T K~ L~ 1 = (K~ 1)^T (L~ 1)
    Note: HSIC_1 is algebraically invariant to row/column translation (X -> X - 1*mu^T).
    """
    n = K.shape[0]
    if n < 4:
        raise ValueError(f"Unbiased HSIC_1 requires batch size n >= 4, got {n}")
    K64 = K.to(torch.float64)
    L64 = L.to(torch.float64)
    K_tilde = K64 - torch.diag_embed(torch.diagonal(K64))
    L_tilde = L64 - torch.diag_embed(torch.diagonal(L64))

    row_sum_K = K_tilde.sum(dim=1)
    row_sum_L = L_tilde.sum(dim=1)
    sum_K = row_sum_K.sum()
    sum_L = row_sum_L.sum()

    term1 = (K_tilde * L_tilde).sum()
    term2 = (sum_K * sum_L) / float((n - 1) * (n - 2))
    term3 = (2.0 / float(n - 2)) * torch.dot(row_sum_K, row_sum_L)
    return (term1 + term2 - term3) / float(n * (n - 3))


def hsic_unbiased_linear(X: np.ndarray | torch.Tensor, Y: np.ndarray | torch.Tensor) -> float:
    """Exact unbiased linear HSIC_1(X X^T, Y Y^T) on centered inputs."""
    Xt = torch.as_tensor(X, dtype=torch.float64)
    Yt = torch.as_tensor(Y, dtype=torch.float64)
    Xt = Xt - Xt.mean(dim=0, keepdim=True)
    Yt = Yt - Yt.mean(dim=0, keepdim=True)
    K = Xt @ Xt.T
    L = Yt @ Yt.T
    return float(hsic_unbiased_gram_torch(K, L).item())


def _gram_pair_hsic(Ki_tilde: torch.Tensor, Kj_tilde: torch.Tensor, rsi: torch.Tensor, rsj: torch.Tensor, si: torch.Tensor, sj: torch.Tensor, nb: int) -> float:
    """Fast O(n^2) HSIC_1 from pre-zeroed-diagonal Gram matrices and precomputed row sums."""
    val = (
        (Ki_tilde * Kj_tilde).sum()
        + (si * sj) / float((nb - 1) * (nb - 2))
        - (2.0 / float(nb - 2)) * torch.dot(rsi, rsj)
    ) / float(nb * (nb - 3))
    return float(val.item())


def debiased_linear_cka(
    X: np.ndarray | torch.Tensor,
    Y: np.ndarray | torch.Tensor,
    batch_size: int = DEFAULT_MINIBATCH_SIZE,
    min_batches: int = DEFAULT_MIN_BATCHES,
    seed: int = 0,
    device: torch.device | str | None = None,
) -> float:
    """Debiased linear CKA (Song et al. 2007; minibatch Nguyen, Raghu & Kornblith ICLR 2021).

    - Always centers X and Y across the full sample first (§5.2 & §12).
    - If N <= batch_size, computes exact full-sample unbiased HSIC_1.
    - If N > batch_size, uses disjoint shuffled minibatches of size `batch_size`, averaging
      numerator and both denominators across batches BEFORE dividing (§5.2 & §12).
    """
    dev = torch.device(device) if device is not None else torch.device("cpu")
    dt = _matmul_dtype(dev)
    Xt = torch.as_tensor(X, dtype=dt, device=dev)
    Yt = torch.as_tensor(Y, dtype=dt, device=dev)
    if Xt.shape[0] != Yt.shape[0]:
        raise ValueError(f"Row mismatch: X has {Xt.shape[0]}, Y has {Yt.shape[0]}")
    n = Xt.shape[0]
    if n < 4:
        return float("nan")

    Xt = Xt - Xt.mean(dim=0, keepdim=True)
    Yt = Yt - Yt.mean(dim=0, keepdim=True)

    if n <= batch_size:
        K = Xt @ Xt.T
        L = Yt @ Yt.T
        num = hsic_unbiased_gram_torch(K, L)
        dx = hsic_unbiased_gram_torch(K, K)
        dy = hsic_unbiased_gram_torch(L, L)
        denom = torch.sqrt(torch.clamp(dx, min=0.0) * torch.clamp(dy, min=0.0))
        if float(denom.item()) <= 1e-18:
            return float("nan")
        return float((num / denom).item())

    rng = np.random.default_rng(seed)
    batches_per_pass = n // batch_size
    n_passes = max(1, int(np.ceil(min_batches / max(batches_per_pass, 1))))

    num_acc = 0.0
    dx_acc = 0.0
    dy_acc = 0.0
    n_batches = 0

    for _ in range(n_passes):
        perm = rng.permutation(n)
        for start in range(0, batches_per_pass * batch_size, batch_size):
            idx = torch.as_tensor(perm[start : start + batch_size], dtype=torch.long, device=dev)
            Xb = Xt[idx]
            Yb = Yt[idx]
            K = Xb @ Xb.T
            L = Yb @ Yb.T
            num_acc += float(hsic_unbiased_gram_torch(K, L).item())
            dx_acc += float(hsic_unbiased_gram_torch(K, K).item())
            dy_acc += float(hsic_unbiased_gram_torch(L, L).item())
            n_batches += 1

    mean_num = num_acc / n_batches
    mean_dx = dx_acc / n_batches
    mean_dy = dy_acc / n_batches
    if mean_dx <= 1e-18 or mean_dy <= 1e-18:
        return float("nan")
    return float(mean_num / np.sqrt(mean_dx * mean_dy))


def debiased_cka_matrix(
    feats_by_model: dict[str, np.ndarray | torch.Tensor],
    model_names: Sequence[str],
    batch_size: int = DEFAULT_MINIBATCH_SIZE,
    min_batches: int = DEFAULT_MIN_BATCHES,
    seed: int = 0,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Compute M x M symmetric matrix of debiased linear CKA sharing identical minibatches across all pairs."""
    dev = torch.device(device) if device is not None else torch.device("cpu")
    dt = _matmul_dtype(dev)
    M = len(model_names)
    Xs = []
    n = None
    for name in model_names:
        Xt = torch.as_tensor(feats_by_model[name], dtype=dt, device=dev)
        if n is None:
            n = Xt.shape[0]
        elif Xt.shape[0] != n:
            raise ValueError(f"Model {name} row count {Xt.shape[0]} != {n}")
        Xs.append(Xt - Xt.mean(dim=0, keepdim=True))

    mat = np.eye(M, dtype=np.float64)
    if n is None or n < 4:
        mat[:] = np.nan
        return mat

    if n <= batch_size:
        idx_batches = [torch.arange(n, dtype=torch.long, device=dev)]
    else:
        rng = np.random.default_rng(seed)
        batches_per_pass = n // batch_size
        n_passes = max(1, int(np.ceil(min_batches / max(batches_per_pass, 1))))
        idx_batches = []
        for _ in range(n_passes):
            perm = rng.permutation(n)
            for start in range(0, batches_per_pass * batch_size, batch_size):
                idx_batches.append(torch.as_tensor(perm[start : start + batch_size], dtype=torch.long, device=dev))

    hsic_acc = np.zeros((M, M), dtype=np.float64)
    for idx in idx_batches:
        grams, row_sums, tot_sums = [], [], []
        nb = int(idx.numel())
        for m in range(M):
            Xb = Xs[m][idx]
            K = (Xb @ Xb.T).to(torch.float64)
            K_tilde = K - torch.diag_embed(torch.diagonal(K))
            rs = K_tilde.sum(dim=1)
            grams.append(K_tilde)
            row_sums.append(rs)
            tot_sums.append(rs.sum())

        for i in range(M):
            for j in range(i, M):
                v = _gram_pair_hsic(grams[i], grams[j], row_sums[i], row_sums[j], tot_sums[i], tot_sums[j], nb)
                hsic_acc[i, j] += v
                if i != j:
                    hsic_acc[j, i] += v

    hsic_mean = hsic_acc / float(len(idx_batches))
    diag = np.diag(hsic_mean)
    for i in range(M):
        for j in range(i + 1, M):
            if diag[i] > 1e-18 and diag[j] > 1e-18:
                val = float(hsic_mean[i, j] / np.sqrt(diag[i] * diag[j]))
            else:
                val = float("nan")
            mat[i, j] = mat[j, i] = val
    return mat


# --------------------------------------------------------------------------- Estimator 2: Orthogonal Procrustes
def pca_reduce(
    X: np.ndarray | torch.Tensor,
    d_star: int = DEFAULT_PROCRUSTES_DIM,
    device: torch.device | str | None = None,
) -> tuple[np.ndarray, float]:
    """Center X and project onto top d_eff = min(d_star, N-1, d) principal components.

    Uses fast Gram eigendecomposition (n x n) when n < d and covariance eigendecomposition (d x d)
    when n >= d, avoiding slow rectangular SVDs on T4 GPU.
    """
    dev = torch.device(device) if device is not None else torch.device("cpu")
    dt = _matmul_dtype(dev)
    Xt = torch.as_tensor(X, dtype=dt, device=dev)
    n, d = Xt.shape
    Xc = Xt - Xt.mean(dim=0, keepdim=True)
    d_eff = min(d_star, max(1, n - 1), d)

    if n >= d:
        cov = (Xc.T @ Xc).to(torch.float64)  # (d, d)
        evals, evecs = torch.linalg.eigh(cov)
        evals = torch.clamp(evals, min=0.0)
        order = torch.argsort(evals, descending=True)
        evals = evals[order]
        evecs = evecs[:, order]
        tot_var = float(evals.sum().item())
        if tot_var <= 1e-18:
            return np.zeros((n, d_eff), dtype=np.float64), 0.0
        evr = float(evals[:d_eff].sum().item() / tot_var)
        Z = (Xc @ evecs[:, :d_eff].to(dt)).to(torch.float64).cpu().numpy()
        return Z, evr
    else:
        # Dual PCA via n x n Gram matrix (much faster than SVD when n < d, e.g. n=250, d=768)
        G = (Xc @ Xc.T).to(torch.float64)  # (n, n)
        evals, U = torch.linalg.eigh(G)
        evals = torch.clamp(evals, min=0.0)
        order = torch.argsort(evals, descending=True)
        evals = evals[order]
        U = U[:, order]
        tot_var = float(evals.sum().item())
        if tot_var <= 1e-18:
            return np.zeros((n, d_eff), dtype=np.float64), 0.0
        evr = float(evals[:d_eff].sum().item() / tot_var)
        Z = (U[:, :d_eff] * torch.sqrt(evals[:d_eff])).cpu().numpy()
        return Z, evr


def orthogonal_procrustes_similarity(
    X: np.ndarray | torch.Tensor,
    Y: np.ndarray | torch.Tensor,
    d_star: int = DEFAULT_PROCRUSTES_DIM,
    device: torch.device | str | None = None,
) -> dict[str, float]:
    """Scale-invariant Orthogonal Procrustes similarity after PCA reduction to d* (Guide §5.2)."""
    n = X.shape[0]
    d_eff = min(d_star, max(1, n - 1), X.shape[1], Y.shape[1])
    A, evr_a = pca_reduce(X, d_star=d_eff, device=device)
    B, evr_b = pca_reduce(Y, d_star=d_eff, device=device)

    na = float(np.linalg.norm(A, ord="fro"))
    nb = float(np.linalg.norm(B, ord="fro"))
    if na <= 1e-18 or nb <= 1e-18:
        return {"similarity": float("nan"), "evr_a": evr_a, "evr_b": evr_b, "d_eff": int(d_eff)}

    A_n = A / na
    B_n = B / nb
    M_mat = B_n.T @ A_n
    U, s, Vt = np.linalg.svd(M_mat, full_matrices=False)
    Q = U @ Vt
    diff_sq = float(np.sum((A_n - B_n @ Q) ** 2))
    denom = float(np.sum(A_n ** 2) + np.sum(B_n ** 2))
    sim = float(1.0 - diff_sq / denom)
    return {
        "similarity": float(np.clip(sim, 0.0, 1.0)),
        "evr_a": float(evr_a),
        "evr_b": float(evr_b),
        "d_eff": int(d_eff),
    }


def procrustes_matrix(
    feats_by_model: dict[str, np.ndarray | torch.Tensor],
    model_names: Sequence[str],
    d_star: int = DEFAULT_PROCRUSTES_DIM,
    device: torch.device | str | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute M x M Orthogonal Procrustes similarity matrix + per-model PCA EVR (batched svdvals)."""
    dev = torch.device(device) if device is not None else torch.device("cpu")
    M = len(model_names)
    d_min = min(feats_by_model[m].shape[1] for m in model_names)
    n = feats_by_model[model_names[0]].shape[0]
    d_eff = min(d_star, max(1, n - 1), d_min)

    pca_normed: list[torch.Tensor] = []
    evr_by_model: dict[str, float] = {}
    svd_dtype = torch.float32 if (dev.type == "cuda" or d_eff > 64) else torch.float64
    for m in model_names:
        Z, evr = pca_reduce(feats_by_model[m], d_star=d_eff, device=device)
        Zt = torch.as_tensor(Z, dtype=svd_dtype, device=dev)
        nz = float(torch.linalg.norm(Zt, ord="fro").item())
        pca_normed.append(Zt / nz if nz > 1e-18 else torch.zeros_like(Zt))
        evr_by_model[m] = float(evr)

    mat = np.eye(M, dtype=np.float64)
    if M < 2:
        return mat, evr_by_model

    pairs = [(i, j) for i in range(M) for j in range(i + 1, M)]
    # Stack all M(M-1)/2 cross-covariance matrices (P, d_eff, d_eff) and run ONE batched svdvals call
    cross = torch.stack([pca_normed[j].T @ pca_normed[i] for i, j in pairs], dim=0)
    svals = torch.linalg.svdvals(cross).sum(dim=-1).clamp(0.0, 1.0).cpu().numpy()
    for (i, j), sim in zip(pairs, svals):
        mat[i, j] = mat[j, i] = float(sim)
    return mat, evr_by_model


# --------------------------------------------------------------------------- Estimator 3: Mutual k-NN Overlap
def cosine_topk_neighbors(
    X: np.ndarray | torch.Tensor,
    k: int = DEFAULT_KNN_K,
    chunk_size: int = 1024,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Compute top-k cosine nearest neighbor indices (excluding self) on centered X. Returns (N, k) int32."""
    dev = torch.device(device) if device is not None else torch.device("cpu")
    Xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
    n = Xt.shape[0]
    if n <= 1:
        raise ValueError(f"Mutual k-NN requires N >= 2, got {n}")
    k_eff = min(k, n - 1)
    Xc = Xt - Xt.mean(dim=0, keepdim=True)
    Xn = Xc / torch.clamp(torch.linalg.norm(Xc, dim=1, keepdim=True), min=1e-12)

    out = np.empty((n, k_eff), dtype=np.int32)
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        sims = Xn[start:end] @ Xn.T
        row_ids = torch.arange(end - start, device=dev)
        col_ids = torch.arange(start, end, device=dev)
        sims[row_ids, col_ids] = -float("inf")
        _, top_idx = torch.topk(sims, k=k_eff, dim=1, largest=True, sorted=False)
        out[start:end] = top_idx.cpu().numpy().astype(np.int32)
    return out


def mutual_knn_from_neighbors(nbrs_a: np.ndarray, nbrs_b: np.ndarray) -> float:
    """Compute average mutual k-NN overlap fraction in [0, 1] from (N, k) neighbor arrays."""
    k = nbrs_a.shape[1]
    if k == 0 or nbrs_a.shape[0] == 0:
        return float("nan")
    matches = (nbrs_a[:, :, None] == nbrs_b[:, None, :]).any(axis=2)
    return float(matches.sum(axis=1).mean() / float(k))


def mutual_knn_overlap(
    X: np.ndarray | torch.Tensor,
    Y: np.ndarray | torch.Tensor,
    k: int = DEFAULT_KNN_K,
    device: torch.device | str | None = None,
) -> float:
    """Mutual k-NN overlap (k = 10, cosine metric on centered features, Guide §5.2)."""
    na = cosine_topk_neighbors(X, k=k, device=device)
    nb = cosine_topk_neighbors(Y, k=k, device=device)
    return mutual_knn_from_neighbors(na, nb)


def mutual_knn_matrix(
    feats_by_model: dict[str, np.ndarray | torch.Tensor],
    model_names: Sequence[str],
    k: int = DEFAULT_KNN_K,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Compute M x M mutual k-NN similarity matrix."""
    M = len(model_names)
    nbrs = [cosine_topk_neighbors(feats_by_model[m], k=k, device=device) for m in model_names]
    mat = np.eye(M, dtype=np.float64)
    for i in range(M):
        for j in range(i + 1, M):
            val = mutual_knn_from_neighbors(nbrs[i], nbrs[j])
            mat[i, j] = mat[j, i] = val
    return mat


# --------------------------------------------------------------------------- Fast shared-Gram R3 CKA (C0 & C1)
def _r3_cka_matrix_shared_gram(
    flat_tok_t: list[torch.Tensor],
    flat_out_t: list[torch.Tensor],
    condition: str,
    batch_size: int,
    min_batches: int,
    seed: int,
    dev: torch.device,
) -> np.ndarray:
    """Fast M x M R3 CKA matrix building only M Gram matrices per minibatch (7x-150x faster for C1).

    For C1, each pair (i, j) slices valid = ~(out_i[idx] | out_j[idx]) from the shared batch Gram
    matrices and accumulates HSIC_1(Ki_sub, Kj_sub), HSIC_1(Ki_sub, Ki_sub), HSIC_1(Kj_sub, Kj_sub)
    across batches before dividing. Exact because HSIC_1 is algebraically translation-invariant.
    """
    M = len(flat_tok_t)
    n = flat_tok_t[0].shape[0]
    mat = np.eye(M, dtype=np.float64)
    if n < 4:
        mat[:] = np.nan
        return mat

    if n <= batch_size:
        idx_batches = [torch.arange(n, dtype=torch.long, device=dev)]
    else:
        rng = np.random.default_rng(seed)
        batches_per_pass = n // batch_size
        n_passes = max(1, int(np.ceil(min_batches / max(batches_per_pass, 1))))
        idx_batches = []
        for _ in range(n_passes):
            perm = rng.permutation(n)
            for start in range(0, batches_per_pass * batch_size, batch_size):
                idx_batches.append(torch.as_tensor(perm[start : start + batch_size], dtype=torch.long, device=dev))

    num_acc = np.zeros((M, M), dtype=np.float64)
    dxi_acc = np.zeros((M, M), dtype=np.float64)
    dxj_acc = np.zeros((M, M), dtype=np.float64)
    cnt_acc = np.zeros((M, M), dtype=np.int64)

    dt_g = _matmul_dtype(dev)
    for idx in idx_batches:
        grams = []
        outs_b = []
        for m in range(M):
            Xb = flat_tok_t[m][idx]
            K = Xb @ Xb.T  # dt_g: fp32 on T4 CUDA (32x faster), fp64 on CPU
            K.fill_diagonal_(0.0)
            grams.append(K)
            outs_b.append(flat_out_t[m][idx])

        if condition == "C0":
            nb = int(idx.numel())
            row_sums = [G.sum(dim=1, dtype=torch.float64) for G in grams]
            tot_sums = [rs.sum() for rs in row_sums]
            diag_v = [
                float((((grams[m] * grams[m]).sum(dtype=torch.float64)
                        + (tot_sums[m] * tot_sums[m]) / float((nb - 1) * (nb - 2))
                        - (2.0 / float(nb - 2)) * torch.dot(row_sums[m], row_sums[m]))
                       / float(nb * (nb - 3))).item())
                for m in range(M)
            ]
            for i in range(M):
                for j in range(i + 1, M):
                    num = float((((grams[i] * grams[j]).sum(dtype=torch.float64)
                                  + (tot_sums[i] * tot_sums[j]) / float((nb - 1) * (nb - 2))
                                  - (2.0 / float(nb - 2)) * torch.dot(row_sums[i], row_sums[j]))
                                 / float(nb * (nb - 3))).item())
                    num_acc[i, j] += num
                    dxi_acc[i, j] += diag_v[i]
                    dxj_acc[i, j] += diag_v[j]
                    cnt_acc[i, j] += 1
        else:
            # Zero-allocation masked HSIC_1 using m in {0,1}^B:
            #   sum(Ki_sub * Kj_sub) = m^T (Ki * Kj) m
            #   row_sum(Ki_sub)      = (Ki @ m) * m
            grams_sq = [G * G for G in grams]
            for i in range(M):
                oi = outs_b[i]
                Gi = grams[i]
                Gi2 = grams_sq[i]
                for j in range(i + 1, M):
                    valid = ~(oi | outs_b[j])
                    nb = int(valid.sum().item())
                    if nb < 4:
                        continue
                    m_vec = valid.to(dt_g)
                    Gj = grams[j]
                    Gj2 = grams_sq[j]
                    rsi = ((Gi @ m_vec) * m_vec).to(torch.float64)
                    rsj = ((Gj @ m_vec) * m_vec).to(torch.float64)
                    si = rsi.sum()
                    sj = rsj.sum()
                    c_div = float(nb * (nb - 3))
                    c_t2 = 1.0 / float((nb - 1) * (nb - 2))
                    c_t3 = 2.0 / float(nb - 2)

                    t1_ij = ((Gi * Gj) @ m_vec).to(torch.float64).dot(m_vec.to(torch.float64))
                    t1_ii = (Gi2 @ m_vec).to(torch.float64).dot(m_vec.to(torch.float64))
                    t1_jj = (Gj2 @ m_vec).to(torch.float64).dot(m_vec.to(torch.float64))

                    num_acc[i, j] += float(((t1_ij + si * sj * c_t2 - c_t3 * torch.dot(rsi, rsj)) / c_div).item())
                    dxi_acc[i, j] += float(((t1_ii + si * si * c_t2 - c_t3 * torch.dot(rsi, rsi)) / c_div).item())
                    dxj_acc[i, j] += float(((t1_jj + sj * sj * c_t2 - c_t3 * torch.dot(rsj, rsj)) / c_div).item())
                    cnt_acc[i, j] += 1

    for i in range(M):
        for j in range(i + 1, M):
            c = cnt_acc[i, j]
            if c > 0 and dxi_acc[i, j] > 1e-18 and dxj_acc[i, j] > 1e-18:
                val = float((num_acc[i, j] / c) / np.sqrt((dxi_acc[i, j] / c) * (dxj_acc[i, j] / c)))
            else:
                val = float("nan")
            mat[i, j] = mat[j, i] = val
    return mat


def r3_pairwise_matrices(
    r3_tokens_by_model: dict[str, np.ndarray],
    r3_out_by_model: dict[str, np.ndarray],
    model_names: Sequence[str],
    condition: str,
    batch_size: int = DEFAULT_MINIBATCH_SIZE,
    min_batches: int = DEFAULT_MIN_BATCHES,
    d_star: int = DEFAULT_PROCRUSTES_DIM,
    knn_k: int = DEFAULT_KNN_K,
    seed: int = 0,
    device: torch.device | str | None = None,
    metrics: Sequence[str] = ("cka", "procrustes", "mknn"),
) -> dict[str, np.ndarray]:
    """Compute M x M matrices on R3 patch tokens (shape (I, N', d)) under C0 or C1."""
    dev = torch.device(device) if device is not None else torch.device("cpu")
    dt = _matmul_dtype(dev)
    M = len(model_names)

    if condition not in ("C0", "C1"):
        raise ValueError(f"Unknown condition {condition!r}; expected 'C0' or 'C1'")

    out: dict[str, np.ndarray] = {}
    flat_tok = {m: r3_tokens_by_model[m].reshape(-1, r3_tokens_by_model[m].shape[-1]) for m in model_names}
    flat_out = {m: r3_out_by_model[m].reshape(-1) for m in model_names}

    # Fast shared-Gram path for CKA (both C0 and C1)
    if "cka" in metrics:
        tok_t, out_t = [], []
        for m in model_names:
            t = torch.as_tensor(flat_tok[m], dtype=dt, device=dev)
            om = torch.as_tensor(flat_out[m], dtype=torch.bool, device=dev)
            if condition == "C1" and (~om).any():
                t = t - t[~om].mean(dim=0, keepdim=True)
            else:
                t = t - t.mean(dim=0, keepdim=True)
            tok_t.append(t)
            out_t.append(om)
        out["cka"] = _r3_cka_matrix_shared_gram(tok_t, out_t, condition, batch_size, min_batches, seed, dev)

    other_metrics = [m for m in metrics if m != "cka"]
    if not other_metrics:
        return out

    if condition == "C0":
        if "procrustes" in other_metrics:
            pmat, _ = procrustes_matrix(flat_tok, model_names, d_star=d_star, device=device)
            out["procrustes"] = pmat
        if "mknn" in other_metrics:
            out["mknn"] = mutual_knn_matrix(flat_tok, model_names, k=knn_k, device=device)
        return out

    for k in other_metrics:
        out[k] = np.eye(M, dtype=np.float64)
    for i in range(M):
        mi = model_names[i]
        for j in range(i + 1, M):
            mj = model_names[j]
            valid = ~(flat_out[mi] | flat_out[mj])
            Xi = flat_tok[mi][valid]
            Xj = flat_tok[mj][valid]
            if "procrustes" in other_metrics:
                val = orthogonal_procrustes_similarity(Xi, Xj, d_star=d_star, device=device)["similarity"]
                out["procrustes"][i, j] = out["procrustes"][j, i] = val
            if "mknn" in other_metrics:
                val = mutual_knn_overlap(Xi, Xj, k=knn_k, device=device)
                out["mknn"][i, j] = out["mknn"][j, i] = val
    return out


# --------------------------------------------------------------------------- Rank stability, tau_noise & Gate G1
def upper_tri_vector(mat: np.ndarray) -> np.ndarray:
    """Return strict upper-triangular entries (M*(M-1)/2 pairs) in deterministic row-major order."""
    iu = np.triu_indices(mat.shape[0], k=1)
    return np.asarray(mat[iu], dtype=np.float64)


def kendall_tau_b_matrices(mat_a: np.ndarray, mat_b: np.ndarray) -> tuple[float, float]:
    """Kendall's tau-b between upper-triangular pairwise similarity vectors."""
    va = upper_tri_vector(mat_a)
    vb = upper_tri_vector(mat_b)
    mask = np.isfinite(va) & np.isfinite(vb)
    if mask.sum() < 2:
        return float("nan"), float("nan")
    res = kendalltau(va[mask], vb[mask], variant="b")
    return float(res.statistic), float(res.pvalue)


def split_half_tau_test(
    c0_by_model: dict[str, np.ndarray],
    c1_by_model: dict[str, np.ndarray],
    model_names: Sequence[str],
    metric: str = "cka",
    n_splits: int = 100,
    batch_size: int = DEFAULT_MINIBATCH_SIZE,
    min_batches: int = DEFAULT_MIN_BATCHES,
    d_star: int = DEFAULT_PROCRUSTES_DIM,
    knn_k: int = DEFAULT_KNN_K,
    seed: int = 0,
    device: torch.device | str | None = None,
    r3_out_by_model: dict[str, np.ndarray] | None = None,
) -> dict:
    """Split-half noise floor tau_noise and paired bootstrap test for tau(C0, C1) < tau_noise (Guide §5.4 & §11.2)."""
    dev = torch.device(device) if device is not None else torch.device("cpu")
    dt = _matmul_dtype(dev)
    is_r3 = r3_out_by_model is not None
    M = len(model_names)
    n_images = c0_by_model[model_names[0]].shape[0]
    half = n_images // 2
    if half < 4:
        raise ValueError(f"Need at least 8 images for split-half test, got {n_images}")

    # Fast pre-cached full Gram matrices for R2 CKA when N <= 2048 and half <= batch_size
    fast_r2_cka_grams = (not is_r3) and (metric == "cka") and (half <= batch_size) and (n_images <= 2048)
    grams_c0_full, grams_c1_full = [], []
    if fast_r2_cka_grams:
        for m in model_names:
            X0 = torch.as_tensor(c0_by_model[m], dtype=dt, device=dev)
            X0 = X0 - X0.mean(dim=0, keepdim=True)
            G0 = (X0 @ X0.T).to(torch.float64)
            G0 = G0 - torch.diag_embed(torch.diagonal(G0))
            grams_c0_full.append(G0)

            X1 = torch.as_tensor(c1_by_model[m], dtype=dt, device=dev)
            X1 = X1 - X1.mean(dim=0, keepdim=True)
            G1 = (X1 @ X1.T).to(torch.float64)
            G1 = G1 - torch.diag_embed(torch.diagonal(G1))
            grams_c1_full.append(G1)

    def _cka_from_cached_grams(gram_list: list[torch.Tensor], img_idx: np.ndarray) -> np.ndarray:
        idx_t = torch.as_tensor(img_idx, dtype=torch.long, device=dev)
        nb = int(idx_t.numel())
        sub_g, rsums, tsums, diag_v = [], [], [], []
        for m in range(M):
            Gm = gram_list[m].index_select(0, idx_t).index_select(1, idx_t)
            rs = Gm.sum(dim=1)
            ts = rs.sum()
            sub_g.append(Gm)
            rsums.append(rs)
            tsums.append(ts)
            diag_v.append(_gram_pair_hsic(Gm, Gm, rs, rs, ts, ts, nb))
        mat = np.eye(M, dtype=np.float64)
        for i in range(M):
            for j in range(i + 1, M):
                if diag_v[i] > 1e-18 and diag_v[j] > 1e-18:
                    num = _gram_pair_hsic(sub_g[i], sub_g[j], rsums[i], rsums[j], tsums[i], tsums[j], nb)
                    mat[i, j] = mat[j, i] = float(num / np.sqrt(diag_v[i] * diag_v[j]))
                else:
                    mat[i, j] = mat[j, i] = float("nan")
        return mat

    # Pre-upload R3 tensors to GPU once for fast split-half R3 CKA
    r3_tok_gpu, r3_out_gpu = None, None
    if is_r3 and metric == "cka":
        r3_tok_gpu = [torch.as_tensor(c0_by_model[m], dtype=dt, device=dev) for m in model_names]  # (I, N', d)
        r3_out_gpu = [torch.as_tensor(r3_out_by_model[m], dtype=torch.bool, device=dev) for m in model_names]  # (I, N')

    def _compute_mat(img_idx: np.ndarray, cond: str, split_seed: int, mb_override: int | None = None) -> np.ndarray:
        mb_eff = min_batches if mb_override is None else mb_override
        if fast_r2_cka_grams and len(img_idx) <= batch_size:
            return _cka_from_cached_grams(grams_c0_full if cond == "C0" else grams_c1_full, img_idx)
        if is_r3:
            if metric == "cka" and r3_tok_gpu is not None and r3_out_gpu is not None:
                idx_t = torch.as_tensor(img_idx, dtype=torch.long, device=dev)
                tok_flat, out_flat = [], []
                for m in range(M):
                    t = r3_tok_gpu[m].index_select(0, idx_t).reshape(-1, r3_tok_gpu[m].shape[-1])
                    om = r3_out_gpu[m].index_select(0, idx_t).reshape(-1)
                    if cond == "C1" and (~om).any():
                        t = t - t[~om].mean(dim=0, keepdim=True)
                    else:
                        t = t - t.mean(dim=0, keepdim=True)
                    tok_flat.append(t)
                    out_flat.append(om)
                return _r3_cka_matrix_shared_gram(tok_flat, out_flat, cond, batch_size, mb_eff, split_seed, dev)
            sub_tok = {m: c0_by_model[m][img_idx] for m in model_names}
            sub_out = {m: r3_out_by_model[m][img_idx] for m in model_names}
            return r3_pairwise_matrices(
                sub_tok, sub_out, model_names, condition=cond,
                batch_size=batch_size, min_batches=mb_eff,
                d_star=d_star, knn_k=knn_k, seed=split_seed, device=device, metrics=(metric,),
            )[metric]
        src = c0_by_model if cond == "C0" else c1_by_model
        sub = {m: src[m][img_idx] for m in model_names}
        if metric == "cka":
            return debiased_cka_matrix(sub, model_names, batch_size=batch_size, min_batches=mb_eff, seed=split_seed, device=device)
        if metric == "procrustes":
            pmat, _ = procrustes_matrix(sub, model_names, d_star=d_star, device=device)
            return pmat
        if metric == "mknn":
            return mutual_knn_matrix(sub, model_names, k=knn_k, device=device)
        raise ValueError(f"Unknown metric {metric!r}")

    all_idx = np.arange(n_images)
    mat_c0_full = _compute_mat(all_idx, "C0", seed, mb_override=min_batches)
    mat_c1_full = _compute_mat(all_idx, "C1", seed, mb_override=min_batches)
    tau_full, p_full = kendall_tau_b_matrices(mat_c0_full, mat_c1_full)

    split_mb = min(min_batches, 12) if is_r3 else min_batches
    rng = np.random.default_rng(seed + 1000)
    tau_noise_arr = np.empty(n_splits, dtype=np.float64)
    tau_same_arr = np.empty(n_splits, dtype=np.float64)
    tau_cross_arr = np.empty(n_splits, dtype=np.float64)

    for b in range(n_splits):
        perm = rng.permutation(n_images)
        h1 = perm[:half]
        h2 = perm[half : 2 * half]
        m_c0_h1 = _compute_mat(h1, "C0", seed + b * 3 + 1, mb_override=split_mb)
        m_c0_h2 = _compute_mat(h2, "C0", seed + b * 3 + 2, mb_override=split_mb)
        m_c1_h1 = _compute_mat(h1, "C1", seed + b * 3 + 1, mb_override=split_mb)
        m_c1_h2 = _compute_mat(h2, "C1", seed + b * 3 + 2, mb_override=split_mb)

        tau_noise_arr[b], _ = kendall_tau_b_matrices(m_c0_h1, m_c0_h2)
        tau_same_arr[b], _ = kendall_tau_b_matrices(m_c0_h1, m_c1_h1)
        tau_cross_arr[b], _ = kendall_tau_b_matrices(m_c0_h1, m_c1_h2)

    def _summ(v: np.ndarray) -> dict:
        vf = v[np.isfinite(v)]
        if vf.size == 0:
            return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
        lo, hi = np.percentile(vf, [2.5, 97.5])
        return {"mean": float(vf.mean()), "ci95_low": float(lo), "ci95_high": float(hi)}

    diff_same = tau_same_arr - tau_noise_arr
    diff_cross = tau_cross_arr - tau_noise_arr
    diff_full = tau_full - tau_noise_arr

    s_same = _summ(diff_same)
    s_cross = _summ(diff_cross)
    s_full = _summ(diff_full)

    return {
        "metric": metric,
        "n_images": int(n_images),
        "half_size": int(half),
        "n_splits": int(n_splits),
        "n_pairs": int(len(model_names) * (len(model_names) - 1) // 2),
        "tau_c0_c1_full": float(tau_full),
        "tau_c0_c1_full_p": float(p_full),
        "tau_noise": _summ(tau_noise_arr),
        "tau_same_half": _summ(tau_same_arr),
        "tau_cross_half": _summ(tau_cross_arr),
        "diff_same_minus_noise": {**s_same, "below_noise_ci_excl0": bool(s_same["ci95_high"] < 0.0)},
        "diff_cross_minus_noise": {**s_cross, "below_noise_ci_excl0": bool(s_cross["ci95_high"] < 0.0)},
        "diff_full_minus_noise": {**s_full, "below_noise_ci_excl0": bool(s_full["ci95_high"] < 0.0)},
        "mat_c0_full": mat_c0_full,
        "mat_c1_full": mat_c1_full,
    }


# --------------------------------------------------------------------------- Grounding (§5.5): Otsu Map IoU
def otsu_threshold(values: np.ndarray, n_bins: int = 256) -> float:
    """Standard Otsu's threshold in pure NumPy on a 1D/2D float array."""
    v = np.asarray(values, dtype=np.float64).ravel()
    vmin, vmax = float(v.min()), float(v.max())
    if vmax - vmin <= 1e-12:
        return vmax
    hist, bin_edges = np.histogram(v, bins=n_bins, range=(vmin, vmax))
    hist = hist.astype(np.float64)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    w1 = np.cumsum(hist)
    w2 = np.cumsum(hist[::-1])[::-1]
    mu1 = np.cumsum(hist * bin_centers) / np.maximum(w1, 1e-12)
    mu2 = (np.cumsum((hist * bin_centers)[::-1]) / np.maximum(w2[::-1], 1e-12))[::-1]
    between = w1[:-1] * w2[1:] * (mu1[:-1] - mu2[1:]) ** 2
    idx = int(np.argmax(between))
    return float(bin_centers[idx])


def binarize_maps_otsu(maps: np.ndarray) -> np.ndarray:
    """Binarize (I, H, W) or (I, N) saliency maps per image using Otsu's threshold."""
    arr = np.asarray(maps, dtype=np.float64)
    flat = arr.reshape(arr.shape[0], -1)
    masks = np.empty_like(flat, dtype=bool)
    for i in range(flat.shape[0]):
        thr = otsu_threshold(flat[i])
        masks[i] = flat[i] >= thr
    return masks.reshape(arr.shape)


def map_iou_matrix(maps_by_model: dict[str, np.ndarray], model_names: Sequence[str]) -> np.ndarray:
    """Compute M x M pairwise mean Otsu-mask IoU over images (Guide §5.5)."""
    M = len(model_names)
    masks = [binarize_maps_otsu(maps_by_model[m]).reshape(maps_by_model[m].shape[0], -1) for m in model_names]
    mat = np.eye(M, dtype=np.float64)
    for i in range(M):
        mi = masks[i]
        for j in range(i + 1, M):
            mj = masks[j]
            inter = np.logical_and(mi, mj).sum(axis=1).astype(np.float64)
            union = np.logical_or(mi, mj).sum(axis=1).astype(np.float64)
            iou = np.where(union > 0, inter / np.maximum(union, 1.0), 1.0)
            val = float(iou.mean())
            mat[i, j] = mat[j, i] = val
    return mat


def correlate_similarity_with_map_iou(sim_mat: np.ndarray, iou_mat: np.ndarray) -> dict[str, float]:
    """Spearman correlation across M(M-1)/2 model pairs between similarity and Otsu map IoU (§5.5)."""
    vs = upper_tri_vector(sim_mat)
    vi = upper_tri_vector(iou_mat)
    mask = np.isfinite(vs) & np.isfinite(vi)
    if mask.sum() < 3:
        return {"spearman_rho": float("nan"), "p_value": float("nan"), "n_pairs": int(mask.sum())}
    res = spearmanr(vs[mask], vi[mask])
    return {"spearman_rho": float(res.statistic), "p_value": float(res.pvalue), "n_pairs": int(mask.sum())}


# --------------------------------------------------------------------------- Calibration floor (§5.6)
def shuffled_floor_calibration(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int = DEFAULT_MINIBATCH_SIZE,
    min_batches: int = DEFAULT_MIN_BATCHES,
    d_star: int = DEFAULT_PROCRUSTES_DIM,
    knn_k: int = DEFAULT_KNN_K,
    seed: int = 42,
    device: torch.device | str | None = None,
) -> dict[str, float]:
    """Compute CKA, Procrustes, and mutual k-NN on randomly row-shuffled Y as the null floor (§5.6)."""
    rng = np.random.default_rng(seed)
    Y_shuf = np.asarray(Y)[rng.permutation(Y.shape[0])]
    cka = debiased_linear_cka(X, Y_shuf, batch_size=batch_size, min_batches=min_batches, seed=seed, device=device)
    proc = orthogonal_procrustes_similarity(X, Y_shuf, d_star=d_star, device=device)["similarity"]
    mknn = mutual_knn_overlap(X, Y_shuf, k=knn_k, device=device)
    return {"cka": float(cka), "procrustes": float(proc), "mknn": float(mknn)}


# --------------------------------------------------------------------------- Matched Random-Drop Control (H3 causality)
def make_matched_random_out_mask(out_mask: np.ndarray, seed: int = 0) -> np.ndarray:
    """Create a boolean mask dropping the exact same count of patches per image at random positions."""
    om = np.asarray(out_mask, dtype=bool)
    I, N = om.shape
    rng = np.random.default_rng(seed)
    rand_mask = np.zeros_like(om, dtype=bool)
    counts = om.sum(axis=1)
    for i in range(I):
        k = int(counts[i])
        if k >= N:
            rand_mask[i, :] = True
        elif k > 0:
            rand_mask[i, rng.choice(N, size=k, replace=False)] = True
    return rand_mask


def r3_matched_random_drop_control(
    r3_tokens_by_model: dict[str, np.ndarray],
    r3_out_by_model: dict[str, np.ndarray],
    model_names: Sequence[str],
    *,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    batch_size: int = DEFAULT_MINIBATCH_SIZE,
    min_batches: int = DEFAULT_MIN_BATCHES,
    seed: int = 0,
    device: torch.device | str | None = None,
) -> dict:
    """Compare C0 vs C1(OUT-masked) against C0 vs C1_rand(matched-count random-dropped) on R3 CKA (5 seeds)."""
    mat_c0 = r3_pairwise_matrices(
        r3_tokens_by_model, r3_out_by_model, model_names, condition="C0",
        batch_size=batch_size, min_batches=min_batches, seed=seed, device=device, metrics=("cka",),
    )["cka"]
    mat_c1_out = r3_pairwise_matrices(
        r3_tokens_by_model, r3_out_by_model, model_names, condition="C1",
        batch_size=batch_size, min_batches=min_batches, seed=seed, device=device, metrics=("cka",),
    )["cka"]
    tau_out, _ = kendall_tau_b_matrices(mat_c0, mat_c1_out)
    max_shift_out = float(np.nanmax(np.abs(mat_c0 - mat_c1_out)))

    rand_taus, rand_shifts = [], []
    for s in seeds:
        rand_out = {m: make_matched_random_out_mask(r3_out_by_model[m], seed=100 * s + idx) for idx, m in enumerate(model_names)}
        mat_c1_rand = r3_pairwise_matrices(
            r3_tokens_by_model, rand_out, model_names, condition="C1",
            batch_size=batch_size, min_batches=min_batches, seed=seed, device=device, metrics=("cka",),
        )["cka"]
        tr, _ = kendall_tau_b_matrices(mat_c0, mat_c1_rand)
        rand_taus.append(float(tr))
        rand_shifts.append(float(np.nanmax(np.abs(mat_c0 - mat_c1_rand))))

    return {
        "n_seeds": len(seeds),
        "tau_c0_c1_out": float(tau_out),
        "max_abs_shift_out": float(max_shift_out),
        "tau_c0_c1_rand_mean": float(np.mean(rand_taus)),
        "tau_c0_c1_rand_std": float(np.std(rand_taus)),
        "tau_c0_c1_rand_per_seed": rand_taus,
        "max_abs_shift_rand_mean": float(np.mean(rand_shifts)),
        "delta_tau_out_minus_rand": float(tau_out - np.mean(rand_taus)),
    }
