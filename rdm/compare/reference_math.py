"""Offline reference-side math shared by the Nystrom reference builders.

The data side of the iRDM objective never moves, so it is compressed once over the
full training set and frozen. This module holds the pieces that compression needs:

* :func:`kmeans_landmarks` -- ``m`` landmarks placed by k-means on the data features;
* :func:`streaming_reference_mean` -- the per-landmark mean embedding
  ``mu_Zr = mean_j k(Z, r_j)`` accumulated in chunks over the full pool;
* :func:`reference_self_kernel_mean` -- the constant ``k_rr = E[k(r, r')]``;
* :func:`build_nystrom_reference` -- the full per-encoder build returning the frozen
  ``{Z, alpha, sigma, k_rr}`` bundle that training consumes.

``alpha = K_ZZ^{-1} mu_Zr`` is solved once (Cholesky) so the train-time attraction is
``k_gr ~= mean_i k(g_i, Z) . alpha`` -- the algebraically-equivalent precomputed form
of the paper's ``psi(g_i)^T mu_bar`` with the eigh feature map (see
:mod:`rdm.compare.nystrom`). All reference math is done in fp64 for accuracy and the
bundle is stored fp32.
"""
import torch

from .kernels import gamma_from_sigma


@torch.no_grad()
def kmeans_landmarks(X: torch.Tensor, n_landmarks: int, iters: int = 20,
                     seed: int = 0, chunk: int = 16384) -> torch.Tensor:
    """``n_landmarks`` k-means centroids over feature pool ``X`` (random init, fp of X).

    Chunked nearest-centroid assignment and ``index_add_`` centroid update; empty
    clusters are reseeded from random data points each iteration.
    """
    dev = X.device
    n = X.shape[0]
    gen = torch.Generator(device=dev).manual_seed(seed)
    C = X[torch.randperm(n, generator=gen, device=dev)[:n_landmarks]].clone()
    for _ in range(iters):
        labels = torch.empty(n, dtype=torch.long, device=dev)
        C2 = (C * C).sum(1)
        for lo in range(0, n, chunk):
            x = X[lo:lo + chunk]
            d2 = (x * x).sum(1)[:, None] + C2[None, :] - 2.0 * (x @ C.T)
            labels[lo:lo + chunk] = d2.argmin(1)
        newC = torch.zeros_like(C)
        cnt = torch.zeros(n_landmarks, device=dev)
        newC.index_add_(0, labels, X)
        cnt.index_add_(0, labels, torch.ones(n, device=dev))
        empty = cnt == 0
        newC = newC / cnt.clamp_min(1.0)[:, None]
        if empty.any():
            ridx = torch.randperm(n, generator=gen, device=dev)[:int(empty.sum())]
            newC[empty] = X[ridx]
        C = newC
    return C


@torch.no_grad()
def streaming_reference_mean(Z: torch.Tensor, Z2: torch.Tensor, pool: torch.Tensor,
                             gamma: float, chunk: int = 50000) -> torch.Tensor:
    """Per-landmark reference mean ``mu_Zr[k] = mean_j k(Z_k, r_j)`` over the full pool.

    Accumulated in row chunks (fp64). ``Z2 = (Z*Z).sum(1)``.
    """
    M = Z.shape[0]
    mu = torch.zeros(M, dtype=torch.float64, device=Z.device)
    n = pool.shape[0]
    for lo in range(0, n, chunk):
        r = pool[lo:lo + chunk].double()
        r2 = (r * r).sum(1)
        d2 = (Z2[:, None] + r2[None, :] - 2.0 * (Z @ r.T)).clamp_min(0)
        mu += torch.exp(-gamma * d2).sum(1)
    return mu / n


@torch.no_grad()
def reference_self_kernel_mean(pool: torch.Tensor, sigma: float, n_sub: int = 100000,
                               a_chunk: int = 4096, b_chunk: int = 50000) -> float:
    """Constant ``k_rr = E[k(r, r')]`` over a subsample (double-chunked, fp same as pool).

    Kept as a detached constant in the loss so the value stays ``>= 0`` (the correct
    sign for the downstream self-normalization). A subsample of the full pool estimates
    this scalar to ~1e-5; double-chunking bounds peak memory to ``O(a_chunk * b_chunk)``.
    """
    r = pool[:min(n_sub, pool.shape[0])].float()
    gamma = gamma_from_sigma(sigma)
    r2 = (r * r).sum(1)
    n = r.shape[0]
    tot = r.new_zeros(())
    for alo in range(0, n, a_chunk):
        a = r[alo:alo + a_chunk]
        a2 = (a * a).sum(1)
        for blo in range(0, n, b_chunk):
            bb = r[blo:blo + b_chunk]
            bb2 = r2[blo:blo + b_chunk]
            d2 = (a2[:, None] + bb2[None, :] - 2.0 * (a @ bb.T)).clamp_min(0)
            tot = tot + torch.exp(-gamma * d2).sum()
    return float(tot / (n * n))


@torch.no_grad()
def build_nystrom_reference(pool: torch.Tensor, sigma: float, n_landmarks: int = 4096,
                            fit_n: int = 200000, kmeans_iters: int = 20,
                            krr_n: int = 100000, seed: int = 0, jitter: float = 1e-6) -> dict:
    """Build the frozen per-encoder Nystrom reference bundle for one feature pool.

    Steps: k-means landmarks ``Z`` (on a ``fit_n`` subsample) -> ``K_ZZ + jitter*I`` ->
    Cholesky -> streaming ``mu_Zr`` over the *full* pool -> ``alpha = K_ZZ^{-1} mu_Zr``
    -> constant ``k_rr``. Landmarks and solve are fp64; the bundle is stored fp32.

    Returns a dict ``{Z, alpha, sigma, k_rr, M, d_in}`` ready for
    :func:`rdm.compare.mmd_nystrom.mmd_nystrom_loss` (which recomputes ``Z2`` at load).
    """
    Y = pool.cuda().float() if torch.cuda.is_available() else pool.float()
    n, d_in = Y.shape
    gamma = gamma_from_sigma(sigma)
    fit = Y[torch.randperm(n, generator=torch.Generator(device=Y.device).manual_seed(seed),
                           device=Y.device)[:min(fit_n, n)]]
    Z = kmeans_landmarks(fit, n_landmarks, kmeans_iters, seed).double()
    Z2 = (Z * Z).sum(1)
    dzz = (Z2[:, None] + Z2[None, :] - 2.0 * (Z @ Z.T)).clamp_min(0)
    K_ZZ = torch.exp(-gamma * dzz) + jitter * torch.eye(n_landmarks, dtype=torch.float64, device=Y.device)
    L = torch.linalg.cholesky(K_ZZ)
    mu_Zr = streaming_reference_mean(Z, Z2, Y, gamma)
    alpha = torch.cholesky_solve(mu_Zr[:, None], L).squeeze(1)
    k_rr = reference_self_kernel_mean(Y, sigma, n_sub=krr_n)
    return dict(Z=Z.float().cpu(), alpha=alpha.float().cpu(), sigma=float(sigma),
                k_rr=float(k_rr), M=int(n_landmarks), d_in=int(d_in))
