"""Correctness gates for the comparison axis (rdm.compare).

Pure-math checks, no GPU / no weights:
  * Nystrom attraction (precomputed-alpha) == the explicit eigh psi^T mu_bar form;
  * mmd_nystrom_loss == k_gg - 2 k_gr + k_rr assembled by hand;
  * self_kernel_mean is the BIASED estimator (includes the diagonal) and chunk-invariant;
  * sliced_wasserstein(X, X) == 0 and sw_loss >= 0;
  * frechet_loss(real-fit) ~ 0; rff_phi shape; grad flows to feat_g.
"""
import torch

from rdm.compare import (build_nystrom_reference, gamma_from_sigma, mmd_nystrom_loss,
                         nystrom_attraction, nystrom_feature_map, rff_phi, self_kernel_mean,
                         sliced_wasserstein, sw_loss, frechet_loss)
from rdm.compare.kernels import cross_kernel_mean

torch.manual_seed(0)


def _toy_reference(n=4000, d=16, m=64, sigma=1.5):
    pool = torch.randn(n, d).double()
    bundle = build_nystrom_reference(pool, sigma, n_landmarks=m, fit_n=n, kmeans_iters=5, krr_n=n)
    return pool, bundle


def test_nystrom_equiv_eigh_psi():
    """Precomputed-alpha attraction == explicit K^{-1/2} psi feature-map form (paper eq. 3)."""
    pool, b = _toy_reference()
    Z = b["Z"].double()
    Z2 = (Z * Z).sum(1)
    alpha = b["alpha"].double()
    gamma = gamma_from_sigma(b["sigma"])
    g = torch.randn(128, Z.shape[1]).double()

    k_gr_alpha = nystrom_attraction(g, Z, Z2, alpha, gamma)
    # explicit psi^T mu_bar with mu_bar = mean_t psi(r_t)
    psi = nystrom_feature_map(Z, gamma)
    mu_bar = psi(pool).mean(0)
    k_gr_psi = (psi(g) @ mu_bar).mean()
    assert torch.allclose(k_gr_alpha, k_gr_psi, atol=1e-6, rtol=1e-5), (k_gr_alpha, k_gr_psi)


def test_mmd_nystrom_assembled():
    """The iRDM loss == k_gg - 2 k_gr + k_rr put together by hand."""
    pool, b = _toy_reference()
    # production bundles are fp32 and the loss casts feat_g to fp32 -> keep everything fp32
    Z = b["Z"]; Z2 = (Z * Z).sum(1); alpha = b["alpha"]
    gamma = gamma_from_sigma(b["sigma"])
    g = torch.randn(200, Z.shape[1]).requires_grad_(True)

    loss = mmd_nystrom_loss(g, Z, Z2, alpha, b["sigma"], b["k_rr"])
    k_gg = self_kernel_mean(g.detach(), gamma)
    k_gr = nystrom_attraction(g.detach(), Z, Z2, alpha, gamma)
    assert torch.allclose(loss.detach(), k_gg - 2 * k_gr + b["k_rr"], atol=1e-6)
    loss.backward()
    assert g.grad is not None and torch.isfinite(g.grad).all()


def test_self_kernel_mean_biased_and_chunk_invariant():
    """Biased: includes diagonal so >= 1/N; and chunking is bit-stable."""
    g = torch.randn(300, 8).double()
    gamma = 0.5
    full = self_kernel_mean(g, gamma, chunk=10_000)
    chunked = self_kernel_mean(g, gamma, chunk=37)
    assert torch.allclose(full, chunked, atol=1e-10)
    # diagonal (i==i, k=1) contributes N/N^2 = 1/N, so the biased mean >= 1/N
    assert full.item() >= 1.0 / g.shape[0] - 1e-9


def test_exact_cross_mean_matches_naive():
    g = torch.randn(64, 8).double(); r = torch.randn(50, 8).double()
    gamma = 0.7
    naive = torch.exp(-gamma * torch.cdist(g, r).pow(2)).mean()
    assert torch.allclose(cross_kernel_mean(g, r, gamma), naive, atol=1e-10)


def test_sliced_wasserstein_self_zero():
    x = torch.randn(512, 12)
    assert sliced_wasserstein(x, x.clone(), n_proj=256, seed=1) < 1e-5
    g = torch.randn(128, 12).requires_grad_(True)
    val = sw_loss(g, torch.randn(128, 12), n_proj=64,
                  generator=torch.Generator().manual_seed(0))
    assert val.item() >= 0
    val.backward(); assert g.grad is not None


def test_frechet_zero_at_match():
    d = 8
    real = torch.randn(5000, d)  # fp32: frechet_loss casts feat_g to fp32, refs must match
    mu_r = real.mean(0)
    rc = real - mu_r
    Sig = (rc.T @ rc) / (real.shape[0] - 1)
    ev, U = torch.linalg.eigh(Sig)
    Sr_half = (U * ev.clamp_min(0).sqrt()) @ U.T
    fd = frechet_loss(real, mu_r, Sr_half, float(torch.diagonal(Sig).sum()))
    assert fd.abs().item() < 1e-2  # generated == real fit -> FD ~ 0


def test_rff_phi_shape():
    x = torch.randn(10, 16); W = torch.randn(16, 64); b = torch.rand(64) * 6.28
    assert rff_phi(x, W, b).shape == (10, 64)
