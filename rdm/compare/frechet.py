"""Frechet (Gaussian 2-Wasserstein) distance: the ``fd`` moment-matching ablation.

The Frechet distance keeps only the first two moments of each side:

    FD^2 = ||mu_g - mu_r||^2 + Tr(Sigma_g) + Tr(Sigma_r)
           - 2 Tr( (Sigma_r^{1/2} Sigma_g Sigma_r^{1/2})^{1/2} )

the squared 2-Wasserstein between the fitted Gaussians ``N(mu_g, Sigma_g)`` and
``N(mu_r, Sigma_r)`` -- the moment-matching alternative to the kernel-MMD / OT losses.
In the Table-4 ablation it sits below the kernel-MMD estimators: two moments cannot
encode a manifold, so matching can saturate while samples stay flawed.

The cross trace is computed stably and differentiably **without** a non-symmetric matrix
square root. With the frozen real half-covariance ``Sigma_r^{1/2}``, the matrix
``M = Sigma_r^{1/2} Sigma_g Sigma_r^{1/2}`` is symmetric PSD and shares the nonzero
spectrum of ``Sigma_g Sigma_r``, so ``Tr((Sigma_g Sigma_r)^{1/2}) = sum_k sqrt(lambda_k(M))``
via a single ``torch.linalg.eigvalsh`` (autograd-supported). Only ``Sigma_g`` carries
gradient. (This is the differentiable training loss; the non-differentiable
evaluation FID over precomputed mean/cov lives in :mod:`rdm.eval`.)
"""
import torch


def frechet_loss(feat_g: torch.Tensor, mu_r: torch.Tensor, Sr_half: torch.Tensor,
                 tr_Sr: float, eps: float = 1e-6) -> torch.Tensor:
    """Differentiable Frechet distance FD^2 between ``N(mu_g, Sigma_g)`` and the frozen real.

    Args:
        feat_g: ``(N_g, d)`` generated features (carry grad).
        mu_r: ``(d,)`` frozen real mean; Sr_half: ``(d, d)`` frozen ``Sigma_r^{1/2}``;
            tr_Sr: precomputed ``Tr(Sigma_r)``.
        eps: diagonal jitter on ``Sigma_g`` for numerical positive-definiteness (it also
            bounds the ``eigvalsh`` gradient near a degenerate spectrum).
    """
    g = feat_g.float()
    n, d = g.shape
    mu_g = g.mean(0)
    gc = g - mu_g
    Sigma_g = (gc.T @ gc) / (n - 1) + eps * torch.eye(d, device=g.device, dtype=g.dtype)
    M = Sr_half @ Sigma_g @ Sr_half
    M = 0.5 * (M + M.T)                                   # symmetrize away fp asymmetry
    tr_sqrt = torch.linalg.eigvalsh(M).clamp_min(0).sqrt().sum()
    mean_term = (mu_g - mu_r).pow(2).sum()
    tr_g = torch.diagonal(Sigma_g).sum()
    return mean_term + tr_g + tr_Sr - 2.0 * tr_sqrt
