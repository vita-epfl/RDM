"""The iRDM training loss: biased MMD^2 with an exact repulsion and a Nystrom attraction.

This is the operative objective of the paper (eq. 3). For a generated batch of features
``g_i = phi(g_theta(z_i))`` under a frozen encoder ``phi``,

    L_phi = (1/B^2) sum_{i,i'} k(g_i, g_i')          # repulsion, exact (within-batch)
            - 2 * mean_i k(g_i, Z) . alpha           # attraction, Nystrom (frozen reference)
            + k_rr                                    # constant data term (dropped from grad)

The repulsion is the **biased** within-batch estimator (diagonal included, divided by
``B^2`` -- see :func:`rdm.compare.kernels.self_kernel_mean`). The attraction is the
frozen Nystrom kernel mean embedding toward the full-data reference
(:func:`rdm.compare.nystrom.nystrom_attraction`). ``k_rr = E[k(r, r')]`` is a precomputed
constant: it sets the loss value (for the downstream self-normalization) but carries no
gradient, since the data term of eq. (2) is constant in ``theta``.

``self_kernel_mean`` is chunkable via ``gen_chunk`` so the gradient-caching optimizer
(:mod:`rdm.train.grad_caching`) can take the full-batch gradient at one-chunk memory.
"""
import torch

from .kernels import gamma_from_sigma, self_kernel_mean
from .nystrom import nystrom_attraction


def mmd_nystrom_loss(feat_g: torch.Tensor, Z: torch.Tensor, Z2: torch.Tensor,
                     alpha: torch.Tensor, sigma: float, k_rr_const: float = 0.0,
                     gen_chunk: int = 8192) -> torch.Tensor:
    """Biased MMD^2 = exact within-batch repulsion - 2 * Nystrom attraction + const ``k_rr``.

    Args:
        feat_g: ``(N_g, d)`` generated features (carry grad).
        Z: ``(M, d)`` frozen landmarks; ``Z2 = (Z*Z).sum(1)``; ``alpha`` ``(M,)`` the
            precomputed ``K_ZZ^{-1} mu_bar`` coefficients (all fp32, frozen).
        sigma: per-encoder RBF bandwidth.
        k_rr_const: precomputed ``E[k(r, r')]`` (>= 0 offset; no gradient).
        gen_chunk: row-chunk size for the checkpointed repulsion (>= N_g -> single block).
    """
    g = feat_g.float()
    gamma = gamma_from_sigma(sigma)
    k_gg = self_kernel_mean(g, gamma, gen_chunk)
    k_gr = nystrom_attraction(g, Z, Z2, alpha, gamma)
    return k_gg - 2.0 * k_gr + k_rr_const
