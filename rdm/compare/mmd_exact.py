"""Exact two-sample MMD ablation (the control for the Nystrom attraction).

Identical in structure to :func:`rdm.compare.mmd_nystrom.mmd_nystrom_loss` except the
attraction cross term is the **exact** full-pairwise RBF mean against an ``N_r``-sized
real-feature sample, not the low-rank Nystrom ``Z . alpha`` form. Comparing the two in
isolation (Table 4) shows the exact MMD does *not* beat its Nystrom approximation -- the
low-rank cross term is a smoother gradient and its frozen reference has zero variance,
while the exact estimator scores the batch against a freshly resampled real batch.

The caller subsamples the frozen real pool to ``feat_r`` each step, seeded for DDP
rank-consistency (the cross-term estimate must be identical across ranks, exactly like
the sliced-Wasserstein path).
"""
import torch

from .kernels import cross_kernel_mean, gamma_from_sigma, self_kernel_mean


def mmd_exact_loss(feat_g: torch.Tensor, feat_r: torch.Tensor, sigma: float,
                   k_rr_const: float = 0.0, gen_chunk: int = 8192) -> torch.Tensor:
    """Biased MMD^2 with an EXACT full-pairwise cross term:

        k_gg - 2 * mean_{i,j} k(g_i, r_j) + k_rr

    Args:
        feat_g: ``(N_g, d)`` generated features (carry grad).
        feat_r: ``(N_r, d)`` real-feature sample (detached target).
        sigma: RBF bandwidth (same as the matched Nystrom arm).
        k_rr_const: precomputed ``E[k(r, r')]`` constant offset.
        gen_chunk: row-chunk for the checkpointed terms.
    """
    g = feat_g.float()
    r = feat_r.float().detach()
    gamma = gamma_from_sigma(sigma)
    k_gg = self_kernel_mean(g, gamma, gen_chunk)
    k_gr = cross_kernel_mean(g, r, gamma, gen_chunk)
    return k_gg - 2.0 * k_gr + k_rr_const
