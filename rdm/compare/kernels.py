"""Stable Gaussian-RBF kernel primitives on raw frozen-encoder embeddings.

This is the single source of the squared-distance and RBF arithmetic used by every
distance in :mod:`rdm.compare`. The kernel is the Gaussian
``k(x, y) = exp(-||x - y||^2 / (2 * sigma^2))`` on *raw* embeddings (no feature
normalization), with one bandwidth ``sigma`` per encoder fixed by the median
heuristic (see :mod:`rdm.compare.median_heuristic`).

All "mean" reductions here are the **biased** estimators that the training loss
actually uses: the within-batch self-kernel mean includes the ``i == i`` diagonal
and divides by ``N**2`` (not the unbiased ``N(N-1)`` off-diagonal U-statistic).
See :mod:`rdm.compare.mmd_nystrom` for the assembled loss.
"""
import torch
import torch.utils.checkpoint as checkpoint


def gamma_from_sigma(sigma: float) -> float:
    """RBF precision ``gamma = 1 / (2 * sigma^2)`` for ``k(x,y)=exp(-gamma||x-y||^2)``."""
    sigma = float(sigma)
    return 1.0 / (2.0 * sigma * sigma)


def squared_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Numerically stable pairwise squared Euclidean distance ``(a.shape[0], b.shape[0])``.

    ``||a_i - b_j||^2 = ||a_i||^2 + ||b_j||^2 - 2 a_i . b_j``; the final ``clamp_min(0)``
    removes the small negative values that the expansion can produce in floating point.
    """
    a2 = (a * a).sum(1)
    b2 = (b * b).sum(1)
    return (a2[:, None] + b2[None, :] - 2.0 * (a @ b.T)).clamp_min(0)


def gaussian_gram(a: torch.Tensor, b: torch.Tensor, gamma: float) -> torch.Tensor:
    """Gaussian-RBF Gram matrix ``exp(-gamma * ||a_i - b_j||^2)``."""
    return torch.exp(-gamma * squared_distance(a, b))


def self_kernel_mean(g: torch.Tensor, gamma: float, chunk: int = 8192) -> torch.Tensor:
    """BIASED within-batch self-kernel mean ``(1 / N^2) * sum_{i,i'} k(g_i, g_i')``.

    This is the iRDM **repulsion** term (the only force preventing collapse onto the
    densest modes). It is the *biased* estimator: the ``i == i'`` diagonal is included
    and the sum is divided by ``N**2``. It is reference-free, so it is always computed
    exactly. Chunked over generated rows and gradient-checkpointed per chunk so the
    peak middle-backward kernel matrix is ``O(chunk * N)``; ``chunk >= N`` collapses to
    a single bit-exact block.
    """
    n = g.shape[0]

    def block(gg, lo, hi):
        return gaussian_gram(gg[lo:hi], gg, gamma).sum()

    acc = g.new_zeros(())
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        acc = acc + checkpoint.checkpoint(block, g, lo, hi, use_reentrant=False)
    return acc / (n * n)


def cross_kernel_mean(g: torch.Tensor, r: torch.Tensor, gamma: float,
                      chunk: int = 8192) -> torch.Tensor:
    """BIASED cross-kernel mean ``(1 / (N_g N_r)) * sum_{i,j} k(g_i, r_j)``.

    The *exact* full-pairwise attraction against a real-feature sample ``r``
    (:mod:`rdm.compare.mmd_exact`); the iRDM loss replaces this with the cheaper
    frozen Nystrom form (:mod:`rdm.compare.nystrom`). Chunked + checkpointed over the
    generated rows; ``r`` is treated as a detached target by the caller.
    """
    n_g, n_r = g.shape[0], r.shape[0]
    r2 = (r * r).sum(1)

    def block(gg, rr, rr2, lo, hi):
        a = gg[lo:hi]
        a2 = (a * a).sum(1)
        d2 = (a2[:, None] + rr2[None, :] - 2.0 * (a @ rr.T)).clamp_min(0)
        return torch.exp(-gamma * d2).sum()

    acc = g.new_zeros(())
    for lo in range(0, n_g, chunk):
        hi = min(lo + chunk, n_g)
        acc = acc + checkpoint.checkpoint(block, g, r, r2, lo, hi, use_reentrant=False)
    return acc / (n_g * n_r)
