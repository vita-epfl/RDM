"""Random-Fourier-Feature MMD: the ``mmd_rff`` ablation arm and the MMDr14 metric core.

The RFF feature map ``phi(x) = sqrt(2/D) cos(x W + b)`` linearizes the RBF kernel with a
data-independent basis (``W ~ N(0, sigma^-2 I)``, ``b ~ U[0, 2pi)``), so
``<phi(x), phi(y)> ~= k(x, y)`` and ``|| mean_i phi(g_i) - mu_r ||^2 ~= MMD^2_RBF``.

Two roles:

* :func:`mmd_rff_loss` -- the ``mmd_rff`` *training* arm of the Table-4 ablation. With
  frozen ``W, b, mu_r`` it is deterministic (no per-step sampling), hence DDP
  rank-identical and gradient-cache bit-exact with no seeding. As an attraction it is the
  control that loses to the data-dependent Nystrom basis (its random cosines leave a
  null space a generator under optimization pressure exploits).
* :func:`rff_phi` -- the same feature map, reused by the off-objective **MMDr14** metric
  (:mod:`rdm.eval.mmd_r14`), built from an independent seed so the metric is not the
  training instance.
"""
import math

import torch


def rff_phi(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """RFF map ``phi(x) = sqrt(2/D) cos(x @ W + b)`` for an RBF kernel (``D = W.shape[1]``)."""
    D = W.shape[1]
    return math.sqrt(2.0 / D) * torch.cos(x @ W + b)


def mmd_rff_loss(feat_g: torch.Tensor, W: torch.Tensor, b: torch.Tensor,
                 mu_r: torch.Tensor) -> torch.Tensor:
    """RFF-MMD^2 = ``|| mean_i phi(g_i) - mu_r ||^2`` (biased), differentiable in ``feat_g``.

    Args:
        feat_g: ``(N_g, d)`` generated features (carry grad).
        W: ``(d, D)`` frozen RFF frequencies; b: ``(D,)`` frozen phases;
            mu_r: ``(D,)`` frozen real RFF mean over the reference pool.
    """
    mu_g = rff_phi(feat_g.float(), W, b).mean(0)
    return (mu_g - mu_r).pow(2).sum()
