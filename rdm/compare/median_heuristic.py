"""Per-encoder RBF bandwidth by the median heuristic.

The bandwidth ``sigma_phi`` is fixed once per encoder and held at a single scale.
This estimator is what the frozen reference bundles were built with (the code is the
source of truth): a subsample of the feature pool, the strict-upper-triangular
squared pairwise distances, then ``sigma = sqrt(median(d^2))`` guarded at ``1e-5``.

The joint (text-to-image) and "cold" image kernels multiply this by ``scale=0.25``.
At train/eval time the bandwidth is read back from the stored bundle, never
recomputed, so only the *build* path uses this function.
"""
import torch


@torch.no_grad()
def median_bandwidth(feats: torch.Tensor, other: torch.Tensor | None = None,
                     max_subsample: int = 2000, scale: float = 1.0) -> float:
    """Median-heuristic bandwidth ``sigma = scale * sqrt(median(off-diagonal d^2))``.

    Args:
        feats: ``(N, d)`` feature pool.
        other: optional second pool; if given, a combined subsample of both is used
            (the two-sample heuristic, matching the eval-time CMMD path).
        max_subsample: cap per pool for the pairwise-distance estimate.
        scale: multiplier on the bandwidth (``0.25`` for the cold / joint kernel).

    Returns:
        The bandwidth ``sigma`` (a Python float), guarded ``>= 1e-5`` before scaling.
    """
    n = min(max_subsample, feats.shape[0])
    pts = feats[torch.randperm(feats.shape[0], device=feats.device)[:n]]
    if other is not None:
        m = min(max_subsample, other.shape[0])
        pts = torch.cat([pts, other[torch.randperm(other.shape[0], device=other.device)[:m]]], dim=0)
    d2 = torch.cdist(pts, pts).pow(2)
    mask = torch.triu(torch.ones(d2.shape[0], d2.shape[0], device=d2.device, dtype=torch.bool),
                      diagonal=1)
    sigma = d2[mask].median().sqrt().item()
    return max(sigma, 1e-5) * float(scale)
