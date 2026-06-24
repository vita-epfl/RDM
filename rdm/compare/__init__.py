"""Comparison axis: kernels, Nystrom math, and every distance.

The iRDM loss is :func:`mmd_nystrom_loss`; the Table-4 ablation distances are
:func:`mmd_exact_loss`, :func:`mmd_rff_loss`, :func:`frechet_loss`, :func:`sw_loss`,
and :func:`drift_loss`. Training and evaluation should go through :mod:`interface`
(``ReferencePack`` / ``DistanceConfig`` / ``build_distance`` / ``aggregate_losses``).
"""
from .drifting import drift_loss
from .frechet import frechet_loss
from .grad_balance import clip_generator_grads, self_normalize
from .interface import (DISTANCES, DistanceConfig, ReferencePack, aggregate_losses,
                        build_distance)
from .kernels import (cross_kernel_mean, gamma_from_sigma, gaussian_gram,
                      self_kernel_mean, squared_distance)
from .median_heuristic import median_bandwidth
from .mmd_exact import mmd_exact_loss
from .mmd_nystrom import mmd_nystrom_loss
from .mmd_rff import mmd_rff_loss, rff_phi
from .nystrom import nystrom_attraction, nystrom_feature_map
from .reference_math import (build_nystrom_reference, kmeans_landmarks,
                             reference_self_kernel_mean, streaming_reference_mean)
from .sliced_wasserstein import sliced_wasserstein, sw_loss

__all__ = [
    "gamma_from_sigma", "squared_distance", "gaussian_gram", "self_kernel_mean",
    "cross_kernel_mean", "median_bandwidth", "kmeans_landmarks", "streaming_reference_mean",
    "reference_self_kernel_mean", "build_nystrom_reference", "nystrom_attraction",
    "nystrom_feature_map", "mmd_nystrom_loss", "mmd_exact_loss", "mmd_rff_loss", "rff_phi",
    "sw_loss", "sliced_wasserstein", "frechet_loss", "drift_loss", "self_normalize",
    "clip_generator_grads", "ReferencePack", "DistanceConfig", "build_distance",
    "aggregate_losses", "DISTANCES",
]
