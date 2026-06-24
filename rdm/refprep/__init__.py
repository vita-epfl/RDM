"""Offline reference precompute: the heavy one-time compute that training loads.

Pipeline: :mod:`extract_features` (raw embeddings + median bandwidth) ->
:mod:`build_nystrom_reference` (the frozen ``{Z, alpha, sigma, k_rr}`` per encoder) and
:mod:`build_joint_reference` (the image-text joint bundle) -> :mod:`compute_floors`
(the PID real-validation floors) -> :mod:`build_eval_reference` (the off-objective RFF /
Frechet / SW evaluation banks, kept disjoint from the trained reference).
"""
from .build_eval_reference import build_fd_bank, build_rff_bank, build_sw_bank
from .build_joint_reference import build_joint_one
from .build_nystrom_reference import build_one
from .compute_floors import compute_all_floors, compute_floor
from .extract_features import cache_median_sigma, extract_features, save_features

__all__ = ["extract_features", "save_features", "cache_median_sigma", "build_one",
           "build_joint_one", "compute_floor", "compute_all_floors", "build_rff_bank",
           "build_fd_bank", "build_sw_bank"]
