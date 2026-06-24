"""The distance contract: one factory + reference container every method plugs into.

Training and evaluation both reach the comparison axis only through this module. A
:class:`ReferencePack` carries the frozen per-encoder artifacts (loaded once, fp32); a
:class:`DistanceConfig` names the discrepancy and its estimator knobs; :func:`build_distance`
returns the callable that maps a generated batch (plus, for the resampled arms, a real
sample) to a raw scalar; :func:`aggregate_losses` applies the per-encoder
self-normalization and the proportional-Lagrangian weights and sums.

Supported modes (``DistanceConfig.mode``):

============  ====================================  ====================================
mode          loss                                   reference fields used
============  ====================================  ====================================
mmd_nystrom   the iRDM loss (eq. 3) -- DEFAULT        Z, Z2, alpha, sigma, k_rr
mmd_exact     exact two-sample MMD control           real_pool, sigma, k_rr
mmd_rff       RFF-MMD                                 rff_W, rff_b, rff_mu_r
fd            Frechet (Gaussian 2-Wasserstein)        fd_mu_r, fd_Sr_half, fd_tr_Sr
sw            sliced-Wasserstein                      real_pool
drifting      coupled drifting force field           real_pool  (bypasses self-norm)
============  ====================================  ====================================
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from . import drifting, frechet, mmd_exact, mmd_nystrom, mmd_rff, sliced_wasserstein
from .grad_balance import self_normalize

# Modes whose loss is used raw (no self-normalization tail); see drifting.
BYPASS_SELF_NORM = frozenset({"drifting"})
DISTANCES = frozenset({"mmd_nystrom", "mmd_exact", "mmd_rff", "fd", "sw", "drifting"})


@dataclass
class ReferencePack:
    """Frozen per-encoder reference artifacts for one distance (loaded once, held fp32)."""

    name: str
    sigma: float = 0.0
    weight: float = 1.0
    beta: Optional[float] = None        # joint text-block scale (sigma_img / s_txt); None = marginal
    # mmd_nystrom (the iRDM default)
    Z: Optional[torch.Tensor] = None
    Z2: Optional[torch.Tensor] = None
    alpha: Optional[torch.Tensor] = None
    k_rr: float = 0.0
    # mmd_rff
    rff_W: Optional[torch.Tensor] = None
    rff_b: Optional[torch.Tensor] = None
    rff_mu_r: Optional[torch.Tensor] = None
    # frechet
    fd_mu_r: Optional[torch.Tensor] = None
    fd_Sr_half: Optional[torch.Tensor] = None
    fd_tr_Sr: float = 0.0
    # mmd_exact / sw / drifting: a real-feature pool to subsample each step
    real_pool: Optional[torch.Tensor] = None

    @classmethod
    def from_nystrom_bundle(cls, name: str, bundle: dict, weight: float = 1.0) -> "ReferencePack":
        """Build from a ``{Z, alpha, sigma, k_rr}`` bundle (see
        :func:`rdm.compare.reference_math.build_nystrom_reference`); ``Z2`` is derived. Joint
        bundles also carry ``beta`` (the text-block scale), kept for the coupled objective."""
        Z = bundle["Z"].float()
        beta = bundle.get("beta")
        return cls(name=name, sigma=float(bundle["sigma"]), weight=weight, Z=Z,
                   Z2=(Z * Z).sum(1), alpha=bundle["alpha"].float(), k_rr=float(bundle["k_rr"]),
                   beta=(float(beta) if beta is not None else None))


@dataclass
class DistanceConfig:
    """The discrepancy + estimator knobs."""

    mode: str = "mmd_nystrom"
    gen_chunk: int = 8192
    sw_n_proj: int = 128
    sw_p: int = 2
    drift_R_list: tuple = (0.2, 0.05, 0.02)


def build_distance(cfg: DistanceConfig) -> Callable[..., torch.Tensor]:
    """Return ``loss_fn(feat_g, ref, real_sample=None) -> raw scalar`` for ``cfg.mode``.

    ``real_sample`` is the per-step real-feature subsample (already on device, detached,
    equal size for the ``sw`` arm); it is ignored by the deterministic modes
    (``mmd_nystrom``, ``mmd_rff``, ``fd``).
    """
    if cfg.mode not in DISTANCES:
        raise ValueError(f"unknown distance mode {cfg.mode!r}; expected one of {sorted(DISTANCES)}")

    def loss_fn(feat_g: torch.Tensor, ref: ReferencePack,
                real_sample: Optional[torch.Tensor] = None) -> torch.Tensor:
        if cfg.mode == "mmd_nystrom":
            return mmd_nystrom.mmd_nystrom_loss(feat_g, ref.Z, ref.Z2, ref.alpha, ref.sigma,
                                                ref.k_rr, gen_chunk=cfg.gen_chunk)
        if cfg.mode == "mmd_exact":
            return mmd_exact.mmd_exact_loss(feat_g, real_sample, ref.sigma, ref.k_rr,
                                            gen_chunk=cfg.gen_chunk)
        if cfg.mode == "mmd_rff":
            return mmd_rff.mmd_rff_loss(feat_g, ref.rff_W, ref.rff_b, ref.rff_mu_r)
        if cfg.mode == "fd":
            return frechet.frechet_loss(feat_g, ref.fd_mu_r, ref.fd_Sr_half, ref.fd_tr_Sr)
        if cfg.mode == "sw":
            return sliced_wasserstein.sw_loss(feat_g, real_sample, n_proj=cfg.sw_n_proj, p=cfg.sw_p)
        # drifting expects grouped [B, C, S] tensors prepared by the caller
        return drifting.drift_loss(feat_g, real_sample, R_list=cfg.drift_R_list)

    return loss_fn


def aggregate_losses(raw_by_name: dict, weights: dict, mode: str = "mmd_nystrom",
                     eps: float = 1e-7) -> torch.Tensor:
    """Self-normalize each per-encoder raw loss and sum under the (PID) weights.

    ``weights`` are the proportional-Lagrangian multipliers from
    :mod:`rdm.train.pid_lagrangian` (default uniform). ``drifting`` is summed raw.
    """
    total = None
    bypass = mode in BYPASS_SELF_NORM
    for name, raw in raw_by_name.items():
        term = raw if bypass else self_normalize(raw, eps)
        contrib = float(weights.get(name, 1.0)) * term
        total = contrib if total is None else total + contrib
    return total
