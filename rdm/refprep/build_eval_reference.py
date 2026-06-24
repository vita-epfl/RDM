"""Off-objective evaluation banks: RFF-MMD (MMDr14), Frechet mean/cov, and SW samples.

These are the *evaluation* references, kept disjoint from the trained Nystrom reference
(a different RFF seed, 99999) so a metric win cannot be the training instance read back.

* :func:`build_rff_bank` -- frozen ``{W, b, mu_r}`` for the RFF-MMD ratio (MMDr14, App D);
* :func:`build_fd_bank` -- frozen ``{mu_r, Sr_half, tr_Sr}`` for the differentiable Frechet
  distance and the Gaussian-Frechet eval;
* :func:`build_sw_bank` -- a frozen real-feature sample for the SW_r14 floor/ratio (the SW
  metric resamples directions on the fly, so only the feature sample is stored).
"""
import math

import torch

from ..compare.mmd_rff import rff_phi
from ..utils.io import save_bundle

EVAL_RFF_SEED = 99999


@torch.no_grad()
def build_rff_bank(features: torch.Tensor, sigma: float, out_path: str, D: int = 4096,
                   seed: int = EVAL_RFF_SEED, device: str = "cuda") -> dict:
    """Frozen RFF bank: ``W ~ N(0, sigma^-2)``, ``b ~ U[0, 2pi)``, ``mu_r = mean phi(r)``."""
    feats = features.float().to(device)
    d = feats.shape[1]
    g = torch.Generator(device=device).manual_seed(seed)
    W = torch.randn(d, D, generator=g, device=device) / float(sigma)
    b = torch.rand(D, generator=g, device=device) * (2.0 * math.pi)
    mu_r = rff_phi(feats, W, b).mean(0)
    bundle = dict(W=W.cpu(), b=b.cpu(), mu_r=mu_r.cpu(), sigma=float(sigma), D=int(D),
                  d_in=int(d), seed=int(seed))
    save_bundle(bundle, out_path)
    return bundle


@torch.no_grad()
def build_fd_bank(features: torch.Tensor, out_path: str, eps: float = 1e-6,
                  device: str = "cuda") -> dict:
    """Frozen Frechet bank: real mean, ``Sigma_r^{1/2}`` (symmetric eigh), and ``Tr(Sigma_r)``."""
    feats = features.float().to(device)
    n, d = feats.shape
    mu_r = feats.mean(0)
    fc = feats - mu_r
    Sigma_r = (fc.T @ fc) / (n - 1)
    ev, U = torch.linalg.eigh(Sigma_r + eps * torch.eye(d, device=device))
    Sr_half = (U * ev.clamp_min(0).sqrt()) @ U.T
    bundle = dict(mu_r=mu_r.cpu(), Sr_half=Sr_half.cpu(), Sigma_r=Sigma_r.cpu(),
                  tr_Sr=float(torch.diagonal(Sigma_r).sum()), d=int(d), eps=float(eps))
    save_bundle(bundle, out_path)
    return bundle


@torch.no_grad()
def build_sw_bank(features: torch.Tensor, out_path: str, n_samples: int = 16384,
                  seed: int = 0) -> dict:
    """Frozen real-feature sample for the SW_r14 ratio (directions resampled at eval time)."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(features.shape[0], generator=g)[:n_samples]
    bundle = dict(features=features[idx].float().cpu(), n=int(min(n_samples, features.shape[0])))
    save_bundle(bundle, out_path)
    return bundle
