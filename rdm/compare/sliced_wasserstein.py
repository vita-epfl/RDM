"""Sliced-Wasserstein: the ``sw`` training ablation and the SW_r14 evaluation core.

The sliced-Wasserstein distance projects both sets onto ``L`` random unit directions
and, along each, takes the exact 1D Wasserstein-``p`` between the two empirical
distributions by sorting (1D optimal transport is sorted matching). It is an
optimal-transport distance that shares no machinery with the kernel MMD, which is why
the primary evaluation metric SW_r14 (:mod:`rdm.eval.sw_r14`) is built on it.

* :func:`sw_loss` -- the differentiable ``sw`` *training* arm (Table 4). Gradient flows
  through the sort (a gather by the sort permutation). Projections are resampled each
  call; the caller subsamples the real pool to the gen rollout size (sorted matching
  needs equal sizes).
* :func:`sliced_wasserstein` -- the no-grad evaluation primitive (``sw_raw``) reused by
  the SW_r14 metric: equal-size sets, a *shared* projection seed across encoders /
  checkpoints / floor so the ratios are comparable.
"""
import torch


def sw_loss(feat_g: torch.Tensor, feat_r: torch.Tensor, n_proj: int = 128, p: int = 2,
            proj_chunk: int = 64, generator: torch.Generator | None = None) -> torch.Tensor:
    """Differentiable SW_p^p between equal-size sets ``feat_g`` (grad) and ``feat_r`` (target).

        SW_p^p = (1/L) sum_k mean_j | sort(theta_k . g)_j - sort(theta_k . r)_j |^p

    Returns the scalar ``SW_p^p`` (``p=2`` gives SW^2, parallel to the biased MMD^2 so the
    caller's self-normalization tail applies identically). Pass a seeded ``generator``
    only for reproducible tests; over training the resampled directions are the standard
    stochastic estimate of the SW expectation.
    """
    g = feat_g.float()
    r = feat_r.float().detach()
    n, d = g.shape
    assert r.shape[0] == n, f"sw_loss needs equal sizes (gen {n} vs real {r.shape[0]})"
    total = g.new_zeros(())
    done = 0
    while done < n_proj:
        k = min(proj_chunk, n_proj - done)
        theta = torch.randn(d, k, generator=generator, device=g.device, dtype=g.dtype)
        theta = theta / theta.norm(dim=0, keepdim=True)
        pg = (g @ theta).sort(dim=0).values
        pr = (r @ theta).sort(dim=0).values
        total = total + (pg - pr).abs().pow(p).mean(dim=0).sum()
        done += k
    return total / n_proj


@torch.no_grad()
def sliced_wasserstein(G: torch.Tensor, R: torch.Tensor, n_proj: int = 1024, p: int = 2,
                       proj_chunk: int = 64, seed: int = 0) -> float:
    """Empirical SW_p between equal-size sets ``G, R`` (each ``N x d``); the eval ``sw_raw``.

    Uses the SAME random directions (seeded by ``seed``) across every call so the gen,
    floor, and per-checkpoint values are directly comparable. Returns the SW_p distance
    (the ``1/p`` power of the mean per-direction Wasserstein-``p``-to-the-``p``).
    """
    dev = G.device
    G = G.float()
    R = R.float()
    n, d = G.shape
    assert R.shape[0] == n, f"sliced_wasserstein expects equal sample sizes ({n} vs {R.shape[0]})"
    gen = torch.Generator(device=dev).manual_seed(seed)
    total = 0.0
    for i in range(0, n_proj, proj_chunk):
        k = min(proj_chunk, n_proj - i)
        theta = torch.randn(d, k, generator=gen, device=dev)
        theta = theta / theta.norm(dim=0, keepdim=True)
        pg = (G @ theta).sort(dim=0).values
        pr = (R @ theta).sort(dim=0).values
        total += (pg - pr).abs().pow(p).mean(dim=0).sum().item()
    return (total / n_proj) ** (1.0 / p)
