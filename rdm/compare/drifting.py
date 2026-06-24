"""Drifting force-field ablation -- a faithful port of the coupled drifting loss.

Port of ``drift_loss`` from *Generative Modeling via Drifting* (Deng et al., 2026,
arXiv:2602.04770). Generated features are drifted toward real positives and away from
generated negatives by a multi-temperature softmax-affinity **force field**, then the
current features are regressed to the (stop-grad) drifted goal:

    goal = old_gen + sum_R F_R / ||F_R||,   loss = mean || gen - goal ||^2   (scale-normalized)

where ``F_R`` is a doubly-stochastic (``sqrt(softmax_row . softmax_col)``) affinity that
per temperature ``R`` pulls each generated point toward positives and pushes it off
other generated / explicit-negative points. The goal is computed under ``no_grad`` (the
reference's ``stop_gradient``); gradient flows only through the final MSE's ``gen`` term.

In the Table-4 ablation the drifting arm is weakest even at the best of a learning-rate
sweep: it is a per-particle normalized form of the same MMD gradient, steadier at small
batches but its resampled per-batch reference confines it to small batches. The drifting
loss is used **raw** by the caller (it bypasses the self-normalization tail), so the
swept learning rate acts on the same gradient scale as the reference implementation.
"""
import math

import torch


def _drift_cdist(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Batched pairwise Euclidean distance: ``x[B,N,D], y[B,M,D] -> [B,N,M]``."""
    xy = torch.einsum("bnd,bmd->bnm", x, y)
    xn = torch.einsum("bnd,bnd->bn", x, x)
    yn = torch.einsum("bmd,bmd->bm", y, y)
    sq = xn[:, :, None] + yn[:, None, :] - 2.0 * xy
    return sq.clamp_min(eps).sqrt()


def drift_loss(gen: torch.Tensor, fixed_pos: torch.Tensor, fixed_neg: torch.Tensor | None = None,
               R_list=(0.2, 0.05, 0.02)) -> torch.Tensor:
    """Multi-scale drifting force-field MSE.

    Args:
        gen: ``[B, C_g, S]`` generated features (carry grad).
        fixed_pos: ``[B, C_p, S]`` real positive features (detached target).
        fixed_neg: ``[B, C_n, S]`` optional explicit negatives (detached); ``None`` ->
            in-batch generated self-repulsion provides the negatives.
        R_list: kernel radii (temperatures) for the multi-scale force.
    """
    gen = gen.float()
    fixed_pos = fixed_pos.float()
    B, C_g, S = gen.shape
    C_p = fixed_pos.shape[1]
    if fixed_neg is None:
        fixed_neg = gen.new_zeros(B, 0, S)
    fixed_neg = fixed_neg.float()
    C_n = fixed_neg.shape[1]
    targets_w = torch.cat([gen.new_ones(B, C_g), gen.new_ones(B, C_n),
                           gen.new_ones(B, C_p)], dim=1)
    old_gen = gen.detach()
    targets = torch.cat([old_gen, fixed_neg.detach(), fixed_pos.detach()], dim=1)

    with torch.no_grad():
        dist = _drift_cdist(old_gen, targets)
        weighted_dist = dist * targets_w[:, None, :]
        scale = weighted_dist.mean() / targets_w.mean()
        scale_inputs = (scale / math.sqrt(S)).clamp_min(1e-3)
        old_gen_scaled = old_gen / scale_inputs
        targets_scaled = targets / scale_inputs
        dist_normed = dist / scale.clamp_min(1e-3)
        diag = torch.eye(C_g, device=gen.device, dtype=gen.dtype)
        block_mask = torch.nn.functional.pad(diag, (0, C_n + C_p))
        dist_normed = dist_normed + block_mask[None] * 100.0       # mask gen self-block
        force_across_R = torch.zeros_like(old_gen_scaled)
        split_idx = C_g + C_n
        for R in R_list:
            logits = -dist_normed / R
            affinity = (torch.softmax(logits, dim=-1) * torch.softmax(logits, dim=-2)).clamp_min(1e-6).sqrt()
            affinity = affinity * targets_w[:, None, :]
            aff_neg = affinity[:, :, :split_idx]                   # gen-self (+ neg) = repulsive
            aff_pos = affinity[:, :, split_idx:]                   # positives = attractive
            sum_pos = aff_pos.sum(-1, keepdim=True)
            sum_neg = aff_neg.sum(-1, keepdim=True)
            R_coeff = torch.cat([-aff_neg * sum_pos, aff_pos * sum_neg], dim=2)
            total_force_R = torch.einsum("biy,byx->bix", R_coeff, targets_scaled)
            total_force_R = total_force_R - R_coeff.sum(-1)[..., None] * old_gen_scaled
            force_scale = (total_force_R ** 2).mean().clamp_min(1e-8).sqrt()
            force_across_R = force_across_R + total_force_R / force_scale
        goal_scaled = old_gen_scaled + force_across_R

    gen_scaled = gen / scale_inputs
    return ((gen_scaled - goal_scaled) ** 2).mean()
