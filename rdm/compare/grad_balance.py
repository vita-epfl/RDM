"""Per-encoder gradient balancing and the generator gradient clip.

Encoders produce raw losses of wildly different magnitude (a biased MMD^2 can span
orders of magnitude across the battery), so summing them naively lets the largest-scale
encoder dominate the gradient. iRDM equalizes them with **self-normalization** rather
than an RMS-tracking scheme:

    self_normalize(raw) = raw / (|raw.detach()| + eps)

whose gradient is ``d/dtheta log|raw|`` -- a scale-free ``grad / value`` per encoder, so
every encoder contributes a comparably-scaled gradient regardless of its raw magnitude
(this is the mechanism the manifest calls "gradient balancing"). The proportional
Lagrangian controller (:mod:`rdm.train.pid_lagrangian`) then weights these balanced
terms. ``|raw|`` (not ``raw``) keeps the descent direction correct when a biased MMD^2
dips slightly below zero.

The generator gradient is clipped to a max norm before each optimizer step.
"""
import torch


def self_normalize(raw: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Scale-free per-encoder loss: ``raw / (|raw.detach()| + eps)`` (the grad-log surrogate)."""
    return raw / (raw.detach().abs() + eps)


def clip_generator_grads(parameters, max_norm: float = 2.0) -> torch.Tensor:
    """Clip the generator gradient to ``max_norm`` (returns the pre-clip total norm)."""
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm)
