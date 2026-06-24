"""One-step ImageNet generator wrapper (pMF-H).

Wraps the released pMF-H denoiser (:mod:`rdm.representation.models.pmfh_fdsim`): calls its
differentiable one-step sampler ``sample_images_with_grad(z, y, sampling_args)`` under
autocast, optionally decodes through a VAE tokenizer, and clamps to ``[0, 1]``. The model
is forced to ``eval()`` for the sampling call (deployment-time generation), restoring the
prior mode afterward. The 1-step Euler sampler itself lives in the denoiser; this is the
thin generator-forward adapter the loss/rollout use.
"""
from __future__ import annotations

from typing import Any

import torch

from .base import Generator


class PMFHGenerator(Generator):
    """ImageNet one-step generator over the pMF-H denoiser."""

    def __init__(self, model, sampling_args: dict, args=None, tokenizer=None):
        self.model = model
        self.sampling_args = sampling_args
        self.tokenizer = tokenizer
        self.enable_amp = bool(getattr(args, "enable_amp", True))
        self.amp_dtype = getattr(args, "amp_dtype", torch.bfloat16)

    def sample(self, noise: torch.Tensor, condition: Any) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.autocast("cuda", enabled=self.enable_amp, dtype=self.amp_dtype):
                x = self.model.sample_images_with_grad(noise, condition,
                                                       sampling_args=self.sampling_args)
        finally:
            if was_training:
                self.model.train()
        if self.tokenizer is not None:                       # latent-space generators
            x = self.tokenizer.decode(self.tokenizer.denormalize_z(x))
        return (x * 0.5 + 0.5).clamp(0, 1)                   # [-1,1] -> [0,1]
