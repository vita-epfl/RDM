"""Generator-forward abstraction shared by the training loop and the rollout refresh.

A :class:`Generator` wraps the actual one-step network (plus autocast, an optional VAE
tokenizer decode, and the ``[0, 1]`` clamp) behind a single ``sample(noise, condition)``
call, so the loss / rollout code never needs to know whether the backend is the ImageNet
pMF-H or the FLUX.2 adapter.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class Generator(ABC):
    """Generic one-step generator interface."""

    @abstractmethod
    def sample(self, noise: torch.Tensor, condition: Any) -> torch.Tensor:
        """Generate images in ``[0, 1]`` from ``(B, C, H, W)`` noise and an opaque condition.

        ``condition`` is whatever the matching conditioner's
        ``encode_for_generator`` returns (class ids for ImageNet, prompt embeddings for FLUX).
        """
