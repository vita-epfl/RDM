"""The encoder-battery facade: build a subset, run one shared batch, return name -> (B, D).

A :class:`Battery` builds the frozen backbones for a chosen subset of the panel
(:func:`Battery.for_training` -> the 10 training encoders; :func:`Battery.for_eval` ->
all 14), runs a single generated/real batch through each, and returns an ordered mapping
``name -> (B, D)`` in panel order. Matching the original feature-extraction discipline,
logits-output encoders (Inception) run in fp32 outside any enclosing autocast, while the
ViT-style encoders run under bf16 autocast; the kernels cast features to fp32 internally.

The same facade serves training and evaluation so the two paths cannot diverge in pool /
resolution / extraction (a documented footgun: e.g. MAE must avg-pool, PE-Core must
attn-pool -- both encoded once in the registry).
"""
from collections import OrderedDict

import torch
import torch.nn as nn

from .backbones import build_backbone
from .registry import all_specs, training_specs


class Battery(nn.Module):
    """Frozen subset of the encoder panel; ``forward`` returns ``OrderedDict[name -> (B, D)]``."""

    def __init__(self, specs, device: str = "cuda"):
        super().__init__()
        self.specs = list(specs)
        self.encoders = nn.ModuleDict({s.name: build_backbone(s, device) for s in self.specs})

    @classmethod
    def for_training(cls, device: str = "cuda") -> "Battery":
        """The 10 training encoders."""
        return cls(training_specs(), device)

    @classmethod
    def for_eval(cls, device: str = "cuda") -> "Battery":
        """All 14 encoders (10 train + 4 held-out)."""
        return cls(all_specs(), device)

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.specs]

    def forward(self, images: torch.Tensor, only=None) -> "OrderedDict[str, torch.Tensor]":
        """Run encoders over one ``[0, 1]`` NCHW batch; return ordered ``name -> (B, D)``.

        ``only`` optionally restricts to a subset of names (still in panel order).
        """
        out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        for s in self.specs:
            if only is not None and s.name not in only:
                continue
            enc = self.encoders[s.name]
            if getattr(enc, "has_logits", False):           # Inception: fp32, no autocast
                with torch.autocast("cuda", enabled=False):
                    out[s.name] = enc(images.float())
            else:                                           # ViT-style: bf16 autocast
                with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    out[s.name] = enc(images)
        return out
