"""Torch-free schema for the encoder battery.

:class:`EncoderSpec` is a pure dataclass (no torch import) so the comparison and
evaluation code can size buffers, name metric columns, and check the train/held-out
split without loading any backbone. :class:`PoolType` is the *conceptual* global pool
of Table 5; the backbone for each family applies it (see :mod:`rdm.representation.pooling`
and :mod:`rdm.representation.backbones`).
"""
from dataclasses import dataclass
from enum import Enum


class PoolType(str, Enum):
    """How a backbone's tokens / feature map are reduced to one ``(B, D)`` embedding."""

    CLS = "cls"                # class / prefix token (token 0)
    AVG = "avg"                # mean over patch tokens, or spatial mean for a CNN
    ATTN = "attn"              # the model's attention-pooling head (its intended output)
    SUMMARY = "summary"        # RADIO summary token
    PATCH_MEAN = "patch_mean"  # FLUX VAE 4x4 latent patch-mean


class Split(str, Enum):
    """Whether an encoder supervises training or is held out for evaluation only."""

    TRAIN = "train"
    HELD_OUT = "held_out"


# Backbone families (one wrapper per family in rdm.representation.backbones).
SOURCES = ("inception", "timm", "webssl", "dreamsim", "radio", "flux_vae")


@dataclass(frozen=True)
class EncoderSpec:
    """One frozen encoder of the panel.

    Attributes:
        name: short label (registry key and metric-table column).
        source: backbone family in :data:`SOURCES` (selects the wrapper).
        model_id: timm id / hub id; ``""`` for families with a fixed checkpoint.
        input_res: native input resolution the generator's 256px output is resized to.
        pool: the conceptual global pool (:class:`PoolType`).
        dim: feature dimension ``D``.
        split: :class:`Split` (10 train / 4 held-out).
    """

    name: str
    source: str
    model_id: str
    input_res: int
    pool: PoolType
    dim: int
    split: Split

    def __post_init__(self):
        if self.source not in SOURCES:
            raise ValueError(f"{self.name}: unknown source {self.source!r}; expected {SOURCES}")
