"""Representation axis: the frozen encoder battery, generators, and the joint feature.

The 14-encoder panel is data in :mod:`registry`; :mod:`encoder_spec` is its torch-free
schema. :mod:`preprocess` and :mod:`pooling` are the autograd-safe transforms; the
per-family wrappers live in :mod:`backbones`; :class:`battery.Battery` runs a subset over
one shared batch. :mod:`joint_feature` builds the coupled image-text feature for the
conditional objective (with :mod:`text_encoder` supplying the frozen text tower).
"""
from .battery import Battery
from .encoder_spec import EncoderSpec, PoolType, Split
from .registry import all_specs, by_name, heldout_specs, training_specs

__all__ = ["EncoderSpec", "PoolType", "Split", "Battery", "all_specs", "training_specs",
           "heldout_specs", "by_name"]
