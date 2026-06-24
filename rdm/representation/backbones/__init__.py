"""Per-family frozen backbone wrappers + the :func:`build_backbone` dispatch.

Each family maps a ``[0, 1]`` NCHW image to one ``(B, D)`` embedding under its conceptual
pool. Imports are lazy so an evaluation-only family's weights are never fetched during a
training run.
"""
from ..encoder_spec import EncoderSpec


def build_backbone(spec: EncoderSpec, device: str = "cuda"):
    """Construct the frozen backbone for ``spec`` (dispatch on ``spec.source``)."""
    src = spec.source
    if src == "inception":
        from .inception_backbone import InceptionBackbone
        return InceptionBackbone(device=device)
    if src == "timm":
        from .timm_backbone import TimmBackbone
        return TimmBackbone(spec, device=device)
    if src == "webssl":
        from .webssl_backbone import WebSSLBackbone
        return WebSSLBackbone(device=device, model_id=spec.model_id, target_size=spec.input_res)
    if src == "dreamsim":
        from .dreamsim_backbone import DreamSimBackbone
        return DreamSimBackbone(device=device, target_size=spec.input_res)
    if src == "radio":
        from .radio_backbone import RadioBackbone
        return RadioBackbone(device=device, model_id=spec.model_id, target_size=spec.input_res)
    if src == "flux_vae":
        from .flux_vae_backbone import FluxVAEBackbone
        return FluxVAEBackbone(device=device, target_size=spec.input_res)
    raise ValueError(f"unknown backbone source {src!r} for encoder {spec.name!r}")


__all__ = ["build_backbone"]
