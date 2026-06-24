"""Autograd-safe resize + normalize from the generator's 256px output to each native input.

The generator emits images in ``[0, 1]`` at 256px. Each encoder resizes that to its own
native resolution (Table 5) and normalizes with its own mean/std. The resize is a
``bicubic`` (``antialias``) ``F.interpolate`` and the normalize is affine, so gradients
flow back to the generator -- nothing here uses ``inference_mode`` or detaches.

Per-encoder mean/std come from the backbone (timm resolves them from the model's
``pretrained_cfg``; the FLUX VAE uses its own ``[-1, 1]`` rescale). The standard constant
tables below are provided for the non-timm wrappers.
"""
import torch
import torch.nn.functional as F

#: Standard normalization constants used by the non-timm backbones.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def resize_to(x: torch.Tensor, target_size: int, mode: str = "bicubic") -> torch.Tensor:
    """Autograd-safe resize ``(B,3,H,W) -> (B,3,target,target)`` (no-op if already sized)."""
    if target_size is not None and (x.shape[-2] != target_size or x.shape[-1] != target_size):
        x = F.interpolate(x, size=(target_size, target_size), mode=mode,
                          align_corners=False, antialias=True)
    return x


def preprocess(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
               target_size: int | None = None, mode: str = "bicubic") -> torch.Tensor:
    """``[0,1]`` image -> resize to ``target_size`` -> ``(x - mean) / std``.

    ``mean`` / ``std`` are broadcastable ``(1,3,1,1)`` buffers on the same device as ``x``.
    """
    x = resize_to(x, target_size, mode)
    return (x - mean) / std
