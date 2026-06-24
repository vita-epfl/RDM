"""FLUX VAE backbone -- 4x4 latent patch-mean -> 1024-d, held-out evaluation encoder.

The FLUX.1 VAE has 16 latent channels at 1/8 resolution; for a 256px input the latent is
(16, 32, 32). A 4x4 spatial patch-mean reduces it to (16, 8, 8) -> 1024-d, matching the
DINOv2/DINOv3-L geometry. Input is ``[0, 1]`` NCHW, rescaled to ``[-1, 1]`` for the VAE; no
ImageNet normalization. The VAE is frozen but autograd flows to the input (no
``inference_mode``). Loaded from an openly-distributable FLUX.1 VAE.
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("rdm")

DEFAULT_FLUX_VAE_REPO = "black-forest-labs/FLUX.1-schnell"


class FluxVAEBackbone(nn.Module):
    """Frozen FLUX VAE with 4x4 latent patch-mean -> (B, 1024) from ``[0, 1]`` NCHW input."""

    has_logits = False

    def __init__(self, device: str = "cuda", target_size: int = 256, patch_size: int = 4,
                 repo_id: str = DEFAULT_FLUX_VAE_REPO, dtype=torch.float32):
        super().__init__()
        from diffusers.models import AutoencoderKL
        logger.info("[flux_vae] loading %s (subfolder='vae')", repo_id)
        self.vae = AutoencoderKL.from_pretrained(repo_id, subfolder="vae", torch_dtype=dtype)
        self.vae.requires_grad_(False).eval()
        self.vae = self.vae.to(device=device, dtype=dtype)
        self.target_size = int(target_size)
        self.patch_size = int(patch_size)
        self.latent_channels = int(self.vae.config.latent_channels)
        self.spatial = self.target_size // 8
        if self.spatial % self.patch_size != 0:
            raise ValueError(f"spatial={self.spatial} not divisible by patch_size={self.patch_size}")
        self.grid = self.spatial // self.patch_size
        self.feat_dim = self.latent_channels * self.grid * self.grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.target_size or x.shape[-2] != self.target_size:
            x = F.interpolate(x, size=(self.target_size, self.target_size),
                              mode="bilinear", align_corners=False, antialias=True)
        x = (x * 2.0 - 1.0).to(dtype=next(self.vae.parameters()).dtype)
        latent = self.vae.encode(x).latent_dist.mean.float()    # (B, 16, H/8, W/8)
        B, C, H, W = latent.shape
        ps = self.patch_size
        latent = latent.reshape(B, C, H // ps, ps, W // ps, ps).mean(dim=(3, 5))
        return latent.flatten(1)                                # (B, 16*grid*grid)
