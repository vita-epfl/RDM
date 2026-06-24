"""Web-SSL DINO 1B backbone (Meta, ViT-1B trained on web images), CLS, held-in encoder.

A DINOv2-style ViT-1B trained on web-scale SSL, loaded via HF transformers (not timm).
The wrapper returns its 1536-d pooled (CLS) embedding from a ``[0, 1]`` NCHW image,
ImageNet-normalized and resized to 224.
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..preprocess import IMAGENET_MEAN, IMAGENET_STD

logger = logging.getLogger("rdm")


class WebSSLBackbone(nn.Module):
    """Frozen Web-SSL DINO 1B -> 1536-d CLS embedding from ``[0, 1]`` NCHW input."""

    has_logits = False

    def __init__(self, device: str = "cuda", model_id: str = "facebook/webssl-dino1b-full2b-224",
                 target_size: int = 224):
        super().__init__()
        from transformers import AutoModel
        logger.info("[webssl_dino_1b] loading %s", model_id)
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.model.to(device).eval().requires_grad_(False)
        self.feat_dim = int(self.model.config.hidden_size)
        self.target_size = int(target_size)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.target_size or x.shape[-2] != self.target_size:
            x = F.interpolate(x, size=(self.target_size, self.target_size),
                              mode="bicubic", align_corners=False, antialias=True)
        x = (x - self.mean) / self.std
        return self.model(x).pooler_output   # (B, 1536)
