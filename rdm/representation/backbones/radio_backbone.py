"""C-RADIOv3-L backbone (NVIDIA multi-teacher agglomerative), summary token, held-out.

C-RADIOv3-L distills DINOv2 + SigLIP + SAM + OpenCLIP into one ViT-L. The wrapper returns
its 3072-d summary token from a ``[0, 1]`` NCHW image (resized to 256); normalization is
handled inside RADIO's own input conditioner. Held out from training (evaluation only).
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("rdm")


class RadioBackbone(nn.Module):
    """Frozen C-RADIOv3-L -> summary token (3072-d) from ``[0, 1]`` NCHW input."""

    has_logits = False

    def __init__(self, device: str = "cuda", model_id: str = "nvidia/C-RADIOv3-L",
                 target_size: int = 256):
        super().__init__()
        from transformers import AutoModel
        logger.info("[cradiov3_l] loading %s", model_id)
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.model.to(device).eval().requires_grad_(False)
        self.target_size = int(target_size)
        with torch.inference_mode():
            out = self.model(torch.zeros(1, 3, target_size, target_size, device=device))
            self.feat_dim = int(out[0].shape[-1])   # summary token dim (3072)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.target_size or x.shape[-2] != self.target_size:
            x = F.interpolate(x, size=(self.target_size, self.target_size),
                              mode="bicubic", align_corners=False, antialias=True)
        out = self.model(x)
        return out[0]   # summary token (patch features out[1] are unused)
