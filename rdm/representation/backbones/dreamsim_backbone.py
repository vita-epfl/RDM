"""DreamSim ensemble backbone (DINO + CLIP + OpenCLIP), 1792-d, held-in training encoder.

DreamSim is a LoRA-tuned ensemble fine-tuned for human visual similarity. The wrapper
returns its 1792-d embedding from a ``[0, 1]`` NCHW image (resized to 224).

DreamSim's torch.hub call for ``facebookresearch/dino`` imports a ``vision_transformer.py``
that does ``from utils import trunc_normal_``. That ``utils`` name collides with this repo's
own package, so during construction we prepend dino's path and temporarily evict
``utils`` / ``utils.*`` from ``sys.modules`` so Python resolves dino's local ``utils.py``;
both are reverted in ``finally``. (Requires ``dreamsim``; a pinned ``peft`` may be needed
depending on the environment.)
"""
import logging
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("rdm")


class DreamSimBackbone(nn.Module):
    """Frozen DreamSim ensemble -> 1792-d embedding from ``[0, 1]`` NCHW input."""

    has_logits = False

    def __init__(self, device: str = "cuda", target_size: int = 224, cache_dir: str | None = None):
        super().__init__()
        self.cache_dir = cache_dir or os.environ.get(
            "DREAMSIM_CACHE_DIR", os.path.expanduser("~/.cache/dreamsim"))
        dino_main_dir = os.path.join(self.cache_dir, "facebookresearch_dino_main")
        sys.path.insert(0, dino_main_dir)
        saved = {k: sys.modules.pop(k) for k in list(sys.modules)
                 if k == "utils" or k.startswith("utils.")}
        try:
            from dreamsim import dreamsim
            logger.info("[dreamsim] loading ensemble (DINO+CLIP+OpenCLIP)")
            self.model, _ = dreamsim(pretrained=True, device=device, cache_dir=self.cache_dir)
        finally:
            for k in list(sys.modules):
                if k == "utils" or k.startswith("utils."):
                    del sys.modules[k]
            sys.modules.update(saved)
            try:
                sys.path.remove(dino_main_dir)
            except ValueError:
                pass
        self.model.eval().requires_grad_(False)
        self.target_size = int(target_size)
        with torch.inference_mode():
            dummy = torch.zeros(1, 3, target_size, target_size, device=device)
            self.feat_dim = int(self.model.embed(dummy).shape[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.target_size or x.shape[-2] != self.target_size:
            x = F.interpolate(x, size=(self.target_size, self.target_size),
                              mode="bicubic", align_corners=False, antialias=True)
        return self.model.embed(x)
