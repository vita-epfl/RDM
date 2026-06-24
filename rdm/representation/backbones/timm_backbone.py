"""Generic frozen-timm backbone covering most of the panel.

Wraps a timm model as a frozen feature extractor that maps a ``[0, 1]`` NCHW image to a
single ``(B, D)`` embedding under the encoder's conceptual pool (Table 5):

* CNN feature map (``feats.ndim == 4``) -> spatial mean (Inception is its own backbone;
  ConvNeXt lands here);
* ViT ``AVG`` -> mean over patch tokens;
* ViT ``CLS`` -> token-0 (prefix), or the model's pool head if there is no prefix token;
* ViT ``ATTN`` -> the attention-pool head. The PE-Core case (a prefix token *and* an
  attn-pool head) is handled by forcing the attn head; SigLIP2 (no prefix) reaches the
  same head through the pool fallback -- this reproduces the original feature selection
  exactly, so the extracted features match the precomputed reference bundles.

mean/std and the resize interpolation are resolved from the model's timm
``pretrained_cfg``; no feature normalization is applied to the output.
"""
import logging

import torch
import torch.nn as nn

from ..encoder_spec import EncoderSpec, PoolType
from ..preprocess import preprocess

logger = logging.getLogger("rdm")


class TimmBackbone(nn.Module):
    """Frozen timm encoder -> ``(B, D)`` per ``spec.pool``."""

    has_logits = False

    def __init__(self, spec: EncoderSpec, device: str = "cuda"):
        super().__init__()
        import timm
        from timm.data import resolve_data_config

        kwargs = dict(pretrained=True, num_classes=0)
        try:
            self.model = timm.create_model(spec.model_id, dynamic_img_size=True,
                                           dynamic_img_pad=True, **kwargs)
        except TypeError:  # not all backbones accept the dynamic-size kwargs
            self.model = timm.create_model(spec.model_id, **kwargs)
        self.model.to(device).eval().requires_grad_(False)

        self.pool = spec.pool
        self.input_res = spec.input_res
        self.num_prefix_tokens = getattr(self.model, "num_prefix_tokens", 0)
        self.has_attn_pool = getattr(self.model, "attn_pool", None) is not None
        # PE-Core has BOTH a prefix token and an attn-pool head; token-0 is not its
        # trained output, so an ATTN spec with prefix tokens must force the attn head.
        self.cls_use_attn_pool = (spec.pool is PoolType.ATTN and self.num_prefix_tokens > 0)
        self.feat_dim = int(self.model.num_features)
        if self.feat_dim != spec.dim:
            logger.warning("[%s] num_features=%d != spec.dim=%d", spec.name, self.feat_dim, spec.dim)

        cfg = resolve_data_config(self.model.pretrained_cfg)
        self.interp = cfg.get("interpolation", "bicubic")
        self.register_buffer("mean", torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = preprocess(x, self.mean, self.std, self.input_res, self.interp)
        feats = self.model.forward_features(x)
        if feats.ndim == 4:                                   # CNN feature map -> spatial mean
            return feats.mean(dim=[2, 3])
        patch = feats[:, self.num_prefix_tokens:]
        if self.pool is PoolType.AVG:
            return patch.mean(1)
        if self.cls_use_attn_pool and self.has_attn_pool:     # PE-Core
            return self.model.attn_pool(feats)
        if self.num_prefix_tokens > 0:                        # CLS token-0
            return feats[:, 0]
        if self.has_attn_pool:                                # SigLIP2 / SigLIP-v1 pool head
            pool_head = getattr(self.model, "pool", None) or getattr(self.model, "_pool", None)
            return pool_head(feats)
        return patch.mean(1)                                  # fallback
