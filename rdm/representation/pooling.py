"""Pooling tokens / feature maps to one ``(B, D)`` embedding -- no normalization.

The five conceptual pools of Table 5 (:class:`rdm.representation.encoder_spec.PoolType`):

* ``CLS``  -- the class / prefix token (token 0) of a ViT;
* ``AVG``  -- mean over patch tokens (ViT) or spatial mean (CNN feature map);
* ``ATTN`` -- the model's own attention-pooling head output (its intended global feature,
  e.g. PE-Core / SigLIP2 -- token 0 is not the trained output for these);
* ``SUMMARY``    -- RADIO's summary token (handled in the RADIO backbone);
* ``PATCH_MEAN`` -- FLUX VAE 4x4 latent patch-mean (handled in the FLUX-VAE backbone).

The token-level pools (``CLS``, ``AVG``, ``ATTN`` over a token sequence, and the CNN
spatial mean) are pure functions here; ``SUMMARY`` and ``PATCH_MEAN`` are intrinsic to
their backbone and applied there. No L2 / feature normalization is ever applied -- the
kernels operate on raw embeddings.
"""
import torch


def cls_token(tokens: torch.Tensor) -> torch.Tensor:
    """Token-0 (class / first prefix token) of a ``(B, T, D)`` sequence."""
    return tokens[:, 0]


def avg_tokens(tokens: torch.Tensor, num_prefix_tokens: int = 0) -> torch.Tensor:
    """Mean over the patch tokens of a ``(B, T, D)`` sequence (skipping prefix tokens)."""
    return tokens[:, num_prefix_tokens:].mean(dim=1)


def spatial_mean(feature_map: torch.Tensor) -> torch.Tensor:
    """Spatial mean of a CNN feature map ``(B, C, H, W) -> (B, C)``."""
    return feature_map.mean(dim=[2, 3])


def attn_pool(model: torch.nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    """The model's attention-pooling head applied to its token sequence ``(B, T, D)``."""
    return model.attn_pool(tokens)
