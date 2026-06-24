"""FLUX.2 [klein] native text-context encoder (Qwen3) -- the in-repo ctx_pool builder.

Turns prompt strings into the conditioning tensor the FLUX.2 MM-DiT consumes:
``(B, L, 7680)``, where ``7680 = 3 x 2560`` is the Qwen3-4B hidden size stacked over layers
``[9, 18, 27]`` and ``L = ctx_len`` is a fixed pad/truncate length. This is the FLUX.2
*generation* conditioning, NOT the SigLIP2 tau(c) used by the joint loss (that lives in
:mod:`rdm.representation.text_encoder`); the two are different text features.

It is the runtime that produces both the offline training ``ctx_pool`` AND the per-prompt
context for evaluation -- the GenEval / Pick-a-Pic prompts must be encoded HERE (not sliced
from the COCO ctx_pool), or the generator renders the wrong captions.

``ctx_len`` MUST match the length the training ctx_pool was built with, so eval contexts have
the same sequence geometry the student was trained on. Faithful to the offline builder:
per-prompt chat template with ``enable_thinking=False``, pad/truncate to ``ctx_len``, hidden
layers ``[9, 18, 27]`` stacked. Lazy-imports the external ``flux2`` package (point to it with
the ``FLUX2_SRC`` env var or ``flux2_src=``), so this module imports even where flux2 is absent.
"""
from __future__ import annotations

import logging
import os
import sys

import torch
import torch.nn as nn

logger = logging.getLogger("rdm")

#: Qwen3-4B context width = 3 stacked hidden layers x 2560 hidden dim.
FLUX2_QWEN3_DIM = 7680


def _ensure_flux2_importable(flux2_src: str | None = None) -> None:
    src = flux2_src or os.environ.get("FLUX2_SRC")
    if src and src not in sys.path:
        sys.path.insert(0, src)


class Flux2TextContextEncoder(nn.Module):
    """Frozen FLUX.2 Qwen3 text encoder: ``list[str]`` -> ``(B, ctx_len, 7680)`` context."""

    def __init__(self, ctx_len: int, variant: str = "4B", flux2_src: str | None = None,
                 device: str = "cuda"):
        super().__init__()
        _ensure_flux2_importable(flux2_src)
        from flux2.text_encoder import load_qwen3_embedder
        logger.info("[flux2-text] loading Qwen3-%s embedder (ctx_len=%d)", variant, ctx_len)
        self.embedder = load_qwen3_embedder(variant=variant, device=device)
        self.embedder.max_length = int(ctx_len)        # fixed pad/truncate length L
        self.ctx_len = int(ctx_len)
        self.device = device

    @torch.no_grad()
    def encode(self, prompts: list[str], batch: int = 16) -> torch.Tensor:
        """Encode prompts to a ``(len(prompts), ctx_len, 7680)`` fp32 context table (on CPU)."""
        if not prompts:
            return torch.empty(0, self.ctx_len, FLUX2_QWEN3_DIM)
        out = []
        for i in range(0, len(prompts), batch):
            ctx = self.embedder([str(p) for p in prompts[i:i + batch]])   # (b, ctx_len, 7680) bf16
            out.append(ctx.float().cpu())
        return torch.cat(out, 0)


@torch.no_grad()
def encode_flux2_context(prompts: list[str], ctx_len: int, *, variant: str = "4B",
                         flux2_src: str | None = None, device: str = "cuda",
                         batch: int = 16) -> torch.Tensor:
    """One-shot prompts -> ``(N, ctx_len, 7680)`` FLUX.2 context (fp32, CPU)."""
    enc = Flux2TextContextEncoder(ctx_len, variant=variant, flux2_src=flux2_src, device=device)
    return enc.encode(prompts, batch=batch)
