"""Weight download / cache resolution for the encoder battery and generators.

Loading is lazy by construction -- each backbone fetches its weights at first
instantiation (timm ``pretrained=True``, HF ``from_pretrained``, torch.hub), so an
evaluation-only encoder's weights are never downloaded during a training run. This module
only centralizes the cache locations and offers a one-shot warm-up / verification.

Set the cache roots once via :func:`configure_caches` (or the standard ``HF_HOME`` /
``TORCH_HOME`` / ``DREAMSIM_CACHE_DIR`` environment variables); the per-encoder source
ids live in :data:`rdm.representation.registry.PANEL`.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("rdm")

#: Non-panel weights the repo also needs (documented for the download script).
EXTRA_WEIGHTS = {
    "inception_fid": "github:toshas/torch-fidelity weights-inception-2015-12-05",
    "flux_vae": "black-forest-labs/FLUX.1-schnell (subfolder=vae)",
    "siglip2_text": "open_clip ViT-SO400M-16-SigLIP2-256 (text tower, tau(c))",
}


def configure_caches(hf_home: str | None = None, torch_home: str | None = None,
                     dreamsim_cache: str | None = None) -> None:
    """Point the HF / torch.hub / DreamSim caches at a writable location (no-op for unset)."""
    if hf_home:
        os.environ["HF_HOME"] = hf_home
    if torch_home:
        os.environ["TORCH_HOME"] = torch_home
    if dreamsim_cache:
        os.environ["DREAMSIM_CACHE_DIR"] = dreamsim_cache
    logger.info("[checkpoints] HF_HOME=%s TORCH_HOME=%s DREAMSIM_CACHE_DIR=%s",
                os.environ.get("HF_HOME"), os.environ.get("TORCH_HOME"),
                os.environ.get("DREAMSIM_CACHE_DIR"))


def warm_cache(specs=None, device: str = "cuda") -> None:
    """Instantiate each backbone once to download + cache its weights (then release it)."""
    from .backbones import build_backbone
    from .registry import all_specs
    for spec in (specs or all_specs()):
        logger.info("[checkpoints] fetching %s (%s)", spec.name, spec.source)
        enc = build_backbone(spec, device=device)
        del enc
