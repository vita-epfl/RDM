"""Frozen text tower for the coupled feature tau(c).

The conditional objective concatenates a frozen text embedding tau(c) onto each image
encoder's features (see :mod:`rdm.representation.joint_feature`). tau is the **text tower**
of a SigLIP2 SO400M model (``ViT-SO400M-16-SigLIP2-256`` via open_clip, matching the
panel's trained SigLIP2 image encoder geometry); ``encode_text`` followed by L2
normalization onto the unit sphere. Only the text tower is used -- no image tower.

In practice the per-caption embeddings are precomputed once over the prompt set and frozen
to a ``(N, d_txt)`` table (:func:`encode_captions`), then memory-mapped at train time and
indexed by caption id. :class:`TextEncoder` is the runtime that builds that table.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("rdm")

#: Preferred SigLIP2 SO400M text models (first available in open_clip wins).
PREFERRED = ("ViT-SO400M-16-SigLIP2-256", "ViT-SO400M-16-SigLIP2-384",
             "ViT-SO400M-14-SigLIP2-378", "ViT-SO400M-16-SigLIP2-512")


def _auto_pick(prefer=PREFERRED):
    """Pick a SigLIP2 SO400M (model, pretrained) pair from open_clip's pretrained list."""
    import open_clip
    pairs = open_clip.list_pretrained()
    sig2 = [(m, p) for (m, p) in pairs if "SigLIP2" in m and "SO400M" in m]
    for pref in prefer:
        for (m, p) in sig2:
            if m == pref:
                return m, p
    if sig2:
        return sig2[0]
    raise RuntimeError(f"no SigLIP2 SO400M model in open_clip {open_clip.__version__}")


class TextEncoder(nn.Module):
    """Frozen SigLIP2 text tower: captions -> L2-normalized tau(c) of shape ``(N, d_txt)``."""

    def __init__(self, model_name: str | None = None, pretrained: str | None = None,
                 device: str = "cuda"):
        super().__init__()
        import open_clip
        if model_name is None:
            model_name, pretrained = _auto_pick()
        logger.info("[text_encoder] %s (%s)", model_name, pretrained)
        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = self.model.to(device).eval().requires_grad_(False)
        self.model_name = model_name
        self.device = device

    @torch.no_grad()
    def encode(self, captions: list[str], batch: int = 512) -> torch.Tensor:
        """Encode captions to an L2-normalized ``(len(captions), d_txt)`` fp32 table."""
        out = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(captions), batch):
                toks = self.tokenizer(captions[i:i + batch]).to(self.device)
                t = F.normalize(self.model.encode_text(toks).float(), dim=-1)
                out.append(t.cpu())
        return torch.cat(out, 0)


@torch.no_grad()
def encode_captions(captions: list[str], out_path: str, model_name: str | None = None,
                    pretrained: str | None = None, batch: int = 512, device: str = "cuda"):
    """Precompute the frozen tau(c) table for a caption list and save it to ``out_path`` (.npy)."""
    import numpy as np
    emb = TextEncoder(model_name, pretrained, device).encode(captions, batch).numpy().astype("float32")
    np.save(out_path, emb)
    logger.info("[text_encoder] wrote %s shape=%s ||row0||=%.4f",
                out_path, emb.shape, float((emb[0] ** 2).sum() ** 0.5))
    return emb


def load_text_table(path: str, device: str = "cpu", mmap: bool = True) -> torch.Tensor:
    """Load a frozen tau(c) table (.npy); ``mmap`` keeps the large prompt pool off-RAM."""
    import numpy as np
    arr = np.load(path, mmap_mode="r" if mmap else None)
    return torch.from_numpy(np.ascontiguousarray(arr)).to(device)
