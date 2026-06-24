"""Offline FLUX render driver for the joint reference (teacher / student generations).

Renders images for a list of prompts with the FLUX.2 adapter and writes them to disk,
sharded across ranks with skip-if-exists resume. The four-step teacher (``num_steps=4``)
generates the FLUX side of the joint reference (a few seeds per prompt); the same driver
with ``num_steps=1`` and a trained checkpoint renders student samples for evaluation.

The prompts here are precomputed text-context tensors ``(N, L, d)`` (the FLUX text encoder
is offline; see :mod:`rdm.data.prompts` for the raw-text side). This is a thin driver over
:class:`rdm.representation.generators.flux_generator.FluxGenerator`; it is intentionally
infrastructure-light (no cluster specifics).
"""
import os

import torch

from ..utils.io import save_uint8_png


@torch.no_grad()
def render_prompts(generator, ctx_table: torch.Tensor, out_dir: str, *, num_steps: int = 4,
                   seeds=(0,), latent_channels: int = 128, latent_size: int = 32,
                   batch_size: int = 8, rank: int = 0, world_size: int = 1,
                   device: str = "cuda") -> int:
    """Render ``ctx_table`` rows (× seeds) with ``generator`` to ``out_dir/<row>_<seed>.png``.

    Sharded round-robin by ``rank``; skips rows whose output already exists. Returns the
    number of images written by this rank.
    """
    os.makedirs(out_dir, exist_ok=True)
    generator.sampling_args = {**generator.sampling_args, "num_steps": num_steps}
    written = 0
    n = ctx_table.shape[0]
    work = [(i, s) for i in range(n) for s in seeds if (i * len(seeds)) % world_size == rank]
    for lo in range(0, len(work), batch_size):
        chunk = work[lo:lo + batch_size]
        todo = [(i, s) for (i, s) in chunk
                if not os.path.exists(os.path.join(out_dir, f"{i:08d}_s{s}.png"))]
        if not todo:
            continue
        ctx = torch.stack([ctx_table[i] for (i, _s) in todo]).to(device)
        gens = []
        for (i, s) in todo:
            g = torch.Generator(device=device).manual_seed(int(s) * 1_000_003 + int(i))
            gens.append(torch.randn(latent_channels, latent_size, latent_size,
                                    generator=g, device=device))
        noise = torch.stack(gens)
        imgs = generator.sample(noise, ctx)                  # (b, 3, H, W) in [0,1]
        for (i, s), img in zip(todo, imgs):
            save_uint8_png(img, os.path.join(out_dir, f"{i:08d}_s{s}.png"))
            written += 1
    return written
