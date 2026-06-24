"""Fresh on-policy noise and condition samplers.

The generated side moves every step and is drawn fresh (estimating it from a stale buffer
biases the gradient off-policy), so each step samples a new batch of noise and conditions.
``sample_noise`` draws standard-normal latents; the conditioners draw class ids (ImageNet)
or index a precomputed text-context pool (FLUX). Conditions are also used to row-align the
joint reference: the FLUX conditioner returns both the ctx tensor (for the generator) and
the caption indices (for the text block tau(c)).
"""
import torch


def sample_noise(batch: int, shape, device: str = "cuda",
                 generator: torch.Generator | None = None) -> torch.Tensor:
    """Standard-normal latent noise ``(batch, *shape)`` (the generator applies any noise scale)."""
    return torch.randn(batch, *shape, device=device, generator=generator)


class ClassConditioner:
    """ImageNet class-id sampler: uniform ``randint(0, num_classes)``."""

    def __init__(self, num_classes: int = 1000):
        self.num_classes = num_classes

    def sample(self, batch: int, device: str = "cuda", generator=None) -> torch.Tensor:
        return torch.randint(0, self.num_classes, (batch,), device=device, generator=generator)

    def encode_for_generator(self, condition):
        return condition  # class ids feed the generator directly


class PromptConditioner:
    """FLUX prompt sampler: draw rows from a precomputed text-context pool ``(N, L, d)``.

    Returns ``(ctx, caption_ids)`` -- ``ctx`` feeds the generator, ``caption_ids`` index the
    frozen tau(c) table for the joint reference (so the generated pair couples each output
    with the prompt that produced it).
    """

    def __init__(self, ctx_pool: torch.Tensor):
        self.ctx_pool = ctx_pool   # (N, L, d), possibly mmap on CPU
        self.num_prompts = ctx_pool.shape[0]

    def sample(self, batch: int, device: str = "cuda", generator=None):
        idx = torch.randint(0, self.num_prompts, (batch,), generator=generator)
        ctx = self.ctx_pool[idx].to(device)
        return ctx, idx

    def encode_for_generator(self, condition):
        ctx, _caption_ids = condition
        return ctx
