"""One-step generator wrappers + the :func:`build_generator` dispatch.

``Generator`` is the shared interface; ``PMFHGenerator`` is the ImageNet one-step wrapper
and ``FluxGenerator`` the FLUX.2 4->1 wrapper (lazy-imported so the external ``flux2``
dependency is only needed for the FLUX path).
"""
from .base import Generator
from .pmfh_generator import PMFHGenerator


def build_generator(mode: str, model, sampling_args: dict, args=None, tokenizer=None) -> Generator:
    """Construct the generator wrapper for ``mode`` ('imagenet' / 'flux')."""
    if mode in ("imagenet", "pmfh"):
        return PMFHGenerator(model, sampling_args, args=args, tokenizer=tokenizer)
    if mode == "flux":
        from .flux_generator import FluxGenerator
        return FluxGenerator(model, sampling_args, tokenizer, args=args)
    raise ValueError(f"unknown generator mode {mode!r}; expected 'imagenet' or 'flux'")


__all__ = ["Generator", "PMFHGenerator", "build_generator"]
