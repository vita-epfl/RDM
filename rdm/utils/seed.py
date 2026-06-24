"""Determinism: global seeding and independent per-metric RNG streams.

``GLOBAL_SEED = 0``. :func:`fix_random_seeds` seeds Python, NumPy, and Torch (CPU + all
CUDA devices). :func:`independent_generator` mints a named, reproducible
:class:`torch.Generator` so each metric / sampler gets its own stream that does not
disturb the global RNG (the evaluation metrics each draw their own projections /
subsamples). For evaluation, :func:`enable_deterministic` turns on deterministic cuDNN.
"""
import hashlib
import os
import random

import numpy as np
import torch

GLOBAL_SEED = 0


def fix_random_seeds(seed: int = GLOBAL_SEED) -> None:
    """Seed Python / NumPy / Torch (CPU and CUDA) for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def independent_generator(name: str, base_seed: int = GLOBAL_SEED,
                          device: str = "cpu") -> torch.Generator:
    """A named, reproducible generator derived from ``(name, base_seed)``.

    Hashing the name keeps each stream independent of the others and stable across runs.
    """
    h = int(hashlib.sha256(f"{name}:{base_seed}".encode()).hexdigest(), 16) % (2 ** 31)
    return torch.Generator(device=device).manual_seed(h)


def enable_deterministic() -> None:
    """Deterministic cuDNN for evaluation (slower; not used in training)."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
