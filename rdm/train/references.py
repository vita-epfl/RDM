"""Runtime loader for the frozen reference artifacts the training loop consumes.

Loads each encoder's Nystrom bundle (``{Z, alpha, sigma, k_rr}``) into a
:class:`rdm.compare.interface.ReferencePack` on the training device (fp32, read-only), plus
the PID real-validation floors. Joint (text-to-image) bundles carry the same fields over the
widened ``[phi | beta*tau]`` feature, so they load through the same path; the frozen text
table tau(c) is loaded separately and memory-mapped.
"""
import torch

from ..compare.interface import ReferencePack
from ..utils.io import load_bundle, read_json


def load_reference_packs(bundle_paths: dict, weights: dict | None = None,
                         device: str = "cuda") -> dict:
    """Load ``{name: path}`` Nystrom bundles into ``{name: ReferencePack}`` on ``device`` (fp32)."""
    weights = weights or {}
    packs = {}
    for name, path in bundle_paths.items():
        b = load_bundle(path)
        rp = ReferencePack.from_nystrom_bundle(name, b, weight=float(weights.get(name, 1.0)))
        rp.Z = rp.Z.to(device)
        rp.Z2 = rp.Z2.to(device)
        rp.alpha = rp.alpha.to(device)
        packs[name] = rp
    return packs


def load_floors(json_path: str) -> dict:
    """Load the per-encoder PID floors ``{name: b_phi}``."""
    return {k: float(v) for k, v in read_json(json_path).items()}


def load_text_table(path: str, device: str = "cpu", mmap: bool = True) -> torch.Tensor:
    """Load the frozen tau(c) text table (.npy); mmap keeps the large prompt pool off-RAM."""
    import numpy as np
    arr = np.load(path, mmap_mode="r" if mmap else None)
    return torch.from_numpy(np.ascontiguousarray(arr)).to(device)
