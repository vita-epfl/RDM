"""Stream images through a frozen encoder and cache raw embeddings + the median bandwidth.

The first stage of the offline reference precompute: run a dataset of ``[0, 1]`` images
through one panel encoder, accumulate the raw ``(N, d)`` features (fp32, CPU), and save them
as a ``{"features": tensor}`` bundle. A companion helper caches the per-encoder median
bandwidth that the Nystrom / RFF / FD builders read back (the bandwidth is fixed once here,
never recomputed at train time). One encoder per call -- shard encoders across GPUs at the
shell level.
"""
import torch

from ..compare.median_heuristic import median_bandwidth
from ..representation.backbones import build_backbone


@torch.no_grad()
def extract_features(spec, dataset, batch_size: int = 256, num_workers: int = 8,
                     device: str = "cuda") -> torch.Tensor:
    """Run ``spec``'s frozen backbone over ``dataset`` -> raw ``(N, d)`` fp32 CPU features."""
    from torch.utils.data import DataLoader
    backbone = build_backbone(spec, device=device)
    use_amp = not getattr(backbone, "has_logits", False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    out = []
    for batch in loader:
        imgs = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
        with torch.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            feats = backbone(imgs.float() if not use_amp else imgs)
        out.append(feats.float().cpu())
    return torch.cat(out, 0)


def save_features(features: torch.Tensor, out_path: str, **meta) -> None:
    """Save a raw-feature pool as ``{"features": (N,d), **meta}``."""
    from ..utils.io import save_bundle
    save_bundle({"features": features, "n": int(features.shape[0]),
                 "d_in": int(features.shape[1]), **meta}, out_path)


def cache_median_sigma(features: torch.Tensor, out_path: str, scale: float = 1.0,
                       max_subsample: int = 10000) -> float:
    """Compute + save the per-encoder median-heuristic bandwidth ``{"sigma": float}``."""
    from ..utils.io import save_bundle
    sigma = median_bandwidth(features, max_subsample=max_subsample, scale=scale)
    save_bundle({"sigma": float(sigma), "scale": float(scale)}, out_path)
    return sigma
