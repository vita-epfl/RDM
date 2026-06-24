"""Per-encoder real-validation floor b_phi -- the PID-Lagrangian constraint target.

The proportional-Lagrangian controller (:mod:`rdm.train.pid_lagrangian`) upweights an
encoder by how far the generator's score sits above the encoder's real floor. That floor
``b_phi`` is the *same* training distance evaluated on held-out real validation features:
the mean over ``k_draws`` random draws of size ``n_draw`` of the Nystrom MMD of a real-val
sample against the frozen train reference. Because it uses the exact training loss, the
floor and the live score are on the same scale (including the constant ``k_rr``), so the
satisfaction gate ``s_phi <= b_phi`` is meaningful.
"""
import torch

from ..compare.mmd_nystrom import mmd_nystrom_loss
from ..utils.io import load_bundle, write_json


@torch.no_grad()
def compute_floor(val_features: torch.Tensor, bundle: dict, n_draw: int = 5120,
                  k_draws: int = 8, device: str = "cuda", seed: int = 0) -> float:
    """Mean Nystrom-MMD of ``k_draws`` real-val samples (size ``n_draw``) vs the train reference."""
    Z = bundle["Z"].float().to(device)
    Z2 = (Z * Z).sum(1)
    alpha = bundle["alpha"].float().to(device)
    sigma, k_rr = float(bundle["sigma"]), float(bundle["k_rr"])
    feats = val_features.float()
    g = torch.Generator().manual_seed(seed)
    vals = []
    for _ in range(k_draws):
        idx = torch.randperm(feats.shape[0], generator=g)[:n_draw]
        sample = feats[idx].to(device)
        vals.append(float(mmd_nystrom_loss(sample, Z, Z2, alpha, sigma, k_rr)))
    return sum(vals) / len(vals)


def compute_all_floors(val_pool_paths: dict, bundle_paths: dict, out_json: str,
                       n_draw: int = 5120, k_draws: int = 8, device: str = "cuda") -> dict:
    """Compute b_phi for each encoder (keyed by short name) and save the PID-baselines JSON.

    ``val_pool_paths`` / ``bundle_paths`` map encoder short-name -> raw-val-pool / Nystrom
    bundle paths.
    """
    floors = {}
    for name, vpath in val_pool_paths.items():
        val = load_bundle(vpath)["features"]
        floors[name] = compute_floor(val, load_bundle(bundle_paths[name]), n_draw, k_draws, device)
    write_json(floors, out_json)
    return floors
