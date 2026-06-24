"""End-to-end ImageNet evaluation: SW_r14 (primary) + MMDr14 (secondary) + PickScore.

Renders class-conditional one-step samples, extracts the 14-encoder eval panel, and computes
the floor-normalized Sliced-Wasserstein ratio (:mod:`rdm.eval.sw_r14`, the primary metric) and
the RFF-MMD ratio (:mod:`rdm.eval.mmd_r14`, App D), plus the class-prompt PickScore (Fig. 6).
Operates on precomputed real train/val feature banks + the frozen eval RFF banks (disjoint
seed); off-objective throughout.
"""
import os

import torch

from .mmd_r14 import mmd_r14
from .panel import build_eval_battery, extract_features, panel_names
from .report import Report
from .sw_r14 import sw_r14


def load_eval_banks(eval_rff_dir: str, sw_bank_dir: str, mmdr14_floors_json: str,
                    names: list | None = None):
    """Load the precomputed eval banks built by :mod:`rdm.refprep.run` (``imagenet`` mode).

    Returns ``(train_feats, val_feats, rff_banks, mmd_val_floors)`` for whichever of the 14
    panel encoders have artifacts on disk -- the tuple :func:`evaluate_imagenet` consumes.
    """
    from ..utils.io import load_bundle, read_json
    names = names or panel_names()
    rff_banks, train_feats, val_feats = {}, {}, {}
    for n in names:
        rff_p = os.path.join(eval_rff_dir, f"{n}.pt")
        tr_p = os.path.join(sw_bank_dir, f"{n}_train.pt")
        va_p = os.path.join(sw_bank_dir, f"{n}_val.pt")
        if os.path.exists(rff_p):
            rff_banks[n] = load_bundle(rff_p)
        if os.path.exists(tr_p):
            train_feats[n] = load_bundle(tr_p)["features"]
        if os.path.exists(va_p):
            val_feats[n] = load_bundle(va_p)["features"]
    floors = {k: float(v) for k, v in read_json(mmdr14_floors_json).items()}
    return train_feats, val_feats, rff_banks, floors


@torch.no_grad()
def render_class_conditional(generator, n: int, num_classes: int, *, img_channels: int = 3,
                             img_size: int = 256, batch: int = 256, device: str = "cuda",
                             seed: int = 0, return_labels: bool = False):
    """Render ``n`` class-conditional one-step samples -> ``[0, 1]`` images (+ labels if asked)."""
    g = torch.Generator(device=device).manual_seed(seed)
    out, labels = [], []
    for lo in range(0, n, batch):
        b = min(batch, n - lo)
        noise = torch.randn(b, img_channels, img_size, img_size, generator=g, device=device)
        y = torch.randint(0, num_classes, (b,), generator=g, device=device)
        out.append(generator.sample(noise, y).cpu())
        labels.append(y.cpu())
    imgs = torch.cat(out, 0)
    return (imgs, torch.cat(labels, 0)) if return_labels else imgs


@torch.no_grad()
def evaluate_imagenet(generator, *, train_feats: dict, val_feats: dict, rff_banks: dict,
                      mmd_val_floors: dict, n_samples: int = 16384, num_classes: int = 1000,
                      img_size: int = 256, device: str = "cuda",
                      class_prompts: list | None = None, pickscore_n: int = 4000) -> dict:
    """Compute SW_r14 + MMDr14 (+ optional class-prompt PickScore) against precomputed banks.

    Args:
        train_feats / val_feats: ``{encoder: (N, d)}`` real train / val features (for SW ratios).
        rff_banks: ``{encoder: {"W","b","mu_r"}}`` frozen eval RFF banks (MMDr14).
        mmd_val_floors: ``{encoder: float}`` the real-val RFF-MMD^2 denominators.
        class_prompts: ``[num_classes]`` ``"a photo of a {classname}"`` prompts (class-index
            order); when given, the off-objective Fig. 6 ImageNet PickScore (mean over
            ``pickscore_n`` class-conditional renders) is computed. Auxiliary, not the headline.
    Returns ``{"sw": {...}, "mmd": {...}, "reports": [Report, Report], ["pickscore": float]}``.
    """
    battery = build_eval_battery(device)
    imgs = render_class_conditional(generator, n_samples, num_classes, img_size=img_size, device=device)
    gen_feats = {}
    for lo in range(0, imgs.shape[0], 256):
        for k, v in extract_features(battery, imgs[lo:lo + 256].to(device)).items():
            gen_feats.setdefault(k, []).append(v.cpu())
    gen_feats = {k: torch.cat(v, 0) for k, v in gen_feats.items()}

    sw = sw_r14(gen_feats, train_feats, val_feats)
    mmd = mmd_r14(gen_feats, rff_banks, mmd_val_floors)
    out = {"sw": sw, "mmd": mmd,
           "reports": [Report("iRDM", "SW_r14", sw["swr14"], sw["per_encoder"]),
                       Report("iRDM", "MMDr14", mmd["mmdr14"], mmd["per_encoder"])]}

    if class_prompts is not None:                              # Fig. 6 PickScore (off-objective)
        from .pickscore_eval import PickScorer, mean_pickscore
        p_imgs, labels = render_class_conditional(generator, pickscore_n, num_classes,
                                                  img_size=img_size, device=device, seed=1,
                                                  return_labels=True)
        prompts = [class_prompts[int(y)] for y in labels]
        out["pickscore"] = mean_pickscore(PickScorer(device=device), p_imgs, prompts)
    return out
