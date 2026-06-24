"""SW_r14 -- the primary metric (eq. 5): a Sliced-Wasserstein ratio over the 14-encoder panel.

For each encoder the per-encoder ratio is the floor-normalized Sliced-Wasserstein distance

    r_e = SW(phi_e(gen), phi_e(train)) / SW(phi_e(val), phi_e(train))

(the denominator is the irreducible SW floor between two independent real draws), and
``SW_r14`` is the arithmetic mean over the 14 encoders (matching MMDr14's aggregate). SW is
an optimal-transport distance sharing no machinery with the kernel MMD we train against, so
a gain cannot be the loss read back -- real validation data scores ~1 by construction.

This module never imports the training loss path: it pulls the SW primitive from
:mod:`rdm.compare.sliced_wasserstein` and operates on extracted feature dicts. All three
feature sets are truncated to the same ``N`` (sorted matching needs equal sizes), and the
same projection seed is reused across encoders / checkpoints / the floor so the values are
comparable.
"""
from ..compare.sliced_wasserstein import sliced_wasserstein


def sw_ratio_per_encoder(gen_feats: dict, train_feats: dict, val_feats: dict,
                         n_proj: int = 1024, p: int = 2, seed: int = 0) -> dict:
    """Per-encoder SW ratio ``SW(gen,train) / SW(val,train)`` for each shared encoder."""
    ratios = {}
    for name in gen_feats:
        if name not in train_feats or name not in val_feats:
            continue
        n = min(gen_feats[name].shape[0], train_feats[name].shape[0], val_feats[name].shape[0])
        g, tr, v = gen_feats[name][:n], train_feats[name][:n], val_feats[name][:n]
        floor = sliced_wasserstein(v, tr, n_proj, p, seed=seed)
        sw = sliced_wasserstein(g, tr, n_proj, p, seed=seed)
        ratios[name] = float(sw / floor) if floor > 0 else float("nan")
    return ratios


def sw_r14(gen_feats: dict, train_feats: dict, val_feats: dict, n_proj: int = 1024,
           p: int = 2, seed: int = 0) -> dict:
    """SW_r14 = arithmetic mean of the per-encoder SW ratios (+ the per-encoder breakdown).

    Returns ``{"swr14": float, "swr_dinov2": float|None, "per_encoder": {name: ratio}}``.
    """
    ratios = sw_ratio_per_encoder(gen_feats, train_feats, val_feats, n_proj, p, seed)
    vals = [r for r in ratios.values() if r == r]                # drop NaNs
    return {"swr14": sum(vals) / len(vals) if vals else float("nan"),
            "swr_dinov2": ratios.get("dinov2"),
            "per_encoder": ratios}
