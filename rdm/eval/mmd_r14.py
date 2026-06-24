"""MMDr14 -- the secondary kernel-MMD metric (App D) validating the SW_r14 ordering.

Per encoder, an RFF-MMD ratio against real training data:

    r_e = RFF-MMD^2(phi_e(gen), train) / RFF-MMD^2(phi_e(val), train)

with the frozen eval RFF bank ``{W, b, mu_r}`` (seed 99999, disjoint from the trained
reference), and ``MMDr14`` is the **arithmetic mean** over the 14 encoders (the published
iRDM value 2.69 is this arithmetic mean). The raw RFF-MMD^2 is scaled by 1000 by convention,
matching the val-floor table. Operates on extracted feature dicts; never imports the
training loss path.
"""
from ..compare.mmd_rff import rff_phi

SCALE = 1000.0


def rff_mmd2(feats, W, b, mu_r) -> float:
    """Biased RFF-MMD^2 ``|| mean phi(feats) - mu_r ||^2`` x 1000 (the metric convention)."""
    mu = rff_phi(feats.float().to(W.device), W, b).mean(0)
    return float((mu - mu_r).pow(2).sum()) * SCALE


def mmd_r14(gen_feats: dict, rff_banks: dict, val_floors: dict) -> dict:
    """MMDr14 = arithmetic mean of per-encoder ``rff_mmd2(gen) / val_floor`` ratios.

    Args:
        gen_feats: ``{name: (N, d)}`` generated features.
        rff_banks: ``{name: {"W","b","mu_r"}}`` frozen eval RFF banks.
        val_floors: ``{name: float}`` the real-val RFF-MMD^2 (x1000) denominators.
    Returns ``{"mmdr14": float, "per_encoder": {name: ratio}}``.
    """
    ratios = {}
    for name, feats in gen_feats.items():
        if name not in rff_banks or name not in val_floors:
            continue
        bank = rff_banks[name]
        m = rff_mmd2(feats, bank["W"].float(), bank["b"].float(), bank["mu_r"].float())
        floor = float(val_floors[name])
        ratios[name] = m / floor if floor > 0 else float("nan")
    vals = [r for r in ratios.values() if r == r]
    return {"mmdr14": sum(vals) / len(vals) if vals else float("nan"), "per_encoder": ratios}
