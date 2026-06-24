"""Toy spiral wiring tests (fast: tiny n_iters; numbers are not the published ones).

Checks the substrate + distance trainers run and produce finite recall/medDist with the
right sample geometry, and that the published EXPECTED grid is well-formed.
"""
import torch

from rdm.toy import EXPECTED, METHODS, REAL_FLOOR, make_spiral, metrics, train_distance
from rdm.toy.run_spiral_grid import ROWS


def test_spiral_substrate():
    pts = make_spiral(500, seed=1)
    assert pts.shape == (500, 2)
    rec, md = metrics(pts)                      # real data scores near the floor
    assert 0.0 <= rec <= 1.0 and md >= 0.0
    assert abs(REAL_FLOOR - 0.033) < 1e-9


def test_distance_trainers_run():
    for m in ["nystrom", "sw", "fd"]:
        p2, rec, md = train_distance(m, bs=32, seed=42, n_iters=20)   # tiny, just wiring
        assert p2.shape == (4000, 2)
        assert torch.isfinite(p2).all()
        assert 0.0 <= rec <= 1.0 and md >= 0.0


def test_expected_grid_wellformed():
    assert len(EXPECTED) == len(ROWS) * len(METHODS) == 18
    # Nystrom is the sharpest (lowest medDist) in every batch row of the published grid.
    for bs in ROWS:
        nys_md = EXPECTED[(bs, "nystrom")][1]
        for m in METHODS:
            assert nys_md <= EXPECTED[(bs, m)][1] + 1e-9, (bs, m)
