"""Spiral diagnostic substrate (Fig. 3): a thin manifold in a high-dimensional space.

Real encoders place data on a thin manifold in a high-dimensional space; this isolates that
regime with a known target -- a two-turn spiral (intrinsic dim 2) orthogonally embedded in
``D = 64`` so the kernels see full ``D`` while the view stays 2D. A shared small MLP generator
is trained under each distance at a matched budget while the batch size is swept; samples are
scored by anchor ``recall`` and ``medDist`` (median distance to the curve), on which real data
floors at ~0.033.
"""
import math

import torch

# spiral geometry
TURNS, R0, R1, SPREAD = 2.0, 0.5, 2.5, 0.05
ZDIM = 2
D = 64                                   # ambient dim (intrinsic dim of the data stays 2)
SCALES = [0.1, 0.2, 0.4, 0.8, 1.6]       # multi-scale sigma multipliers x median pairwise dist
M_FEAT = 512                             # RFF features / Nystrom landmarks per scale (m/D = 8)
L_SW = 1000                              # SW projections
RECALL_RADIUS = 0.15
REAL_FLOOR = 0.033


def make_spiral(n, seed=0):
    """``n`` noisy points on the 2-turn spiral in R^2."""
    g = torch.Generator().manual_seed(seed)
    t = torch.rand(n, generator=g)
    th = TURNS * 2 * math.pi * t
    r = R0 + (R1 - R0) * t
    return torch.stack([r * torch.cos(th), r * torch.sin(th)], 1) + torch.randn(n, 2, generator=g) * SPREAD


def curve_pts(k):
    """``k`` points along the noiseless spiral curve (for anchors / medDist)."""
    t = torch.linspace(0, 1, k)
    th = TURNS * 2 * math.pi * t
    r = R0 + (R1 - R0) * t
    return torch.stack([r * torch.cos(th), r * torch.sin(th)], 1)


ANCH = curve_pts(60)
DENSE = curve_pts(1500)
real_tr = make_spiral(2000, 0)           # the reference / training spiral


class MLP(torch.nn.Module):
    """Shared toy generator: z in R^2 -> point in R^Dout."""

    def __init__(self, Dout, h=256):
        super().__init__()
        layers = [torch.nn.Linear(ZDIM, h), torch.nn.ReLU()]
        for _ in range(3):
            layers += [torch.nn.Linear(h, h), torch.nn.ReLU()]
        layers += [torch.nn.Linear(h, Dout)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


def real_sigma(realD, nsub=2000):
    """Median pairwise distance (sqrt of median squared distance) on a subsample."""
    nsub = min(nsub, len(realD))
    sub = realD[torch.randperm(len(realD))[:nsub]]
    d2 = torch.cdist(sub, sub) ** 2
    iu = torch.triu_indices(len(sub), len(sub), 1)
    return d2[iu[0], iu[1]].median().sqrt().clamp(min=1e-5).item()


def metrics(p2):
    """(recall, medDist): anchor recall within RECALL_RADIUS, median distance to the curve."""
    rec = (torch.cdist(ANCH, p2).min(1).values < RECALL_RADIUS).float().mean().item()
    md = torch.cdist(p2, DENSE).min(1).values.median().item()
    return rec, md
