"""Low-dimensional diagnostics on the spiral substrate (Fig. 3, Fig. 4, Table 4).

:mod:`spiral` is the substrate; :mod:`distances_toy` trains the six distances on it;
:mod:`run_spiral_grid` / :mod:`run_batch_axis` / :mod:`run_ablations` are the drivers.
"""
from .distances_toy import train_distance
from .spiral import D, MLP, REAL_FLOOR, curve_pts, make_spiral, metrics
from .run_spiral_grid import EXPECTED, METHODS
from .run_spiral_grid import run as run_spiral_grid
from .run_batch_axis import run as run_batch_axis
from .run_ablations import run as run_ablations

__all__ = ["make_spiral", "curve_pts", "MLP", "metrics", "D", "REAL_FLOOR", "train_distance",
           "run_spiral_grid", "run_batch_axis", "run_ablations", "EXPECTED", "METHODS"]
