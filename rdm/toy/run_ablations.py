"""Distance ablation on the spiral (the low-dim analogue of Table 4).

Runs all six distances at a fixed batch size and ranks them by medDist (lower = closer to
the curve). On the spiral, the kernel-MMD estimators lead, moment-matching (Frechet) and
sliced-Wasserstein follow, and drifting trails at large batch -- the same ordering the
full-scale single-encoder DINOv2 ablation (Table 4) reports. The headline Table 3
(gated-vs-uniform PID) and Table 4 (real encoder) runs are config-driven via
``configs/ablation_*.yaml`` over the training stack; this is the self-contained version.
"""
from .distances_toy import train_distance
from .run_spiral_grid import METHODS, seed_for


def run(batch=128, n_iters=8000) -> dict:
    """Run all six distances at ``batch`` -> ``{method: (recall, medDist)}`` plus the ranking."""
    res = {m: train_distance(m, batch, seed=seed_for(m, batch), n_iters=n_iters)[1:] for m in METHODS}
    ranking = sorted(res, key=lambda m: res[m][1])           # ascending medDist (best first)
    return {"results": res, "ranking": ranking}


def main():
    import sys
    n_iters = 120 if "--smoke" in sys.argv else 8000
    out = run(n_iters=n_iters)
    for m in out["ranking"]:
        rec, md = out["results"][m]
        print(f"  {m:8}: recall {rec:.2f}  medDist {md:.3f}")
    print("ranking (best->worst medDist):", " > ".join(out["ranking"]))


if __name__ == "__main__":
    main()
