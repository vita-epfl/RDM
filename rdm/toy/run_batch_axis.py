"""Batch-size axis on the spiral (the low-dim analogue of Fig. 4 / Table 6).

Sweeps the generation batch ``B`` for the Nystrom-MMD arm on the spiral substrate and reports
recall / medDist; quality climbs with ``B`` to a broad optimum and the smallest batch is noise-
dominated. (The headline Fig. 4 / Table 6 is the same sweep at matched wall-clock on a single
real DINOv2 encoder; that full-scale run is config-driven via ``configs/toy_batch.yaml`` over
the training stack -- this module is the self-contained low-dim version.)
"""
from .distances_toy import train_distance

DEFAULT_BATCHES = [8, 16, 32, 64, 128]


def run(batches=DEFAULT_BATCHES, n_iters=8000, seed=42) -> dict:
    """Sweep ``B`` for the Nystrom arm -> ``{B: (recall, medDist)}``."""
    out = {}
    for bs in batches:
        _, rec, md = train_distance("nystrom", bs, seed=seed, n_iters=n_iters)
        out[bs] = (rec, md)
    return out


def main():
    import sys
    n_iters = 120 if "--smoke" in sys.argv else 8000
    for bs, (rec, md) in run(n_iters=n_iters).items():
        print(f"  B={bs:4}: recall {rec:.2f}  medDist {md:.3f}")


if __name__ == "__main__":
    main()
