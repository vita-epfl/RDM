"""Publication-figure dispatcher (fig1c / fig3 / fig4 / fig6).

A thin dispatcher behind ``python -m rdm.figures.make_figures <fig>``:

* ``fig3`` -- the spiral 3x6 grid (delegates to :mod:`rdm.toy.run_spiral_grid`);
* ``fig4`` -- the batch-size axis (:mod:`rdm.toy.run_batch_axis`), quality vs B with the
  broad optimum shaded past 2560 (full-scale) / past the toy optimum (low-dim);
* ``fig1c`` / ``fig6`` -- GenEval & PickScore over post-training compute, and the PickScore
  preference bars; these consume cached evaluation result dicts (JSON) rather than recompute.

Figures use the iRDM orange ``#D97757``.
"""
import os

IRDM_ORANGE = "#D97757"


def _ax():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig3(out="figures/fig3_spiral.png", n_iters=8000):
    from ..toy.run_spiral_grid import render, run
    return render(run(n_iters=n_iters), out)


def fig4(out="figures/fig4_batch_axis.png", n_iters=8000):
    from ..toy.run_batch_axis import run
    res = run(n_iters=n_iters)
    plt = _ax()
    xs = sorted(res)
    ys = [res[b][1] for b in xs]                      # medDist vs B
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(xs, ys, "-o", color=IRDM_ORANGE)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("generation batch B")
    ax.set_ylabel("medDist (lower=better)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=300)
    return out


def fig_compute_curve(results: dict, out="figures/fig1c_compute.png",
                      keys=("geneval", "pickscore")):
    """fig1c: a metric vs post-training-compute curve from a cached ``{step: {metric: val}}`` dict."""
    plt = _ax()
    steps = sorted(results)
    fig, ax = plt.subplots(figsize=(4, 3))
    for k in keys:
        ax.plot(steps, [results[s].get(k) for s in steps], "-o", label=k, color=IRDM_ORANGE)
    ax.set_xlabel("post-training compute (steps)")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=300)
    return out


def fig6_pickscore_bars(winrates: dict, out="figures/fig6_pickscore.png"):
    """fig6: PickScore win-rate bars from a cached ``{label: winrate}`` dict."""
    plt = _ax()
    fig, ax = plt.subplots(figsize=(4, 3))
    labels = list(winrates)
    ax.bar(labels, [winrates[k] for k in labels], color=IRDM_ORANGE)
    ax.set_ylabel("PickScore win rate")
    ax.axhline(0.5, ls="--", c="gray", lw=0.8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=300)
    return out


def main():
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "fig3"
    n_iters = 120 if "--smoke" in sys.argv else 8000
    if which == "fig3":
        print("saved", fig3(n_iters=n_iters))
    elif which == "fig4":
        print("saved", fig4(n_iters=n_iters))
    else:
        print(f"{which}: provide cached results via fig_compute_curve / fig6_pickscore_bars")


if __name__ == "__main__":
    main()
