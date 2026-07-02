"""End-to-end FLUX text-to-image evaluation: GenEval + PickScore (the paper's main FLUX evals).

Renders one-step samples with the FLUX generator and scores them with the two headline
text-to-image metrics:

* **GenEval** (Ghosh et al., 2023) -- object/count/color/position/attribute binding, scored by
  the official protocol (:mod:`rdm.eval.geneval_harness`); iRDM 0.805 vs the 4-step 0.794.
* **PickScore** (Kirstain et al., 2023) -- a learned human-preference proxy never optimized,
  on the 499 Pick-a-Pic test prompts (:mod:`rdm.eval.pickscore_eval`); iRDM 21.69 vs 4-step 21.85.

These are the paper's own-run numbers. ``configs/eval_flux.yaml`` defaults ``load_from`` to the
**released** geALLcoco s180 student, which scores higher (GenEval 0.826 / PickScore-pa 21.82) --
see ``docs/evaluating_released_checkpoints.md``.

Off-objective: imports only the metric harnesses + the generator, never the training loss.
"""
import os

import torch

from .geneval_harness import render_geneval, run_official_scorer
from .pickscore_eval import PickScorer, mean_pickscore


@torch.no_grad()
def render_for_prompts(generator, ctx_table, indices, *, batch: int = 16, latent_channels: int = 128,
                       latent_size: int = 32, device: str = "cuda", seed: int = 0):
    """Render one one-step sample per prompt index -> ``[0, 1]`` NCHW batch.

    The full noise tensor is drawn once (seed ``seed``) then forwarded in chunks of ``batch``,
    so the result is identical to a single-pass render (samples are independent) but a 4B
    model at 512px does not OOM on the few-hundred-prompt Pick-a-Pic set.
    """
    idx = list(indices)
    ctx = ctx_table[idx].to(device)
    g = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(len(idx), latent_channels, latent_size, latent_size,
                        generator=g, device=device)
    outs = [generator.sample(noise[lo:lo + batch], ctx[lo:lo + batch])
            for lo in range(0, len(idx), batch)]
    return torch.cat(outs, 0)


@torch.no_grad()
def evaluate_flux(generator, *, pickscore_prompts=None, pickscore_ctx=None,
                  geneval_metadata=None, geneval_ctx=None, geneval_repo=None,
                  out_dir: str = "work_dirs/flux_eval", n_geneval_per_prompt: int = 4,
                  latent_channels: int = 128, latent_size: int = 32, device: str = "cuda") -> dict:
    """Run PickScore (mean over the Pick-a-Pic prompts) and GenEval (official protocol).

    Args:
        generator: a one-step :class:`FluxGenerator`.
        pickscore_prompts / pickscore_ctx: the prompt strings and their precomputed text context.
        geneval_metadata / geneval_ctx: the GenEval per-prompt metadata and text context.
        geneval_repo: path to the official ``djghosh13/geneval`` scorer (None -> render only).
    Returns ``{"pickscore": float, "geneval": {category: ..., "overall": ...}}``.
    """
    os.makedirs(out_dir, exist_ok=True)
    results = {}

    if pickscore_prompts is not None and pickscore_ctx is not None:
        imgs = render_for_prompts(generator, pickscore_ctx, range(len(pickscore_prompts)),
                                  latent_channels=latent_channels, latent_size=latent_size, device=device)
        results["pickscore"] = mean_pickscore(PickScorer(device=device), imgs, pickscore_prompts)

    if geneval_metadata is not None and geneval_ctx is not None:
        gdir = render_geneval(generator, geneval_ctx, geneval_metadata, os.path.join(out_dir, "geneval"),
                              num_steps=1, n_per_prompt=n_geneval_per_prompt,
                              latent_channels=latent_channels, latent_size=latent_size, device=device)
        if geneval_repo:
            results["geneval"] = run_official_scorer(gdir, geneval_repo,
                                                     os.path.join(out_dir, "geneval_summary.json"))
        else:
            results["geneval"] = {"_note": f"rendered to {gdir}; run the official scorer to score"}
    return results
