"""GenEval harness: render the GenEval prompts and score with the official protocol.

GenEval (Ghosh et al., 2023) scores object presence, counting, colors, position, and
attribute binding per prompt with an object detector. The canonical numbers are from the
official ``djghosh13/geneval`` scorer (an mmdet detector), which is an **external** tool run
in its own environment -- this module renders the images in the layout that scorer expects
and parses its JSONL output into per-category + overall scores. (A reproduction may also
vendor that scorer; it is not bundled here because of its heavy detector dependencies.)
"""
import os

import torch

from ..utils.io import read_jsonl, write_json

GENEVAL_CATEGORIES = ["single_object", "two_object", "counting", "colors", "position",
                      "color_attr"]


@torch.no_grad()
def render_geneval(generator, ctx_table, metadata: list[dict], out_dir: str, *,
                   num_steps: int = 1, n_per_prompt: int = 4, latent_channels: int = 128,
                   latent_size: int = 32, device: str = "cuda") -> str:
    """Render ``n_per_prompt`` samples for each GenEval prompt into the official layout.

    ``out_dir/<idx:05d>/{metadata.jsonl, samples/<j>.png}``; ``ctx_table[idx]`` is the
    precomputed text context for prompt ``idx``. Returns ``out_dir``.
    """
    from ..utils.io import save_uint8_png
    generator.sampling_args = {**generator.sampling_args, "num_steps": num_steps}
    for idx, meta in enumerate(metadata):
        pdir = os.path.join(out_dir, f"{idx:05d}")
        os.makedirs(os.path.join(pdir, "samples"), exist_ok=True)
        write_json(meta, os.path.join(pdir, "metadata.jsonl"))
        ctx = ctx_table[idx:idx + 1].to(device).expand(n_per_prompt, *ctx_table.shape[1:])
        g = torch.Generator(device=device).manual_seed(idx)
        noise = torch.randn(n_per_prompt, latent_channels, latent_size, latent_size,
                            generator=g, device=device)
        imgs = generator.sample(noise, ctx)
        for j, img in enumerate(imgs):
            save_uint8_png(img, os.path.join(pdir, "samples", f"{j:04d}.png"))
    return out_dir


def summarize_official_results(results_jsonl: str) -> dict:
    """Aggregate the official scorer's per-image JSONL into per-category + overall scores.

    Each row must carry ``tag`` (a GenEval category) and ``correct`` (bool). Overall is the
    mean over all rows; per-category is the mean within each tag.
    """
    rows = read_jsonl(results_jsonl)
    by_cat: dict[str, list] = {c: [] for c in GENEVAL_CATEGORIES}
    allv = []
    for r in rows:
        ok = float(bool(r.get("correct", r.get("score", 0))))
        allv.append(ok)
        tag = r.get("tag", r.get("category"))
        if tag in by_cat:
            by_cat[tag].append(ok)
    out = {c: (sum(v) / len(v) if v else float("nan")) for c, v in by_cat.items()}
    out["overall"] = sum(allv) / len(allv) if allv else float("nan")
    return out


def run_official_scorer(image_dir: str, geneval_repo: str, out_json: str,
                        python_bin: str = "python") -> dict:
    """Shell out to the external ``djghosh13/geneval`` scorer, then summarize.

    Requires the geneval repo (with its mmdet detector weights) on disk in a compatible
    environment. Writes ``out_json`` and returns the summary dict.
    """
    import subprocess
    results = os.path.join(os.path.dirname(out_json) or ".", "geneval_results.jsonl")
    subprocess.run([python_bin, os.path.join(geneval_repo, "evaluation", "evaluate_images.py"),
                    image_dir, "--outfile", results], check=True)
    summary = summarize_official_results(results)
    write_json(summary, out_json)
    return summary
