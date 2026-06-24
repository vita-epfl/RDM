# Eval-only prompt assets

The FLUX text-to-image evaluation (`python reproduce.py eval-flux`, `configs/eval_flux.yaml`)
reads two prompt files, **bundled here** for convenience:

| File | What | Source |
|---|---|---|
| `geneval_prompts.jsonl` | 553 GenEval prompts, one JSON object per line with `prompt` + the `tag` / `include` object-spec the scorer needs | the official [GenEval](https://github.com/djghosh13/geneval) prompt set (`prompts/evaluation_metadata.jsonl`, MIT) |
| `pickapic_test_prompts.jsonl` | 499 Pick-a-Pic test prompts (one `{"prompt": ...}` per line) | a held-out test subset of [Pick-a-Pic](https://huggingface.co/datasets/yuvalkirstain/pickapic_v1) |

These are short prompt strings only (no images); they redistribute the prompt text of the
respective benchmarks — cite GenEval (Ghosh et al., 2023) and Pick-a-Pic (Kirstain et al.,
2023) accordingly.

GenEval *scoring* (not just rendering) additionally needs a local clone of the official
scorer; point `geneval_repo` in `configs/eval_flux.yaml` at it. Without it, `eval-flux`
renders the samples and tells you where they landed.

**Text context.** The FLUX.2 (Qwen3) context for each eval prompt is encoded on the fly from
the prompt string (`rdm.representation.flux_text_context`); no precomputed pool is needed for
`eval-flux` — just set `flux_ctx_len` to the length the training `ctx_pool` used. The training
`ctx_pool` (`data/fid_stats/flux2/qwen3_ctx_coco.npy`) is the same encoder over the **COCO
captions**, built with `scripts/build_flux2_ctx.py` (see `docs/flux_reference.md`).
