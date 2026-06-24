# Eval-only prompt assets

The FLUX text-to-image evaluation (`python reproduce.py eval-flux`, `configs/eval_flux.yaml`)
reads two external prompt files that are **not** shipped with this repo (they come from the
respective benchmarks). Place them here:

| File | What | Source |
|---|---|---|
| `geneval_prompts.jsonl` | GenEval per-prompt metadata (one JSON object per line, each with a `prompt` key + the object/count/color/position spec the scorer needs) | the official [GenEval](https://github.com/djghosh13/geneval) prompt set |
| `pickapic_test_prompts.jsonl` | Pick-a-Pic test prompts (one `{"prompt": ...}` per line) | the [Pick-a-Pic](https://huggingface.co/datasets/yuvalkirstain/pickapic_v1) test split |

GenEval *scoring* (not just rendering) additionally needs a local clone of the official
scorer; point `geneval_repo` in `configs/eval_flux.yaml` at it. Without it, `eval-flux`
renders the samples and tells you where they landed.

The generator text-context pool referenced by `ctx_pool`
(`data/fid_stats/flux2/qwen3_ctx_coco.npy`) is produced by the FLUX text encoder over the
prompt set, not by the reference precompute — see `docs/reproduction_map.md`.
