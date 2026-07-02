# Building the headline "geALLcoco" FLUX assets from scratch

`docs/flux_reference.md` builds the **baseline** joint reference: real COCO images as the target,
10 `nystrom_joint/<enc>_joint_M4096.pt` bundles, and a COCO `ctx_pool`. That is enough to train
and reproduce the Table-2 joint number.

The paper's **headline** FLUX run uses a richer, fully *teacher-rendered + curated* reference —
the "geALLcoco" mix — which lifts GenEval composition. Those assets are large (a 74 GB / 69 GiB
generator context, ~300 K reference rows × 10 encoders), so a deployment usually **transfers** a prebuilt
copy rather than rebuilding. This document records how to build them **from scratch** with the
in-repo tooling, and the exact asset spec so a transferred copy can be verified.

> Everything here builds on the COCO download + pairing in `docs/flux_reference.md` §1–2 and the
> `flux2` / klein-4B prerequisites in the README. The teacher-render + curation steps are
> compute-heavy (millions of 4-step renders) and the curation needs the **external scorers**
> (`djghosh13/geneval` for correctness, `rdm/eval/pickscore_eval.py` for PickScore) — these are
> not bundled. The baseline COCO-image path needs none of this.

## Asset spec (what a correct build / transfer contains)

| asset | shape / size | composition |
|---|---|---|
| reference image pool, per encoder | `pools_joint/<enc>.pt`, 302 149 × d_img | **53 800** GenEval-correct teacher imgs + **248 349** COCO-caption teacher imgs (top-3-of-24), order `[GenEval \| COCO]` |
| reference text table τ(c) | `siglip2_text_mix.npy`, 302 149 × 1152 | SigLIP2 **text** tower over the row-aligned prompts/captions |
| joint Nyström bundle, per enc | `<enc>_joint_M8192.pt` `{Z, alpha, sigma, k_rr, beta}` | concat `[φ(x) \| β·τ(c)]`, **M = 8192**, cold σ = 0.25·median, w=1 ⇒ β = σ |
| generator ctx pool | `qwen3_ctx_<tag>.npy`, **100 479** × 48 × 7680 (f16, ~74 GB) | COCO **82 783** + GenEval **553 × 32** (replication ×32) |
| generator text table | `siglip2_text_<tag>.npy`, 100 479 × 1152 | row-aligned to the ctx pool |

GenEval mass fraction = 53 800 / 302 149 = **17.8 %** (reference) and 17 696 / 100 479 = **17.6 %**
(generator) — the two must be kept ≈ equal ("generator ratio matches reference ratio"). The
generator pool size **must** equal `num_classes` (and the τ-table rows) in the train config.

## Recommended: download the bundles, build the context locally

The two pieces split by how expensive they are to recreate:

- **The joint Nyström bundles** (773 MB) encode the teacher-render + GenEval/PickScore **curation** —
  the expensive, hard-to-reproduce part. **Download them.**
- **The generator context** (~74 GB `qwen3_ctx`) and **text table** (442 MB `siglip2_text`) are just
  Qwen3 / SigLIP2 **text encodings** of the COCO captions + GenEval prompts — cheap and deterministic
  to recompute. **Build them locally** (don't download 74 GB).

```bash
# 1. bundles + reference psi  <-  Hugging Face  (~1.2 GB)
hf download Lanl11/irdm-flux-geall-assets --repo-type dataset \
    --local-dir data/fid_stats_dl
mkdir -p data/fid_stats/bundles/nystrom_joint data/fid_stats/flux2
cp data/fid_stats_dl/bundles/nystrom_joint/*_joint_M8192.pt data/fid_stats/bundles/nystrom_joint/
cp data/fid_stats_dl/flux2/siglip2_text_geall_g32.npy        data/fid_stats/flux2/

# 2. generator context  <-  build locally (needs COCO pairing + the FLUX.2 Qwen3 encoder; see below)
python scripts/build_flux2_ctx.py --captions data/coco/coco_pairs.npz \
    --out data/fid_stats/flux2/qwen3_ctx_coco.npy --ctx-len 48
python scripts/build_flux2_ctx.py --jsonl assets/geneval_prompts.jsonl \
    --out data/fid_stats/flux2/qwen3_ctx_geneval553.npy --ctx-len 48
python - <<'PY'   # COCO 82783 + GenEval 553 x32 = 100479, order [COCO | GenEval x32]
import numpy as np
coco = np.asarray(np.load("data/fid_stats/flux2/qwen3_ctx_coco.npy", mmap_mode="r"))
ge32 = np.tile(np.load("data/fid_stats/flux2/qwen3_ctx_geneval553.npy"), (32, 1, 1))
np.save("data/fid_stats/flux2/qwen3_ctx_geall_g32.npy", np.concatenate([coco, ge32], 0))
PY
```
The downloaded bundle file names already match `configs/flux_geall.yaml` (iRDM encoder names). After
both steps, `python scripts/check_artifacts.py configs/flux_geall.yaml` should pass and you can train.

> If you also want the prebuilt generator context (to skip step 2) or the trained s20 student, ask the
> dataset owner — they are not in the public repo by default because of their size (74 GB / 15 GB).

## From-scratch build (full pipeline, if you need to rebuild the bundles too)

### 0. Prerequisites
COCO pairing (`data/coco/coco_pairs.npz`) from `docs/flux_reference.md` §1–2; the GenEval prompt
list `assets/geneval_prompts.jsonl` (553); a built FLUX generator (klein-4B base via `FLUX2_SRC`).

### 1. Render the 4-step teacher (the reference is teacher outputs, not real photos)
Encode the prompts to Qwen3 contexts, then render with `num_steps=4`:
```bash
# GenEval prompts (many seeds; you will keep only the GenEval-correct ones)
python scripts/build_flux2_ctx.py --jsonl assets/geneval_prompts.jsonl \
    --out data/fid_stats/flux2/qwen3_ctx_geneval.npy --ctx-len 48
# COCO captions (24 seeds each; you will keep PickScore top-3)
python scripts/build_flux2_ctx.py --captions data/coco/coco_pairs.npz \
    --out data/fid_stats/flux2/qwen3_ctx_coco.npy --ctx-len 48
# render with rdm.data.flux_gen_driver.render_prompts (function API; sharded round-robin by rank,
# skip-if-exists resume). Drive it under torchrun for multi-GPU; e.g.:
python - <<'PY'
import torch
from rdm.train.launch import build_generator_from_config, load_config
from rdm.data.flux_gen_driver import render_prompts
gen, _ = build_generator_from_config(load_config("configs/flux_geall.yaml"))
for ctx, out, seeds in [("data/fid_stats/flux2/qwen3_ctx_geneval.npy", "data/teacher/geneval", range(32)),
                        ("data/fid_stats/flux2/qwen3_ctx_coco.npy",    "data/teacher/coco",    range(24))]:
    render_prompts(gen, torch.from_numpy(__import__("numpy").load(ctx)), out, num_steps=4, seeds=tuple(seeds))
PY
```

### 2. Curate (needs the external scorers)
- **GenEval block:** score every `data/teacher/geneval/*.png` with the official
  `djghosh13/geneval` mmdet scorer (see `docs/flux_reference.md` §5);
  keep the renders whose prompt is satisfied, balanced to ≤ 100 per prompt → **53 800** images.
- **COCO block:** score the 24 seeds/caption with `rdm/eval/pickscore_eval.py`, keep the **top 3**
  by PickScore → **248 349** images.
- Write a manifest (image paths) for each block in the fixed order `[GenEval | COCO]`.

### 3. Reference: per-encoder image features, τ, σ, joint bundles
```bash
# per-encoder image features over the curated [GenEval|COCO] pool (rdm.refprep.extract_features),
# saved row-aligned to data/fid_stats/pools_joint/<enc>.pt  (302149 x d_img)
# SigLIP2 text table over the row-aligned prompts -> data/fid_stats/flux2/siglip2_text_mix.npy
# per-encoder median sigma -> data/fid_stats/sigma_joint/<enc>.pt  (cache_median_sigma)
# joint Nystrom bundle, M=8192, cold sigma x0.25, w=1 (beta=sigma):
python - <<'PY'
from rdm.refprep.build_joint_reference import build_joint_one
from rdm.representation.registry import training_specs
for s in training_specs():
    build_joint_one(f"data/fid_stats/pools_joint/{s.name}.pt",
                    "data/fid_stats/flux2/siglip2_text_mix.npy",
                    f"data/fid_stats/sigma_joint/{s.name}.pt",
                    f"data/fid_stats/bundles/nystrom_joint/{s.name}_joint_M8192.pt",
                    n_landmarks=8192, sigma_scale=0.25, s_txt=1.0)
PY
```
(`rdm.refprep.run joint` automates the COCO-image baseline at M4096; for the mix you drive
`build_joint_one` directly with the curated pool path and `n_landmarks=8192`, as above.)

### 4. Generator pool: COCO + GenEval × 32 (ctx + τ, row-aligned)
```bash
python scripts/build_flux2_ctx.py --jsonl assets/geneval_prompts.jsonl \
    --out data/fid_stats/flux2/qwen3_ctx_geneval553.npy --ctx-len 48     # 553 x 48 x 7680
python - <<'PY'
import numpy as np
coco = np.load("data/fid_stats/flux2/qwen3_ctx_coco.npy", mmap_mode="r")        # 82783
ge   = np.load("data/fid_stats/flux2/qwen3_ctx_geneval553.npy")                 # 553
ge32 = np.tile(ge, (32, 1, 1))                                                  # 17696
out  = np.concatenate([np.asarray(coco), ge32], 0)                             # 100479, [COCO|GE x32]
np.save("data/fid_stats/flux2/qwen3_ctx_geall_g32.npy", out)
# build siglip2_text_geall_g32.npy the same way (SigLIP2 text over COCO captions + GenEval 553 x32)
PY
```
The student samples `y ∈ [0, 100479)`: `y < 82783` → COCO caption, else GenEval prompt
`(y − 82783) % 553`. **`ctx_len` (48) must equal `flux_ctx_len` at eval** (long Pick-a-Pic eval
prompts use 232 — see `docs/flux_reference.md` and the PickScore note).

### 5. Point the config at the mix assets and train
See `configs/flux_geall.yaml` (the headline recipe: M8192 mix bundles, the g32 generator pool,
`num_classes: 100479`). On a 94 GB H100 (vs the original 141 GB H200) set `battery_bf16: true`
and lower `batch_size` (GradCache keeps the result bit-exact at fixed `rollout_size`).

## Verifying a transferred copy
```python
import torch, numpy as np
b = torch.load("…/inception_joint_M8192.pt", map_location="cpu", weights_only=False)
assert b["Z"].shape[0] == 8192 and b["d_txt"] == 1152 and abs(b["beta"]-b["sigma"]) < 1e-3  # w=1
ctx = np.load("…/qwen3_ctx_geall_g32.npy", mmap_mode="r")
assert ctx.shape == (100479, 48, 7680)
assert np.any(ctx[90000] != 0)        # GenEval tail present (a known silent-truncation failure mode)
```
