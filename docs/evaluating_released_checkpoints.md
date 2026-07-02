# Evaluating the released checkpoints

Two one-step generators are released; this page is the end-to-end recipe for reproducing their
headline scores from the published weights — **GenEval** and **PickScore** for the FLUX student,
**SW_r14** (primary), **MMDr14**, and PickScore for the ImageNet generator.

| checkpoint | HF repo | file (`load_from`) | metrics |
|---|---|---|---|
| FLUX.2 klein-4B one-step (geALLcoco s180) | [`epfl-vita/flux2-klein-1step-rdm`](https://huggingface.co/epfl-vita/flux2-klein-1step-rdm) | `flux2_klein_1step_rdm_geallcoco_s180.pth` (15.5 GB fp32) | GenEval, PickScore |
| ImageNet-256 pMF-H FD-SIM (σ0.7, 4k) | [`Lanl11/pMF-H-FDSIM-imagenet256-sigma07-4k`](https://huggingface.co/Lanl11/pMF-H-FDSIM-imagenet256-sigma07-4k) | `model.pth` → `checkpoints/pMF-H_FD-SIM.pth` (3.8 GB fp32) | SW_r14, MMDr14, PickScore |

Both files are ordinary training checkpoints (a dict with the weights under the `model` key), so
they drop straight into the eval configs' `load_from` — `rdm/train/launch.py:load_generator_weights`
unwraps the `model` key and `load_state_dict(..., strict=False)`. `scripts/download_checkpoints.py`
links each into the flat name the config already points at, so **download → eval** works with no
config edits.

> Every metric here is **off-objective**: `rdm/eval/` never imports the training loss
> (`tests/test_offobjective_floor.py` enforces it), and SW_r14 shares no machinery with the
> trained MMD. See the [reproduction map](reproduction_map.md) for the artifact → paper-table map.

---

## A. FLUX.2 one-step — GenEval + PickScore

### Prerequisites

- The external **`flux2`** package (Black Forest Labs) + the klein-4B / AE base-weight snapshots.
  The student replaces only the DiT weights; klein's VAE/AE and the Qwen3 text encoder are reused.
  Set these up once (compute nodes are often offline, so pre-download into `HF_HOME`):

  ```bash
  git clone https://github.com/black-forest-labs/flux2 && export FLUX2_SRC="$PWD/flux2/src"
  export HF_HOME=/path/to/hf_cache
  huggingface-cli download black-forest-labs/FLUX.2-klein-4B flux-2-klein-4b.safetensors
  huggingface-cli download black-forest-labs/FLUX.2-dev      ae.safetensors
  ```

  Full explanation in [`docs/flux_reference.md`](flux_reference.md).
- **Text encoder id.** Training used `Qwen/Qwen3-4B-FP8` at **`ctx_len 48`** — the eval config
  matches (`flux_ctx_len: 48`). On an offline / air-gapped node the FP8 model can't fetch its
  finegrained-fp8 (deep-gemm) kernels; set `flux_text_model: Qwen/Qwen3-4B` in
  `configs/eval_flux.yaml` (bf16, identical hidden states, no kernels).

### Run

```bash
python scripts/download_checkpoints.py --flux        # -> checkpoints/flux2_klein_1step_rdm_geallcoco_s180.pth
```

The two FLUX axes are **two separate runs at different context lengths** — each of the 553 GenEval
and 499 Pick-a-Pic prompts is encoded into its *own* Qwen3 context on the fly (the COCO `ctx_pool`
is never sliced for eval), and the right `ctx_len` differs per axis:

```bash
# GenEval axis — ctx_len 48 (the training geometry; the GenEval prompts are short)
python reproduce.py eval-flux                                       # configs/eval_flux.yaml
# PickScore-pa axis — ctx_len 232 (avoids truncating ~53 long Pick-a-Pic prompts)
python reproduce.py eval-flux --config configs/eval_flux_pspa.yaml  # configs/eval_flux_pspa.yaml
```

- **PickScore** — runs out of the box (`rdm/eval/pickscore_eval.py`), mean over the 499 Pick-a-Pic
  test prompts, printed as the `pickscore` field. **Use `eval_flux_pspa.yaml` (ctx_len 232) for the
  headline number**: at ctx_len 48 the long prompts truncate and PickScore drops to ~21.2.
- **GenEval** — `configs/eval_flux.yaml` renders `geneval_n_per_prompt` (default 4) samples per
  prompt into the official layout under `output_dir/geneval/`, with the **canonical per-sample noise
  seed** `46_000_000 + prompt_idx*100 + sample_idx` (so the render matches the released eval bit-for-bit).
  Scoring uses the **external** official scorer, which is *not* bundled (heavy mmdet detector deps).
  Clone [`djghosh13/geneval`](https://github.com/djghosh13/geneval), install its detector env, and set:

  ```yaml
  geneval_repo: /path/to/geneval        # local clone with its mmdet weights (in configs/eval_flux.yaml)
  ```

  With `geneval_repo: null` (default) `eval-flux` renders and stops, printing the render dir — score
  those images with the mmdet scorer. **The number is scorer-sensitive**: use the mmdet-**builtin**
  Mask2Former config (not the geneval-repo standalone) and the open_clip ViT-L-14 (openai) color
  classifier. The full generation + scoring spec (detector build, config, thresholds, known scorer
  offsets) is in **[`docs/geneval_protocol.md`](geneval_protocol.md)** — read it before quoting a number.

### Expected (model card, checkpoint = s180)

| metric | axis / prompt set | ctx_len | this 1-step student | 4-step klein teacher |
|---|---|---|---|---|
| **GenEval** (avg over 6 tasks, 553 prompts) | `configs/eval_flux.yaml` + mmdet scorer | 48 | **0.8258** | 0.7944 |
| **PickScore** — Pick-a-Pic (499) | `configs/eval_flux_pspa.yaml` | 232 | 21.817 | 21.848 |
| PickScore — COCO-val | *card-reported; not the repo default* | — | **22.755** | 22.576 |

- `reproduce.py eval-flux` reproduces the **GenEval** and **Pick-a-Pic PickScore** columns. The
  **COCO-val** PickScore (22.755) uses a different prompt set (COCO validation captions) not wired
  into the default config; it is included here only to match the model card in full.
- The student beats the teacher on GenEval (+3.1 pp) and COCO-val PickScore (+0.18) and is within
  noise on Pick-a-Pic (21.817 vs 21.848).
- Card caveat: the 553 GenEval prompts are ~17.6% in the training generator pool, so the GenEval
  figure is partly in-distribution.

> **Locally reproduced on the released s180** (this repo, one H100, `Qwen/Qwen3-4B` bf16 text
> encoder, canonical protocol per [`geneval_protocol.md`](geneval_protocol.md)):
> - **PickScore-pa @ ctx232 = 21.83** — matches the card's 21.817 to 0.01.
> - **GenEval @ ctx48, n=4 = 0.830** (single_object 99.4 · two_object 92.2 · colors 91.8 · counting
>   77.5 · color_attr 71.0 · position 66.0) — reproduces the card's 0.826 to within scorer noise.
>
> Reproducing GenEval requires two protocol details (both now baked into the repo): the **per-sample
> noise seed** `46_000_000 + prompt_idx*100 + sample_idx` and the mmdet **builtin** detector config.
> An earlier run with a one-per-prompt seed stream + the geneval-repo standalone config read 0.814
> (colors/color_attr depressed ~3–4 pp) — that was a reproduction bug, not the checkpoint.

---

## B. ImageNet-256 — SW_r14 (primary) + MMDr14 + PickScore

### Prerequisites

- **Access.** The ImageNet repo is public — no login needed. Override with `--pmfh-repo`
  (or the `PMFH_REPO` env var) if you re-host or mirror it.
- **Eval reference banks.** SW_r14 / MMDr14 compare the generator's 14-encoder features against
  frozen **real** train/val banks. These are *not* shipped (they are large and derived from
  ImageNet); build them once over your ImageNet roots:

  ```bash
  IMAGENET_TRAIN=/data/imagenet/train IMAGENET_VAL=/data/imagenet/val bash scripts/run_refprep.sh
  ```

  This writes what `configs/eval_imagenet.yaml` expects — `data/fid_stats/bundles/eval_sw/<enc>_{train,val}.pt`
  (SW_r14), `data/fid_stats/bundles/eval_rff/<enc>.pt` + `data/fid_stats/mmdr14_val_floors.json`
  (MMDr14). Image-feature extraction dominates; shard it across GPUs with disjoint `--encoders`.
  Verify with `python scripts/check_artifacts.py configs/eval_imagenet.yaml`.
- The 14 panel encoders download lazily on first use; `python scripts/download_checkpoints.py
  --warm-encoders --eval` prefetches all of them (incl. the 4 held-out).

### Run

```bash
python scripts/download_checkpoints.py --pmfh        # -> checkpoints/pMF-H_FD-SIM.pth
python reproduce.py eval-imagenet                     # configs/eval_imagenet.yaml
```

`eval-imagenet` renders class-conditional one-step samples (1-step, cfg 7.0), extracts the panel,
and prints `SW_r14`, `MMDr14`, the per-encoder tables, and the off-objective PickScore. PickScore
uses `"a photo of a {classname}"` over torchvision's ImageNet class names by default (offline, no
asset) — set `imagenet_class_prompts` to a JSON list to override.

### Expected

| metric | value | paper |
|---|---|---|
| **SW_r14** (N=16384, M=1024 proj) | **1.30** | Table 1 |
| MMDr14 (arithmetic mean over 14 enc) | 2.69 | Table 7 (App. D) |
| PickScore (class prompts) | off-objective auxiliary (Fig. 6) | — |

- `eval-imagenet` computes SW_r14 and MMDr14 over the **same** 16384-sample render (`sw_n_samples`);
  SW_r14 — the primary metric — follows the paper protocol exactly (N=16384, M=1024). The paper's
  Table 7 MMDr14 used a larger N=50000 offline render, so the shipped-command MMDr14 differs slightly
  in variance (`mmd_n_samples` in the config is a documented reference value, not consumed by the
  shipped `eval-imagenet`).

---

## Notes

- **Missing `load_from`** never crashes: a missing checkpoint path warns and evaluates the
  *untrained base* (so a long prompt-encoding pass isn't wasted on a typo). If your numbers look
  like an untrained model, check the download linked the file where the config points.
- **`config` override.** Both entry points accept `--config` to point at a copy with your own
  paths/overrides, e.g. `python reproduce.py eval-flux --config configs/eval_flux.yaml`.
- **safetensors vs .pth (FLUX).** The FLUX repo also ships `model.safetensors` (bf16, bare klein
  DiT keys) for external inference stacks. For *this* repo's loader use the `.pth` (the exact
  `Flux2AdapterModel` state_dict) — that is what `--flux` fetches.
