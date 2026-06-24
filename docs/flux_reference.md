# FLUX text-to-image: data, reference pack, and contexts

The ImageNet path is self-contained once you point `run_refprep.sh` at ImageNet. The FLUX
path needs three extra things, none of which is fetched for you:

1. the **COCO** images + captions (the reference distribution),
2. the **joint reference pack** — 10 `nystrom_joint/<enc>_joint_M4096.pt` bundles + the SigLIP2
   text table `siglip2_text_coco.npy` (the τ(c) the joint feature concatenates),
3. the **generator text context** — the FLUX.2 Qwen3 `ctx_pool` over the COCO captions (what
   the loop conditions on while training) — plus, at eval time, the Qwen3 context of each
   eval prompt (encoded on the fly).

All of (2) and (3) are built **in-repo** from the COCO download with the steps below; nothing
needs to be uploaded. Everything except the image download runs on one GPU (a few GPU-hours,
dominated by the per-encoder image-feature extraction).

> Prerequisite: the external `flux2` package (Black Forest Labs) and the klein-4B / AE weight
> snapshots, reachable via `FLUX2_SRC` + `HF_HOME` (see the main README). The FLUX.2 Qwen3
> text encoder (`Qwen/Qwen3-4B-FP8`) downloads lazily on first use.

## 1. Download COCO (train2014)

The canonical reference is COCO `train2014` (≈82,783 images) with `captions_train2014.json`:

```bash
mkdir -p data/coco && cd data/coco
wget http://images.cocodataset.org/zips/train2014.zip          && unzip -q train2014.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip && unzip -q annotations_trainval2014.zip
cd ../..
# -> data/coco/train2014/*.jpg , data/coco/annotations/captions_train2014.json
```

## 2. Build the canonical image–caption pairing

One caption per image, ordered by image id — everything joint is row-aligned to this:

```bash
python scripts/prepare_datasets.py coco \
    --captions data/coco/annotations/captions_train2014.json \
    --images   data/coco/train2014 \
    --out      data/coco/coco_pairs.npz
```

## 3. Build the joint reference pack (τ + 10 joint Nyström bundles)

```bash
python -m rdm.refprep.run joint --coco-pairs data/coco/coco_pairs.npz \
    --out data/fid_stats --skip-existing
```

This writes, at the paths `configs/flux.yaml` references:

- `data/fid_stats/flux2/siglip2_text_coco.npy` — τ(c), the frozen SigLIP2 **text** tower over
  the captions (`joint_text_psi`);
- `data/fid_stats/bundles/nystrom_joint/<enc>_joint_M4096.pt` — the joint Nyström bundle over
  `[φ(x) | β·τ(c)]` for each of the 10 training encoders (`nystrom_paths`).

The reference distribution here is the **real COCO images**. The paper's headline run mixes in
a few FLUX-teacher seeds per prompt (~400K samples); that is an optional enrichment of the
reference, not required to reproduce the pipeline — to use it, render teacher samples with
`rdm.data.flux_gen_driver` (`num_steps=4`) and extend the image pool before `build_joint_one`.

Shard the heavy extraction across GPUs by launching this several times with disjoint
`--encoders` on different `CUDA_VISIBLE_DEVICES` (per-encoder bundles are independent).

## 4. Build the training text context (`ctx_pool`)

The loop conditions the generator on the FLUX.2 Qwen3 context of the COCO captions:

```bash
python scripts/build_flux2_ctx.py --captions data/coco/coco_pairs.npz \
    --out data/fid_stats/flux2/qwen3_ctx_coco.npy --ctx-len 48
```

`--ctx-len` is the sequence length L baked into the table (`(N, 48, 7680)` float16). **It must
match `flux_ctx_len` in `configs/eval_flux.yaml`** so the student is evaluated at the same
sequence geometry it trained on. 48 is the default on both sides.

## 5. Train, then evaluate

```bash
python scripts/check_artifacts.py configs/flux.yaml          # all joint artifacts present?
GPUS=8 bash scripts/train.sh configs/flux.yaml               # joint run; joint_enable:false = marginal
# set load_from in configs/eval_flux.yaml to the trained student checkpoint, then:
python reproduce.py eval-flux                                # GenEval + PickScore
```

At evaluation the GenEval (553) and Pick-a-Pic (499) prompts are encoded into their **own**
Qwen3 contexts on the fly (same `Flux2TextContextEncoder`, same `ctx_len`) — the COCO
`ctx_pool` is **not** sliced for eval. GenEval *scoring* additionally needs a local clone of
the official [`djghosh13/geneval`](https://github.com/djghosh13/geneval) scorer; point
`geneval_repo` at it (without it, `eval-flux` renders the samples and stops).
