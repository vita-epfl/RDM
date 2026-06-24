#!/usr/bin/env bash
# Build the frozen reference banks the training loop + evaluators consume (one-time, heavy).
# Set the dataset roots below (or pass them in the environment), then:
#   bash scripts/run_refprep.sh
#
# This is a thin wrapper around `python -m rdm.refprep.run`, which streams the train/val
# images through each frozen encoder and builds the Nystrom bundles, PID floors and eval
# (RFF / SW) banks at the exact paths the configs reference. The extraction dominates the
# runtime; to shard it across GPUs, launch the driver several times with disjoint
# `--encoders` on different CUDA_VISIBLE_DEVICES (per-encoder artifacts are independent).
set -euo pipefail

IMAGENET_TRAIN="${IMAGENET_TRAIN:?set IMAGENET_TRAIN to the ImageNet-256 train root (ImageFolder)}"
IMAGENET_VAL="${IMAGENET_VAL:?set IMAGENET_VAL to the ImageNet-256 val root}"
OUT="${OUT:-data/fid_stats}"

echo "[refprep] ImageNet reference -> ${OUT} (train=${IMAGENET_TRAIN} val=${IMAGENET_VAL})"
python -m rdm.refprep.run imagenet \
  --train "${IMAGENET_TRAIN}" --val "${IMAGENET_VAL}" --out "${OUT}" --skip-existing

# ---- FLUX joint reference (text-to-image; optional, needs COCO -- see docs/flux_reference.md) ----
# 1) pairing:  python scripts/prepare_datasets.py coco --captions <captions_train2014.json> \
#                  --images <train2014> --out data/coco/coco_pairs.npz
# 2) joint pack (tau + 10 joint Nystrom bundles):
#      python -m rdm.refprep.run joint --coco-pairs data/coco/coco_pairs.npz --out "${OUT}" --skip-existing
# 3) training text context (ctx_pool) -- needs flux2 + GPU:
#      python scripts/build_flux2_ctx.py --captions data/coco/coco_pairs.npz \
#          --out "${OUT}/flux2/qwen3_ctx_coco.npy" --ctx-len 48
#    then: python scripts/check_artifacts.py configs/flux.yaml

echo "[refprep] done. Sanity check: python scripts/check_artifacts.py configs/imagenet.yaml"
