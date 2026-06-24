#!/usr/bin/env bash
# torchrun launcher for the iRDM training loop.
#   GPUS=8 bash scripts/train.sh configs/imagenet.yaml
# Multi-node: set NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT in the environment.
set -euo pipefail
CONFIG="${1:?usage: train.sh <config.yaml>}"
GPUS="${GPUS:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

torchrun \
  --nnodes="${NNODES}" --node_rank="${NODE_RANK}" \
  --nproc_per_node="${GPUS}" \
  --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
  -m rdm.train.launch "${CONFIG}"
