#!/usr/bin/env bash
# Run distributed DDP training (default: 2 GPUs).
# Usage: NGPUS=2 bash scripts/run_distributed.sh [extra train.py args]
set -euo pipefail

NGPUS=${NGPUS:-2}

torchrun \
  --nproc_per_node="${NGPUS}" \
  --master_port=29500 \
  src/train.py \
  --config configs/distributed.yaml \
  "$@"
