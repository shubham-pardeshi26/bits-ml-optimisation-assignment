#!/usr/bin/env bash
# Run single-GPU baseline training.
# Usage: bash scripts/run_baseline.sh [extra train.py args]
set -euo pipefail

python src/train.py --config configs/baseline.yaml "$@"
