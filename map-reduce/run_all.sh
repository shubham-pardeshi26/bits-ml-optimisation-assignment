#!/usr/bin/env bash
# run_all.sh — Generate datasets and run all five MapReduce examples.
#
# Usage:
#   bash map-reduce/run_all.sh
#
# Run from the project root (MLSysOp/).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

echo "========================================"
echo " MapReduce Educational Examples"
echo " Working directory: ${PROJECT_ROOT}"
echo "========================================"
echo ""

# ── Step 0: Generate datasets ─────────────────────────────────────────────
echo ">>> Step 0: Generating synthetic datasets"
python map-reduce/data/generate_datasets.py --output map-reduce/data --seed 42
echo ""

# ── Example 01: Word Count ────────────────────────────────────────────────
echo "========================================"
echo " Example 01: Word Count"
echo "========================================"
python map-reduce/examples/01_word_count.py --engine sequential --top 10
echo ""

# ── Example 02: Sales Aggregation ────────────────────────────────────────
echo "========================================"
echo " Example 02: Sales Aggregation"
echo "========================================"
python map-reduce/examples/02_sales_aggregation.py --engine sequential
echo ""

# ── Example 03: Inverted Index ───────────────────────────────────────────
echo "========================================"
echo " Example 03: Inverted Index"
echo "========================================"
python map-reduce/examples/03_inverted_index.py --engine sequential --lookup gradient
echo ""

# ── Example 04: Matrix Multiply ──────────────────────────────────────────
echo "========================================"
echo " Example 04: Matrix Multiply"
echo "========================================"
python map-reduce/examples/04_matrix_multiply.py --size 16 --engine sequential
echo ""

# ── Example 05: Gradient Aggregation (DDP bridge) ────────────────────────
echo "========================================"
echo " Example 05: Gradient Aggregation (DDP)"
echo "========================================"
python map-reduce/examples/05_gradient_aggregation.py --engine sequential
echo ""

echo "========================================"
echo " All examples completed successfully."
echo "========================================"
