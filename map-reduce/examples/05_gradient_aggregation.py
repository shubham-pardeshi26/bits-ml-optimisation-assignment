"""
05_gradient_aggregation.py — DDP All-Reduce via MapReduce.

This example is the pedagogical bridge between MapReduce and PyTorch DDP.

What DDP does (from src/train.py, ~line 137):
----------------------------------------------
    # Backward pass triggers all-reduce automatically:
    loss.backward()
    # Internally, DDP calls:
    #   dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
    #   param.grad /= world_size

What THIS example does with MapReduce:
---------------------------------------
    Mapper:  worker_record  ->  (layer_name, gradient_array)
    Reducer: (layer_name, [grad_0, grad_1, ..., grad_W])
                           ->  (layer_name, mean_gradient)

The reducer IS the all-reduce: it averages gradients across workers,
exactly what dist.all_reduce(...SUM) + grad /= world_size does.

Output
------
  - Per-layer gradient norms (before and after averaging)
  - Inter-worker gradient variance per layer (measures gradient staleness)
  - Side-by-side comparison with the DDP code pattern

Usage
-----
    python map-reduce/examples/05_gradient_aggregation.py
    python map-reduce/examples/05_gradient_aggregation.py --workers 4 --layers 8 --engine parallel
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from framework import ParallelMapReduce, SequentialMapReduce

# ---------------------------------------------------------------------------
# Mapper / Reducer  (module-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

def mapper(line: str):
    """
    Input line: 'worker_id<TAB>layer_name<TAB>space-separated floats'
    Output: (layer_name, gradient_array)
    """
    parts = line.strip().split("\t")
    if len(parts) != 3:
        return
    _worker_id, layer_name, grad_str = parts
    grad = np.array([float(v) for v in grad_str.split()], dtype=np.float32)
    yield (layer_name, grad)


def reducer(layer_name: str, values):
    """
    Average gradients across all workers for this layer.
    Equivalent to: dist.all_reduce(grad, op=SUM) ; grad /= world_size
    """
    grad_list  = list(values)
    mean_grad  = np.mean(np.stack(grad_list, axis=0), axis=0)
    yield (layer_name, mean_grad)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MapReduce gradient aggregation (DDP bridge)")
    parser.add_argument("--engine",  choices=["sequential", "parallel"],
                        default="sequential")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data",    default="map-reduce/data/gradients.txt")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[error] Dataset not found: {data_path}")
        print("  Run: python map-reduce/data/generate_datasets.py --output map-reduce/data")
        sys.exit(1)

    lines = [l for l in data_path.read_text().splitlines() if l.strip()]
    print(f"\n=== Gradient Aggregation via MapReduce ({args.engine}) ===")
    print(f"Input: {len(lines):,} gradient records\n")

    # Parse raw records to count workers/layers
    raw = {}
    for line in lines:
        parts = line.strip().split("\t")
        if len(parts) == 3:
            w, layer, _ = parts
            raw.setdefault(layer, set()).add(w)
    n_workers = max(len(v) for v in raw.values()) if raw else 0
    n_layers  = len(raw)
    print(f"Configuration: {n_workers} workers × {n_layers} layers\n")

    engine = (ParallelMapReduce(num_workers=args.workers)
              if args.engine == "parallel" else SequentialMapReduce())

    results = engine.run(lines, mapper, reducer, verbose=True)

    # Build per-layer dict
    averaged = {layer: grad for layer, grad in results}

    # Per-worker gradients (for variance computation)
    per_worker: dict = {}
    for line in lines:
        parts = line.strip().split("\t")
        if len(parts) != 3:
            continue
        worker_id, layer_name, grad_str = parts
        g = np.array([float(v) for v in grad_str.split()], dtype=np.float32)
        per_worker.setdefault(layer_name, []).append(g)

    # --- Results table ---
    print(f"\n{'Layer':<12} {'Avg Grad Norm':>14} {'Worker Variance':>16}  {'Gradient shape'}")
    print(f"{'-'*12} {'-'*14} {'-'*16}  {'-'*16}")
    for layer in sorted(averaged.keys()):
        mean_g    = averaged[layer]
        avg_norm  = float(np.linalg.norm(mean_g))
        worker_gs = np.stack(per_worker[layer], axis=0)  # shape [W, D]
        variance  = float(np.mean(np.var(worker_gs, axis=0)))
        print(f"  {layer:<10} {avg_norm:>14.6f} {variance:>16.8f}  {list(mean_g.shape)}")

    # --- Side-by-side comparison with DDP code ---
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║          PyTorch DDP  vs.  MapReduce — same operation, different scale       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  PyTorch DDP (src/train.py ~line 137):    MapReduce (this file):             ║
║  ─────────────────────────────────────    ─────────────────────────────────  ║
║  # DDP hooks into .backward()             # mapper extracts gradients        ║
║  loss.backward()                          mapper(line) -> (layer, grad)      ║
║                                                                              ║
║  # NCCL ring-allreduce:                   # reducer averages them            ║
║  dist.all_reduce(                         reducer(layer, [g0,g1,g2,g3])     ║
║    param.grad,                              -> np.mean(stack, axis=0)        ║
║    op=dist.ReduceOp.SUM                                                      ║
║  )                                        # shuffle = routing gradients      ║
║  param.grad /= world_size                 # to the right reducer key         ║
║                                                                              ║
║  Why NCCL is faster than MapReduce shuffle:                                  ║
║  • NCCL ring-allreduce: O(2*(N-1)/N * data) — no central bottleneck         ║
║  • MapReduce shuffle:   all data routes through a sort-and-merge step        ║
║  • NCCL uses direct GPU-to-GPU transfers (NVLink / IB) — no serialization   ║
║  • MapReduce serialises every value to bytes for disk/network transit        ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
