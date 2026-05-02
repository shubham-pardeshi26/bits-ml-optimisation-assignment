"""
04_matrix_multiply.py — Matrix multiplication via MapReduce.

Algorithm (standard MapReduce matrix multiply for A × B = C)
------------------------------------------------------------
For an M×K matrix A and K×N matrix B:

  Mapper for A[i][k] = v:
      For each j in 0..N-1:  emit key=(i,j), value=("A", k, v)

  Mapper for B[k][j] = v:
      For each i in 0..M-1:  emit key=(i,j), value=("B", k, v)

  Reducer for key=(i,j):
      Separate values into A_vals[(k)] and B_vals[(k)]
      C[i][j] = sum_k( A_vals[k] * B_vals[k] )

Key lesson
----------
This example INTENTIONALLY shows that MapReduce is SLOWER than numpy.matmul
for in-memory data. The overhead of:
  - serializing and deserializing (key, value) pairs
  - the shuffle phase sorting M*K*N + K*M*N keys
  - process spawning (parallel engine)
... completely dominates for in-memory matrices.

MapReduce shines when data is TOO LARGE to fit on a single machine, not
for in-memory compute. This is the same reason PyTorch uses NCCL ring-allreduce
(O(2*(N-1)/N * data) communication) rather than a MapReduce-style shuffle.

Usage
-----
    python map-reduce/examples/04_matrix_multiply.py
    python map-reduce/examples/04_matrix_multiply.py --size 32 --engine sequential
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from framework import ParallelMapReduce, SequentialMapReduce

# ---------------------------------------------------------------------------
# Module-level globals (set in main, read by mapper — avoids closure pickling)
# ---------------------------------------------------------------------------

_N_COLS_B = 0   # number of columns in B (= N)
_M_ROWS_A = 0   # number of rows in A    (= M)


# ---------------------------------------------------------------------------
# Mapper / Reducer  (module-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

def mapper(tagged_row):
    """
    Input: ("A", i, [v0, v1, ...]) or ("B", k, [v0, v1, ...])
    Output: ((i,j), (tag, k, v)) pairs
    """
    tag, idx, values = tagged_row
    if tag == "A":
        # A[i][k] = values[k]; emit for each output column j
        for j in range(_N_COLS_B):
            for k, v in enumerate(values):
                yield ((idx, j), ("A", k, v))
    else:
        # B[k][j] = values[j]; emit for each output row i
        for i in range(_M_ROWS_A):
            for j, v in enumerate(values):
                yield ((i, j), ("B", idx, v))


def reducer(key, values):
    """Dot product: sum_k A[i][k] * B[k][j]"""
    a_vals = {}
    b_vals = {}
    for tag, k, v in values:
        if tag == "A":
            a_vals[k] = v
        else:
            b_vals[k] = v
    result = sum(a_vals.get(k, 0.0) * b_vals.get(k, 0.0)
                 for k in set(a_vals) | set(b_vals))
    yield (key, result)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _N_COLS_B, _M_ROWS_A

    parser = argparse.ArgumentParser(description="MapReduce matrix multiply")
    parser.add_argument("--size",    type=int, default=16,
                        help="Use a SIZE×SIZE submatrix (default 16; file has 64×64)")
    parser.add_argument("--engine",  choices=["sequential", "parallel"],
                        default="sequential")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data-a",  default="map-reduce/data/matrix_a.txt")
    parser.add_argument("--data-b",  default="map-reduce/data/matrix_b.txt")
    args = parser.parse_args()

    pa, pb = Path(args.data_a), Path(args.data_b)
    if not pa.exists() or not pb.exists():
        print("[error] Matrix files not found.")
        print("  Run: python map-reduce/data/generate_datasets.py --output map-reduce/data")
        sys.exit(1)

    # Load and clip to requested size
    A = np.array([[float(x) for x in row.split()]
                  for row in pa.read_text().splitlines()])[:args.size, :args.size]
    B = np.array([[float(x) for x in row.split()]
                  for row in pb.read_text().splitlines()])[:args.size, :args.size]

    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match"

    _M_ROWS_A = M
    _N_COLS_B = N

    print(f"\n=== Matrix Multiply ({args.engine}) ===")
    print(f"A: {M}×{K}   B: {K}×{N}   ->   C: {M}×{N}")
    print(f"MapReduce will emit {M*K*N + K*M*N:,} key-value pairs in the map phase.\n")

    # Build input records
    records = [("A", i, A[i].tolist()) for i in range(M)] + \
              [("B", k, B[k].tolist()) for k in range(K)]

    engine = (ParallelMapReduce(num_workers=args.workers)
              if args.engine == "parallel" else SequentialMapReduce())

    # MapReduce multiply
    t0 = time.perf_counter()
    mr_results = engine.run(records, mapper, reducer, verbose=True)
    t_mr = time.perf_counter() - t0

    # Reconstruct matrix
    C_mr = np.zeros((M, N))
    for (i, j), v in mr_results:
        C_mr[i, j] = v

    # numpy reference
    t0 = time.perf_counter()
    C_np = A @ B
    t_np = time.perf_counter() - t0

    # Verify
    max_err = float(np.max(np.abs(C_mr - C_np)))
    match   = max_err < 1e-4

    print(f"\nResult matches numpy.matmul: {match}  (max element error: {max_err:.2e})")
    print(f"\nTiming comparison:")
    print(f"  MapReduce ({args.engine}): {t_mr*1000:>8.1f} ms")
    print(f"  numpy.matmul:             {t_np*1000:>8.3f} ms")
    print(f"  Slowdown factor:          {t_mr/max(t_np,1e-9):>8.1f}x")
    print(f"\n>>> MapReduce is {t_mr/max(t_np,1e-9):.0f}x SLOWER than numpy for "
          f"in-memory {M}×{M} matrices.")
    print(">>> Lesson: MapReduce overhead (serialization, shuffle, IPC) dominates")
    print(">>> for data that fits in memory. It shines for distributed storage.")


if __name__ == "__main__":
    main()
