"""
01_word_count.py — Classic MapReduce word frequency count.

Concepts demonstrated
---------------------
  - Basic mapper / reducer pattern
  - Combiner: local sum before shuffle (reduces inter-node traffic by ~10x)
  - Sequential vs. parallel engine comparison

Usage
-----
    python map-reduce/examples/01_word_count.py
    python map-reduce/examples/01_word_count.py --engine parallel --workers 4 --top 20
"""

import argparse
import sys
import time
from pathlib import Path

# Make the map-reduce/ directory importable so `framework` resolves correctly
sys.path.insert(0, str(Path(__file__).parent.parent))

from framework import ParallelMapReduce, SequentialMapReduce

# ---------------------------------------------------------------------------
# Mapper / Combiner / Reducer  (must be module-level for multiprocessing)
# ---------------------------------------------------------------------------

def mapper(line: str):
    """Tokenize a line; emit (word, 1) for each token."""
    for word in line.lower().split():
        # Strip basic punctuation
        word = word.strip(".,!?;:\"'()[]{}")
        if word:
            yield (word, 1)


def combiner(key: str, values):
    """Local sum — same as reducer but runs before shuffle."""
    yield sum(values)


def reducer(key: str, values):
    """Global sum across all (word, 1) pairs from all mappers."""
    yield (key, sum(values))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MapReduce word count")
    parser.add_argument("--engine",  choices=["sequential", "parallel"],
                        default="sequential")
    parser.add_argument("--workers", type=int, default=4,
                        help="Worker processes (parallel engine only)")
    parser.add_argument("--top",     type=int, default=10,
                        help="Show top-K most frequent words")
    parser.add_argument("--data",    default="map-reduce/data/word_count.txt")
    args = parser.parse_args()

    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[error] Dataset not found: {data_path}")
        print("  Run: python map-reduce/data/generate_datasets.py --output map-reduce/data")
        sys.exit(1)

    lines = data_path.read_text().splitlines()
    print(f"\n=== Word Count ({args.engine}) ===")
    print(f"Input: {len(lines):,} lines from {data_path.name}\n")

    # Build engine
    if args.engine == "parallel":
        engine = ParallelMapReduce(num_workers=args.workers)
    else:
        engine = SequentialMapReduce()

    # Run WITHOUT combiner
    t0 = time.perf_counter()
    results_no_combine = engine.run(lines, mapper, reducer, verbose=True)
    t_no_combine = time.perf_counter() - t0
    print(f"  Total (no combiner):   {t_no_combine*1000:.1f} ms\n")

    # Run WITH combiner
    t0 = time.perf_counter()
    results_combine = engine.run(lines, mapper, reducer, combiner=combiner, verbose=True)
    t_combine = time.perf_counter() - t0
    print(f"  Total (with combiner): {t_combine*1000:.1f} ms\n")

    # Top-K
    top = sorted(results_combine, key=lambda kv: kv[1][1], reverse=True)[:args.top]

    print(f"Top {args.top} words:")
    print(f"  {'Word':<20} {'Count':>8}")
    print(f"  {'-'*20} {'-'*8}")
    for word, count in top:
        print(f"  {word:<20} {count:>8,}")

    print(f"\nCombiner speedup: {t_no_combine/t_combine:.2f}x")
    print(f"Unique words: {len(results_combine):,}")


if __name__ == "__main__":
    main()
