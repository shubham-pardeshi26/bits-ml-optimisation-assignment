"""
parallel.py — multiprocessing.Pool-based MapReduce engine.

Architecture
------------
Map phase  : Pool.map over all (mapper_fn, record) pairs — embarrassingly parallel
Shuffle    : runs in the MAIN PROCESS — this is intentional and pedagogically
             correct: shuffle is the real-world bottleneck because it requires
             moving data across the network in a real cluster.
Reduce     : Pool.map over all (reducer_fn, key, values) groups — parallel per key

Two separate Pool contexts are used (one for map, one for reduce) so phase
boundaries are clean and timing is accurate.

macOS / spawn constraint
------------------------
Python on macOS uses 'spawn' (not 'fork') for multiprocessing. This means every
argument passed to a worker must be picklable. Lambda functions and closures
defined inside other functions are NOT picklable under spawn.

Solution: all mapper / reducer functions MUST be defined at module level in the
calling script. The shim functions _apply_mapper and _apply_reducer below are
module-level and therefore always picklable.
"""

import multiprocessing as mp
from .base import MapReduceBase, MapperFn, ReducerFn


# ---------------------------------------------------------------------------
# Module-level shims (required for macOS spawn pickling)
# ---------------------------------------------------------------------------

def _apply_mapper(args):
    """Unpack (mapper_fn, record) and return list of (key, value) pairs."""
    mapper_fn, record = args
    return list(mapper_fn(record))


def _apply_reducer(args):
    """Unpack (reducer_fn, key, values) and return list of output pairs."""
    reducer_fn, key, values = args
    return list(reducer_fn(key, iter(values)))


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ParallelMapReduce(MapReduceBase):
    """
    multiprocessing.Pool-based MapReduce.

    Parameters
    ----------
    num_workers : int, optional
        Number of worker processes. Defaults to os.cpu_count().
    """

    def __init__(self, num_workers: int = None):
        import os
        self.num_workers = num_workers or os.cpu_count()

    # ------------------------------------------------------------------
    # Map phase — parallel over records
    # ------------------------------------------------------------------

    def _run_map(self, records: list, mapper: MapperFn) -> list:
        args = [(mapper, r) for r in records]
        with mp.Pool(self.num_workers) as pool:
            nested = pool.map(_apply_mapper, args)
        # Flatten list-of-lists
        return [pair for sublist in nested for pair in sublist]

    # ------------------------------------------------------------------
    # Reduce phase — parallel over keys
    # ------------------------------------------------------------------

    def _run_reduce(self, grouped: dict, reducer: ReducerFn) -> list:
        args = [(reducer, key, vals) for key, vals in grouped.items()]
        with mp.Pool(self.num_workers) as pool:
            nested = pool.map(_apply_reducer, args)
        return [pair for sublist in nested for pair in sublist]
