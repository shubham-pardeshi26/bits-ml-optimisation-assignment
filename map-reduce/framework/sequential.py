"""
sequential.py — Single-threaded MapReduce engine (reference implementation).

All phases run in the calling thread. Use this to:
  - understand the algorithm step-by-step
  - validate correctness before switching to the parallel engine
  - establish a timing baseline
"""

from .base import MapReduceBase, MapperFn, ReducerFn


class SequentialMapReduce(MapReduceBase):
    """Single-threaded MapReduce. Every phase runs in the main process."""

    # ------------------------------------------------------------------
    # Map phase
    # ------------------------------------------------------------------

    def _run_map(self, records: list, mapper: MapperFn) -> list:
        """Apply mapper to each record sequentially; flatten to pair list."""
        pairs = []
        for record in records:
            for pair in mapper(record):
                pairs.append(pair)
        return pairs

    # ------------------------------------------------------------------
    # Reduce phase
    # ------------------------------------------------------------------

    def _run_reduce(self, grouped: dict, reducer: ReducerFn) -> list:
        """Apply reducer to each group sequentially; collect output pairs."""
        output = []
        for key, values in grouped.items():
            for pair in reducer(key, iter(values)):
                output.append(pair)
        return output
