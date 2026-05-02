"""
base.py — Abstract base class and shared type aliases for the MapReduce framework.

Type aliases
------------
KeyValuePair : tuple[Any, Any]
    A single (key, value) pair emitted by the mapper or reducer.

MapperFn : Callable[[Any], Iterable[KeyValuePair]]
    Takes one input record, yields (key, value) pairs.

CombinerFn : Callable[[Any, Iterator], Iterator]
    fn(key, Iterator[V]) -> Iterator[V]
    Local aggregation before shuffle (same constraints as reducer:
    must be associative and commutative).

ReducerFn : Callable[[Any, Iterator], Iterable[KeyValuePair]]
    fn(key, Iterator[V]) -> Iterable[(key, result)]
    Takes a key and all its values, returns output (key, value) pairs.
"""

import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Iterable, Iterator, Optional, Tuple

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

KeyValuePair = Tuple[Any, Any]
MapperFn     = Callable[[Any], Iterable[KeyValuePair]]
CombinerFn   = Callable[[Any, Iterator], Iterator]
ReducerFn    = Callable[[Any, Iterator], Iterable[KeyValuePair]]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class MapReduceBase(ABC):
    """Shared orchestrator. Subclasses implement the three core phases."""

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        data: Iterable[Any],
        mapper: MapperFn,
        reducer: ReducerFn,
        combiner: Optional[CombinerFn] = None,
        verbose: bool = False,
    ) -> list:
        """
        Execute a full MapReduce job.

        Parameters
        ----------
        data     : iterable of input records
        mapper   : fn(record) -> Iterable[(key, value)]
        reducer  : fn(key, Iterator[value]) -> Iterable[(key, result)]
        combiner : optional fn(key, Iterator[value]) -> Iterator[value]
                   for local pre-aggregation before the shuffle
        verbose  : if True, print phase timings and record counts

        Returns
        -------
        list of (key, value) output pairs, sorted by key
        """
        records = list(data)
        timings = {}

        # MAP
        t0 = time.perf_counter()
        mapped = self._run_map(records, mapper)
        timings["map"] = time.perf_counter() - t0
        if verbose:
            print(f"  [map]     {len(records):>7,} input records "
                  f"-> {len(mapped):>7,} pairs  ({timings['map']*1000:.1f} ms)")

        # COMBINE (optional, runs in main process for both engines)
        if combiner is not None:
            t0 = time.perf_counter()
            mapped = self._run_combine(mapped, combiner)
            timings["combine"] = time.perf_counter() - t0
            if verbose:
                print(f"  [combine] after local aggregation "
                      f"-> {len(mapped):>7,} pairs  ({timings['combine']*1000:.1f} ms)")

        # SHUFFLE
        t0 = time.perf_counter()
        grouped = self._run_shuffle(mapped)
        timings["shuffle"] = time.perf_counter() - t0
        if verbose:
            print(f"  [shuffle] {len(grouped):>7,} unique keys           "
                  f"({timings['shuffle']*1000:.1f} ms)")

        # REDUCE
        t0 = time.perf_counter()
        output = self._run_reduce(grouped, reducer)
        timings["reduce"] = time.perf_counter() - t0
        if verbose:
            print(f"  [reduce]  {len(output):>7,} output pairs          "
                  f"({timings['reduce']*1000:.1f} ms)")

        self._last_timings = timings
        return output

    # ------------------------------------------------------------------
    # Phase hooks (to be implemented by subclasses)
    # ------------------------------------------------------------------

    @abstractmethod
    def _run_map(self, records: list, mapper: MapperFn) -> list:
        """Apply mapper to every record; return flat list of (key, value)."""

    @abstractmethod
    def _run_reduce(self, grouped: dict, reducer: ReducerFn) -> list:
        """Apply reducer to each (key, [values]) group; return output pairs."""

    # ------------------------------------------------------------------
    # Shared phases (identical for both engines)
    # ------------------------------------------------------------------

    def _run_combine(self, pairs: list, combiner: CombinerFn) -> list:
        """Group pairs locally then apply the combiner, flattening results."""
        local: dict = defaultdict(list)
        for k, v in pairs:
            local[k].append(v)
        result = []
        for k, vals in local.items():
            for v in combiner(k, iter(vals)):
                result.append((k, v))
        return result

    def _run_shuffle(self, pairs: list) -> dict:
        """Sort and group pairs by key into {key: [values]} dict."""
        grouped: dict = defaultdict(list)
        for k, v in pairs:
            grouped[k].append(v)
        # Sort by string representation for generality (handles tuple keys)
        return dict(sorted(grouped.items(), key=lambda kv: str(kv[0])))

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def last_timings(self) -> dict:
        """Return timing breakdown (seconds) from the most recent run()."""
        return getattr(self, "_last_timings", {})
