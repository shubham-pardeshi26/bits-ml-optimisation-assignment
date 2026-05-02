# MapReduce — Educational Reference for Distributed ML

This folder provides a self-contained, progressive introduction to the MapReduce
paradigm — from the original 2004 Google model through to how gradient aggregation
in `src/train.py` is conceptually identical to a MapReduce job.

---

## Table of Contents

1. [What is MapReduce](#1-what-is-mapreduce)
2. [The Three Phases](#2-the-three-phases)
3. [The Combiner Optimization](#3-the-combiner-optimization)
4. [Fault Tolerance](#4-fault-tolerance)
5. [MapReduce vs. Modern Alternatives](#5-mapreduce-vs-modern-alternatives)
6. [Framework Usage](#6-framework-usage)
7. [Running the Examples](#7-running-the-examples)
8. [Examples Overview](#8-examples-overview)
9. [Connecting to PyTorch DDP](#9-connecting-to-pytorch-ddp)
10. [Further Reading](#10-further-reading)

---

## 1. What is MapReduce

MapReduce (Dean & Ghemawat, Google 2004) is a programming model for processing
large datasets across a cluster of commodity machines. The core insight:

> **Any computation that can be expressed as "apply a function to each record,
> then aggregate by key" can be distributed automatically.**

The runtime handles parallelism, fault tolerance, and data movement. The programmer
writes only two functions: `map` and `reduce`.

```
Input records
     │
     ▼
┌─────────┐     ┌──────────┐     ┌──────────┐
│   MAP   │────▶│ SHUFFLE  │────▶│  REDUCE  │────▶ Output
│(parallel│     │(sort+    │     │(parallel │
│per rec) │     │ group)   │     │per key)  │
└─────────┘     └──────────┘     └──────────┘
  Worker 0         Network          Worker 0
  Worker 1         (bottleneck)     Worker 1
  Worker 2                          Worker 2
```

---

## 2. The Three Phases

### Map

Each input record is processed independently by the mapper function:

```
record  ──▶  mapper(record)  ──▶  [(key_1, val_1), (key_2, val_2), ...]
```

Records are distributed across workers — no inter-worker communication needed.
This phase is **embarrassingly parallel**.

### Shuffle

The runtime collects all `(key, value)` pairs from all mappers, groups them by
key, and routes each group to a reducer worker:

```
All (key, value) pairs
       │
       ▼  sort by key
  key_A → [v1, v3, v7]    ──▶  reducer worker 0
  key_B → [v2, v5]        ──▶  reducer worker 1
  key_C → [v4, v6, v8]    ──▶  reducer worker 2
```

This phase requires **moving data across the network** — it is the primary
bottleneck in real MapReduce clusters.

### Reduce

Each reducer receives a single key and all its associated values:

```
(key, Iterator[values])  ──▶  reducer(key, values)  ──▶  [(key, result), ...]
```

Reducers for different keys run in parallel.

---

## 3. The Combiner Optimization

A **combiner** is a mini-reducer that runs locally on the mapper's output
*before* the shuffle. It performs partial aggregation on a single machine,
reducing the volume of data sent over the network.

```
Without combiner:
  Mapper → 1,000,000 ("the", 1) pairs → network → reducer sums them

With combiner:
  Mapper → 1,000,000 pairs → combiner → ("the", 45821) → network → reducer
                                         ^--- 1 pair instead of 45,821
```

**Constraint**: the combiner must be associative and commutative — the same
requirement as the reducer. This mirrors Hadoop's combiner API.

Traffic reduction: typically **10–100x** fewer bytes over the network for
count/sum operations.

---

## 4. Fault Tolerance

Real MapReduce (Hadoop, Google MR) achieves fault tolerance by:

- **Checkpointing map outputs** to local disk before shuffle
- **Re-executing failed tasks** on another machine (map tasks are idempotent)
- **Detecting stragglers** and speculatively re-executing slow tasks
- **Heartbeat monitoring** — coordinator re-assigns tasks from dead workers

Our implementation does **not** implement fault tolerance. If a worker dies, the
job fails. This is acceptable for educational purposes and mirrors how PyTorch
DDP behaves: a failed worker crashes the entire training job.

---

## 5. MapReduce vs. Modern Alternatives

| System | Data location | Fault tolerance | Latency | Best for |
|---|---|---|---|---|
| Hadoop MapReduce | HDFS (disk) | Yes (recompute) | Minutes | TB-scale ETL |
| Apache Spark | Memory + disk | Yes (lineage) | Seconds | Iterative ML |
| PyTorch DDP | GPU memory | No | Microseconds | DNN training |
| MPI (all-reduce) | RAM / GPU | No | Microseconds | HPC simulations |

**Why not MapReduce for DNN training?**

1. **Iterative**: DNNs train over hundreds of epochs. MapReduce was designed
   for single-pass jobs; reading from HDFS each iteration is prohibitive.
2. **Latency**: GPU-to-GPU NCCL transfers are 10–1000x faster than disk-based shuffle.
3. **Granularity**: DNN gradients are dense arrays; MapReduce serializes to
   `(key, bytes)` pairs, wasting bandwidth.

---

## 6. Framework Usage

### Sequential engine (for learning and debugging)

```python
from framework import SequentialMapReduce

def mapper(line):
    for word in line.split():
        yield (word, 1)

def reducer(key, values):
    yield (key, sum(values))

engine  = SequentialMapReduce()
results = engine.run(data, mapper, reducer, verbose=True)
```

### Parallel engine (multiprocessing.Pool)

```python
from framework import ParallelMapReduce

# IMPORTANT: mapper / reducer must be module-level functions
# (no lambdas or nested functions) for macOS spawn pickling.

engine  = ParallelMapReduce(num_workers=4)
results = engine.run(data, mapper, reducer, verbose=True)
```

### With a combiner

```python
def combiner(key, values):
    yield sum(values)          # same signature as reducer, yields values (not pairs)

results = engine.run(data, mapper, reducer, combiner=combiner, verbose=True)
```

### Inspecting timings

```python
engine.run(data, mapper, reducer)
print(engine.last_timings)
# {'map': 0.123, 'shuffle': 0.045, 'reduce': 0.089}  # seconds
```

---

## 7. Running the Examples

### Prerequisites

Datasets must be generated before running any example:

```bash
# From project root (MLSysOp/)
python map-reduce/data/generate_datasets.py --output map-reduce/data --seed 42
```

### Run all examples at once

```bash
bash map-reduce/run_all.sh
```

### Run individually

```bash
# Word count — sequential, show top 10 words
python map-reduce/examples/01_word_count.py --engine sequential --top 10

# Word count — parallel with 4 workers
python map-reduce/examples/01_word_count.py --engine parallel --workers 4 --top 20

# Sales aggregation
python map-reduce/examples/02_sales_aggregation.py --engine sequential

# Inverted index with word lookup
python map-reduce/examples/03_inverted_index.py --engine sequential --lookup gradient

# Matrix multiply (deliberately slow — teaches the lesson)
python map-reduce/examples/04_matrix_multiply.py --size 32 --engine sequential

# Gradient aggregation (bridge to DDP)
python map-reduce/examples/05_gradient_aggregation.py --engine sequential
```

---

## 8. Examples Overview

| # | File | Problem type | Key concept |
|---|------|-------------|-------------|
| 01 | `01_word_count.py` | Frequency count | Combiner reduces shuffle traffic |
| 02 | `02_sales_aggregation.py` | Group-by aggregation | Composite tuple keys; job chaining |
| 03 | `03_inverted_index.py` | Set construction | Set-valued reducers; deduplication |
| 04 | `04_matrix_multiply.py` | Linear algebra | MapReduce is **slower** than numpy in-memory |
| 05 | `05_gradient_aggregation.py` | DNN gradient sync | MapReduce IS the DDP all-reduce |

---

## 9. Connecting to PyTorch DDP

The gradient aggregation step in `src/train.py` is structurally identical to
a MapReduce job. Here is the side-by-side:

```
MapReduce (05_gradient_aggregation.py)    PyTorch DDP (src/train.py ~line 137)
──────────────────────────────────────    ────────────────────────────────────
mapper(record)                            loss.backward()
  -> (layer_name, grad_array)             # DDP hook fires on each param.grad

reducer(layer_name, [g0, g1, g2, g3])     dist.all_reduce(
  -> mean_grad = np.mean(stack, axis=0)     param.grad, op=ReduceOp.SUM)
                                           param.grad /= world_size

# shuffle routes each layer's grads       # NCCL routes tensors via ring
# to the correct reducer                  # topology — no central bottleneck
```

### Why NCCL ring-allreduce is faster than MapReduce shuffle

```
MapReduce shuffle:
  Worker 0 ──▶ Sort & Merge ──▶ Worker 0
  Worker 1 ──▶ (central     ──▶ Worker 1   O(N * data) bytes total
  Worker 2 ──▶  bottleneck) ──▶ Worker 2

NCCL ring-allreduce:
  Worker 0 ──▶ Worker 1 ──▶ Worker 2 ──▶ Worker 0
              (ring; 2 passes)             O(2*(N-1)/N * data) bytes total
              No central node!
```

NCCL also:
- Sends raw binary tensors (no key serialisation overhead)
- Uses NVLink / InfiniBand for direct GPU-to-GPU memory transfer
- Overlaps communication with the backward pass (DDP `bucket` mechanism)

---

## 10. Further Reading

- **Original paper**: Dean & Ghemawat (2004), *MapReduce: Simplified Data Processing on Large Clusters* — https://research.google/pubs/pub62/
- **Hadoop**: https://hadoop.apache.org/docs/current/hadoop-mapreduce-client/hadoop-mapreduce-client-core/MapReduceTutorial.html
- **PyTorch DDP internals**: https://pytorch.org/docs/stable/notes/ddp.html
- **NCCL ring-allreduce**: Baidu Research (2017), *Bringing HPC Techniques to Deep Learning*
- **Spark vs. MapReduce**: Zaharia et al. (2012), *Resilient Distributed Datasets*
