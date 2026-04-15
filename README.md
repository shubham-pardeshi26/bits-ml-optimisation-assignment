# Distributed CIFAR-10 Training — ML System Optimization Assignment

Group assignment: Design, implement, and evaluate a parallelized ML training system for CIFAR-10 image classification using PyTorch Distributed Data Parallel (DDP).

---

## Project Overview

| Item | Detail |
|---|---|
| Dataset | CIFAR-10 — 50,000 train / 10,000 test, 32×32 RGB, 10 classes |
| Model | ResNet-50 adapted for CIFAR-10 (3×3 stem, no early MaxPool) |
| Framework | PyTorch DDP with NCCL backend |
| Parallelism | Synchronous Data Parallelism (All-Reduce) |
| Configurations | 1 GPU baseline vs. 2 GPU distributed |

---

## Repository Structure

```
MLSysOp/
├── src/
│   ├── train.py        # Main entry point (single-GPU and DDP)
│   ├── model.py        # ResNet-50 adapted for CIFAR-10
│   ├── dataset.py      # CIFAR-10 data loading + DistributedSampler
│   ├── metrics.py      # CUDA-event timing, accuracy, AverageMeter
│   ├── utils.py        # Distributed helpers, logging, checkpointing
│   └── analyze.py      # Post-training metrics + plot generation
├── configs/
│   ├── baseline.yaml   # 1-GPU config (batch=128, lr=0.1)
│   └── distributed.yaml# 2-GPU DDP config (batch=256, lr=0.2)
├── scripts/
│   ├── run_baseline.sh       # Launch single-GPU training
│   └── run_distributed.sh    # Launch DDP training via torchrun
├── results/            # Output logs, metrics.json, checkpoints, plots
├── docs/               # Assignment report documents
├── requirements.txt
└── README.md
```

---

## Setup

### Local (with GPU)

```bash
git clone https://github.com/shubham-pardeshi26/bits-ml-optimisation-assignment.git
cd bits-ml-optimisation-assignment
pip install -r requirements.txt
```

### Kaggle Notebook

1. Create a new notebook at [kaggle.com](https://kaggle.com)
2. In the right sidebar: **Settings → Internet → ON**
3. In the right sidebar: **Session Options → Accelerator → GPU T4 x2**
4. Run in a notebook cell:

```python
!git clone https://github.com/shubham-pardeshi26/bits-ml-optimisation-assignment.git
%cd bits-ml-optimisation-assignment
!pip install -r requirements.txt -q
```

> CIFAR-10 is downloaded automatically by torchvision on first run (~170 MB). No manual data upload needed.

---

## Running Experiments

### Step 1 — Baseline (1 GPU)

```bash
# Via script
bash scripts/run_baseline.sh

# Or directly
python src/train.py --config configs/baseline.yaml
```

Saves results to `./results/baseline/`:
- `metrics.json` — per-epoch stats (loss, accuracy, timing)
- `checkpoints/best.pth` — best validation accuracy checkpoint
- `checkpoints/last.pth` — latest checkpoint (for resume)

To resume an interrupted run:
```bash
python src/train.py --config configs/baseline.yaml --resume results/baseline/checkpoints/last.pth
```

---

### Step 2 — Distributed (2 GPUs)

```bash
# Via script
NGPUS=2 bash scripts/run_distributed.sh

# Or directly
torchrun --nproc_per_node=2 src/train.py --config configs/distributed.yaml
```

Saves results to `./results/distributed/`.

To resume:
```bash
torchrun --nproc_per_node=2 src/train.py \
  --config configs/distributed.yaml \
  --resume results/distributed/checkpoints/last.pth
```

---

### Step 3 — Analysis

Run **after both experiments complete**:

```bash
python src/analyze.py \
  --baseline    results/baseline/metrics.json \
  --distributed results/distributed/metrics.json \
  --output      results/analysis
```

Outputs in `./results/analysis/`:
- `summary.json` — all five required performance metrics
- `convergence.png` — validation accuracy + training loss curves
- `epoch_times.png` — wall-clock time per epoch
- `step_breakdown.png` — stacked bar (forward / backward / optimizer)

---

### Saving Results on Kaggle

Kaggle sessions expire. Before the session ends, zip and download results:

```python
import shutil
shutil.make_archive('results', 'zip', 'results')
# Then download results.zip from the Output panel (right sidebar)
```

---

## Configuration Reference

| Parameter | Baseline | Distributed |
|---|---|---|
| `batch_size` | 128 | 256 (linear scaling) |
| `lr` | 0.1 | 0.2 (linear scaling) |
| `epochs` | 100 | 100 |
| `optimizer` | SGD + Nesterov | SGD + Nesterov |
| `scheduler` | Cosine + 5-epoch warmup | Cosine + 5-epoch warmup |
| `amp` | true | true |
| `world_size` | 1 | 2 |

---

## Required Metrics (computed by analyze.py)

| Metric | Formula |
|---|---|
| Speedup S(N) | T₁ / T_N |
| Parallel Efficiency E(N) | S(N) / N × 100% |
| Communication Overhead | backward_ms(DDP) − backward_ms(baseline) |
| Response Time | Mean wall-clock time per epoch |
| Accuracy Gap | val_acc(baseline) − val_acc(DDP) |

---

## Model Architecture

ResNet-50 with two modifications for CIFAR-10's 32×32 input:

1. **Stem conv**: 7×7 stride-2 → 3×3 stride-1 (preserves spatial resolution)
2. **MaxPool**: Replaced with Identity (removes early downsampling)

Standard ImageNet ResNet-50 would reduce 32×32 to 1×1 before the classifier; these changes keep meaningful spatial features.

---

## Design Decisions

**Data Parallelism (DDP):** Each GPU holds a full model replica and processes a unique data shard. Gradients are averaged via All-Reduce (NCCL) after each backward pass. Chosen for simplicity and suitability to CIFAR-10's small model size.

**Synchronous All-Reduce:** Guarantees identical model state across all ranks after each step. Deterministic and easier to reason about than async approaches.

**Linear Scaling Rule:** Batch size and learning rate scale proportionally with world size (128→256, 0.1→0.2 for 2 GPUs) to maintain training dynamics.

**AMP (Automatic Mixed Precision):** FP16 forward/backward pass with FP32 master weights via GradScaler. Reduces memory and increases throughput on modern GPUs.
