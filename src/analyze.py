"""
Compute assignment performance metrics and generate plots from training results.

Usage:
    python src/analyze.py \
        --baseline    results/baseline/metrics.json \
        --distributed results/distributed/metrics.json \
        --output      results/analysis

Outputs:
    results/analysis/summary.json     -- speedup table (all five required metrics)
    results/analysis/convergence.png  -- val accuracy + train loss curves
    results/analysis/epoch_times.png  -- wall-clock time per epoch
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless; safe for Kaggle / Colab
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_metrics(path: str) -> list:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Speedup table
# ---------------------------------------------------------------------------

def compute_summary(baseline: list, distributed: list) -> dict:
    """
    Compute the five required metrics using the last 10 steady-state epochs.

    Metrics:
        S(N)  = T1 / TN                  (speedup)
        E(N)  = S(N) / N * 100           (parallel efficiency %)
        comm  ≈ bwd_ddp - bwd_baseline   (all-reduce overhead per step)
        RT    = mean wall-clock / epoch   (response time)
        gap   = acc_baseline - acc_ddp   (accuracy gap)
    """
    tail = 10
    base_records = baseline[-tail:]
    dist_records = distributed[-tail:]

    base_epoch_time  = sum(r["epoch_time_s"]    for r in base_records) / tail
    dist_epoch_time  = sum(r["epoch_time_s"]    for r in dist_records) / tail
    base_bwd_ms      = sum(r["avg_backward_ms"] for r in base_records) / tail
    dist_bwd_ms      = sum(r["avg_backward_ms"] for r in dist_records) / tail

    world_size  = distributed[0]["world_size"]
    speedup     = base_epoch_time / dist_epoch_time
    efficiency  = speedup / world_size * 100.0
    comm_ms     = max(0.0, dist_bwd_ms - base_bwd_ms)

    base_acc = baseline[-1]["val_acc"]
    dist_acc = distributed[-1]["val_acc"]

    return {
        "world_size":           world_size,
        "baseline_epoch_time_s":  round(base_epoch_time, 3),
        "dist_epoch_time_s":      round(dist_epoch_time, 3),
        "speedup_S_N":            round(speedup, 4),
        "parallel_efficiency_pct":round(efficiency, 2),
        "comm_overhead_ms":       round(comm_ms, 3),
        "baseline_val_acc":       round(base_acc, 3),
        "dist_val_acc":           round(dist_acc, 3),
        "accuracy_gap":           round(base_acc - dist_acc, 3),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_convergence(baseline: list, distributed: list, out: Path):
    epochs_b = [r["epoch"] for r in baseline]
    epochs_d = [r["epoch"] for r in distributed]
    n        = distributed[0]["world_size"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(epochs_b, [r["val_acc"]    for r in baseline],    label="1 GPU (baseline)")
    ax1.plot(epochs_d, [r["val_acc"]    for r in distributed], label=f"{n} GPUs (DDP)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Accuracy (%)")
    ax1.set_title("Convergence — Validation Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_b, [r["train_loss"] for r in baseline],    label="1 GPU (baseline)")
    ax2.plot(epochs_d, [r["train_loss"] for r in distributed], label=f"{n} GPUs (DDP)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Training Loss")
    ax2.set_title("Convergence — Training Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out / "convergence.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out / 'convergence.png'}")


def plot_epoch_times(baseline: list, distributed: list, out: Path):
    epochs_b = [r["epoch"] for r in baseline]
    epochs_d = [r["epoch"] for r in distributed]
    n        = distributed[0]["world_size"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs_b, [r["epoch_time_s"] for r in baseline],    label="1 GPU (baseline)")
    ax.plot(epochs_d, [r["epoch_time_s"] for r in distributed], label=f"{n} GPUs (DDP)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Wall-Clock Time (s)")
    ax.set_title("Epoch Wall-Clock Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out / "epoch_times.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out / 'epoch_times.png'}")


def plot_step_breakdown(baseline: list, distributed: list, out: Path):
    """Stacked bar: forward / backward / optimizer time per epoch (mean over run)."""
    def means(records):
        fwd = sum(r["avg_forward_ms"]   for r in records) / len(records)
        bwd = sum(r["avg_backward_ms"]  for r in records) / len(records)
        opt = sum(r["avg_optimizer_ms"] for r in records) / len(records)
        return fwd, bwd, opt

    n = distributed[0]["world_size"]
    labels = ["1 GPU\n(baseline)", f"{n} GPUs\n(DDP)"]
    fwds, bwds, opts = zip(means(baseline), means(distributed))

    x   = range(len(labels))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x, fwds, label="Forward",   color="#4C72B0")
    ax.bar(x, bwds, label="Backward",  color="#DD8452", bottom=fwds)
    ax.bar(x, opts, label="Optimizer", color="#55A868",
           bottom=[f + b for f, b in zip(fwds, bwds)])

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Step Time (ms)")
    ax.set_title("Per-Step Time Breakdown")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(out / "step_breakdown.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out / 'step_breakdown.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline",    required=True, help="Path to baseline metrics.json")
    p.add_argument("--distributed", required=True, help="Path to distributed metrics.json")
    p.add_argument("--output",      default="results/analysis")
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    baseline    = load_metrics(args.baseline)
    distributed = load_metrics(args.distributed)

    summary = compute_summary(baseline, distributed)

    print("\n=== Performance Summary ===")
    print(f"  World size (N):           {summary['world_size']}")
    print(f"  Baseline epoch time:      {summary['baseline_epoch_time_s']:.2f}s")
    print(f"  Distributed epoch time:   {summary['dist_epoch_time_s']:.2f}s")
    print(f"  Speedup S(N):             {summary['speedup_S_N']:.3f}x")
    print(f"  Parallel efficiency E(N): {summary['parallel_efficiency_pct']:.1f}%")
    print(f"  Comm overhead:            {summary['comm_overhead_ms']:.1f}ms/step")
    print(f"  Baseline val acc:         {summary['baseline_val_acc']:.2f}%")
    print(f"  Distributed val acc:      {summary['dist_val_acc']:.2f}%")
    print(f"  Accuracy gap:             {summary['accuracy_gap']:.2f}%")

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out / 'summary.json'}")

    plot_convergence(baseline, distributed, out)
    plot_epoch_times(baseline, distributed, out)
    plot_step_breakdown(baseline, distributed, out)


if __name__ == "__main__":
    main()
