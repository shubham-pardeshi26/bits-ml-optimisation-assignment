"""
Timing and accuracy utilities.

StepTimer wraps CUDA events to measure forward / backward / optimizer step
times with GPU-accurate resolution. On CPU (local Mac dev) it falls back to
time.perf_counter.

Communication overhead is derived offline in analyze.py as:
    comm_ms ≈ distributed_backward_ms − baseline_backward_ms
This avoids injecting extra backward passes into the hot training loop.
"""

import time
from collections import defaultdict
from dataclasses import dataclass

import torch


@dataclass
class EpochMetrics:
    epoch: int
    world_size: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    epoch_time_s: float     # wall-clock seconds for the full epoch
    avg_forward_ms: float   # mean CUDA time: forward pass
    avg_backward_ms: float  # mean CUDA time: backward pass (includes all-reduce in DDP)
    avg_optimizer_ms: float # mean CUDA time: optimizer step
    lr: float


class _TimerContext:
    """Times a block with CUDA events (or perf_counter on CPU)."""

    def __init__(self, name: str, store: dict, use_cuda: bool):
        self.name     = name
        self.store    = store
        self.use_cuda = use_cuda

    def __enter__(self):
        if self.use_cuda:
            self._start = torch.cuda.Event(enable_timing=True)
            self._end   = torch.cuda.Event(enable_timing=True)
            self._start.record()
        else:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        if self.use_cuda:
            self._end.record()
            torch.cuda.synchronize()
            self.store[self.name].append(self._start.elapsed_time(self._end))
        else:
            self.store[self.name].append((time.perf_counter() - self._t0) * 1e3)


class StepTimer:
    """Accumulates per-step CUDA timings and exposes per-key means.

    Usage:
        timer = StepTimer(device)
        with timer.record("forward"):
            out = model(x)
        print(timer.mean("forward"))  # ms
        timer.reset()                 # call at epoch boundary
    """

    def __init__(self, device: torch.device):
        self._use_cuda = device.type == "cuda"
        self._store: dict = defaultdict(list)

    def record(self, name: str) -> _TimerContext:
        return _TimerContext(name, self._store, self._use_cuda)

    def mean(self, name: str) -> float:
        vals = self._store.get(name, [])
        return sum(vals) / len(vals) if vals else 0.0

    def reset(self):
        self._store.clear()


class AverageMeter:
    """Tracks a running weighted mean — the classic PyTorch training idiom."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


def top1_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Top-1 accuracy as a percentage (no grad required)."""
    with torch.no_grad():
        pred    = output.argmax(dim=1)
        correct = pred.eq(target).sum().item()
        return correct / target.size(0) * 100.0
