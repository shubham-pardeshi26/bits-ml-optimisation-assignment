"""
Entry point for single-GPU and distributed (DDP) CIFAR-10 training.

Single GPU:
    python src/train.py --config configs/baseline.yaml

Distributed (2 GPUs via torchrun):
    torchrun --nproc_per_node=2 src/train.py --config configs/distributed.yaml

The script auto-detects distributed mode from the presence of the LOCAL_RANK
environment variable set by torchrun.
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

# Make src/ importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset import build_dataloaders
from metrics import AverageMeter, StepTimer, top1_accuracy
from model import build_model
from utils import (
    append_result,
    get_rank,
    get_world_size,
    is_main_process,
    load_checkpoint,
    load_config,
    save_checkpoint,
    set_seed,
    setup_logger,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="CIFAR-10 DDP training")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Distributed init
# ---------------------------------------------------------------------------

def setup_distributed(backend: str):
    """Initialise the process group. torchrun sets LOCAL_RANK automatically."""
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


# ---------------------------------------------------------------------------
# Optimizer / Scheduler
# ---------------------------------------------------------------------------

def build_optimizer(cfg: dict, model: nn.Module) -> torch.optim.Optimizer:
    opt = cfg["training"]["optimizer"]
    if opt["type"] == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=opt["lr"],
            momentum=opt.get("momentum", 0.9),
            weight_decay=opt.get("weight_decay", 5e-4),
            nesterov=opt.get("nesterov", True),
        )
    if opt["type"] == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=opt["lr"],
            weight_decay=opt.get("weight_decay", 1e-4),
        )
    raise ValueError(f"Unknown optimizer: {opt['type']}")


def build_scheduler(cfg: dict, optimizer: torch.optim.Optimizer):
    """Linear warmup followed by cosine annealing."""
    total_epochs  = cfg["training"]["epochs"]
    warmup_epochs = cfg["training"]["scheduler"].get("warmup_epochs", 5)

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Train / Validate
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, loader, optimizer, scaler, criterion,
    timer, device, cfg, epoch: int,
) -> dict:
    model.train()
    use_amp = cfg["training"]["amp"] and device.type == "cuda"

    # DistributedSampler must be re-seeded each epoch for proper shuffling
    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)

    loss_meter = AverageMeter()
    acc_meter  = AverageMeter()

    for inputs, targets in loader:
        inputs  = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Forward
        with timer.record("forward"):
            with autocast(enabled=use_amp):
                outputs = model(inputs)
                loss    = criterion(outputs, targets)

        # Backward (all-reduce is triggered here in DDP mode)
        optimizer.zero_grad(set_to_none=True)
        with timer.record("backward"):
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        # Optimizer step
        with timer.record("optimizer"):
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

        n = inputs.size(0)
        loss_meter.update(loss.item(), n)
        acc_meter.update(top1_accuracy(outputs, targets), n)

    return {
        "loss":          loss_meter.avg,
        "accuracy":      acc_meter.avg,
        "fwd_ms":        timer.mean("forward"),
        "bwd_ms":        timer.mean("backward"),
        "opt_ms":        timer.mean("optimizer"),
    }


@torch.no_grad()
def validate(model, loader, criterion, device, cfg) -> dict:
    model.eval()
    use_amp = cfg["training"]["amp"] and device.type == "cuda"

    loss_meter = AverageMeter()
    acc_meter  = AverageMeter()

    for inputs, targets in loader:
        inputs  = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            outputs = model(inputs)
            loss    = criterion(outputs, targets)
        n = inputs.size(0)
        loss_meter.update(loss.item(), n)
        acc_meter.update(top1_accuracy(outputs, targets), n)

    return {"loss": loss_meter.avg, "accuracy": acc_meter.avg}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg  = load_config(args.config)

    # --- Distributed setup ---
    distributed = cfg["distributed"]["enabled"]
    if distributed:
        local_rank = setup_distributed(cfg["distributed"]["backend"])
        device     = torch.device(f"cuda:{local_rank}")
    elif torch.cuda.is_available():
        device     = torch.device("cuda")
        local_rank = 0
    else:
        device     = torch.device("cpu")
        local_rank = 0

    rank       = get_rank()
    world_size = get_world_size()
    logger     = setup_logger(cfg["logging"]["log_dir"], rank)

    set_seed(cfg["training"]["seed"], rank)

    # --- Model ---
    model = build_model(num_classes=cfg["model"]["num_classes"]).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank])

    # --- Data ---
    train_loader, val_loader = build_dataloaders(
        data_dir    = cfg["data"]["data_dir"],
        batch_size  = cfg["training"]["batch_size"],
        num_workers = cfg["data"]["num_workers"],
        distributed = distributed,
        pin_memory  = cfg["data"]["pin_memory"],
    )

    # --- Training components ---
    criterion = nn.CrossEntropyLoss(
        label_smoothing=cfg["training"].get("label_smoothing", 0.0)
    )
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    use_amp   = cfg["training"]["amp"] and device.type == "cuda"
    scaler    = GradScaler() if use_amp else None

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer) + 1
        logger.info(f"Resumed from {args.resume}, epoch {start_epoch}")

    timer    = StepTimer(device)
    best_acc = 0.0

    if is_main_process():
        logger.info(
            f"world_size={world_size} | device={device} | "
            f"batch={cfg['training']['batch_size']} | amp={use_amp} | "
            f"epochs={cfg['training']['epochs']}"
        )

    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        timer.reset()
        t0 = time.perf_counter()

        train_m = train_one_epoch(
            model, train_loader, optimizer, scaler,
            criterion, timer, device, cfg, epoch,
        )
        val_m = validate(model, val_loader, criterion, device, cfg)

        epoch_time = time.perf_counter() - t0
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        if is_main_process():
            logger.info(
                f"[{epoch:3d}/{cfg['training']['epochs']}] "
                f"train loss={train_m['loss']:.4f} acc={train_m['accuracy']:.2f}% | "
                f"val loss={val_m['loss']:.4f} acc={val_m['accuracy']:.2f}% | "
                f"time={epoch_time:.1f}s lr={current_lr:.5f} | "
                f"fwd={train_m['fwd_ms']:.1f}ms "
                f"bwd={train_m['bwd_ms']:.1f}ms "
                f"opt={train_m['opt_ms']:.1f}ms"
            )

            append_result(cfg["logging"]["log_dir"], {
                "epoch":          epoch,
                "world_size":     world_size,
                "train_loss":     train_m["loss"],
                "train_acc":      train_m["accuracy"],
                "val_loss":       val_m["loss"],
                "val_acc":        val_m["accuracy"],
                "epoch_time_s":   epoch_time,
                "avg_forward_ms": train_m["fwd_ms"],
                "avg_backward_ms":train_m["bwd_ms"],
                "avg_optimizer_ms":train_m["opt_ms"],
                "lr":             current_lr,
            })

            # Best checkpoint
            if val_m["accuracy"] > best_acc:
                best_acc = val_m["accuracy"]
                raw_model = model.module if distributed else model
                save_checkpoint(
                    {"epoch": epoch, "model": raw_model.state_dict(),
                     "optimizer": optimizer.state_dict(), "best_acc": best_acc},
                    cfg["logging"]["checkpoint_dir"],
                    "best.pth",
                )

            # Rolling last checkpoint (for resume)
            raw_model = model.module if distributed else model
            save_checkpoint(
                {"epoch": epoch, "model": raw_model.state_dict(),
                 "optimizer": optimizer.state_dict()},
                cfg["logging"]["checkpoint_dir"],
                "last.pth",
            )

    if is_main_process():
        logger.info(f"Done. Best val accuracy: {best_acc:.2f}%")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
