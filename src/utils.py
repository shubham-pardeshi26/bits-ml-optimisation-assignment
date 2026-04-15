"""
Shared utilities: distributed helpers, seeding, logging, config, checkpointing,
and results persistence.
"""

import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def is_dist_available() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_available() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_available() else 1


def is_main_process() -> bool:
    return get_rank() == 0


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int, rank: int = 0):
    """Give each rank a unique seed so data loading is diverse across workers."""
    effective = seed + rank
    random.seed(effective)
    np.random.seed(effective)
    torch.manual_seed(effective)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(log_dir: str, rank: int = 0) -> logging.Logger:
    logger = logging.getLogger("mlsysop")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | rank%(name_ext)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Attach rank to every record so multi-rank logs are distinguishable
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.name_ext = str(rank)
        return record

    logging.setLogRecordFactory(record_factory)

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Console handler: rank 0 only to avoid interleaved output
    if rank == 0 and not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    # File handler: every rank writes its own log
    fh = logging.FileHandler(Path(log_dir) / f"rank{rank}.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, checkpoint_dir: str, filename: str):
    """Only rank 0 writes checkpoints to avoid race conditions."""
    if not is_main_process():
        return
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    torch.save(state, Path(checkpoint_dir) / filename)


def load_checkpoint(path: str, model: torch.nn.Module, optimizer=None) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0)


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

def append_result(results_dir: str, record: dict):
    """Append a single epoch record to metrics.json (rank 0 only)."""
    if not is_main_process():
        return
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    path = Path(results_dir) / "metrics.json"
    records = []
    if path.exists():
        with open(path) as f:
            records = json.load(f)
    records.append(record)
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
