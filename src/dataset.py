"""
CIFAR-10 data loading with support for distributed training.

When DDP is active, DistributedSampler ensures each rank receives a
non-overlapping shard. The caller must call
    train_loader.sampler.set_epoch(epoch)
at the start of each epoch so shuffling differs across epochs.
"""

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

# Per-channel statistics computed over the CIFAR-10 training set
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


def _train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def _val_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def build_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    distributed: bool,
    pin_memory: bool = True,
):
    """Return (train_loader, val_loader).

    torchvision looks for data at {data_dir}/cifar-10-batches-py/, so
    data_dir should be the project root if the data is already extracted there.
    """
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=_train_transform(),
    )
    val_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=_val_transform(),
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True)  if distributed else None
    val_sampler   = DistributedSampler(val_dataset,   shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader
