"""
ResNet-50 adapted for CIFAR-10 (32x32 input).

Standard ResNet-50 is designed for ImageNet (224x224). Two modifications are
needed for CIFAR-10 so the 32x32 spatial resolution is not destroyed early:
  1. Replace the 7x7 conv (stride=2) with a 3x3 conv (stride=1).
  2. Remove the MaxPool layer (replaced with Identity).

This is the same adaptation used in He et al. (2016) for their CIFAR experiments.
"""

import torch.nn as nn
from torchvision.models import resnet50


def build_model(num_classes: int = 10) -> nn.Module:
    model = resnet50(weights=None)

    # Stem: preserve 32x32 spatial resolution through early layers
    model.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    # Classifier head: 2048 -> num_classes
    model.fc = nn.Linear(2048, num_classes)

    return model
