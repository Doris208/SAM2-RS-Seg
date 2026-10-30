"""Lightweight three-scale classifier for official SAM2 FpnNeck features."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class SemanticSegmentationHead(nn.Module):
    """Align and concatenate three backbone_fpn levels before classification."""

    def __init__(
        self, in_channels: int = 256, head_channels: int = 128, num_classes: int = 6
    ) -> None:
        super().__init__()
        if min(in_channels, head_channels, num_classes) <= 0:
            raise ValueError("Channel counts and num_classes must be positive")
        self.in_channels = in_channels
        self.layers = nn.Sequential(
            nn.Conv2d(3 * in_channels, head_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_channels, num_classes, 1),
        )

    def forward(
        self, features: Sequence[torch.Tensor], output_size: tuple[int, int]
    ) -> torch.Tensor:
        if isinstance(features, torch.Tensor) or len(features) != 3:
            raise ValueError("Expected three backbone_fpn feature tensors")
        if any(x.ndim != 4 or x.shape[1] != self.in_channels for x in features):
            raise ValueError("Each backbone_fpn feature must be [B, in_channels, H, W]")
        size = features[0].shape[-2:]
        aligned = [features[0]] + [
            F.interpolate(x, size=size, mode="bilinear", align_corners=False)
            for x in features[1:]
        ]
        logits = self.layers(torch.cat(aligned, dim=1))
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
