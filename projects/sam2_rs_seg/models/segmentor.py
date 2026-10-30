"""Dataset- and training-framework-independent multiclass SAM2 segmentation."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .backbones.sam2_image_encoder import DEFAULT_SAM2_CONFIG, SAM2ImageEncoder
from .decode_heads.semantic_head import SemanticSegmentationHead


class SAM2SemanticSegmentor(nn.Module):
    """Official ImageEncoder -> three-scale feature fusion -> semantic logits."""

    def __init__(
        self,
        num_classes: int = 6,
        head_channels: int = 128,
        pretrained: bool = True,
        checkpoint_path: str | Path | None = None,
        sam2_config: str = DEFAULT_SAM2_CONFIG,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        if min(num_classes, head_channels) <= 0:
            raise ValueError("num_classes and head_channels must be positive")
        self.encoder = SAM2ImageEncoder(
            sam2_config=sam2_config,
            checkpoint_path=checkpoint_path,
            pretrained=pretrained,
            device=device,
        )
        self.decode_head = SemanticSegmentationHead(
            in_channels=self.encoder.out_channels,
            head_channels=head_channels,
            num_classes=num_classes,
        ).to(device)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)["backbone_fpn"]
        return self.decode_head(features, output_size=images.shape[-2:])
