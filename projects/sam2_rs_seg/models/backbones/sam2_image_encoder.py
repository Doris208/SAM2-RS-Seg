"""Thin adapter around the pinned Meta SAM2 ImageEncoder; see docs/task03a.md."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"


class SAM2ImageEncoder(nn.Module):
    """Keep the official trunk, neck, positional encoding and scalp unchanged.

    Inputs are floating RGB tensors already ordered and normalized by the caller.
    Config names are relative to the installed official ``sam2`` config package.
    """

    def __init__(
        self,
        sam2_config: str = DEFAULT_SAM2_CONFIG,
        checkpoint_path: str | Path | None = None,
        pretrained: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        checkpoint = None
        if pretrained:
            if checkpoint_path is None:
                raise ValueError("pretrained=True requires checkpoint_path")
            checkpoint = Path(checkpoint_path).expanduser()
            if not checkpoint.is_file():
                raise FileNotFoundError(f"SAM2 checkpoint does not exist: {checkpoint}")
        elif checkpoint_path is not None:
            raise ValueError("checkpoint_path requires pretrained=True")

        from sam2.build_sam import build_sam2

        # The official builder strictly loads the FULL model before extraction.
        sam_model = build_sam2(
            sam2_config,
            ckpt_path=str(checkpoint) if checkpoint is not None else None,
            device=device,
            mode="train",
            apply_postprocessing=False,
        )
        self.image_encoder = sam_model.image_encoder
        self.out_channels = self.image_encoder.neck.d_model
        self.pretrained_loaded = pretrained

    def forward(self, images: torch.Tensor) -> dict:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected images with shape [B, 3, H, W]")
        if not images.is_floating_point():
            raise TypeError("Images must be floating point and normalized by the caller")
        if any(size < 32 or size % 32 for size in images.shape[-2:]):
            raise ValueError("This adapter supports H and W divisible by 32 (at least 32)")
        return self.image_encoder(images)
