"""Tensor label validation shared by losses, evaluation and training adapters."""

import torch

from datasets.label_contract import validate_ids


def validate_masks(mask, num_classes, ignore_index, *, require_valid=False):
    if mask.dtype != torch.long or mask.ndim != 3:
        raise ValueError("mask must be torch.long [B,H,W]")
    validate_ids(mask.unique().tolist(), num_classes, ignore_index)
    valid = mask != ignore_index
    if require_valid and not valid.any().item():
        raise ValueError("All training pixels are ignored; no supervised update is possible")
    return valid
