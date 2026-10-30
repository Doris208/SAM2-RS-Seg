"""Synthetic supervision and gradient diagnostics shared by task 03A checks."""

import torch
from torch import nn

from datasets.label_contract import IGNORE_INDEX, validate_ids


def validate_masks(masks: torch.Tensor, num_classes: int) -> None:
    """Reject invalid IDs and all-ignored batches before mean cross entropy."""
    if masks.ndim != 3 or masks.dtype != torch.long:
        raise ValueError("Masks must be torch.long [B, H, W]")
    validate_ids(masks.unique().tolist(), num_classes, IGNORE_INDEX)
    if not (masks != IGNORE_INDEX).any().item():
        raise ValueError("All pixels are ignored; at least one supervised pixel is required")


def synthetic_batch(batch_size, height, width, num_classes, device="cpu"):
    """Generate float32 images and class IDs with a guaranteed ignored region."""
    if min(batch_size, height, width, num_classes) <= 0 or height * width <= num_classes:
        raise ValueError("Need positive dimensions and more pixels than classes")
    images = torch.randn(batch_size, 3, height, width, device=device, dtype=torch.float32)
    masks = torch.randint(num_classes, (batch_size, height, width), device=device)
    masks[:, : max(1, height // 8), : max(1, width // 8)] = IGNORE_INDEX
    # Reserve the last pixels for every valid class, including class zero.
    masks.flatten()[-num_classes:] = torch.arange(num_classes, device=device)
    validate_masks(masks, num_classes)
    return images, masks


def gradient_report(module: nn.Module, require_all: bool = False) -> dict:
    """Separate frozen, missing, zero, nonfinite and finite nonzero gradients.

    With require_all, every parameter must be trainable with a finite gradient.
    Individual zero gradients remain visible; required weights are checked
    separately by multiscale_gradient_report.
    """
    groups = {name: [] for name in ("frozen", "none", "zero", "nonfinite", "nonzero")}
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            status = "frozen"
        elif parameter.grad is None:
            status = "none"
        elif not torch.isfinite(parameter.grad).all().item():
            status = "nonfinite"
        elif not torch.count_nonzero(parameter.grad).item():
            status = "zero"
        else:
            status = "nonzero"
        groups[status].append(name)
    return {"counts": {key: len(value) for key, value in groups.items()},
            "parameters": groups,
            "ok": bool(groups["nonzero"]) and not groups["nonfinite"]
                  and (not require_all or not (groups["frozen"] or groups["none"]))}


def multiscale_gradient_report(trunk: nn.Module, neck: nn.Module, head: nn.Module) -> dict:
    """Require complete Hiera/neck connectivity and effective branch weights."""
    stage_ends = list(trunk.stage_ends)
    if (len(stage_ends) != 4 or stage_ends != sorted(set(stage_ends))
            or stage_ends[0] < 0 or stage_ends[-1] != len(trunk.blocks) - 1
            or len(neck.convs) != 4):
        raise ValueError("Expected four Hiera stages and four official neck projections")

    reports = {name: gradient_report(module, require_all=True) for name, module in (
        ("hiera", trunk), ("fpn_neck", neck), ("semantic_head", head))}
    reports["hiera_blocks"] = {
        f"blocks.{i}": gradient_report(block, require_all=True)
        for i, block in enumerate(trunk.blocks)
    }
    reports["fpn_neck_branches"] = {
        f"convs.{i}": gradient_report(projection, require_all=True)
        for i, projection in enumerate(neck.convs)
    }
    reports["hiera_stages"] = {}
    start = 0
    for stage, end in enumerate(stage_ends, start=1):
        blocks = [f"blocks.{i}" for i in range(start, end + 1)]
        reports["hiera_stages"][f"stage_{stage}"] = {
            "blocks": blocks,
            "ok": all(reports["hiera_blocks"][name]["ok"] for name in blocks),
        }
        start = end + 1

    required = [("hiera.patch_embed.proj.weight", trunk.patch_embed.proj.weight)]
    required += [(f"hiera.blocks.{i}.attn.qkv.weight", block.attn.qkv.weight)
                 for i, block in enumerate(trunk.blocks)]
    required += [(f"fpn_neck.convs.{i}.conv.weight", projection.conv.weight)
                 for i, projection in enumerate(neck.convs)]
    required += [("semantic_head.layers.0.weight", head.layers[0].weight),
                 ("semantic_head.layers.3.weight", head.layers[3].weight)]
    reports["required_weights"] = {
        name: {"ok": parameter.requires_grad and parameter.grad is not None
               and torch.isfinite(parameter.grad).all().item()
               and bool(torch.count_nonzero(parameter.grad).item())}
        for name, parameter in required
    }
    failures = [name for name in ("hiera", "fpn_neck", "semantic_head")
                if not reports[name]["ok"]]
    for group in ("hiera_stages", "hiera_blocks", "fpn_neck_branches", "required_weights"):
        failures.extend(name for name, result in reports[group].items() if not result["ok"])
    reports["failures"] = failures
    reports["ok"] = not failures
    return reports


def parameter_counts(module: nn.Module) -> dict:
    return {"total": sum(p.numel() for p in module.parameters()),
            "trainable": sum(p.numel() for p in module.parameters() if p.requires_grad)}
