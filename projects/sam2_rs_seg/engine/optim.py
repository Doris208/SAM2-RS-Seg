"""Official AdamW and PolynomialLR construction with parameter coverage checks."""

from copy import deepcopy
import inspect

import torch


def build_optimizer(model, config):
    options = deepcopy(config)
    if options.pop("name") != "AdamW":
        raise ValueError("03B supports AdamW only")
    learning_rate = options.pop("learning_rate")
    group_specs = options.pop("groups", [])
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters")
    assigned, groups = set(), []
    for spec in group_specs:
        if set(spec) != {"prefix", "lr_multiplier"} or not spec["prefix"] or spec["lr_multiplier"] <= 0:
            raise ValueError("Each parameter group needs prefix and positive lr_multiplier")
        selected = [p for name, p in parameters if name == spec["prefix"]
                    or name.startswith(spec["prefix"] + ".")]
        if not selected or any(id(p) in assigned for p in selected):
            raise ValueError(f"Empty or overlapping optimizer group: {spec['prefix']}")
        assigned.update(id(p) for p in selected)
        groups.append({"params": selected, "lr": learning_rate * spec["lr_multiplier"]})
    remainder = [p for _, p in parameters if id(p) not in assigned]
    if remainder:
        groups.append({"params": remainder, "lr": learning_rate})
    try:
        inspect.signature(torch.optim.AdamW).bind(groups, lr=learning_rate, **options)
    except TypeError as exc:
        raise ValueError(f"Invalid AdamW options: {exc}") from exc
    optimizer = torch.optim.AdamW(groups, lr=learning_rate, **options)
    actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
    expected = {id(p) for _, p in parameters}
    if set(actual) != expected or len(actual) != len(expected):
        raise RuntimeError("Optimizer contains missing or duplicated trainable parameters")
    return optimizer


def build_scheduler(optimizer, config):
    options = dict(config)
    if options.pop("interval", "step") not in ("step", "epoch"):
        raise ValueError("Scheduler interval must be step or epoch")
    name = options.pop("name")
    if name is None or name == "none":
        if options:
            raise ValueError("No scheduler accepts no additional options")
        return None
    if name != "PolynomialLR" or set(options) != {"total_iters", "power"}:
        raise ValueError("Use none or PolynomialLR(total_iters, power)")
    if (not isinstance(options["total_iters"], int) or isinstance(options["total_iters"], bool)
            or options["total_iters"] <= 0 or options["power"] <= 0):
        raise ValueError("PolynomialLR requires positive total_iters and power")
    return torch.optim.lr_scheduler.PolynomialLR(optimizer, **options)
