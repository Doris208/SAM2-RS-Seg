"""Safe YAML loading and explicit validation; no dynamic imports or evaluation."""

from copy import deepcopy
from pathlib import Path

import yaml

from datasets.label_contract import validate_ids
from ..losses.config import validate_loss_config

SECTIONS = {"model", "data", "loss", "optimizer", "scheduler", "trainer",
            "evaluation", "checkpoint", "seed"}


def validate_config(config: dict) -> dict:
    if not isinstance(config, dict) or set(config) != SECTIONS:
        raise ValueError(f"Experiment must contain exactly these sections: {sorted(SECTIONS)}")
    if (not isinstance(config["seed"], int) or isinstance(config["seed"], bool)
            or not 0 <= config["seed"] < 2**32):
        raise ValueError("seed must be an integer in [0,2**32)")
    for section in SECTIONS - {"seed"}:
        if not isinstance(config[section], dict):
            raise ValueError(f"{section} must be a mapping")
    data = config["data"]
    classes, ignore = data["num_classes"], data["ignore_index"]
    if not isinstance(classes, int) or isinstance(classes, bool) or classes < 2:
        raise ValueError("num_classes must be an integer >=2")
    if not isinstance(ignore, int) or isinstance(ignore, bool):
        raise ValueError("ignore_index must be an integer")
    validate_ids([], classes, ignore)
    size = data["image_size"]
    if not isinstance(size, (list, tuple)) or len(size) != 2 or any(
            not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in size):
        raise ValueError("image_size must contain two positive integers")
    for key in ("input_channels", "batch_size", "train_samples", "val_samples", "test_samples"):
        if not isinstance(data[key], int) or isinstance(data[key], bool) or data[key] <= 0:
            raise ValueError(f"data.{key} must be a positive integer")
    if not isinstance(data["num_workers"], int) or data["num_workers"] < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    if not isinstance(data["pin_memory"], bool):
        raise ValueError("pin_memory must be boolean")
    if not 0 <= data["ignore_fraction"] < 1:
        raise ValueError("ignore_fraction must be in [0,1)")
    validate_loss_config(config["loss"], ignore)
    if config["evaluation"] != {"absent_class": config["evaluation"].get("absent_class")}:
        raise ValueError("evaluation only accepts absent_class")
    if config["evaluation"]["absent_class"] not in ("exclude", "zero"):
        raise ValueError("absent_class must be exclude or zero")
    trainer = config["trainer"]
    if (trainer.get("devices") != 1 or trainer.get("strategy", "auto") != "auto"
            or trainer.get("num_nodes", 1) != 1):
        raise ValueError("03B supports one device and strategy=auto; DDP is not implemented")
    limits = {key: trainer.get(key, -1) for key in ("max_steps", "max_epochs")}
    for key, value in limits.items():
        if not isinstance(value, int) or isinstance(value, bool) or (value != -1 and value <= 0):
            raise ValueError(f"trainer.{key} must be a positive integer or -1")
    if all(value == -1 for value in limits.values()):
        raise ValueError("A positive max_steps or max_epochs is required")
    if config["scheduler"].get("interval", "step") not in ("step", "epoch"):
        raise ValueError("Scheduler interval must be step or epoch")
    if trainer.get("accumulate_grad_batches", 1) <= 0:
        raise ValueError("accumulate_grad_batches must be positive")
    checkpoint = config["checkpoint"]
    if (checkpoint.get("monitor") != "val_mIoU" or checkpoint.get("mode") != "max"
            or checkpoint.get("save_top_k") != 1 or checkpoint.get("save_last") is not True):
        raise ValueError("03B checkpoint requires val_mIoU/max/top-1/save_last=True")
    if set(checkpoint) != {"monitor", "mode", "save_top_k", "save_last"}:
        raise ValueError("Unknown checkpoint options")
    if trainer.get("limit_val_batches", 1) == 0 or trainer.get("check_val_every_n_epoch", 1) != 1:
        raise ValueError("03B requires validation in every training epoch")
    return deepcopy(config)


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        return validate_config(yaml.safe_load(stream))


def resume_signature(config: dict) -> dict:
    """Training limits/output locations may change; scientific protocol may not."""
    signature = {key: deepcopy(config[key]) for key in
                 ("model", "data", "loss", "optimizer", "scheduler", "evaluation", "seed")}
    if signature["scheduler"].get("name") in (None, "none"):
        signature["scheduler"].pop("interval", None)
    else:
        signature["scheduler"].setdefault("interval", "step")
    return signature
