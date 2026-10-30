"""Public Lightning APIs compose an experiment; no custom training loop."""

from copy import deepcopy
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import torch
import yaml

from .lightning_module import SegmentationLightningModule
from ..datasets.datamodule import SegmentationDataModule
from ..models.registry import build_model
from ..utils.config import validate_config


def json_value(value):
    """Serialize diagnostics with missing-class NaN represented by JSON null."""
    if isinstance(value, torch.Tensor):
        return json_value(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_value(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def environment_info():
    git = shutil.which("git")
    info = {"python": sys.version, "torch": str(torch.__version__),
            "lightning": str(pl.__version__), "torchmetrics": importlib.metadata.version("torchmetrics"),
            "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
    if git:
        result = subprocess.run([git, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        info["git_commit"] = result.stdout.strip() if result.returncode == 0 else None
    return info


def build_experiment(config, output_dir, *, resume=None, callbacks=None):
    config = validate_config(config)
    output_dir = Path(output_dir).resolve()
    if resume is not None and not Path(resume).is_file():
        raise FileNotFoundError(f"Lightning checkpoint missing: {resume}")
    pl.seed_everything(config["seed"], workers=True)
    model = build_model(config["model"], config["data"], resume=resume is not None)
    module = SegmentationLightningModule(
        model, config["data"]["num_classes"], config["loss"]["ignore_index"],
        optimizer=config["optimizer"], scheduler=config["scheduler"],
        absent_class=config["evaluation"]["absent_class"], experiment_config=config, loss=config["loss"])
    data = SegmentationDataModule(**config["data"], seed=config["seed"])
    output_dir.mkdir(parents=True, exist_ok=True)
    # Keep both protocols: the SAM2 initialization file and Lightning full resume.
    (output_dir / ("effective_resume.yaml" if resume else "effective_config.yaml")).write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    write_json(output_dir / ("environment_resume.json" if resume else "environment.json"),
               {**environment_info(), "initialization": "lightning_full_resume" if resume else "model_builder",
                "resume_from": str(resume) if resume else None, "seed": config["seed"]})
    checkpoint = ModelCheckpoint(
        dirpath=output_dir / "checkpoints", filename="best-{epoch:02d}-{step:04d}",
        auto_insert_metric_name=False, enable_version_counter=False,
        save_on_train_epoch_end=True, **config["checkpoint"])
    trainer_options = deepcopy(config["trainer"])
    if set(trainer_options) & {"callbacks", "logger", "default_root_dir", "enable_checkpointing"}:
        raise ValueError("Callbacks/logger/checkpoint locations are managed by the experiment")
    trainer = pl.Trainer(**trainer_options, default_root_dir=str(output_dir),
                         logger=CSVLogger(str(output_dir), name="logs"),
                         callbacks=[*(callbacks or []), checkpoint], enable_checkpointing=True)
    return trainer, module, data, checkpoint


def fit_experiment(trainer, module, data, output_dir, *, resume=None):
    trainer.fit(module, datamodule=data, ckpt_path=str(resume) if resume else None)
    # Public API ensures last.ckpt contains the FINAL optimizer/scheduler step,
    # including max_steps stops between validations and versions that only save
    # ModelCheckpoint.save_last when a new top-k checkpoint was written.
    last = Path(output_dir).resolve() / "checkpoints" / "last.ckpt"
    trainer.save_checkpoint(last)
    return last
