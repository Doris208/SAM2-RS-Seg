"""Generic Lightning automatic optimization for ordinary segmentation modules."""

from copy import deepcopy

import lightning.pytorch as pl
import torch
from torch import nn

from .optim import build_optimizer, build_scheduler
from ..evaluation.segmentation_metrics import SegmentationMetrics
from ..losses.segmentation import build_loss
from ..utils.labels import validate_masks
from ..utils.config import resume_signature


class SegmentationLightningModule(pl.LightningModule):
    def __init__(self, model: nn.Module, num_classes: int, ignore_index=255,
                 optimizer=None, scheduler=None, absent_class="exclude", experiment_config=None, loss=None):
        super().__init__()
        self.model = model
        self.num_classes, self.ignore_index = num_classes, ignore_index
        self.optimizer_config = deepcopy(optimizer or dict(name="AdamW", learning_rate=1e-4,
                                                          weight_decay=0.01))
        self.scheduler_config = deepcopy(scheduler or dict(name="none"))
        self.experiment_config = deepcopy(experiment_config or {})
        self.save_hyperparameters(ignore=["model"])
        loss_config = deepcopy(loss if loss is not None else dict(name="CrossEntropyLoss", ignore_index=ignore_index))
        if loss_config.get("ignore_index") != ignore_index:
            raise ValueError("Loss ignore_index must match the training/evaluation protocol")
        self.criterion = build_loss(loss_config, num_classes)
        self.metrics = nn.ModuleDict({f"{split}_phase": SegmentationMetrics(num_classes, ignore_index, absent_class)
                                      for split in ("train", "val", "test")})
        self.epoch_results = {}
        for split in ("val", "test"):
            self.register_buffer(f"_{split}_loss_sum", torch.tensor(0., dtype=torch.float64), persistent=False)
            self.register_buffer(f"_{split}_valid_count", torch.tensor(0, dtype=torch.long), persistent=False)

    def forward(self, images):
        return self.model(images)

    def batch_loss(self, batch, *, training):
        images, mask = batch["image"], batch["mask"]
        valid = validate_masks(mask, self.num_classes, self.ignore_index, require_valid=training)
        if images.ndim != 4 or not images.is_floating_point() or images.shape[0] != mask.shape[0]:
            raise ValueError("image must be floating [B,C,H,W] with the mask batch size")
        if images.shape[-2:] != mask.shape[-2:]:
            raise ValueError("Image and mask spatial sizes differ")
        logits = self(images)
        expected = (images.shape[0], self.num_classes, *images.shape[-2:])
        if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != expected:
            raise ValueError(f"Model must return one logits Tensor with shape {expected}")
        if not torch.isfinite(logits).all().item():
            raise FloatingPointError("Nonfinite model logits")
        count = valid.sum()
        if training:
            loss = self.criterion(logits, mask)
            loss_sum = loss * count
        else:
            loss_sum, count = self.criterion.evaluation_terms(logits, mask)
            loss = loss_sum / count.clamp_min(1)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Nonfinite segmentation loss")
        return logits, loss, loss_sum, count

    def training_step(self, batch, batch_idx):
        logits, loss, _, _ = self.batch_loss(batch, training=True)
        self.metrics["train_phase"].update(logits.detach().argmax(1), batch["mask"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=batch["image"].shape[0])
        return loss

    def _evaluation_step(self, batch, split):
        logits, _, loss_sum, count = self.batch_loss(batch, training=False)
        self.metrics[f"{split}_phase"].update(logits.argmax(1), batch["mask"])
        getattr(self, f"_{split}_loss_sum").add_(loss_sum.detach().to(torch.float64))
        getattr(self, f"_{split}_valid_count").add_(count)

    def validation_step(self, batch, batch_idx):
        self._evaluation_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._evaluation_step(batch, "test")

    def _reset(self, split):
        self.metrics[f"{split}_phase"].reset()
        if split != "train":
            getattr(self, f"_{split}_loss_sum").zero_()
            getattr(self, f"_{split}_valid_count").zero_()

    def on_train_epoch_start(self):
        self._reset("train")

    def on_validation_epoch_start(self):
        self._reset("val")

    def on_test_epoch_start(self):
        self._reset("test")

    def _finish_epoch(self, split):
        result = self.metrics[f"{split}_phase"].compute()
        if result["valid_pixels"].item() == 0:
            raise ValueError(f"{split} has no valid GT pixels; metrics/checkpoint selection unavailable")
        self.epoch_results[split] = {key: value.detach().cpu().clone() for key, value in result.items()}
        self.log_dict({f"{split}_{key}": result[key] for key in ("mIoU", "mF1", "OA")},
                      on_step=False, on_epoch=True)
        if split != "train":
            self.log(f"{split}_loss", getattr(self, f"_{split}_loss_sum")
                     / getattr(self, f"_{split}_valid_count"), on_step=False, on_epoch=True)

    def on_train_epoch_end(self):
        self._finish_epoch("train")

    def on_validation_epoch_end(self):
        self._finish_epoch("val")

    def on_test_epoch_end(self):
        self._finish_epoch("test")

    def configure_optimizers(self):
        optimizer = build_optimizer(self.model, self.optimizer_config)
        scheduler = build_scheduler(optimizer, self.scheduler_config)
        if scheduler is None:
            return optimizer
        return {"optimizer": optimizer, "lr_scheduler": {
            "scheduler": scheduler, "interval": self.scheduler_config.get("interval", "step"), "frequency": 1}}

    def on_save_checkpoint(self, checkpoint):
        checkpoint["experiment_config"] = deepcopy(self.experiment_config)

    def on_load_checkpoint(self, checkpoint):
        previous = checkpoint.get("experiment_config")
        if self.experiment_config and (not previous or resume_signature(previous)
                                       != resume_signature(self.experiment_config)):
            raise ValueError("Full resume requires the same model/data/loss/optimizer/scheduler/seed protocol")
