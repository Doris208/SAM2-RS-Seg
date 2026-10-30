"""Standard CE and multiclass soft Dice, independent of Lightning/TorchMetrics.

Dice algorithm references (no third-party implementation is copied):
https://github.com/qubvel-org/segmentation_models.pytorch/blob/v0.5.0/segmentation_models_pytorch/losses/dice.py
https://github.com/open-mmlab/mmsegmentation/blob/v1.2.2/mmseg/models/losses/dice_loss.py
This version explicitly masks BOTH probabilities and one-hot targets, computes
per-image/class Dice, and defines its absent-class and empty-target policies.
"""

import torch
from torch import nn
from torch.nn import functional as F

from datasets.label_contract import validate_ids
from .config import validate_loss_config
from ..utils.labels import validate_masks


class SegmentationLoss(nn.Module):
    """Scalar objective plus additive evaluation numerator/normalizer terms.

    forward rejects all-ignored batches; evaluation_terms may return (0,0) so
    evaluators can skip empty batches and reject a whole empty dataset later.
    """

    def __init__(self, num_classes, ignore_index=255):
        super().__init__()
        if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes < 2:
            raise ValueError("num_classes must be an integer >=2")
        if not isinstance(ignore_index, int) or isinstance(ignore_index, bool):
            raise ValueError("ignore_index must be an integer")
        validate_ids([], num_classes, ignore_index)
        self.num_classes, self.ignore_index = num_classes, ignore_index

    def _validate(self, logits, target):
        valid = validate_masks(target, self.num_classes, self.ignore_index)
        if (logits.ndim != 4 or not logits.is_floating_point()
                or tuple(logits.shape) != (target.shape[0], self.num_classes, *target.shape[-2:])):
            raise ValueError("Loss expects floating logits [B,K,H,W] and long targets [B,H,W]")
        if not torch.isfinite(logits).all().item():
            raise FloatingPointError("Nonfinite logits passed to loss")
        return valid

    def forward(self, logits, target):
        numerator, normalizer = self.evaluation_terms(logits, target)
        if normalizer.item() == 0:
            raise ValueError("All target pixels are ignored; loss is undefined")
        return numerator / normalizer

    def evaluation_terms(self, logits, target):
        raise NotImplementedError


class CrossEntropyLoss(SegmentationLoss):
    evaluation_unit = "valid_pixels"

    def forward(self, logits, target):
        valid = self._validate(logits, target)
        if not valid.any().item():
            raise ValueError("All target pixels are ignored; loss is undefined")
        # Preserve the original 03B PyTorch mean kernel, including its gradients.
        return F.cross_entropy(logits, target, ignore_index=self.ignore_index)

    def evaluation_terms(self, logits, target):
        valid = self._validate(logits, target)
        return (F.cross_entropy(logits, target, ignore_index=self.ignore_index, reduction="sum"),
                valid.sum())


class DiceLoss(SegmentationLoss):
    evaluation_unit = "supervised_images"

    def __init__(self, num_classes, ignore_index=255, smooth=0.0, eps=1e-7,
                 absent_class="exclude"):
        super().__init__(num_classes, ignore_index)
        validate_loss_config(dict(name="DiceLoss", ignore_index=ignore_index,
                                  smooth=smooth, eps=eps, absent_class=absent_class), ignore_index)
        self.smooth, self.eps, self.absent_class = smooth, eps, absent_class

    def evaluation_terms(self, logits, target):
        valid = self._validate(logits, target)
        work = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
        probabilities = work.log_softmax(dim=1).exp()
        safe_target = target.masked_fill(~valid, 0)
        one_hot = F.one_hot(safe_target, self.num_classes).permute(0, 3, 1, 2).to(work.dtype)
        mask = valid.unsqueeze(1)
        probabilities, one_hot = probabilities * mask, one_hot * mask
        intersection = (probabilities * one_hot).sum(dim=(2, 3))
        gt_mass = one_hot.sum(dim=(2, 3))
        denominator = probabilities.sum(dim=(2, 3)) + gt_mass
        score = (2 * intersection + self.smooth) / (denominator + self.smooth).clamp_min(self.eps)
        score = torch.where(denominator > 0, score, torch.ones_like(score))
        selected = gt_mass > 0 if self.absent_class == "exclude" else torch.ones_like(gt_mass, dtype=torch.bool)
        per_image = ((1 - score) * selected).sum(1) / selected.sum(1).clamp_min(1)
        supervised = valid.flatten(1).any(1)
        return (per_image * supervised).sum(), supervised.sum()


def build_loss(config, num_classes):
    validate_loss_config(config, config.get("ignore_index") if isinstance(config, dict) else None)
    constructors = {"CrossEntropyLoss": CrossEntropyLoss, "DiceLoss": DiceLoss}
    options = {key: value for key, value in config.items() if key != "name"}
    return constructors[config["name"]](num_classes=num_classes, **options)
