"""Epoch confusion matrices via TorchMetrics, explicit absent-class reduction."""

import torch
from torch import nn
from torchmetrics.classification import MulticlassConfusionMatrix

from datasets.label_contract import validate_ids
from ..utils.labels import validate_masks  # Re-export the existing public helper.


def metrics_from_confusion_matrix(matrix, absent_class="exclude"):
    if absent_class not in ("exclude", "zero"):
        raise ValueError("absent_class must be exclude or zero")
    counts = matrix.to(torch.float64)
    tp = counts.diag()
    gt, prediction = counts.sum(1), counts.sum(0)
    union, f1_denominator = gt + prediction - tp, gt + prediction
    fill = float("nan") if absent_class == "exclude" else 0.0
    iou = torch.where(union > 0, tp / union, fill)
    f1 = torch.where(f1_denominator > 0, 2 * tp / f1_denominator, fill)
    total = counts.sum()
    return {"confusion_matrix": matrix.clone(), "IoU": iou, "F1": f1,
            "mIoU": iou.nanmean(), "mF1": f1.nanmean(),
            "OA": torch.where(total > 0, tp.sum() / total, float("nan")),
            "valid_pixels": total}


class SegmentationMetrics(nn.Module):
    def __init__(self, num_classes, ignore_index=255, absent_class="exclude"):
        super().__init__()
        validate_ids([], num_classes, ignore_index)
        if absent_class not in ("exclude", "zero"):
            raise ValueError("Unknown absent class policy")
        self.num_classes, self.ignore_index = num_classes, ignore_index
        self.absent_class = absent_class
        self.confusion = MulticlassConfusionMatrix(num_classes, ignore_index=ignore_index,
                                                  normalize=None, validate_args=True)

    @torch.no_grad()
    def update(self, prediction, mask):
        validate_masks(mask, self.num_classes, self.ignore_index)
        if prediction.shape != mask.shape or prediction.dtype != torch.long:
            raise ValueError("prediction must be torch.long with the mask shape")
        if ((prediction < 0) | (prediction >= self.num_classes)).any().item():
            raise ValueError("Predicted class IDs are invalid; valid GT cannot be filtered")
        self.confusion.update(prediction, mask)

    def compute(self):
        return metrics_from_confusion_matrix(self.confusion.compute(), self.absent_class)

    def reset(self):
        self.confusion.reset()
