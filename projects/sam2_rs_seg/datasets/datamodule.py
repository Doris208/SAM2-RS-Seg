"""Dataset-agnostic Lightning loader organization with synthetic defaults."""

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader

from .synthetic_dataset import SyntheticSegmentationDataset


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, image_size, input_channels, num_classes, ignore_index=255,
                 ignore_fraction=0.05, batch_size=1, num_workers=0, pin_memory=True,
                 train_samples=2, val_samples=2, test_samples=2, seed=42,
                 datasets=None):
        super().__init__()
        self.data_args = dict(image_size=image_size, input_channels=input_channels,
                              num_classes=num_classes, ignore_index=ignore_index,
                              ignore_fraction=ignore_fraction, seed=seed)
        self.batch_size, self.num_workers, self.pin_memory = batch_size, num_workers, pin_memory
        self.seed = seed
        self.sizes = dict(train=train_samples, val=val_samples, test=test_samples)
        self.external_datasets = datasets is not None
        self.datasets = dict(datasets or {})

    def setup(self, stage=None):
        splits = {None: ("train", "val", "test"), "fit": ("train", "val"),
                  "validate": ("val",), "test": ("test",)}
        if stage not in splits:
            raise ValueError(f"Unsupported setup stage: {stage}")
        for split in splits[stage]:
            if split not in self.datasets:
                if self.external_datasets:
                    raise ValueError(f"Missing injected {split} dataset; synthetic fallback is disabled")
                self.datasets[split] = SyntheticSegmentationDataset(
                    num_samples=self.sizes[split], split=split, **self.data_args)

    def _loader(self, split):
        return DataLoader(self.datasets[split], batch_size=self.batch_size,
                          shuffle=split == "train", num_workers=self.num_workers,
                          pin_memory=self.pin_memory, drop_last=False,
                          generator=torch.Generator().manual_seed(self.seed))

    def train_dataloader(self):
        return self._loader("train")

    def val_dataloader(self):
        return self._loader("val")

    def test_dataloader(self):
        return self._loader("test")
