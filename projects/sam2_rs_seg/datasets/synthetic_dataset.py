"""Independent deterministic synthetic splits; no real imagery or label shifts."""

import torch
from torch.utils.data import Dataset

from datasets.label_contract import validate_ids


class SyntheticSegmentationDataset(Dataset):
    def __init__(self, num_samples, image_size, input_channels=3, num_classes=6,
                 ignore_index=255, ignore_fraction=0.05, seed=42, split="train"):
        validate_ids([], num_classes, ignore_index)
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split: {split}")
        if num_samples <= 0 or min(*image_size, input_channels) <= 0 or not 0 <= ignore_fraction < 1:
            raise ValueError("Invalid synthetic data dimensions or ignore fraction")
        self.num_samples, self.image_size = num_samples, tuple(image_size)
        self.input_channels, self.num_classes = input_channels, num_classes
        self.ignore_index, self.ignore_fraction = ignore_index, ignore_fraction
        self.seed, self.split = seed, split
        self.metadata = {"input_channels": input_channels, "num_classes": num_classes,
                         "ignore_index": ignore_index, "split": split, "normalized": True}

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if not 0 <= index < self.num_samples:
            raise IndexError(index)
        # Disjoint sample RNG seeds across all three splits, independent of order/workers.
        sample_seed = (self.seed + 3 * index + ("train", "val", "test").index(self.split)) % (2**63 - 1)
        generator = torch.Generator().manual_seed(sample_seed)
        image = torch.randn(self.input_channels, *self.image_size, generator=generator, dtype=torch.float32)
        mask = torch.randint(self.num_classes, self.image_size, generator=generator, dtype=torch.long)
        ignored = torch.rand(self.image_size, generator=generator) < self.ignore_fraction
        mask[ignored] = self.ignore_index
        # Ensure every sample has supervision; class zero always remains valid.
        mask.view(-1)[0] = 0
        return {"image": image, "mask": mask, "sample_id": f"{self.split}_{index}"}
