"""
Dataset utilities for the ISL image dataset.

Images are loaded from one directory per class. Preprocessing preserves
aspect ratio before center cropping to the target resolution. Training
optionally applies affine and color augmentation.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from .config import ExperimentConfig

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg")


def normalize_to_minus_one_one(x: torch.Tensor) -> torch.Tensor:
    return x * 2 - 1


def list_classes(data_root: str) -> list[str]:
    root = Path(data_root)
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    if not classes:
        raise FileNotFoundError(f"No class subfolders found under {data_root}")
    return classes


def resize_aspect_preserving(
    img: Image.Image,
    target_size: int,
    margin_factor: float = 1.15,
) -> Image.Image:
    """Resizes the shorter side while preserving aspect ratio."""
    short_side = max(target_size, int(round(target_size * margin_factor)))
    return TF.resize(img, short_side, antialias=True)


class ResizeAspectPreserving:
    """Resize the shorter side while preserving aspect ratio."""

    def __init__(self, target_size: int, margin_factor: float):
        self.target_size = target_size
        self.margin_factor = margin_factor

    def __call__(self, img: Image.Image) -> Image.Image:
        return resize_aspect_preserving(img, self.target_size, self.margin_factor)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(target_size={self.target_size}, "
            f"margin_factor={self.margin_factor})"
        )


def _augmentation_transforms() -> list:
    """
    Returns the training augmentation pipeline.

    Horizontal flips are omitted because they may change the sign label.
    """
    return [
        T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        T.ColorJitter(brightness=0.15, contrast=0.15),
    ]


def build_transform(cfg: ExperimentConfig, train: bool) -> T.Compose:
    """Builds the preprocessing pipeline."""
    use_aug = train and cfg.use_augmentation
    transforms: list = [
        ResizeAspectPreserving(
            cfg.image_size, cfg.resize_margin_factor if use_aug else 1.0
        ),
        T.CenterCrop(cfg.image_size),
    ]
    if use_aug:
        transforms.extend(_augmentation_transforms())
    transforms += [T.ToTensor(), normalize_to_minus_one_one]
    return T.Compose(transforms)


class ISLDataset(Dataset):
    """ISL image dataset."""

    def __init__(self, cfg: ExperimentConfig, train: bool = True):
        self.cfg = cfg
        self.mode = "L" if cfg.grayscale else "RGB"
        classes = list_classes(cfg.data_root)
        assert len(classes) == cfg.num_classes, (
            f"Expected {cfg.num_classes} classes, found {len(classes)}: {classes}"
        )
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class_map = {v: k for k, v in self.class_to_idx.items()}

        rng = random.Random(cfg.seed)
        self.paths: list[Path] = []
        self.labels: list[int] = []
        for cls in classes:
            cls_dir = Path(cfg.data_root) / cls
            imgs = sorted(
                p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS
            )
            if len(imgs) < cfg.samples_per_class:
                raise ValueError(
                    f"Class '{cls}' has only {len(imgs)} images, "
                    f"need {cfg.samples_per_class}"
                )
            selected = rng.sample(imgs, cfg.samples_per_class)  # NOSONAR
            self.paths.extend(selected)
            self.labels.extend([self.class_to_idx[cls]] * len(selected))

        self.transform = build_transform(cfg, train=train)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple:
        img = Image.open(self.paths[idx]).convert(self.mode)
        return self.transform(img), self.labels[idx]

    def idx_to_class(self, idx: int) -> str:
        return self.idx_to_class_map[idx]
