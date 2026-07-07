"""
Dataset handling for the local, already-downloaded ISL image folder.

Expects `data_root` to contain one subdirectory per class:
    data_root/
        1/*.jpg
        2/*.jpg
        ...
        A/*.jpg
        ...
        Z/*.jpg

Deterministically samples exactly `samples_per_class` images per class (seeded),
so every experiment trains on the identical set of images regardless of which
config is running -- this is what makes the ablations comparable.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from .config import ExperimentConfig

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg")


def list_classes(data_root: str) -> List[str]:
    root = Path(data_root)
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    if not classes:
        raise FileNotFoundError(f"No class subfolders found under {data_root}")
    return classes


def _augmentation_transforms() -> List:
    """
    Rotation +-10deg, translation 5%, scale 0.95-1.05, brightness/contrast jitter.
    Deliberately NO horizontal flip: ISL handshapes are not left/right symmetric
    and flipping can silently change or invalidate the class label.
    """
    return [
        T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        T.ColorJitter(brightness=0.15, contrast=0.15),
    ]


class ISLDataset(Dataset):
    def __init__(self, cfg: ExperimentConfig, train: bool = True):
        self.cfg = cfg
        self.mode = "L" if cfg.grayscale else "RGB"
        classes = list_classes(cfg.data_root)
        assert len(classes) == cfg.num_classes, (
            f"Expected {cfg.num_classes} classes, found {len(classes)}: {classes}"
        )
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        rng = random.Random(cfg.seed)
        self.paths: List[Path] = []
        self.labels: List[int] = []
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
            selected = rng.sample(imgs, cfg.samples_per_class)
            self.paths.extend(selected)
            self.labels.extend([self.class_to_idx[cls]] * len(selected))

        aug = _augmentation_transforms() if (train and cfg.use_augmentation) else []
        self.transform = T.Compose(
            [T.Resize((cfg.image_size, cfg.image_size))]
            + aug
            + [T.ToTensor(), T.Lambda(lambda x: x * 2 - 1)]  # normalize to [-1, 1]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple:
        img = Image.open(self.paths[idx]).convert(self.mode)
        return self.transform(img), self.labels[idx]

    def idx_to_class(self, idx: int) -> str:
        inv = {v: k for k, v in self.class_to_idx.items()}
        return inv[idx]
