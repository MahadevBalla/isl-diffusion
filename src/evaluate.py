"""
Evaluation utilities for trained diffusion models.

Computes image-generation quality metrics (FID, KID, PRDC, LPIPS), semantic
accuracy, classifier-free guidance sweeps, sampler comparisons, EMA/raw model
comparisons, and denoising visualizations.

Evaluation results are written incrementally to
`<output_dir>/results.json`.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import warnings
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from cleanfid.features import build_feature_extractor
from cleanfid.fid import get_folder_features
from PIL import Image
from prdc import compute_prdc
from torch.utils.data import DataLoader, random_split
from torchvision import models
from torchvision import transforms as TV
from tqdm import tqdm

from .config import ExperimentConfig
from .data import (
    ISLDataset,
    ResizeAspectPreserving,
    list_classes,
    normalize_to_minus_one_one,
    resize_aspect_preserving,
)
from .sampling import sample_images

warnings.filterwarnings(
    "ignore", message=".*pretrained.*deprecated.*", category=UserWarning
)

REAL_CACHE_ROOT = Path("./experiments/_real_fid_cache")
_TORCH_HUB_CACHE_DIR = REAL_CACHE_ROOT / "_torch_hub_cache"
os.environ.setdefault("TORCH_HOME", str(_TORCH_HUB_CACHE_DIR))

_INCEPTION_FILENAME = "inception-2015-12-05.pt"
_PERSISTENT_INCEPTION_PATH = REAL_CACHE_ROOT / _INCEPTION_FILENAME
_LPIPS_MODEL_CACHE: dict[str, lpips.LPIPS] = {}


def _update_results(cfg: ExperimentConfig, key: str, value) -> None:
    out = Path(cfg.output_dir) / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(out.read_text()) if out.exists() else {}
    data[key] = value
    out.write_text(json.dumps(data, indent=2, default=str))


def ensure_inception_weights() -> None:
    tmp_path = Path("/tmp") / _INCEPTION_FILENAME
    if tmp_path.exists():
        return
    if _PERSISTENT_INCEPTION_PATH.exists():
        shutil.copy2(_PERSISTENT_INCEPTION_PATH, tmp_path)
        return
    from cleanfid.downloads_helper import check_download_inception

    check_download_inception(fpath="/tmp")
    REAL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tmp_path, _PERSISTENT_INCEPTION_PATH)


def ensure_lpips_weights() -> None:
    """Downloads LPIPS model weights if not already cached."""
    _TORCH_HUB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    lpips.LPIPS(net="alex")


def ensure_resnet_weights() -> None:
    """Downloads pretrained ResNet18 weights if not already cached."""
    _TORCH_HUB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    models.resnet18(weights=models.ResNet18_Weights.DEFAULT)


def warm_all_caches() -> None:
    """Downloads all pretrained evaluation models into the local cache."""
    print("Downloading Inception weights (clean-fid)...")
    ensure_inception_weights()
    print("Downloading AlexNet weights (LPIPS)...")
    ensure_lpips_weights()
    print("Downloading ResNet18 weights (semantic classifier)...")
    ensure_resnet_weights()
    print("All caches warmed.")


# Dataset caching
def _real_cache_dir(cfg: ExperimentConfig) -> Path:
    tag = f"{cfg.image_size}_{'gray' if cfg.grayscale else 'rgb'}"
    cache_dir = REAL_CACHE_ROOT / tag
    if cache_dir.exists() and len(list(cache_dir.glob("*.png"))) >= cfg.n_fid_samples:
        return cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = ISLDataset(cfg, train=False)
    rng = random.Random(cfg.seed)
    idxs = rng.sample(range(len(dataset.paths)), cfg.n_fid_samples)  # NOSONAR
    for i, idx in enumerate(idxs):
        img = Image.open(dataset.paths[idx]).convert(
            "RGB" if not cfg.grayscale else "L"
        )
        img = resize_aspect_preserving(img, cfg.image_size, margin_factor=1.0)
        img = TV.CenterCrop(cfg.image_size)(img)
        img.convert("RGB").save(cache_dir / f"{i:05d}.png")
    return cache_dir


def _generate_fake_dir(
    model,
    train_scheduler_config,
    cfg: ExperimentConfig,
    out_dir: Path,
    n_images: int,
    device,
    guidance_scale: float,
    sampler: str | None = None,
    num_inference_steps: int | None = None,
    seed_offset: int = 0,
) -> Path:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    idx = 0
    bs = cfg.eval_batch_size
    with tqdm(total=n_images, desc=f"generating {out_dir.name}") as pbar:
        while idx < n_images:
            n = min(bs, n_images - idx)
            class_labels = None
            if cfg.conditional:
                class_labels = (
                    torch.arange(idx, idx + n, device=device) % cfg.num_classes
                )
            images = sample_images(
                model,
                train_scheduler_config,
                cfg,
                num_images=n,
                device=device,
                class_labels=class_labels,
                guidance_scale=guidance_scale,
                sampler=sampler,
                num_inference_steps=num_inference_steps,
                seed=cfg.seed + seed_offset + idx,
            )
            for img in images:
                img.convert("RGB").save(out_dir / f"{idx:05d}.png")
                idx += 1
            pbar.update(n)
    return out_dir


# Image quality metrics
def _frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    """Computes the Fréchet distance between two Gaussian distributions."""
    from scipy import linalg

    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    diff = mu1 - mu2

    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"Imaginary component {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real

    return float(
        diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    )


def _kernel_distance(
    feats1: np.ndarray,
    feats2: np.ndarray,
    num_subsets: int = 100,
    max_subset_size: int = 1000,
) -> float:
    """Computes the polynomial-kernel MMD estimator used for KID."""
    n = feats1.shape[1]
    m = min(min(feats1.shape[0], feats2.shape[0]), max_subset_size)
    t = 0.0
    rng = np.random.default_rng(42)
    for _ in range(num_subsets):
        x = feats2[rng.choice(feats2.shape[0], m, replace=False)]
        y = feats1[rng.choice(feats1.shape[0], m, replace=False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    return float(t / num_subsets / m)


def _extract_features(folder: Path, device, feat_model) -> np.ndarray:
    return get_folder_features(
        str(folder), model=feat_model, mode="clean", device=device, num_workers=0
    )


def fid_kid_precision_recall(
    real_dir: Path, fake_dir: Path, device, nearest_k: int = 5
) -> dict[str, float]:
    """
    Computes FID, KID, Precision, Recall, Density, and Coverage from a
    shared set of Inception features.
    """
    ensure_inception_weights()
    feat_model = build_feature_extractor(mode="clean", device=device)

    real_feats = _extract_features(real_dir, device, feat_model)
    fake_feats = _extract_features(fake_dir, device, feat_model)

    mu1, sigma1 = np.mean(real_feats, axis=0), np.cov(real_feats, rowvar=False)
    mu2, sigma2 = np.mean(fake_feats, axis=0), np.cov(fake_feats, rowvar=False)

    metrics = {
        "fid": _frechet_distance(mu1, sigma1, mu2, sigma2),
        "kid": _kernel_distance(real_feats, fake_feats),
    }
    prdc_metrics = compute_prdc(
        real_features=real_feats, fake_features=fake_feats, nearest_k=nearest_k
    )
    metrics.update({k: float(v) for k, v in prdc_metrics.items()})
    return metrics


def _get_lpips_model(device) -> lpips.LPIPS:
    key = str(device)
    if key not in _LPIPS_MODEL_CACHE:
        ensure_lpips_weights()
        _LPIPS_MODEL_CACHE[key] = lpips.LPIPS(net="alex").to(device)
    return _LPIPS_MODEL_CACHE[key]


def lpips_diversity(fake_dir: Path, device, n_pairs: int = 200) -> float:
    """Computes mean pairwise LPIPS distance between generated images."""
    loss_fn = _get_lpips_model(device)
    paths = sorted(Path(fake_dir).glob("*.png"))
    transform = TV.Compose([TV.ToTensor(), normalize_to_minus_one_one])
    rng = random.Random(0)
    max_pairs = len(paths) * (len(paths) - 1) // 2
    pairs = [
        rng.sample(range(len(paths)), 2)  # NOSONAR
        for _ in range(min(n_pairs, max_pairs))
    ]
    dists = []
    with torch.no_grad():
        for i, j in pairs:
            img_i = (
                transform(Image.open(paths[i]).convert("RGB")).unsqueeze(0).to(device)
            )
            img_j = (
                transform(Image.open(paths[j]).convert("RGB")).unsqueeze(0).to(device)
            )
            dists.append(loss_fn(img_i, img_j).item())
    return float(np.mean(dists))


def evaluate_checkpoint(
    model,
    train_scheduler_config,
    cfg: ExperimentConfig,
    device,
    guidance_scale: float = 1.0,
    tag: str = "main",
    include_lpips: bool = True,
) -> dict[str, float]:
    real_dir = _real_cache_dir(cfg)
    fake_dir = Path(cfg.output_dir) / f"fid_fake_{tag}"
    _generate_fake_dir(
        model,
        train_scheduler_config,
        cfg,
        fake_dir,
        n_images=cfg.n_fid_samples,
        device=device,
        guidance_scale=guidance_scale,
    )
    metrics = fid_kid_precision_recall(real_dir, fake_dir, device=device)
    if include_lpips:
        metrics["lpips_diversity"] = lpips_diversity(fake_dir, device=device)
    _update_results(cfg, f"fid_kid_{tag}", metrics)
    return metrics


def cfg_scale_sweep(
    model,
    train_scheduler_config,
    cfg: ExperimentConfig,
    device,
    scales: list[float] = (1.0, 2.0, 3.0, 5.0, 7.0, 10.0),
) -> dict[str, dict[str, float]]:
    """Evaluates multiple classifier-free guidance scales."""
    assert cfg.conditional, "CFG sweep only applies to conditional models"
    results = {}
    for scale in tqdm(scales, desc="CFG scale sweep"):
        metrics = evaluate_checkpoint(
            model,
            train_scheduler_config,
            cfg,
            device,
            guidance_scale=scale,
            tag=f"cfg_{scale}",
        )
        results[str(scale)] = metrics
    _update_results(cfg, "cfg_sweep", results)
    return results


def select_best_guidance_scale(cfg: ExperimentConfig) -> float:
    """Selects the guidance scale with the lowest FID."""
    results_path = Path(cfg.output_dir) / "results.json"
    data = json.loads(results_path.read_text())
    sweep = data["cfg_sweep"]
    best_scale = min(sweep, key=lambda k: sweep[k]["fid"])
    _update_results(cfg, "best_guidance_scale", float(best_scale))
    return float(best_scale)


def sampler_comparison(
    model,
    train_scheduler_config,
    cfg: ExperimentConfig,
    device,
    samplers: list[str] = ("ddim", "dpm", "unipc"),
    steps_list: list[int] = (50,),
) -> dict[str, dict[str, float]]:
    """Evaluates multiple inference samplers."""
    real_dir = _real_cache_dir(cfg)
    results = {}
    combos = [(s, n) for s in samplers for n in steps_list]
    for sampler, steps in tqdm(combos, desc="Sampler comparison"):
        fake_dir = Path(cfg.output_dir) / f"fid_fake_sampler_{sampler}_{steps}"
        _generate_fake_dir(
            model,
            train_scheduler_config,
            cfg,
            fake_dir,
            n_images=cfg.n_fid_samples,
            device=device,
            guidance_scale=cfg.default_guidance_scale if cfg.conditional else 1.0,
            sampler=sampler,
            num_inference_steps=steps,
        )
        metrics = fid_kid_precision_recall(real_dir, fake_dir, device=device)
        results[f"{sampler}_{steps}"] = metrics
    _update_results(cfg, "sampler_comparison", results)
    return results


def ema_on_off_comparison(
    raw_model,
    ema_model,
    train_scheduler_config,
    cfg: ExperimentConfig,
    device,
) -> dict[str, dict[str, float]]:
    """Compares raw and EMA model weights."""
    guidance = cfg.default_guidance_scale if cfg.conditional else 1.0
    raw_metrics = evaluate_checkpoint(
        raw_model,
        train_scheduler_config,
        cfg,
        device,
        guidance_scale=guidance,
        tag="ema_off",
    )
    ema_metrics = evaluate_checkpoint(
        ema_model,
        train_scheduler_config,
        cfg,
        device,
        guidance_scale=guidance,
        tag="ema_on",
    )
    result = {"ema_off": raw_metrics, "ema_on": ema_metrics}
    _update_results(cfg, "ema_comparison", result)
    return result


# Semantic accuracy
def train_semantic_classifier(cfg: ExperimentConfig, device, epochs: int = 8) -> Path:
    """
    Trains (or loads) the semantic classifier used for conditional
    generation evaluation.
    """
    ckpt_path = REAL_CACHE_ROOT / f"classifier_{cfg.image_size}.pt"
    if ckpt_path.exists():
        return ckpt_path

    ensure_resnet_weights()
    classes = list_classes(cfg.data_root)
    transform = TV.Compose(
        [
            ResizeAspectPreserving(cfg.image_size, margin_factor=1.0),
            TV.CenterCrop(cfg.image_size),
            TV.ToTensor(),
            TV.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ]
    )

    class _RGBClsDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.paths, self.labels = [], []
            for i, c in enumerate(classes):
                for p in sorted((Path(cfg.data_root) / c).iterdir()):
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                        self.paths.append(p)
                        self.labels.append(i)

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            return transform(img), self.labels[idx]

    full_ds = _RGBClsDataset()
    n_val = max(1, int(0.1 * len(full_ds)))
    train_ds, val_ds = random_split(
        full_ds,
        [len(full_ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, cfg.num_classes)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

    for epoch in range(epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(imgs), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.numel()
        print(
            f"[semantic-classifier] epoch {epoch + 1}/{epochs} val_acc={correct / total:.4f}"
        )

    REAL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path


def _semantic_accuracy_at_scale(
    model,
    train_scheduler_config,
    cfg: ExperimentConfig,
    device,
    guidance_scale: float,
    n_per_class: int,
) -> dict:
    classifier_path = train_semantic_classifier(cfg, device)
    classifier = models.resnet18(weights=None)
    classifier.fc = torch.nn.Linear(classifier.fc.in_features, cfg.num_classes)
    classifier.load_state_dict(
        torch.load(classifier_path, map_location=device, weights_only=True)
    )
    classifier = classifier.to(device).eval()

    norm = TV.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    correct, total, per_class_correct, per_class_total = 0, 0, {}, {}
    for cls_idx in tqdm(
        range(cfg.num_classes), desc=f"semantic-acc @ guidance={guidance_scale}"
    ):
        labels = torch.full((n_per_class,), cls_idx, device=device)
        images = sample_images(
            model,
            train_scheduler_config,
            cfg,
            num_images=n_per_class,
            device=device,
            class_labels=labels,
            guidance_scale=guidance_scale,
        )
        with torch.no_grad():
            batch = torch.stack(
                [norm(TV.functional.to_tensor(img)) for img in images]
            ).to(device)
            preds = classifier(batch).argmax(dim=1)
        n_correct = (preds == cls_idx).sum().item()
        correct += n_correct
        total += n_per_class
        per_class_correct[cls_idx] = n_correct
        per_class_total[cls_idx] = n_per_class

    return {
        "overall_accuracy": correct / total,
        "guidance_scale": guidance_scale,
        "per_class_accuracy": {
            str(k): per_class_correct[k] / per_class_total[k] for k in per_class_correct
        },
    }


def semantic_accuracy(
    model,
    train_scheduler_config,
    cfg: ExperimentConfig,
    device,
    n_per_class: int = 100,
    best_guidance_scale: float | None = None,
) -> dict[str, dict]:
    """
    Evaluates semantic accuracy with and without classifier-free guidance.
    """
    assert cfg.conditional, "Semantic accuracy requires a conditional model"
    scale = (
        best_guidance_scale
        if best_guidance_scale is not None
        else cfg.default_guidance_scale
    )

    no_cfg = _semantic_accuracy_at_scale(
        model,
        train_scheduler_config,
        cfg,
        device,
        guidance_scale=1.0,
        n_per_class=n_per_class,
    )
    with_cfg = _semantic_accuracy_at_scale(
        model,
        train_scheduler_config,
        cfg,
        device,
        guidance_scale=scale,
        n_per_class=n_per_class,
    )

    result = {"no_cfg": no_cfg, "best_cfg": with_cfg}
    _update_results(cfg, "semantic_accuracy", result)
    return result


def save_denoising_visualization(
    model,
    train_scheduler_config,
    cfg: ExperimentConfig,
    device,
    class_idx: int | None = None,
    tag: str = "denoise",
) -> Path:
    """Saves intermediate denoising predictions during sampling."""
    from .sampling import make_grid_image

    class_labels = None
    if cfg.conditional:
        cidx = class_idx if class_idx is not None else 0
        class_labels = torch.full((4,), cidx, device=device)

    _, intermediates = sample_images(
        model,
        train_scheduler_config,
        cfg,
        num_images=4,
        device=device,
        class_labels=class_labels,
        guidance_scale=cfg.default_guidance_scale if cfg.conditional else 1.0,
        seed=cfg.seed,
        return_intermediates=True,
        intermediate_every=5,
    )
    out_dir = Path(cfg.output_dir) / "denoising_viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    for t, imgs in intermediates:
        grid = make_grid_image(imgs, rows=1, cols=len(imgs))
        grid.save(out_dir / f"{tag}_t{t:04d}.png")
    return out_dir
