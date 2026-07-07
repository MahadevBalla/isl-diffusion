"""
Evaluation suite. Everything here writes its result to
<output_dir>/results.json incrementally (read-modify-write) so a long
evaluation sweep can be interrupted and resumed without losing completed
numbers, and so nothing needs to be regenerated just to look at a metric
again.

Metrics:
  - FID and KID via `clean-fid` (standardized resizing/antialiasing, unlike
    the raw pytorch-fid the earlier script used).
  - Semantic accuracy: a ResNet18 classifier trained once on the real
    training images, then used to check whether generated images for class X
    are actually classified as X. This is the metric that most directly
    answers "does conditioning actually work", which FID/KID alone cannot.
  - CFG guidance-scale sweep and sampler comparison are inference-only against
    an existing checkpoint -- no retraining involved.

A real-image reference cache is built once per (image_size, grayscale) pair
and reused across every experiment that shares those settings.
"""

from __future__ import annotations

import os
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms as TV
import lpips
from cleanfid.fid import get_folder_features
from prdc import compute_prdc

from .config import ExperimentConfig
from .data import ISLDataset, list_classes
from .sampling import sample_images

REAL_CACHE_ROOT = Path("./experiments/_real_fid_cache")
os.environ.setdefault("TORCH_HOME", str(REAL_CACHE_ROOT / "_torch_hub_cache"))

_INCEPTION_FILENAME = "inception-2015-12-05.pt"
_PERSISTENT_INCEPTION_PATH = REAL_CACHE_ROOT / _INCEPTION_FILENAME

_LPIPS_MODEL_CACHE: Dict[str, "lpips.LPIPS"] = {}
_TORCH_HUB_CACHE_DIR = REAL_CACHE_ROOT / "_torch_hub_cache"

def _update_results(cfg: ExperimentConfig, key: str, value) -> None:
    out = Path(cfg.output_dir) / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(out.read_text()) if out.exists() else {}
    data[key] = value
    out.write_text(json.dumps(data, indent=2, default=str))


def _real_cache_dir(cfg: ExperimentConfig) -> Path:
    tag = f"{cfg.image_size}_{'gray' if cfg.grayscale else 'rgb'}"
    cache_dir = REAL_CACHE_ROOT / tag
    if cache_dir.exists() and len(list(cache_dir.glob("*.png"))) >= cfg.n_fid_samples:
        return cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = ISLDataset(cfg, train=False)
    # deterministic seeding required for reproducible eval splits, not security-sensitive
    rng = random.Random(cfg.seed)
    idxs = rng.sample(range(len(dataset.paths)), cfg.n_fid_samples)  # NOSONAR
    for i, idx in enumerate(idxs):
        img = Image.open(dataset.paths[idx]).convert(
            "RGB" if not cfg.grayscale else "L"
        )
        img = img.resize((cfg.image_size, cfg.image_size), Image.LANCZOS)
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
    sampler: Optional[str] = None,
    num_inference_steps: Optional[int] = None,
    seed_offset: int = 0,
) -> Path:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    idx = 0
    bs = cfg.eval_batch_size
    while idx < n_images:
        n = min(bs, n_images - idx)
        class_labels = None
        if cfg.conditional:
            class_labels = torch.arange(idx, idx + n, device=device) % cfg.num_classes
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
    return out_dir


def ensure_inception_weights() -> None:
    """
    clean-fid caches its Inception weights at /tmp/inception-2015-12-05.pt,
    which is typically node-local and ephemeral on HPC clusters, and compute
    nodes frequently have no internet access to (re-)download it. This stages
    a persistent copy (under experiments/_real_fid_cache, which should
    live on shared/project storage) into /tmp before clean-fid runs.

    One-time setup on a node WITH internet access (e.g. the login node):
        python -c "from isl_diffusion.evaluate import ensure_inception_weights; ensure_inception_weights()"
    After that, every subsequent call (including from compute nodes with no
    internet) will find the persistent copy and just copy it into /tmp.
    """
    tmp_path = Path("/tmp") / _INCEPTION_FILENAME
    if tmp_path.exists():
        return
    if _PERSISTENT_INCEPTION_PATH.exists():
        shutil.copy2(_PERSISTENT_INCEPTION_PATH, tmp_path)
        return
    # not cached anywhere yet -- let clean-fid download it (needs internet),
    # then snapshot it to persistent storage for next time / other nodes.
    from cleanfid.downloads_helper import check_download_inception

    check_download_inception(fpath="/tmp")
    REAL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tmp_path, _PERSISTENT_INCEPTION_PATH)


def ensure_lpips_weights() -> None:
    """
    lpips.LPIPS(net='alex') pulls torchvision's AlexNet ImageNet weights via
    torch.hub, plus the LPIPS linear-layer weights bundled in the `lpips`
    package itself. torch.hub's default cache (~/.cache/torch/hub) is
    typically node-local/ephemeral on HPC clusters, same problem as the
    Inception weights. Point TORCH_HOME at persistent storage and warm the
    cache once from a node with internet access before any compute-node job
    tries to instantiate LPIPS.

    One-time setup from the login node:
        python -c "from src.evaluate import ensure_lpips_weights; ensure_lpips_weights()"
    """
    checkpoint = (
        _TORCH_HUB_CACHE_DIR / "hub" / "checkpoints" / "alexnet-owt-7be5be79.pth"
    )
    if checkpoint.exists():
        return
    os.environ.setdefault("TORCH_HOME", str(_TORCH_HUB_CACHE_DIR))
    _TORCH_HUB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # triggers the torch.hub download of AlexNet weights + lpips' own linear
    # weights into TORCH_HOME, where they'll persist across jobs/nodes.
    lpips.LPIPS(net="alex")


def compute_fid_kid(real_dir: Path, fake_dir: Path) -> Dict[str, float]:
    from cleanfid import fid as cleanfid

    ensure_inception_weights()
    fid_value = cleanfid.compute_fid(str(real_dir), str(fake_dir), mode="clean")
    kid_value = cleanfid.compute_kid(str(real_dir), str(fake_dir), mode="clean")
    return {"fid": float(fid_value), "kid": float(kid_value)}


def evaluate_checkpoint(
    model,
    train_scheduler_config,
    cfg: ExperimentConfig,
    device,
    guidance_scale: float = 1.0,
    tag: str = "main",
) -> Dict[str, float]:
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
    metrics = compute_fid_kid(real_dir, fake_dir)
    metrics.update(precision_recall(real_dir, fake_dir))
    metrics["lpips_diversity"] = lpips_diversity(fake_dir, device=device)
    _update_results(cfg, f"fid_kid_{tag}", metrics)
    return metrics


def cfg_scale_sweep(
    model,
    train_scheduler_config,
    cfg: ExperimentConfig,
    device,
    scales: List[float] = (1.0, 2.0, 3.0, 5.0, 7.0, 10.0),
) -> Dict[str, Dict[str, float]]:
    """Inference-only: reuses the same checkpoint, only guidance_scale changes."""
    assert cfg.conditional, "CFG sweep only applies to conditional models"
    results = {}
    for scale in scales:
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


def sampler_comparison(
    model,
    train_scheduler_config,
    cfg: ExperimentConfig,
    device,
    samplers: List[str] = ("ddim", "dpm", "unipc"),
    steps_list: List[int] = (50,),
) -> Dict[str, Dict[str, float]]:
    """Inference-only: same checkpoint, only the inference scheduler changes."""
    real_dir = _real_cache_dir(cfg)
    results = {}
    for sampler in samplers:
        for steps in steps_list:
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
            metrics = compute_fid_kid(real_dir, fake_dir)
            metrics.update(precision_recall(real_dir, fake_dir))
            metrics["lpips_diversity"] = lpips_diversity(fake_dir, device=device)
            results[f"{sampler}_{steps}"] = metrics
    _update_results(cfg, "sampler_comparison", results)
    return results


def ema_on_off_comparison(
    raw_model,
    ema_model,
    train_scheduler_config,
    cfg: ExperimentConfig,
    device,
) -> Dict[str, Dict[str, float]]:
    """
    Free comparison: both raw and EMA weights come from the SAME training run.
    No retraining involved.
    """
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


# ---------------------------------------------------------------------------
# Semantic accuracy classifier
# ---------------------------------------------------------------------------


def train_semantic_classifier(cfg: ExperimentConfig, device, epochs: int = 8) -> Path:
    """
    Trains a ResNet18 on the real training images (RGB, cfg.image_size) to
    classify the 35 ISL classes. Trained once, reused for every model's
    semantic-accuracy evaluation. Saved under a shared cache path keyed by
    image_size so it doesn't get retrained per experiment.
    """
    ckpt_path = REAL_CACHE_ROOT / f"classifier_{cfg.image_size}.pt"
    if ckpt_path.exists():
        return ckpt_path

    classes = list_classes(cfg.data_root)
    tfm = TV.Compose(
        [
            TV.Resize((cfg.image_size, cfg.image_size)),
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
            return tfm(img), self.labels[idx]

    full_ds = _RGBClsDataset()
    n_val = max(1, int(0.1 * len(full_ds)))
    train_ds, val_ds = random_split(
        full_ds,
        [len(full_ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=8)

    model = models.resnet18(weights=None, num_classes=cfg.num_classes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-2,
    )

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


def save_denoising_visualization(
    model,
    train_scheduler_config,
    cfg: ExperimentConfig,
    device,
    class_idx: Optional[int] = None,
    tag: str = "denoise",
) -> Path:
    """
    Saves a grid showing the model's predicted-clean-image (x0) estimate at
    several timesteps through the reverse process, for a fixed seed. Cheap,
    and useful for the qualitative section of the paper -- especially for the
    conditional model, where you can show the class label steering the shape
    from an early timestep.
    """
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


def semantic_accuracy(
    model,
    train_scheduler_config,
    cfg: ExperimentConfig,
    device,
    n_per_class: int = 50,
    guidance_scale: Optional[float] = None,
) -> Dict[str, float]:
    assert cfg.conditional, "Semantic accuracy requires a conditional model"
    classifier_path = train_semantic_classifier(cfg, device)
    classifier = models.resnet18(weights=None, num_classes=cfg.num_classes).to(device)
    classifier.load_state_dict(
        torch.load(classifier_path, map_location=device, weights_only=True)
    )
    classifier.eval()

    guidance_scale = (
        guidance_scale if guidance_scale is not None else cfg.default_guidance_scale
    )
    norm = TV.Normalize(mean=[0.5] * 3, std=[0.5] * 3)

    correct, total, per_class_correct, per_class_total = 0, 0, {}, {}
    for cls_idx in range(cfg.num_classes):
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

    result = {
        "overall_accuracy": correct / total,
        "guidance_scale": guidance_scale,
        "per_class_accuracy": {
            str(k): per_class_correct[k] / per_class_total[k] for k in per_class_correct
        },
    }
    _update_results(cfg, f"semantic_accuracy_cfg{guidance_scale}", result)
    return result


def precision_recall(
    real_dir: Path, fake_dir: Path, nearest_k: int = 5
) -> Dict[str, float]:
    """
    Precision/Recall for generative models (Kynkäänniemi et al.), computed on
    the same clean-fid Inception features already used for FID/KID -- no
    separate feature extractor, so this is nearly free given compute_fid_kid
    already ran.
    """
    ensure_inception_weights()
    real_feats = get_folder_features(str(real_dir), mode="clean")
    fake_feats = get_folder_features(str(fake_dir), mode="clean")
    metrics = compute_prdc(
        real_features=real_feats, fake_features=fake_feats, nearest_k=nearest_k
    )
    return {
        k: float(v) for k, v in metrics.items()
    }  # precision, recall, density, coverage


def normalize_to_minus_one_one(x: torch.Tensor) -> torch.Tensor:
    return x * 2 - 1


def lpips_diversity(fake_dir: Path, n_pairs: int = 200, device: str = "cuda") -> float:
    """
    Mean pairwise LPIPS distance among generated images for one class/run.
    Directly measures whether high guidance scales are collapsing diversity --
    complements the CFG sweep, since FID/KID alone won't distinguish "sharper"
    from "less diverse."
    """
    loss_fn = _get_lpips_model(device)
    paths = sorted(Path(fake_dir).glob("*.png"))
    tfm = TV.Compose([TV.ToTensor(), TV.Lambda(normalize_to_minus_one_one)])
    # deterministic pairing for reproducible diversity metric, not security-sensitive
    rng = random.Random(0)
    pairs = [
        rng.sample(range(len(paths)), 2)  # NOSONAR
        for _ in range(min(n_pairs, len(paths) * (len(paths) - 1) // 2))
    ]

    dists = []
    with torch.no_grad():
        for i, j in pairs:
            with Image.open(paths[i]) as img:
                img_i = tfm(img.convert("RGB")).unsqueeze(0).to(device)
            with Image.open(paths[j]) as img:
                img_j = tfm(img.convert("RGB")).unsqueeze(0).to(device)
            dists.append(loss_fn(img_i, img_j).item())
    return float(np.mean(dists))

def _get_lpips_model(device: str) -> lpips.LPIPS:
    if device not in _LPIPS_MODEL_CACHE:
        ensure_lpips_weights()
        _LPIPS_MODEL_CACHE[device] = lpips.LPIPS(net="alex").to(device)
    return _LPIPS_MODEL_CACHE[device]
