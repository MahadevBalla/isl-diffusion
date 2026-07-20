"""
Sampling utilities for diffusion models.
"""

from __future__ import annotations

import torch
from PIL import Image

from .config import ExperimentConfig
from .model import build_inference_scheduler


def _to_pil(tensor: torch.Tensor) -> Image.Image:
    """Converts a normalized tensor to a PIL image."""
    img = (tensor / 2 + 0.5).clamp(0, 1)
    img = (img.cpu().permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    if img.shape[-1] == 1:
        img = img[:, :, 0]
        return Image.fromarray(img, mode="L")
    return Image.fromarray(img, mode="RGB")


def _rescale_cfg(
    noise_cfg: torch.Tensor,
    noise_cond: torch.Tensor,
    guidance_rescale: float,
) -> torch.Tensor:
    """
    Rescales the classifier-free guidance prediction to reduce
    over-saturation at high guidance scales.

    Implements the guidance rescaling method proposed by
    Lin et al. (2023). A value of 0.0 disables rescaling,
    while 1.0 applies the full correction.
    """
    dims = list(range(1, noise_cfg.ndim))
    std_cond = noise_cond.std(dim=dims, keepdim=True)
    std_cfg = noise_cfg.std(dim=dims, keepdim=True)
    noise_pred_rescaled = noise_cfg * (std_cond / std_cfg.clamp_min(1e-8))
    return guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg


@torch.no_grad()
def sample_images(
    model,
    train_scheduler_config,
    cfg: ExperimentConfig,
    num_images: int,
    device,
    class_labels: torch.Tensor | None = None,
    guidance_scale: float = 1.0,
    guidance_rescale: float = 0.7,
    sampler: str | None = None,
    num_inference_steps: int | None = None,
    seed: int = 42,
    return_intermediates: bool = False,
    intermediate_every: int = 10,
) -> list[Image.Image] | tuple[list[Image.Image], list]:
    """
    Generates images by iteratively denoising Gaussian noise using
    the configured inference scheduler.

    Supports conditional sampling, classifier-free guidance (CFG),
    optional CFG rescaling, deterministic seeding, and intermediate
    denoising snapshots for visualization.
    """
    sampler = sampler or cfg.sampler
    num_inference_steps = num_inference_steps or cfg.num_inference_steps
    scheduler = build_inference_scheduler(sampler, train_scheduler_config)
    scheduler.set_timesteps(num_inference_steps)

    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(
        (num_images, model.config.in_channels, cfg.image_size, cfg.image_size),
        generator=generator,
        device=device,
    )

    use_cfg = (
        cfg.conditional
        and class_labels is not None
        and guidance_scale != 1.0  # NOSONAR
    )
    null_labels = None
    if use_cfg:
        null_labels = torch.full_like(class_labels, cfg.null_class_idx)

    intermediates = []
    for i, t in enumerate(scheduler.timesteps):
        model_input = scheduler.scale_model_input(x, t)

        if cfg.conditional and class_labels is not None:
            if use_cfg:
                eps_cond = model(
                    model_input, t, class_labels=class_labels, return_dict=False
                )[0]
                eps_uncond = model(
                    model_input, t, class_labels=null_labels, return_dict=False
                )[0]
                noise_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
                if guidance_rescale > 0.0:
                    noise_pred = _rescale_cfg(noise_pred, eps_cond, guidance_rescale)
            else:
                noise_pred = model(
                    model_input, t, class_labels=class_labels, return_dict=False
                )[0]
        else:
            noise_pred = model(model_input, t, return_dict=False)[0]

        step_out = scheduler.step(noise_pred, t, x)
        x = step_out.prev_sample

        if return_intermediates and (
            i % intermediate_every == 0 or i == len(scheduler.timesteps) - 1
        ):
            x0_pred = getattr(step_out, "pred_original_sample", x)
            intermediates.append((int(t), [_to_pil(im) for im in x0_pred[:4]]))

    images = [_to_pil(im) for im in x]
    if return_intermediates:
        return images, intermediates
    return images


def make_grid_image(images: list[Image.Image], rows: int, cols: int) -> Image.Image:
    """Creates a grid from a list of images."""
    w, h = images[0].size
    mode = images[0].mode
    grid = Image.new(
        mode, (cols * w, rows * h), color=255 if mode == "L" else (255, 255, 255)
    )
    for i, image in enumerate(images[: rows * cols]):
        grid.paste(image, box=((i % cols) * w, (i // cols) * h))
    return grid
