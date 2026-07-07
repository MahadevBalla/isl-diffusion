"""
Manual sampling loop.

diffusers' built-in DDPMPipeline/DDIMPipeline don't support classifier-free
guidance out of the box for a plain UNet2DModel, so this implements the CFG
denoising loop directly:

    eps_uncond = model(x_t, t, class_labels=NULL)
    eps_cond   = model(x_t, t, class_labels=y)
    eps        = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

guidance_scale=1.0 recovers plain conditional sampling (no CFG push);
guidance_scale=0.0 would be pure unconditional. Unconditional models simply
skip class_labels entirely and guidance_scale is ignored.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from PIL import Image

from .config import ExperimentConfig
from .model import build_inference_scheduler


def _to_pil(tensor: torch.Tensor) -> Image.Image:
    # tensor in [-1, 1], shape (C, H, W)
    img = (tensor / 2 + 0.5).clamp(0, 1)
    img = (img.cpu().permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    if img.shape[-1] == 1:
        img = img[:, :, 0]
        return Image.fromarray(img, mode="L")
    return Image.fromarray(img, mode="RGB")


@torch.no_grad()
def sample_images(
    model,
    train_scheduler_config,
    cfg: ExperimentConfig,
    num_images: int,
    device,
    class_labels: Optional[torch.Tensor] = None,
    guidance_scale: float = 1.0,
    sampler: Optional[str] = None,
    num_inference_steps: Optional[int] = None,
    seed: int = 42,
    return_intermediates: bool = False,
    intermediate_every: int = 10,
) -> List[Image.Image] | tuple[List[Image.Image], list]:
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

    use_cfg = cfg.conditional and class_labels is not None and guidance_scale != 1.0  # NOSONAR
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


def make_grid_image(images: List[Image.Image], rows: int, cols: int) -> Image.Image:
    w, h = images[0].size
    mode = images[0].mode
    grid = Image.new(
        mode, (cols * w, rows * h), color=255 if mode == "L" else (255, 255, 255)
    )
    for i, image in enumerate(images[: rows * cols]):
        grid.paste(image, box=((i % cols) * w, (i // cols) * h))
    return grid
