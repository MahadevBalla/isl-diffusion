"""
Model and scheduler construction utilities.
"""

from __future__ import annotations

from typing import Any

from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    UNet2DModel,
    UniPCMultistepScheduler,
)

from .config import NOISE_SCHEDULE_TO_BETA, ExperimentConfig


def build_model(cfg: ExperimentConfig) -> UNet2DModel:
    """Builds the diffusion UNet."""
    n_blocks = len(cfg.block_out_channels)
    attn_from = n_blocks - cfg.attn_stages_from_end
    down_block_types = tuple(
        "AttnDownBlock2D" if i >= attn_from else "DownBlock2D" for i in range(n_blocks)
    )
    up_block_types = tuple(
        "AttnUpBlock2D" if i < cfg.attn_stages_from_end else "UpBlock2D"
        for i in range(n_blocks)
    )

    kwargs: dict[str, Any] = {
        "sample_size": cfg.image_size,
        "in_channels": cfg.in_channels,
        "out_channels": cfg.in_channels,
        "layers_per_block": cfg.layers_per_block,
        "block_out_channels": cfg.block_out_channels,
        "down_block_types": down_block_types,
        "up_block_types": up_block_types,
    }
    if cfg.conditional:
        # +1 for the reserved NULL class used by classifier-free guidance
        kwargs["num_class_embeds"] = cfg.num_classes + 1

    return UNet2DModel(**kwargs)


def build_train_scheduler(cfg: ExperimentConfig) -> DDPMScheduler:
    """Builds the training noise scheduler."""
    beta_schedule = NOISE_SCHEDULE_TO_BETA[cfg.noise_schedule]
    kwargs: dict[str, Any] = {
        "num_train_timesteps": cfg.num_train_timesteps,
        "beta_schedule": beta_schedule,
    }
    if beta_schedule == "linear":
        kwargs.update(beta_start=1e-4, beta_end=0.02)
    return DDPMScheduler(**kwargs)


def build_inference_scheduler(sampler: str, train_scheduler_config) -> Any:
    """Builds an inference scheduler from the training configuration."""
    if sampler == "ddim":
        return DDIMScheduler.from_config(train_scheduler_config)
    elif sampler == "dpm":
        return DPMSolverMultistepScheduler.from_config(
            train_scheduler_config, algorithm_type="dpmsolver++"
        )
    elif sampler == "unipc":
        return UniPCMultistepScheduler.from_config(train_scheduler_config)
    raise ValueError(f"Unknown sampler: {sampler}")


def load_unet_from_pretrained(path: str) -> UNet2DModel:
    """Loads a pretrained UNet from disk."""
    return UNet2DModel.from_pretrained(path)
