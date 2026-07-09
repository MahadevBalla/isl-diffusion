"""
Configuration for ISL conditional diffusion experiments.

Each experiment is defined by an `ExperimentConfig` instance. All experiments
use a fixed training-step budget to enable direct comparison across runs.
EMA and data augmentation are enabled for all experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

NOISE_SCHEDULE_TO_BETA = {
    "linear": "linear",
    "cosine": "squaredcos_cap_v2",
}


@dataclass
class ExperimentConfig:
    # identity
    name: str

    # data
    # Local path to the already-downloaded dataset root, containing one
    # subfolder per class (e.g. data_root/A/, data_root/B/, ..., data_root/9/).
    data_root: str = "./data"
    grayscale: bool = False  # RGB by default for all main runs
    image_size: int = 128
    samples_per_class: int = 1200
    use_augmentation: bool = True

    # Resize the shorter side before center cropping. Values >1 preserve a
    # margin around the subject before cropping.
    resize_margin_factor: float = 1.0

    # conditioning
    conditional: bool = True
    num_classes: int = 35
    cfg_dropout_prob: float = (
        0.15  # prob. of replacing true label with NULL during training
    )
    default_guidance_scale: float = 3.0  # used for routine sample grids during training

    # model
    block_out_channels: tuple[int, ...] = (128, 128, 256, 256, 512, 512)
    layers_per_block: int = 2
    attn_stages_from_end: int = 2  # attention on the two coarsest resolution stages

    # noise schedule (training)
    noise_schedule: Literal["linear", "cosine"] = "cosine"
    num_train_timesteps: int = 1000

    # training
    train_batch_size: int = 64
    eval_batch_size: int = 64
    # fixed step budget instead of epoch-based early stopping (see module docstring)
    max_train_steps: int = 120_000
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    lr_warmup_steps: int = 1000
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "fp16"
    ema_decay: float = 0.9999
    max_grad_norm: float = 1.0

    # checkpointing / resume
    # Checkpoint and sample generation intervals.
    checkpoint_every_steps: int = 5000
    sample_every_steps: int = 10000
    resume_from_checkpoint: bool = True

    # inference (routine sampling during training)
    sampler: Literal["ddim", "dpm", "unipc"] = "ddim"
    num_inference_steps: int = 50

    # final evaluation
    n_fid_samples: int = 2048
    fid_inference_steps: int = 50

    # misc
    seed: int = 42
    num_workers: int = 8
    output_root: str = "./experiments"

    @property
    def in_channels(self) -> int:
        return 1 if self.grayscale else 3

    @property
    def output_dir(self) -> str:
        return str(Path(self.output_root) / self.name)

    @property
    def checkpoint_dir(self) -> str:
        return str(Path(self.output_dir) / "checkpoints")

    @property
    def null_class_idx(self) -> int:
        return self.num_classes

    @property
    def steps_per_epoch(self) -> int:
        n_images = self.samples_per_class * self.num_classes
        return max(1, n_images // self.train_batch_size)

    @property
    def approx_epochs(self) -> float:
        """
        Approximate number of epochs implied by the configured training-step budget.
        Provided for reporting only; training is step-based.
        """
        return self.max_train_steps / self.steps_per_epoch

    def __post_init__(self):
        assert self.noise_schedule in NOISE_SCHEDULE_TO_BETA


def _base(name: str, **overrides) -> ExperimentConfig:
    return ExperimentConfig(name=name, **overrides)


EXPERIMENTS = {
    # Unconditional baseline.
    "baseline_uncond": _base(
        "baseline_uncond",
        conditional=False,
    ),
    # Conditional model.
    "main_conditional": _base(
        "main_conditional",
        conditional=True,
    ),
    # Linear noise schedule ablation.
    "schedule_linear": _base(
        "schedule_linear",
        conditional=True,
        noise_schedule="linear",
    ),
    # 64×64 resolution ablation.
    "main_conditional_64": _base(
        "main_conditional_64",
        conditional=True,
        image_size=64,
        block_out_channels=(128, 128, 256, 256),
        attn_stages_from_end=2,
        train_batch_size=128,
        max_train_steps=90_000,
    ),
}
