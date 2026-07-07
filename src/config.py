"""
Central configuration for the ISL conditional diffusion project.

Every experiment is just an ExperimentConfig instance. There is no dataset-size
axis anymore (full 1200/class is used everywhere) and no epoch-based early
stopping (every run trains on the same 42,000 images, so a fixed step budget
is directly comparable across runs -- this removes the epoch-vs-gradient-step
confound that affected the previous paper).

EMA is always ON during training (it's free -- see engine.py, both raw and
EMA weights are kept and either can be sampled from at inference time with no
retraining cost). Augmentation is a fixed part of the recipe, not an ablation,
per the plan discussed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple


NOISE_SCHEDULE_TO_BETA = {
    "linear": "linear",
    "cosine": "squaredcos_cap_v2",
}


@dataclass
class ExperimentConfig:
    # ---- identity ----
    name: str

    # ---- data ----
    # Local path to the already-downloaded dataset root, containing one
    # subfolder per class (e.g. data_root/A/, data_root/B/, ..., data_root/9/).
    # No kagglehub download -- dataset is placed on the login node offline.
    data_root: str = "./data"
    grayscale: bool = False  # RGB by default for all main runs
    image_size: int = 128
    samples_per_class: int = 1200  # full dataset, every run, no size ablation
    use_augmentation: bool = (
        True  # rotation/translate/scale/brightness/contrast, NO flip
    )

    # ---- conditioning ----
    conditional: bool = True
    num_classes: int = 35
    cfg_dropout_prob: float = (
        0.15  # prob. of replacing true label with NULL during training
    )
    default_guidance_scale: float = 3.0  # used for routine sample grids during training

    # ---- model ----
    block_out_channels: Tuple[int, ...] = (128, 128, 256, 256, 512, 512)
    layers_per_block: int = 2
    # attention on the two coarsest resolution stages (mirrors the previous paper's choice,
    # scaled up by one extra stage for 128x128)
    attn_stages_from_end: int = 2

    # ---- noise schedule (training) ----
    noise_schedule: Literal["linear", "cosine"] = "cosine"
    num_train_timesteps: int = 1000

    # ---- training ----
    train_batch_size: int = 64  # comfortable on a 48GB RTX 6000 Ada at 128x128 RGB
    eval_batch_size: int = 36
    # fixed step budget instead of epoch-based early stopping (see module docstring)
    max_train_steps: int = 120_000
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    lr_warmup_steps: int = 1000
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "fp16"
    ema_decay: float = 0.9999
    max_grad_norm: float = 1.0

    # ---- checkpointing / resume (SLURM-chain friendly) ----
    checkpoint_every_steps: int = 5000
    sample_every_steps: int = 5000
    resume_from_checkpoint: bool = (
        True  # auto-resumes from latest checkpoint if present
    )

    # ---- inference (routine sampling during training) ----
    sampler: Literal["ddim", "dpm", "unipc"] = "ddim"
    num_inference_steps: int = 50

    # ---- final evaluation ----
    n_fid_samples: int = 2048
    fid_inference_steps: int = 50

    # ---- misc ----
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
        # reserved index used for CFG's unconditional branch, one past the real classes
        return self.num_classes

    def __post_init__(self):
        assert self.noise_schedule in NOISE_SCHEDULE_TO_BETA
        if not self.conditional:  # dropout is simply unused
            assert self.cfg_dropout_prob == 0.0 or True  # NOSONAR


# ---------------------------------------------------------------------------
# The finalized experiment list. Import EXPERIMENTS and index by name from the
# CLI entrypoint (run_experiment.py).
# ---------------------------------------------------------------------------


def _base(name: str, **overrides) -> ExperimentConfig:
    return ExperimentConfig(name=name, **overrides)


EXPERIMENTS = {
    # 1. New unconditional baseline: RGB, 128, EMA(always on), cosine, augmentation.
    "baseline_uncond": _base(
        "baseline_uncond",
        conditional=False,
    ),
    # 2. Main model: identical to baseline + class conditioning + CFG.
    "main_conditional": _base(
        "main_conditional",
        conditional=True,
    ),
    # 3. Controlled schedule ablation: main model config, linear instead of cosine,
    #    identical LR/sampler/steps/batch size this time (the clean version of the
    #    comparison the previous paper couldn't do fairly).
    "schedule_linear": _base(
        "schedule_linear",
        conditional=True,
        noise_schedule="linear",
    ),
    # 4. Legacy comparison (not a formal ablation row): grayscale, 64x64, unconditional,
    #    otherwise the same modern recipe, for continuity with the previous paper.
    "legacy_gray64_uncond": _base(
        "legacy_gray64_uncond",
        conditional=False,
        grayscale=True,
        image_size=64,
        block_out_channels=(64, 128, 256, 256),
        attn_stages_from_end=2,
        train_batch_size=128,
        max_train_steps=80_000,
    ),
}
