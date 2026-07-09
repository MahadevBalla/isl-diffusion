"""
Command-line interface for training, evaluation, checkpoint export, and
analysis utilities.

Examples:
    uv run run_experiment.py train --exp main_conditional
    uv run run_experiment.py evaluate --exp main_conditional
    uv run run_experiment.py sampler-compare --exp main_conditional
    uv run run_experiment.py plot-curves --exps baseline_uncond main_conditional
    uv run run_experiment.py warm-caches
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import DDPMScheduler
from diffusers.training_utils import EMAModel
from safetensors.torch import load_file as load_safetensors

from plot_training_curves import plot_loss_curves
from src import evaluate as ev
from src.config import EXPERIMENTS, ExperimentConfig
from src.engine import _find_latest_checkpoint, export_final_model, train
from src.model import build_model, build_train_scheduler, load_unet_from_pretrained


def _load_models_from_checkpoint(
    cfg: ExperimentConfig, device, step: int | None = None
):
    """
    Loads raw and EMA model weights.

    If available, loads the exported diffusers pipeline from
    `final_model/` and `final_model_raw/`. Otherwise reconstructs the
    models from the requested checkpoint (or the latest checkpoint if no
    step is specified).
    """
    final_dir = Path(cfg.output_dir) / "final_model"
    final_raw_dir = Path(cfg.output_dir) / "final_model_raw"
    if (
        step is None
        and (final_dir / "unet").exists()
        and (final_raw_dir / "unet").exists()
    ):
        ema_model = load_unet_from_pretrained(str(final_dir / "unet")).to(device).eval()
        raw_model = (
            load_unet_from_pretrained(str(final_raw_dir / "unet")).to(device).eval()
        )
        train_scheduler_config = DDPMScheduler.from_pretrained(
            str(final_dir / "scheduler")
        ).config
        return raw_model, ema_model, train_scheduler_config, final_dir

    if step is not None:
        ckpt_dir = Path(cfg.checkpoint_dir) / f"step_{step:07d}"
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"No checkpoint at step {step} for '{cfg.name}'")
    else:
        ckpt_dir = _find_latest_checkpoint(cfg)
        if ckpt_dir is None:
            raise FileNotFoundError(
                f"No final_model or checkpoint found for experiment '{cfg.name}'"
            )

    raw_model = build_model(cfg).to(device)
    safetensors_path = ckpt_dir / "model.safetensors"
    if safetensors_path.exists():
        raw_state = load_safetensors(str(safetensors_path), device=str(device))
    else:
        raw_state = torch.load(
            ckpt_dir / "pytorch_model.bin", map_location=device, weights_only=True
        )
    raw_model.load_state_dict(raw_state)
    raw_model.eval()

    ema_model = build_model(cfg).to(device)
    ema_model.load_state_dict(raw_state)
    ema_wrapper = EMAModel(
        ema_model.parameters(), model_cls=type(ema_model), model_config=ema_model.config
    )
    ema_wrapper.load_state_dict(
        torch.load(ckpt_dir / "ema.pt", map_location="cpu", weights_only=True)
    )
    ema_wrapper.copy_to(ema_model.parameters())
    ema_model.eval()

    # Reconstruct the training scheduler configuration.
    train_scheduler_config = build_train_scheduler(cfg).config
    return raw_model, ema_model, train_scheduler_config, ckpt_dir


def cmd_train(args):
    cfg = EXPERIMENTS[args.exp]
    train(cfg)


def cmd_export(args):
    """
    Exports diffusers-compatible EMA and raw checkpoints from an existing
    training checkpoint.
    """
    cfg = EXPERIMENTS[args.exp]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw_model, ema_model, _, ckpt_dir = _load_models_from_checkpoint(
        cfg, device, step=args.step
    )
    train_scheduler = build_train_scheduler(cfg)
    # Construct an EMA wrapper from the loaded EMA weights.
    ema_wrapper = EMAModel(
        raw_model.parameters(), model_cls=type(raw_model), model_config=raw_model.config
    )
    ema_wrapper.shadow_params = [p.clone().detach() for p in ema_model.parameters()]
    export_final_model(cfg, raw_model, ema_wrapper, train_scheduler)
    print(f"[{cfg.name}] exported from {ckpt_dir}")


def cmd_evaluate(args):
    cfg = EXPERIMENTS[args.exp]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    raw_model, ema_model, train_scheduler_config, source = _load_models_from_checkpoint(
        cfg, device, step=args.step
    )
    print(f"[{cfg.name}] evaluating {source}")

    ev.ema_on_off_comparison(raw_model, ema_model, train_scheduler_config, cfg, device)
    ev.save_denoising_visualization(ema_model, train_scheduler_config, cfg, device)

    if cfg.conditional:
        ev.cfg_scale_sweep(ema_model, train_scheduler_config, cfg, device)
        best_scale = ev.select_best_guidance_scale(cfg)
        print(f"[{cfg.name}] best guidance scale (min FID): {best_scale}")
        ev.semantic_accuracy(
            ema_model,
            train_scheduler_config,
            cfg,
            device,
            best_guidance_scale=best_scale,
        )

    print(f"[{cfg.name}] results written to {Path(cfg.output_dir) / 'results.json'}")


def cmd_sampler_compare(args):
    cfg = EXPERIMENTS[args.exp]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, ema_model, train_scheduler_config, _ = _load_models_from_checkpoint(cfg, device)
    ev.sampler_comparison(ema_model, train_scheduler_config, cfg, device)


def cmd_plot_curves(args):
    plot_loss_curves(args.exps, out_path=args.out)


def cmd_warm_caches(args):
    ev.warm_all_caches()


def cmd_list(args):
    for name in EXPERIMENTS:
        print(name)


def main():
    parser = argparse.ArgumentParser(
        description="Training and evaluation utilities for the ISL diffusion models."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--exp", required=True, choices=EXPERIMENTS.keys())
    p_train.set_defaults(func=cmd_train)

    p_export = sub.add_parser("export")
    p_export.add_argument("--exp", required=True, choices=EXPERIMENTS.keys())
    p_export.add_argument("--step", type=int, default=None)
    p_export.set_defaults(func=cmd_export)

    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--exp", required=True, choices=EXPERIMENTS.keys())
    p_eval.add_argument(
        "--step",
        type=int,
        default=None,
        help="Evaluate a specific checkpoint instead of the latest model.",
    )
    p_eval.set_defaults(func=cmd_evaluate)

    p_sampler = sub.add_parser("sampler-compare")
    p_sampler.add_argument("--exp", required=True, choices=EXPERIMENTS.keys())
    p_sampler.set_defaults(func=cmd_sampler_compare)

    p_plot = sub.add_parser("plot-curves")
    p_plot.add_argument("--exps", nargs="+", required=True, choices=EXPERIMENTS.keys())
    p_plot.add_argument("--out", default="./experiments/loss_curves.png")
    p_plot.set_defaults(func=cmd_plot_curves)

    p_warm = sub.add_parser("warm-caches")
    p_warm.set_defaults(func=cmd_warm_caches)

    p_list = sub.add_parser("list")
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
