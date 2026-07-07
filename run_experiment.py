"""
Entry point.

Usage
-----
Train an experiment (resumes automatically if a checkpoint exists):
    uv run run_experiment.py train --exp baseline_uncond

Run the full evaluation suite against an experiment's latest checkpoint
(FID/KID, EMA on/off, denoising viz, and -- for conditional models -- the CFG
sweep and semantic accuracy):
    uv run run_experiment.py evaluate --exp main_conditional

Sampler comparison against a specific experiment's checkpoint (inference
only, no retraining, works for any experiment):
    uv run run_experiment.py sampler-compare --exp main_conditional

List experiment names:
    uv run run_experiment.py list
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers.training_utils import EMAModel
from safetensors.torch import load_file as load_safetensors

from src.config import EXPERIMENTS, ExperimentConfig
from src.engine import _find_latest_checkpoint, train
from src.model import build_model, load_unet_from_pretrained
from src import evaluate as ev

def _load_models_from_checkpoint(cfg: ExperimentConfig, device):
    """Loads both the raw and EMA weights from the latest checkpoint into two
    separate model instances, so both can be sampled from without retraining."""
    final_dir = Path(cfg.output_dir) / "final_model"
    final_raw_dir = Path(cfg.output_dir) / "final_model_raw"
    if final_dir.exists() and final_raw_dir.exists():
        ema_model = load_unet_from_pretrained(str(final_dir / "unet")).to(device).eval()
        raw_model = (
            load_unet_from_pretrained(str(final_raw_dir / "unet")).to(device).eval()
        )
        train_scheduler_config = json.loads(
            (final_dir / "experiment_config.json").read_text()
        )
        return raw_model, ema_model, train_scheduler_config, final_dir

    # fallback: reconstruct from the latest resumable checkpoint (e.g. mid-run checks)
    ckpt_dir = _find_latest_checkpoint(cfg)
    if ckpt_dir is None:
        raise FileNotFoundError(f"No checkpoint found for experiment '{cfg.name}'")

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

    train_scheduler_config = json.loads(
        (ckpt_dir.parent.parent / "config.json").read_text()
    )
    return raw_model, ema_model, train_scheduler_config, ckpt_dir


def cmd_train(args):
    cfg = EXPERIMENTS[args.exp]
    train(cfg)


def cmd_evaluate(args):
    cfg = EXPERIMENTS[args.exp]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from src.model import build_train_scheduler

    raw_model, ema_model, _, ckpt_dir = _load_models_from_checkpoint(cfg, device)
    train_scheduler_config = build_train_scheduler(cfg).config

    print(f"[{cfg.name}] evaluating checkpoint {ckpt_dir}")

    ev.ema_on_off_comparison(raw_model, ema_model, train_scheduler_config, cfg, device)
    ev.save_denoising_visualization(ema_model, train_scheduler_config, cfg, device)

    if cfg.conditional:
        ev.cfg_scale_sweep(ema_model, train_scheduler_config, cfg, device)
        ev.semantic_accuracy(ema_model, train_scheduler_config, cfg, device)

    print(f"[{cfg.name}] done. See {Path(cfg.output_dir) / 'results.json'}")


def cmd_sampler_compare(args):
    cfg = EXPERIMENTS[args.exp]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from src.model import build_train_scheduler

    _, ema_model, _, _ = _load_models_from_checkpoint(cfg, device)
    train_scheduler_config = build_train_scheduler(cfg).config
    ev.sampler_comparison(ema_model, train_scheduler_config, cfg, device)


def cmd_list(args):
    for name in EXPERIMENTS:
        print(name)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--exp", required=True, choices=EXPERIMENTS.keys())
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--exp", required=True, choices=EXPERIMENTS.keys())
    p_eval.set_defaults(func=cmd_evaluate)

    p_sampler = sub.add_parser("sampler-compare")
    p_sampler.add_argument("--exp", required=True, choices=EXPERIMENTS.keys())
    p_sampler.set_defaults(func=cmd_sampler_compare)

    p_list = sub.add_parser("list")
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
