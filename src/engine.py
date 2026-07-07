"""
Training loop.

Key design choices, each addressing a specific issue from the previous paper:

- Step-based training budget (max_train_steps), not epoch-based early stopping.
  Since every experiment now uses the full 1200/class dataset, one epoch is the
  same number of steps everywhere, so a fixed step budget is directly
  comparable across runs with no epoch-vs-gradient-step confound.

- EMA is tracked unconditionally during training. Both raw and EMA weights are
  saved at every checkpoint, so an "EMA on vs off" comparison at evaluation
  time costs zero additional GPU-hours -- it's two inference passes off the
  same checkpoint, not two training runs.

- CFG label dropout: for conditional runs, each training label is replaced
  with the reserved NULL class (cfg.null_class_idx) with probability
  cfg.cfg_dropout_prob, so the model also learns the unconditional score
  needed at inference time for classifier-free guidance.

- Checkpointing is resumable: every checkpoint_every_steps, model, EMA,
  optimizer, lr_scheduler and step count are all saved together under
  <output_dir>/checkpoints/step_<N>/. If resume_from_checkpoint is True and a
  checkpoint exists, training picks up exactly where it left off -- this is
  what makes SLURM `--dependency=afterok` chaining safe (a job that gets
  killed by a walltime limit can be resubmitted and will resume, not restart).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import ISLDataset
from .model import build_model, build_train_scheduler
from .sampling import make_grid_image, sample_images


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _find_latest_checkpoint(cfg: ExperimentConfig) -> Optional[Path]:
    ckpt_root = Path(cfg.checkpoint_dir)
    if not ckpt_root.exists():
        return None
    steps = sorted(
        (p for p in ckpt_root.iterdir() if p.is_dir() and p.name.startswith("step_")),
        key=lambda p: int(p.name.split("_")[1]),
    )
    return steps[-1] if steps else None


def train(cfg: ExperimentConfig) -> dict:
    set_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(cfg.__dict__, indent=2, default=str)
    )

    dataset = ISLDataset(cfg, train=True)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    steps_per_epoch = len(dataloader)

    model = build_model(cfg)
    train_scheduler = build_train_scheduler(cfg)

    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=str(out_dir / "logs"),
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(cfg.name)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps,
        num_training_steps=cfg.max_train_steps,
    )
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    ema = EMAModel(
        model.parameters(),
        decay=cfg.ema_decay,
        model_cls=UNet2DModel,
        model_config=accelerator.unwrap_model(model).config,
    )
    ema.to(accelerator.device)

    global_step = 0
    loss_history: list[dict] = []

    if cfg.resume_from_checkpoint:
        latest = _find_latest_checkpoint(cfg)
        if latest is not None:
            accelerator.print(f"[{cfg.name}] resuming from {latest}")
            accelerator.load_state(str(latest))
            state = json.loads((latest / "train_state.json").read_text())
            global_step = state["global_step"]
            ema.load_state_dict(
                torch.load(latest / "ema.pt", map_location="cpu", weights_only=True)
            )
            ema.to(accelerator.device)
            loss_history = state.get("loss_history", [])

    accelerator.print(
        f"[{cfg.name}] starting at step {global_step}/{cfg.max_train_steps} "
        f"({steps_per_epoch} steps/epoch, conditional={cfg.conditional})"
    )

    t0 = time.time()
    model.train()
    data_iter = iter(dataloader)
    while global_step < cfg.max_train_steps:
        try:
            batch_imgs, batch_labels = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch_imgs, batch_labels = next(data_iter)

        bs = batch_imgs.shape[0]
        noise = torch.randn_like(batch_imgs)
        timesteps = torch.randint(
            0,
            train_scheduler.config.num_train_timesteps,
            (bs,),
            device=batch_imgs.device,
        ).long()
        noisy_imgs = train_scheduler.add_noise(batch_imgs, noise, timesteps)

        class_labels = None
        if cfg.conditional:
            labels = batch_labels.clone().to(batch_imgs.device)
            if cfg.cfg_dropout_prob > 0:
                drop_mask = torch.rand(bs, device=labels.device) < cfg.cfg_dropout_prob
                labels = torch.where(
                    drop_mask, torch.full_like(labels, cfg.null_class_idx), labels
                )
            class_labels = labels

        with accelerator.accumulate(model):
            model_kwargs = {}
            if class_labels is not None:
                model_kwargs["class_labels"] = class_labels
            noise_pred = model(
                noisy_imgs, timesteps, return_dict=False, **model_kwargs
            )[0]
            loss = F.mse_loss(noise_pred, noise)
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        ema.step(model.parameters())
        global_step += 1

        loss_val = loss.detach().item()
        if global_step % 50 == 0:
            loss_history.append(
                {
                    "step": global_step,
                    "loss": loss_val,
                    "lr": lr_scheduler.get_last_lr()[0],
                }
            )
            accelerator.log(
                {"loss": loss_val, "lr": lr_scheduler.get_last_lr()[0]},
                step=global_step,
            )
            elapsed = time.time() - t0
            accelerator.print(
                f"[{cfg.name}] step {global_step}/{cfg.max_train_steps} "
                f"loss={loss_val:.5f} lr={lr_scheduler.get_last_lr()[0]:.2e} "
                f"({global_step / max(elapsed, 1e-6):.2f} steps/s)"
            )

        if accelerator.is_main_process and (
            global_step % cfg.sample_every_steps == 0
            or global_step == cfg.max_train_steps
        ):
            _save_sample_grid(cfg, accelerator, model, ema, train_scheduler, global_step)

        if accelerator.is_main_process and (
            global_step % cfg.checkpoint_every_steps == 0
            or global_step == cfg.max_train_steps
        ):
            _save_checkpoint(cfg, accelerator, ema, global_step, loss_history)

    accelerator.end_training()
    if accelerator.is_main_process:
        _export_final_model(cfg, accelerator, model, ema, train_scheduler)
    np.save(out_dir / "loss_history.npy", np.array(loss_history))
    return {
        "name": cfg.name,
        "output_dir": str(out_dir),
        "final_step": global_step,
        "train_scheduler_config": train_scheduler.config,
        "loss_history": loss_history,
    }


def _save_checkpoint(cfg, accelerator, ema, global_step, loss_history) -> None:
    ckpt_dir = Path(cfg.checkpoint_dir) / f"step_{global_step:07d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    accelerator.save_state(str(ckpt_dir))
    torch.save(ema.state_dict(), ckpt_dir / "ema.pt")
    (ckpt_dir / "train_state.json").write_text(
        json.dumps({"global_step": global_step, "loss_history": loss_history})
    )
    accelerator.print(f"[{cfg.name}] checkpoint saved: {ckpt_dir}")


def _save_sample_grid(
    cfg, accelerator, model, ema, train_scheduler, global_step
) -> None:
    unwrapped = accelerator.unwrap_model(model)
    ema.store(unwrapped.parameters())
    ema.copy_to(unwrapped.parameters())
    unwrapped.eval()

    class_labels = None
    if cfg.conditional:
        n = cfg.eval_batch_size
        class_labels = torch.arange(n, device=accelerator.device) % cfg.num_classes

    images = sample_images(
        unwrapped,
        train_scheduler.config,
        cfg,
        num_images=cfg.eval_batch_size,
        class_labels=class_labels,
        guidance_scale=cfg.default_guidance_scale if cfg.conditional else 1.0,
        device=accelerator.device,
        seed=cfg.seed,
    )
    side = int(cfg.eval_batch_size**0.5)
    grid = make_grid_image(images, rows=side, cols=cfg.eval_batch_size // side)
    samples_dir = Path(cfg.output_dir) / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    grid.save(str(samples_dir / f"step_{global_step:07d}.png"))

    ema.restore(unwrapped.parameters())
    unwrapped.train()

def _export_final_model(
    cfg: ExperimentConfig, accelerator, model, ema, train_scheduler
) -> None:
    """
    Plain diffusers-native export: unet/ + scheduler/ via save_pretrained().
    No custom pipeline class -- sampling.py remains the one place that knows
    how to run CFG inference, and it can load the exported unet with
    UNet2DModel.from_pretrained(<final_model>/unet) just as easily as it
    currently loads a raw state dict. This just makes the weights portable
    and Hub-compatible without adding a second thing to maintain.
    """
    from diffusers import DDIMScheduler

    unwrapped = accelerator.unwrap_model(model)
    inference_scheduler = DDIMScheduler.from_config(train_scheduler.config)

    def _export(tag: str):
        out_dir = Path(cfg.output_dir) / tag
        unwrapped.save_pretrained(str(out_dir / "unet"))
        inference_scheduler.save_pretrained(str(out_dir / "scheduler"))
        (out_dir / "experiment_config.json").write_text(
            json.dumps(cfg.__dict__, indent=2, default=str)
        )

    # EMA weights: the canonical export you'd use for inference / push to Hub
    ema.store(unwrapped.parameters())
    ema.copy_to(unwrapped.parameters())
    _export("final_model")
    ema.restore(unwrapped.parameters())

    # raw weights too, since you evaluate both anyway
    _export("final_model_raw")

    accelerator.print(f"[{cfg.name}] exported final_model/ and final_model_raw/")
