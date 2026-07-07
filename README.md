# ISL Conditional Diffusion

A class-conditioned diffusion model for Indian Sign Language (ISL) handshape generation, built on `diffusers`' `UNet2DModel` with classifier-free guidance.

## Setup

```bash
uv sync
```

Refer [official docs](https://docs.astral.sh/uv/getting-started/installation/) if not installed.

Optionally using pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision diffusers accelerate clean-fid pillow numpy tqdm tensorboard prdc lpips
```

Point `data_root` in `src/config.py` to the dataset root containing one subdirectory per class (e.g. `A/`, `B/`, ..., `9/`).

## Running

```bash
python run_experiment.py list
python run_experiment.py train --exp baseline_uncond
python run_experiment.py train --exp main_conditional
python run_experiment.py train --exp schedule_linear
python run_experiment.py train --exp legacy_gray64_uncond
python run_experiment.py evaluate --exp main_conditional
python run_experiment.py sampler-compare --exp main_conditional
```

Training automatically resumes from the latest checkpoint if one exists for the selected experiment.

## Exported models

After training, the final EMA and non-EMA model weights are exported in Diffusers format:

```txt
<output_dir>/
├── final_model/
│   ├── unet/               # EMA weights
│   └── scheduler/          # inference scheduler config
└── final_model_raw/        # non-EMA weights, same layout
    ├── unet/
    └── scheduler/
```

## Experiment list (see `src/config.py::EXPERIMENTS`)

| # | Name | Purpose |
| --- | --- | --- |
| 1 | `baseline_uncond` | Unconditional baseline: RGB, 128px, EMA, cosine schedule, augmentation, full 1200/class dataset. The control for the conditioning comparison. |
| 2 | `main_conditional` | Same as (1) + class conditioning (`num_class_embeds=36`) + CFG label dropout. Only variable changed vs (1) is conditioning. |
| 3 | `schedule_linear` | Same as (2), linear noise schedule instead of cosine, identical LR/steps/batch/sampler otherwise — an isolated schedule comparison. |
| 4 | `legacy_gray64_uncond` | Grayscale, 64px, unconditional, same recipe otherwise. Optional secondary comparison point; whether it's reported depends on how the write-up is scoped. |

## Evaluation

The evaluation pipeline includes:

- FID / KID (`clean-fid`)
- Precision / Recall / Density / Coverage (`prdc`)
- LPIPS diversity
- EMA vs. non-EMA checkpoint comparison
- Semantic accuracy using a ResNet18 classifier
- CFG guidance-scale sweep (conditional models)
- Denoising visualization
