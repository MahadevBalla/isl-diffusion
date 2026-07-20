# ISL Conditional Diffusion

Conditional diffusion baseline for Indian Sign Language (ISL) image generation: RGB, 128x128, class-conditional `UNet2DModel` with classifier-free guidance (CFG).

## Setup

Using `uv` (recommended):

For installation refer [official docs](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync
```

or, with `pip`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Point `data_root` in `src/config.py` (or override per-run) at your already-downloaded dataset: one subfolder per class (`A/`, `B/`, ..., `1/`, ..., `9/`). No dataset download happens anywhere in this codebase.

## Cache pretrained evaluation models

Run once on a machine with internet access:

```bash
uv run run_experiment.py warm-caches
```

This downloads and caches the pretrained weights required for evaluation (AlexNet for LPIPS and ResNet18 for semantic classification). `torch-fidelity` automatically manages the InceptionV3 weights used for FID, KID, and Precision/Recall.

## Running

```bash
uv run run_experiment.py list

uv run run_experiment.py train --exp baseline_uncond
uv run run_experiment.py train --exp main_conditional
uv run run_experiment.py train --exp schedule_linear
uv run run_experiment.py train --exp main_conditional_64

uv run run_experiment.py evaluate --exp main_conditional
uv run run_experiment.py sampler-compare --exp main_conditional
uv run run_experiment.py plot-curves --exps baseline_uncond main_conditional schedule_linear
```

Training automatically resumes from the latest checkpoint under `<output_dir>/checkpoints/` if one exists.

Completed training exports both EMA and raw checkpoints as diffusers-compatible pipelines:

- `<output_dir>/final_model/`
- `<output_dir>/final_model_raw/`

These can be reloaded using `DiffusionPipeline.from_pretrained()`.

## Experiment list (`src/config.py::EXPERIMENTS`)

| Name | Description |
| ------ | ------------- |
| baseline_uncond | Unconditional baseline. |
| main_conditional | Class-conditional model with classifier-free guidance. |
| schedule_linear | Linear noise schedule ablation. |
| main_conditional_64 | 64×64 resolution ablation. |

## Evaluation

```bash
uv run run_experiment.py evaluate --exp main_conditional
uv run run_experiment.py sampler-compare --exp main_conditional
```

Evaluation includes:

- EMA vs. raw weights
- CFG guidance-scale sweep
- Semantic accuracy
- Sampler comparison

## Metrics

- FID
- KID
- Precision / Recall
- Semantic Accuracy
- LPIPS Diversity
