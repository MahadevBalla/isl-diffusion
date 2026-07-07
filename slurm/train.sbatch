#!/bin/bash
#SBATCH --job-name=isl_diff_%a
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=logs/train_%a_%j.out
#SBATCH --error=logs/train_%a_%j.err
#SBATCH --array=0-3

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Train ISL conditional diffusion experiments.
#
# Array index → experiment name:
#   0 → baseline_uncond
#   1 → main_conditional
#   2 → schedule_linear
#   3 → legacy_gray64_uncond
#
# Each task auto-resumes from its own <output_dir>/checkpoints/step_* if
# present, so this script is safe to resubmit (e.g. after a walltime kill)
# without restarting from scratch. For chained submission across a walltime
# limit, use --dependency=afterany so the resume logic in engine.py picks up
# where the previous attempt left off:
#
#   JOB1=$(sbatch --parsable --array=1 slurm/train.sbatch)
#   sbatch --dependency=afterany:$JOB1 --array=1 slurm/train.sbatch
#
# Submit all four experiments:
#   sbatch slurm/train.sbatch
#
# Single experiment:
#   sbatch --array=1 slurm/train.sbatch
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENTS=(baseline_uncond main_conditional schedule_linear legacy_gray64_uncond)
EXP_NAME="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"

PROJECT_DIR="/home/mahadevballa23/scratch_space/isl-diffusion/code"
VENV_DIR="$PROJECT_DIR/.venv"
DATA_ROOT="$PROJECT_DIR/data"
OUTPUT_DIR="$PROJECT_DIR/experiments/${EXP_NAME}"

cd "$PROJECT_DIR"
mkdir -p logs

# ── Environment ───────────────────────────────────────────────────────────────
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "ERROR: Missing venv at $VENV_DIR"
    exit 1
fi
source "$VENV_DIR/bin/activate"
echo ">>> Python: $(which python) | $(python --version)"
python -c "import torch; print(f'>>> torch {torch.__version__} | CUDA {torch.cuda.is_available()}')"

# ── Check dataset ──────────────────────────────────────────────────────────────
if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: data_root not found: $DATA_ROOT"
    exit 1
fi

# ── HuggingFace / offline safety (diffusers pulls model configs via hub otherwise) ──
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
mkdir -p "$HF_HOME"

# ── Thread settings ───────────────────────────────────────────────────────────
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "========================================================"
echo "  Job:         ISL Diffusion Training"
echo "  Experiment:  $EXP_NAME"
echo "  Output dir:  $OUTPUT_DIR"
echo "  GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "  Node:        $(hostname)"
echo "  Start:       $(date)"
echo "========================================================"

python run_experiment.py train --exp "$EXP_NAME"

echo "========================================================"
echo "  Finished: $(date)"
echo "  Results:  $OUTPUT_DIR"
echo "========================================================"