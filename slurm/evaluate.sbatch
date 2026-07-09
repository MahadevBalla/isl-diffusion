#!/bin/bash
#SBATCH --job-name=isl_eval_%a
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=logs/eval_%a_%j.out
#SBATCH --error=logs/eval_%a_%j.err
#SBATCH --array=0-3

set -euo pipefail

EXPERIMENTS=(baseline_uncond main_conditional schedule_linear main_conditional_64)
EXP_NAME="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"

PROJECT_DIR="/home/mahadevballa23/scratch_space/isl-diffusion"
VENV_DIR="$PROJECT_DIR/.venv"
OUTPUT_DIR="$PROJECT_DIR/experiments/${EXP_NAME}"

cd "$PROJECT_DIR"
mkdir -p logs

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "ERROR: missing venv at $VENV_DIR"
    exit 1
fi
source "$VENV_DIR/bin/activate"
echo ">>> python: $(which python) | $(python --version)"
python -c "import torch; print(f'>>> torch {torch.__version__} | CUDA {torch.cuda.is_available()}')"

# evaluate needs the pretrained weight caches (Inception/LPIPS/ResNet18)
# already warmed on a node with internet -- run this once from the login
# node before submitting: uv run run_experiment.py warm-caches
CACHE_DIR="$PROJECT_DIR/experiments/_real_fid_cache"
if [ ! -f "$CACHE_DIR/inception-2015-12-05.pt" ]; then
    echo "ERROR: $CACHE_DIR/inception-2015-12-05.pt not found."
    echo "Run 'uv run run_experiment.py warm-caches' from the login node first."
    exit 1
fi

export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "job:        ISL diffusion evaluation"
echo "experiment: $EXP_NAME"
echo "output:     $OUTPUT_DIR"
echo "node:       $(hostname)"
echo "start:      $(date)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total --format=csv

python run_experiment.py evaluate --exp "$EXP_NAME"

echo "finished: $(date)"
echo "results:  $OUTPUT_DIR/results.json"
