#!/bin/bash
#SBATCH --job-name=snn_combo_a
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/combo_a_%j.out
#SBATCH --error=logs/combo_a_%j.err

module load cuda/12.6.2 libs/cuda/12.8.1 python/3.13.1
source ~/scratch/snn-esc50/.venv/bin/activate
cd ~/scratch/snn-esc50
export PYTHONUNBUFFERED=1

echo "============================================"
echo "  COMBO BATCH A (15 experiments x 5 folds)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "  Start: $(date)"
echo "============================================"

run_combo() {
    local name="$1"; shift
    echo ""; echo ">>>>>>>>>> $name ($(date)) <<<<<<<<<<"
    python -m experiments.combo_experiment "$@" --device cuda 2>&1
    echo ">>>>>>>>>> $name DONE ($(date)) <<<<<<<<<<"
}

