#!/bin/bash
#SBATCH --job-name=snn_combos
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/combos_%j.out
#SBATCH --error=logs/combos_%j.err

# Massive combo experiment batch: ~30 experiments, all 5 folds each
# Queued to run after batch1 and batch2 finish

module load cuda/12.6.2 libs/cuda/12.8.1 python/3.13.1
source ~/scratch/snn-esc50/.venv/bin/activate
cd ~/scratch/snn-esc50

export PYTHONUNBUFFERED=1

echo "============================================"
echo "  COMBO EXPERIMENTS BATCH"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "  Start: $(date)"
echo "============================================"

run_combo() {
    local name="$1"
    shift
