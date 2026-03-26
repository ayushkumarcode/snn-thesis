#!/bin/bash
#SBATCH --job-name=snn_unblocked
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/unblocked_%j.out
#SBATCH --error=logs/unblocked_%j.err

# Fast experiments that were blocked behind slow delay jobs
# These don't use delays — each takes ~30 min for 5 folds

module load cuda/12.6.2 libs/cuda/12.8.1 python/3.13.1
source ~/scratch/snn-esc50/.venv/bin/activate
cd ~/scratch/snn-esc50
export PYTHONUNBUFFERED=1

echo "============================================"
echo "  UNBLOCKED FAST EXPERIMENTS"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "  Start: $(date)"
echo "============================================"

run_combo() {
    local name="$1"; shift
    # Skip if already has 5-fold summary
    for d in results/experiments/combo_*; do
