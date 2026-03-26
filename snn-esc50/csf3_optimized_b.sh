#!/bin/bash
#SBATCH --job-name=snn_opt_b
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/opt_b_%j.out
#SBATCH --error=logs/opt_b_%j.err

# Optimized batch B: delay combos + energy sweeps + KD sweeps

module load cuda/12.6.2 libs/cuda/12.8.1 python/3.13.1
source ~/scratch/snn-esc50/.venv/bin/activate
cd ~/scratch/snn-esc50
export PYTHONUNBUFFERED=1

echo "============================================"
echo "  OPTIMIZED BATCH B"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "  Start: $(date)"
echo "============================================"

run_combo() {
    local name="$1"; shift
    for d in results/experiments/combo_*; do
        if echo "$d" | grep -q "$name" && [ -f "$d/summary.json" ]; then
            echo ">>>>>>>>>> $name ALREADY DONE <<<<<<<<<<"
