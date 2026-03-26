#!/bin/bash
#SBATCH --job-name=snn_exp_batch1
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/batch1_%j.out
#SBATCH --error=logs/batch1_%j.err

# Batch 1: Experiments 1-5, all 5 folds each
# learnable_beta, enhanced_snn, tet_training, rhythm_snn, hybrid_ann_snn

module load cuda/12.6.2 libs/cuda/12.8.1 python/3.13.1
source ~/scratch/snn-esc50/.venv/bin/activate
cd ~/scratch/snn-esc50

export PYTHONUNBUFFERED=1

echo "============================================"
echo "  BATCH 1: Experiments 1-5 (all folds)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "  Start: $(date)"
echo "============================================"

EXPERIMENTS=("learnable_beta" "enhanced_snn" "tet_training" "rhythm_snn" "hybrid_ann_snn")

for exp in "${EXPERIMENTS[@]}"; do
