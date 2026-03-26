#!/bin/bash
#SBATCH --job-name=snn_batch3a
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/batch3a_%j.out
#SBATCH --error=logs/batch3a_%j.err

# Batch 3a: Novel technique experiments (5-fold each)
# Spiking-LEAF, ANN-to-SNN conversion, Stochastic Resonance (+Rhythm variant),
# Predictive Coding, Astrocyte

module load cuda/12.6.2 libs/cuda/12.8.1 python/3.13.1
source ~/scratch/snn-esc50/.venv/bin/activate
cd ~/scratch/snn-esc50
export PYTHONUNBUFFERED=1

echo "============================================"
echo "  BATCH 3a: Novel techniques"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "  Start: $(date)"
echo "============================================"

# 1. ANN-to-SNN conversion (no training, just inference — fast)
echo ""; echo ">>>>>>>>>> ann_to_snn_conversion ($(date)) <<<<<<<<<<"
python -m experiments.ann_to_snn_conversion --device cuda 2>&1
