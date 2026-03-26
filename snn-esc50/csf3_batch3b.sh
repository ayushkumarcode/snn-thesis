#!/bin/bash
#SBATCH --job-name=snn_batch3b
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/batch3b_%j.out
#SBATCH --error=logs/batch3b_%j.err

# Batch 3b: Spiking-LEAF + Info Bottleneck sweeps
# These are heavier experiments (Spiking-LEAF processes raw waveforms)

module load cuda/12.6.2 libs/cuda/12.8.1 python/3.13.1
source ~/scratch/snn-esc50/.venv/bin/activate
cd ~/scratch/snn-esc50
export PYTHONUNBUFFERED=1

echo "============================================"
echo "  BATCH 3b: Spiking-LEAF + Info Bottleneck"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "  Start: $(date)"
echo "============================================"

# 1. Spiking-LEAF SNN (raw waveform → learnable frontend → SNN)
echo ""; echo ">>>>>>>>>> spiking_leaf SNN ($(date)) <<<<<<<<<<"
python -m experiments.spiking_leaf --model snn --device cuda 2>&1
echo ">>>>>>>>>> spiking_leaf SNN DONE ($(date)) <<<<<<<<<<"
