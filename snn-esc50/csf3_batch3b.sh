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

# 2. Spiking-LEAF ANN (raw waveform → learnable frontend → ANN, for comparison)
echo ""; echo ">>>>>>>>>> spiking_leaf ANN ($(date)) <<<<<<<<<<"
python -m experiments.spiking_leaf --model ann --device cuda 2>&1
echo ">>>>>>>>>> spiking_leaf ANN DONE ($(date)) <<<<<<<<<<"

# 3. Info Bottleneck beta=1e-4
echo ""; echo ">>>>>>>>>> info_bottleneck beta=1e-4 ($(date)) <<<<<<<<<<"
python -m experiments.info_bottleneck_snn --beta-ib 0.0001 --device cuda 2>&1
echo ">>>>>>>>>> info_bottleneck beta=1e-4 DONE ($(date)) <<<<<<<<<<"

# 4. Info Bottleneck beta=1e-3
echo ""; echo ">>>>>>>>>> info_bottleneck beta=1e-3 ($(date)) <<<<<<<<<<"
python -m experiments.info_bottleneck_snn --beta-ib 0.001 --device cuda 2>&1
echo ">>>>>>>>>> info_bottleneck beta=1e-3 DONE ($(date)) <<<<<<<<<<"

# 5. Info Bottleneck beta=1e-2
echo ""; echo ">>>>>>>>>> info_bottleneck beta=1e-2 ($(date)) <<<<<<<<<<"
python -m experiments.info_bottleneck_snn --beta-ib 0.01 --device cuda 2>&1
echo ">>>>>>>>>> info_bottleneck beta=1e-2 DONE ($(date)) <<<<<<<<<<"

echo ""
echo "============================================"
echo "  BATCH 3b COMPLETE: $(date)"
echo "============================================"

echo ""
echo "=== RESULTS ==="
