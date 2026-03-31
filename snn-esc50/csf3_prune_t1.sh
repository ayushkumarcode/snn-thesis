#!/bin/bash
#SBATCH --job-name=prune_t1
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --output=prune_t1_%j.out
#SBATCH --error=prune_t1_%j.err

# Prune T=1 Rhythm-SNN models and extract weights for SpiNNaker
# Expected runtime: ~20-30 minutes on A100

module load cuda/12.6.2
module load libs/cuda/12.8.1
module load python/3.13.1

cd ~/scratch/snn-esc50
source .venv/bin/activate

echo "============================================"
echo "T=1 Pruning for SpiNNaker"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================"

python -m experiments.prune_t1_for_spinnaker --device cuda

echo ""
echo "============================================"
echo "COMPLETE: $(date)"
echo "============================================"
