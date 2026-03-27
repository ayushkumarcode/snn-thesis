#!/bin/bash
#SBATCH --job-name=energy_c
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --mem=32G
#SBATCH --output=energy_c_%j.out
#SBATCH --error=energy_c_%j.err

module load cuda/12.6.2
module load libs/cuda/12.8.1
module load python/3.13.1

cd ~/scratch/snn-esc50
source .venv/bin/activate

echo "============================================"
echo "Energy Experiments Group C"
echo "Started: $(date)"
echo "============================================"

echo "=== Spike Budget Sweep ==="
python -m experiments.spike_budget_snn --device cuda

echo "Group C COMPLETE: $(date)"
