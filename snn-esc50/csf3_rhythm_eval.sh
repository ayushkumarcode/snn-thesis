#!/bin/bash
#SBATCH --job-name=rhythm_eval
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --output=rhythm_eval_%j.out
#SBATCH --error=rhythm_eval_%j.err

# Rhythm-SNN comprehensive evaluation: adversarial, NeuroBench, CL, noise
# Expected runtime: ~2-3 hours on A100

module load cuda/12.6.2
module load libs/cuda/12.8.1
module load python/3.13.1

cd ~/scratch/snn-esc50
source .venv/bin/activate
