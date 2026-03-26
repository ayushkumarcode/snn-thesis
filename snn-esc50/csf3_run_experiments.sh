#!/bin/bash
# CSF3 SLURM array job: run all experiments, all folds.
# Submit from ~/scratch/snn-esc50/ on CSF3.
#
# Usage:
#   # First, sync code to CSF3:
#   rsync -av --exclude='.venv*' --exclude='data/' --exclude='__pycache__' \
#     snn-esc50/ r36859ak@csf3.itservices.manchester.ac.uk:~/scratch/snn-esc50/
#
#   # Then on CSF3:
#   cd ~/scratch/snn-esc50
#   sbatch csf3_run_experiments.sh

#SBATCH --job-name=snn-experiments
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-44
#SBATCH --output=logs/exp_%A_%a.out
#SBATCH --error=logs/exp_%A_%a.err

# Load modules
module load cuda/12.6.2 libs/cuda/12.8.1 python/3.13.1

# Activate venv
source ~/scratch/snn-esc50/.venv/bin/activate
