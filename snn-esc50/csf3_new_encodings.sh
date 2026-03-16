#!/bin/bash
#SBATCH --job-name=snn-encodings
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/new_encodings_%j.out
#SBATCH --error=logs/new_encodings_%j.err

# SNN-ESC50: New encoding methods (burst + phase), all 5 folds each
# Submit with: sbatch csf3_new_encodings.sh

set -e
export PYTHONUNBUFFERED=1

echo "========================================"
echo "New Encodings: burst + phase"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

module load cuda/12.6.2
module load libs/cuda/12.8.1
module load python/3.13.1

source venv/bin/activate
mkdir -p logs results

nvidia-smi
echo ""

echo "############################################################"
echo "#  Burst Encoding -- all 5 folds                          #"
echo "############################################################"
python -m src.train --model snn --encoding burst
echo ""

echo "############################################################"
echo "#  Phase Encoding -- all 5 folds                          #"
echo "############################################################"
python -m src.train --model snn --encoding phase
echo ""

echo "========================================"
echo "DONE: $(date)"
echo "========================================"
