#!/bin/bash
#SBATCH --job-name=snn-%j
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/train_%x_%j.out
#SBATCH --error=logs/train_%x_%j.err

# SNN-ESC50: Train single encoding on CSF3 GPU
# Usage: sbatch --job-name=snn-rate csf3_train_encoding.sh rate
#        sbatch --job-name=snn-delta csf3_train_encoding.sh delta
#        sbatch --job-name=snn-latency csf3_train_encoding.sh latency
#        sbatch --job-name=snn-direct csf3_train_encoding.sh direct

set -e
export PYTHONUNBUFFERED=1

ENCODING=${1:?Usage: sbatch csf3_train_encoding.sh <encoding>}

echo "========================================"
echo "SNN-ESC50: ${ENCODING} encoding"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"

# Load modules
module load cuda/12.6.2
module load libs/cuda/12.8.1
module load python/3.13.1

source venv/bin/activate
mkdir -p logs results

# Verify GPU
nvidia-smi --query-gpu=name --format=csv,noheader
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

echo ""
echo "Starting training: SNN with ${ENCODING} encoding (5-fold CV)"
echo ""
python -m src.train --model snn --encoding ${ENCODING}

echo ""
echo "DONE: ${ENCODING} encoding"
echo "Date: $(date)"
