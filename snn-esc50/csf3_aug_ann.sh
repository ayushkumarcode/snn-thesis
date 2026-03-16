#!/bin/bash
#SBATCH --job-name=ann-aug
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/aug_ann_%j.out
#SBATCH --error=logs/aug_ann_%j.err

# SNN-ESC50: Augmented ANN training (100 epochs)
# Submit with: sbatch csf3_aug_ann.sh

set -e
export PYTHONUNBUFFERED=1

echo "========================================"
echo "Augmented ANN Training (100 epochs)"
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

python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
echo ""

echo "############################################################"
echo "#  Augmented ANN -- 100 epochs, aug                        #"
echo "############################################################"
python -m src.train --model ann \
    --augment --epochs 100 --run-suffix _aug

echo ""
echo "========================================"
echo "DONE: $(date)"
echo "========================================"
