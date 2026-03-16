#!/bin/bash
#SBATCH --job-name=snn-esc50
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_all_%j.out
#SBATCH --error=logs/train_all_%j.err

# SNN-ESC50: Train all models on CSF3 GPU
# Submit with: sbatch csf3_train_all.sh

set -e
export PYTHONUNBUFFERED=1

echo "========================================"
echo "SNN-ESC50 Training - All Experiments"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

# Load modules - BOTH compiler and library modules for CUDA
module load cuda/12.6.2
module load libs/cuda/12.8.1
module load python/3.13.1

echo "CUDA toolkit: $(nvcc --version 2>/dev/null | grep release || echo 'not found')"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Activate venv
source venv/bin/activate

# Create logs directory
mkdir -p logs results

# GPU info
nvidia-smi
echo ""

# Verify PyTorch can see GPU BEFORE training
echo "=== PyTorch GPU Check ==="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    # Quick test
    x = torch.randn(100, 100, device='cuda')
    print(f'GPU tensor test: OK (sum={x.sum().item():.2f})')
else:
    print('WARNING: CUDA not available! Falling back to CPU.')
    print('This will be slow. Check CUDA module loading.')
"
echo ""

# ============================================================
# 1. ANN Baseline (5-fold CV)
# ============================================================
echo ""
echo "############################################################"
echo "#  EXPERIMENT 1: ANN Baseline                              #"
echo "############################################################"
python -m src.train --model ann
echo ""

# ============================================================
# 2. SNN with Direct Encoding (5-fold CV)
# ============================================================
echo ""
echo "############################################################"
echo "#  EXPERIMENT 2: SNN Direct Encoding                       #"
echo "############################################################"
python -m src.train --model snn --encoding direct
echo ""

# ============================================================
# 3. SNN with Rate Encoding (5-fold CV)
# ============================================================
echo ""
echo "############################################################"
echo "#  EXPERIMENT 3: SNN Rate Encoding                         #"
echo "############################################################"
python -m src.train --model snn --encoding rate
echo ""

# ============================================================
# 4. SNN with Delta Encoding (5-fold CV)
# ============================================================
echo ""
echo "############################################################"
echo "#  EXPERIMENT 4: SNN Delta Encoding                        #"
echo "############################################################"
python -m src.train --model snn --encoding delta
echo ""

# ============================================================
# 5. SNN with Latency Encoding (5-fold CV)
# ============================================================
echo ""
echo "############################################################"
echo "#  EXPERIMENT 5: SNN Latency Encoding                      #"
echo "############################################################"
python -m src.train --model snn --encoding latency
echo ""

echo "========================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "Date: $(date)"
echo "========================================"
