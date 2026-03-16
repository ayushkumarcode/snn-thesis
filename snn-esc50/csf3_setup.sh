#!/bin/bash
# CSF3 Setup Script for SNN-ESC50 Project
# Run this AFTER SSH-ing into CSF3
#
# Usage:
#   ssh r36859ak@csf3.itservices.manchester.ac.uk
#   cd ~/scratch/snn-esc50
#   bash csf3_setup.sh

set -e

echo "========================================"
echo "SNN-ESC50 CSF3 Setup"
echo "========================================"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

# Load modules
echo "=== Loading modules ==="
module load cuda/12.6.2 2>/dev/null || echo "CUDA module load failed"
module load python/3.13.1 2>/dev/null || echo "Python module load failed"
echo "Python: $(python3 --version)"
echo "CUDA: $(nvcc --version 2>/dev/null | grep release || echo 'not on login node')"
echo ""

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "=== Creating virtual environment ==="
    python3 -m venv venv
    echo "Created venv"
else
    echo "=== Virtual environment already exists ==="
fi

source venv/bin/activate
echo "Python in venv: $(which python)"

# Install dependencies
echo ""
echo "=== Installing dependencies ==="
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install snntorch librosa numpy pandas matplotlib seaborn scikit-learn

# Verify GPU access
echo ""
echo "=== Verifying PyTorch GPU access ==="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    print('WARNING: No GPU detected. Are you on a GPU node?')
"

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To train, submit a GPU job:"
echo "  sbatch csf3_train_all.sh"
echo ""
echo "Or request an interactive GPU session:"
echo "  srun --partition=gpuA --gres=gpu:1 --time=04:00:00 --pty bash"
echo ""
