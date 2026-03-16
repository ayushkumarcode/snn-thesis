#!/bin/bash
#SBATCH --job-name=snn-surrogate
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/surrogate_%j.out
#SBATCH --error=logs/surrogate_%j.err

# SNN-ESC50: Surrogate gradient ablation study
# All 8 surrogates x fold 1 only (for speed)
# Direct encoding, 50 epochs, no augmentation (controlled comparison)
# Submit with: sbatch csf3_surrogate_ablation.sh

set -e
export PYTHONUNBUFFERED=1

echo "========================================"
echo "Surrogate Gradient Ablation Study"
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

# All 8 surrogates, direct encoding, fold 1, run-suffix to distinguish results
for GRAD in fast_sigmoid atan sigmoid ste triangular sre lso sfs; do
    echo "############################################################"
    echo "#  Surrogate: $GRAD                                        #"
    echo "############################################################"
    python -m src.train --model snn --encoding direct \
        --fold 1 --spike-grad "$GRAD" --run-suffix "_sg_${GRAD}"
    echo ""
done

echo "========================================"
echo "DONE: $(date)"
echo "========================================"
