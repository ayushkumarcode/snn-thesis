#!/bin/bash
#SBATCH --job-name=snn_5fold
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=csf3_results/5fold_upgrades_%j.out
#SBATCH --error=csf3_results/5fold_upgrades_%j.err

# Upgrade D-graded experiments from fold-1-only to 5-fold
# This upgrades methodology grades from D to C+

module load cuda/12.6.2
module load libs/cuda/12.8.1
module load python/3.13.1

source ~/scratch/snn-esc50/.venv/bin/activate
cd ~/scratch/snn-esc50

echo "Starting 5-fold upgrades: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# 1. Pruning resilience — all 5 folds
echo "=== Pruning Resilience (5-fold) ==="
python -m experiments.pruning_resilience

# 2. Neuron ablation — all 5 folds
echo "=== Neuron Ablation (5-fold) ==="
python -m experiments.neuron_ablation

# 3. Stochastic resonance — all 5 folds (need more seeds too)
echo "=== Stochastic Resonance (5-fold) ==="
python -m experiments.stochastic_resonance

# 4. Encoding transfer matrix — folds 1-5
echo "=== Encoding Transfer Matrix (5-fold) ==="
for fold in 1 2 3 4 5; do
    echo "  Transfer matrix fold $fold"
    python -m experiments.encoding_transfer_matrix --fold $fold
done

# 5. Saliency maps — 100 samples instead of 10
echo "=== Saliency Maps (100 samples) ==="
python -m experiments.snn_saliency_maps --fold 1 --num-samples 100

echo "All 5-fold upgrades complete: $(date)"
