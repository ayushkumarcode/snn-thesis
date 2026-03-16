#!/bin/bash
#SBATCH --job-name=snn_new_exps
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=csf3_results/new_exps_%j.out
#SBATCH --error=csf3_results/new_exps_%j.err

# ============================================================
# CSF3 batch script: Run ALL GPU-heavy new experiments
# Submit with: sbatch csf3_all_new_experiments.sh
# ============================================================

module load cuda/12.6.2
module load libs/cuda/12.8.1
module load python/3.13.1

source ~/snn-esc50-venv/bin/activate
cd ~/snn-esc50/ 2>/dev/null || cd /scratch/$USER/snn-esc50/ 2>/dev/null

mkdir -p csf3_results

echo "============================================================"
echo "Starting all new experiments: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================================"

# --- 1. ADVERSARIAL ROBUSTNESS 5-FOLD (CRITICAL) ---
echo ""
echo "=== [1/5] Adversarial Robustness (5-fold, 400 samples each) ==="
for fold in 1 2 3 4 5; do
    echo "  Fold $fold starting: $(date)"
    python experiments/adversarial_robustness.py --fold $fold --num-samples 400
    echo "  Fold $fold done: $(date)"
done

# --- 2. NOISE ROBUSTNESS (5-fold) ---
echo ""
echo "=== [2/5] Noise Robustness (5-fold) ==="
python -m experiments.noise_robustness

# --- 3. FEW-SHOT LEARNING CURVES (fold 1 only — 5 data fractions × 2 models) ---
echo ""
echo "=== [3/5] Few-Shot Learning Curves (fold 1) ==="
python -m experiments.few_shot_learning_curves --fold 1

# --- 4. SPIKE EFFICIENCY PARETO (fold 1 only — 7 lambdas) ---
echo ""
echo "=== [4/5] Spike Efficiency Pareto (fold 1) ==="
python -m experiments.spike_efficiency_pareto --fold 1

# --- 5. TEMPORAL ABLATION (all 5 folds) ---
echo ""
echo "=== [5/5] Temporal Ablation (5-fold) ==="
python -m experiments.temporal_ablation --encoding direct

echo ""
echo "============================================================"
echo "All experiments complete: $(date)"
echo "============================================================"
