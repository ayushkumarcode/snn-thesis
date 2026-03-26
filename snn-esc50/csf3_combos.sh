#!/bin/bash
#SBATCH --job-name=snn_combos
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/combos_%j.out
#SBATCH --error=logs/combos_%j.err

# Massive combo experiment batch: ~30 experiments, all 5 folds each
# Queued to run after batch1 and batch2 finish

module load cuda/12.6.2 libs/cuda/12.8.1 python/3.13.1
source ~/scratch/snn-esc50/.venv/bin/activate
cd ~/scratch/snn-esc50

export PYTHONUNBUFFERED=1

echo "============================================"
echo "  COMBO EXPERIMENTS BATCH"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "  Start: $(date)"
echo "============================================"

run_combo() {
    local name="$1"
    shift
    echo ""
    echo ">>>>>>>>>> $name ($(date)) <<<<<<<<<<"
    python -m experiments.combo_experiment "$@" --device cuda 2>&1
    echo ">>>>>>>>>> $name DONE ($(date)) <<<<<<<<<<"
}

# ============================================
# TIER 1: Build on Rhythm-SNN (best at 61.1%)
# ============================================

echo ""
echo "===== TIER 1: RHYTHM-SNN COMBOS ====="

# 1. Rhythm + KD (most promising for beating ANN)
run_combo "rhythm_kd" --rhythm --kd --learn-beta --dropout --sre

# 2. Rhythm + hybrid ANN init
run_combo "rhythm_hybrid" --rhythm --hybrid-init --learn-beta --dropout --sre --epochs 30

# 3. Rhythm + hybrid + KD (triple stack)
run_combo "rhythm_hybrid_kd" --rhythm --hybrid-init --kd --learn-beta --dropout --sre --epochs 30

# 4. Rhythm + TET
run_combo "rhythm_tet" --rhythm --tet --learn-beta --dropout --sre

# 5. Rhythm + TET + KD
run_combo "rhythm_tet_kd" --rhythm --tet --kd --learn-beta --dropout --sre

# 6. Rhythm + cochleagram
run_combo "rhythm_cochleagram" --rhythm --cochleagram --learn-beta --dropout --sre

# 7. Rhythm + delays
run_combo "rhythm_delays" --rhythm --delays --learn-beta --dropout --sre

# ============================================
# TIER 2: Dendritic combos
# ============================================

echo ""
echo "===== TIER 2: DENDRITIC COMBOS ====="

# 8. Dendritic + KD
run_combo "dendritic_kd" --dendritic --kd --dropout --sre

# 9. Dendritic + hybrid init
run_combo "dendritic_hybrid" --dendritic --hybrid-init --dropout --sre --epochs 30

# 10. Dendritic + delays
run_combo "dendritic_delays" --dendritic --delays --dropout --sre

# 11. Dendritic + cochleagram
run_combo "dendritic_cochleagram" --dendritic --cochleagram --dropout --sre

# ============================================
# TIER 3: Other promising combos
# ============================================
