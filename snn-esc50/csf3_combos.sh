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

echo ""
echo "===== TIER 3: OTHER COMBOS ====="

# 12. Hybrid init + KD (no rhythm, baseline neurons)
run_combo "hybrid_kd" --hybrid-init --kd --learn-beta --learn-threshold --dropout --sre --epochs 30

# 13. Hybrid init + TET + KD
run_combo "hybrid_tet_kd" --hybrid-init --tet --kd --learn-beta --learn-threshold --dropout --sre --epochs 30

# 14. Cochleagram + KD (bio-inspired input + distillation)
run_combo "cochleagram_kd" --cochleagram --kd --learn-beta --learn-threshold --dropout --sre

# 15. Delays + KD
run_combo "delays_kd" --delays --kd --learn-beta --learn-threshold --dropout --sre

# ============================================
# TIER 4: Kitchen sink combos
# ============================================

echo ""
echo "===== TIER 4: KITCHEN SINK ====="

# 16. Rhythm + dendritic + KD
run_combo "rhythm_dendritic_kd" --rhythm --kd --dropout --sre

# 17. All techniques (rhythm + delays + KD + TET)
run_combo "all_rhythm" --rhythm --delays --kd --tet --learn-beta --dropout --sre

# 18. All techniques (dendritic + delays + KD + TET)
run_combo "all_dendritic" --dendritic --delays --kd --tet --dropout --sre

# ============================================
# TIER 5: Energy optimization (on best accuracy models)
# ============================================

echo ""
echo "===== TIER 5: ENERGY OPTIMIZATION ====="

# 19-22. Rhythm + L1 regularization sweep
run_combo "rhythm_l1_1e5" --rhythm --learn-beta --dropout --sre --l1-reg 0.00001
run_combo "rhythm_l1_1e4" --rhythm --learn-beta --dropout --sre --l1-reg 0.0001
run_combo "rhythm_l1_1e3" --rhythm --learn-beta --dropout --sre --l1-reg 0.001
run_combo "rhythm_l1_1e2" --rhythm --learn-beta --dropout --sre --l1-reg 0.01

# 23-24. Rhythm + KD + L1 (accuracy + energy)
run_combo "rhythm_kd_l1_1e4" --rhythm --kd --learn-beta --dropout --sre --l1-reg 0.0001
run_combo "rhythm_kd_l1_1e3" --rhythm --kd --learn-beta --dropout --sre --l1-reg 0.001

# ============================================
# TIER 6: KD temperature/alpha sweeps
# ============================================

echo ""
echo "===== TIER 6: KD HYPERPARAMETER SWEEPS ====="

# 25-28. Temperature sweep with Rhythm + KD
run_combo "rhythm_kd_T1" --rhythm --kd --temperature 1.0 --learn-beta --dropout --sre
run_combo "rhythm_kd_T5" --rhythm --kd --temperature 5.0 --learn-beta --dropout --sre
run_combo "rhythm_kd_T10" --rhythm --kd --temperature 10.0 --learn-beta --dropout --sre

# 29-30. Alpha sweep
run_combo "rhythm_kd_a03" --rhythm --kd --alpha 0.3 --learn-beta --dropout --sre
run_combo "rhythm_kd_a07" --rhythm --kd --alpha 0.7 --learn-beta --dropout --sre

echo ""
echo "============================================"
echo "  ALL COMBO EXPERIMENTS COMPLETE: $(date)"
echo "============================================"

# Final summary
echo ""
echo "=== FINAL RESULTS ==="
for dir in results/experiments/combo_*/; do
    summary="$dir/summary.json"
    if [ -f "$summary" ]; then
        python -c "import json; d=json.load(open('$summary')); print(f'  {d[\"experiment\"]: <40} {d[\"mean_accuracy\"]*100:.2f}% ± {d[\"std_accuracy\"]*100:.2f}%')"
    fi
done | sort -t'%' -k1 -rn
