#!/bin/bash
#SBATCH --job-name=snn_opt_a
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/opt_a_%j.out
#SBATCH --error=logs/opt_a_%j.err

# Optimized batch A: all remaining combo experiments
# Uses vectorized DelayedLinear, BF16 autocast, vectorized loss

module load cuda/12.6.2 libs/cuda/12.8.1 python/3.13.1
source ~/scratch/snn-esc50/.venv/bin/activate
cd ~/scratch/snn-esc50
export PYTHONUNBUFFERED=1

echo "============================================"
echo "  OPTIMIZED BATCH A"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "  Start: $(date)"
echo "============================================"

run_combo() {
    local name="$1"; shift
    for d in results/experiments/combo_*; do
        if echo "$d" | grep -q "$name" && [ -f "$d/summary.json" ]; then
            echo ">>>>>>>>>> $name ALREADY DONE <<<<<<<<<<"
            return
        fi
    done
    echo ""; echo ">>>>>>>>>> $name ($(date)) <<<<<<<<<<"
    python -m experiments.combo_experiment "$@" --device cuda 2>&1
    echo ">>>>>>>>>> $name DONE ($(date)) <<<<<<<<<<"
}

# Rhythm combos (from combo_a TODO list)
run_combo "rhythm_hybrid_kd" --rhythm --hybrid-init --kd --learn-beta --dropout --sre --epochs 30
run_combo "rhythm_tet_kd" --rhythm --tet --kd --learn-beta --dropout --sre
run_combo "rhythm_delays" --rhythm --delays --learn-beta --dropout --sre
run_combo "rhythm_kd_T1" --rhythm --kd --temperature 1.0 --learn-beta --dropout --sre
run_combo "rhythm_kd_T5" --rhythm --kd --temperature 5.0 --learn-beta --dropout --sre

# Non-rhythm combos
run_combo "hybrid_kd" --hybrid-init --kd --learn-beta --learn-threshold --dropout --sre --epochs 30
run_combo "hybrid_tet_kd" --hybrid-init --tet --kd --learn-beta --learn-threshold --dropout --sre --epochs 30
run_combo "cochleagram_kd" --cochleagram --kd --learn-beta --learn-threshold --dropout --sre
run_combo "delays_kd" --delays --kd --learn-beta --learn-threshold --dropout --sre

# Kitchen sink
run_combo "all_rhythm" --rhythm --delays --kd --tet --learn-beta --dropout --sre
run_combo "all_dendritic" --dendritic --delays --kd --tet --dropout --sre

echo ""; echo "BATCH A COMPLETE: $(date)"
echo "=== RESULTS ==="
