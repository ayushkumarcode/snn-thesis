#!/bin/bash
#SBATCH --job-name=snn_opt_b
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/opt_b_%j.out
#SBATCH --error=logs/opt_b_%j.err

# Optimized batch B: delay combos + energy sweeps + KD sweeps

module load cuda/12.6.2 libs/cuda/12.8.1 python/3.13.1
source ~/scratch/snn-esc50/.venv/bin/activate
cd ~/scratch/snn-esc50
export PYTHONUNBUFFERED=1

echo "============================================"
echo "  OPTIMIZED BATCH B"
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

# Dendritic combos
run_combo "dendritic_delays" --dendritic --delays --dropout --sre
run_combo "dendritic_cochleagram" --dendritic --cochleagram --dropout --sre

# L1 energy optimization sweeps (Rhythm-SNN)
run_combo "rhythm_l1_1e5" --rhythm --learn-beta --dropout --sre --l1-reg 0.00001
run_combo "rhythm_l1_1e4" --rhythm --learn-beta --dropout --sre --l1-reg 0.0001
run_combo "rhythm_l1_1e3" --rhythm --learn-beta --dropout --sre --l1-reg 0.001
run_combo "rhythm_l1_1e2" --rhythm --learn-beta --dropout --sre --l1-reg 0.01

# L1 + KD combos
run_combo "rhythm_kd_l1_1e4" --rhythm --kd --learn-beta --dropout --sre --l1-reg 0.0001
run_combo "rhythm_kd_l1_1e3" --rhythm --kd --learn-beta --dropout --sre --l1-reg 0.001

# KD hyperparameter sweeps
run_combo "rhythm_kd_T10" --rhythm --kd --temperature 10.0 --learn-beta --dropout --sre
run_combo "rhythm_kd_a03" --rhythm --kd --alpha 0.3 --learn-beta --dropout --sre
run_combo "rhythm_kd_a07" --rhythm --kd --alpha 0.7 --learn-beta --dropout --sre

# Cochleagram combos
