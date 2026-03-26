#!/bin/bash
#SBATCH --job-name=snn_combo_a
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/combo_a_%j.out
#SBATCH --error=logs/combo_a_%j.err

module load cuda/12.6.2 libs/cuda/12.8.1 python/3.13.1
source ~/scratch/snn-esc50/.venv/bin/activate
cd ~/scratch/snn-esc50
export PYTHONUNBUFFERED=1

echo "============================================"
echo "  COMBO BATCH A (15 experiments x 5 folds)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "  Start: $(date)"
echo "============================================"

run_combo() {
    local name="$1"; shift
    echo ""; echo ">>>>>>>>>> $name ($(date)) <<<<<<<<<<"
    python -m experiments.combo_experiment "$@" --device cuda 2>&1
    echo ">>>>>>>>>> $name DONE ($(date)) <<<<<<<<<<"
}

# TIER 1: Rhythm combos (best base at 61.1%)
run_combo "rhythm_kd" --rhythm --kd --learn-beta --dropout --sre
run_combo "rhythm_hybrid" --rhythm --hybrid-init --learn-beta --dropout --sre --epochs 30
run_combo "rhythm_hybrid_kd" --rhythm --hybrid-init --kd --learn-beta --dropout --sre --epochs 30
run_combo "rhythm_tet" --rhythm --tet --learn-beta --dropout --sre
run_combo "rhythm_tet_kd" --rhythm --tet --kd --learn-beta --dropout --sre
run_combo "rhythm_cochleagram" --rhythm --cochleagram --learn-beta --dropout --sre
run_combo "rhythm_delays" --rhythm --delays --learn-beta --dropout --sre

# TIER 2: Hybrid/KD combos
run_combo "hybrid_kd" --hybrid-init --kd --learn-beta --learn-threshold --dropout --sre --epochs 30
run_combo "hybrid_tet_kd" --hybrid-init --tet --kd --learn-beta --learn-threshold --dropout --sre --epochs 30
run_combo "cochleagram_kd" --cochleagram --kd --learn-beta --learn-threshold --dropout --sre
run_combo "delays_kd" --delays --kd --learn-beta --learn-threshold --dropout --sre

# TIER 3: Kitchen sink
run_combo "all_rhythm" --rhythm --delays --kd --tet --learn-beta --dropout --sre
run_combo "all_dendritic" --dendritic --delays --kd --tet --dropout --sre

# KD temp sweep on rhythm
run_combo "rhythm_kd_T1" --rhythm --kd --temperature 1.0 --learn-beta --dropout --sre
run_combo "rhythm_kd_T5" --rhythm --kd --temperature 5.0 --learn-beta --dropout --sre

echo ""; echo "BATCH A COMPLETE: $(date)"
echo "=== RESULTS ==="
for dir in results/experiments/combo_*/; do
    s="$dir/summary.json"
    [ -f "$s" ] && python -c "import json; d=json.load(open('$s')); print(f'  {d[\"experiment\"]: <45} {d[\"mean_accuracy\"]*100:.2f}% ± {d[\"std_accuracy\"]*100:.2f}%')"
done | sort -t'±' -k1 -rn
