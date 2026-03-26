#!/bin/bash
#SBATCH --job-name=snn_combo_b
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/combo_b_%j.out
#SBATCH --error=logs/combo_b_%j.err

module load cuda/12.6.2 libs/cuda/12.8.1 python/3.13.1
source ~/scratch/snn-esc50/.venv/bin/activate
cd ~/scratch/snn-esc50
export PYTHONUNBUFFERED=1

echo "============================================"
echo "  COMBO BATCH B (15 experiments x 5 folds)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "  Start: $(date)"
echo "============================================"

run_combo() {
    local name="$1"; shift
    echo ""; echo ">>>>>>>>>> $name ($(date)) <<<<<<<<<<"
    python -m experiments.combo_experiment "$@" --device cuda 2>&1
    echo ">>>>>>>>>> $name DONE ($(date)) <<<<<<<<<<"
}

# TIER 2: Dendritic combos
run_combo "dendritic_kd" --dendritic --kd --dropout --sre
run_combo "dendritic_hybrid" --dendritic --hybrid-init --dropout --sre --epochs 30
run_combo "dendritic_delays" --dendritic --delays --dropout --sre
run_combo "dendritic_cochleagram" --dendritic --cochleagram --dropout --sre

# TIER 5: Energy optimization (L1 sweep on rhythm)
run_combo "rhythm_l1_1e5" --rhythm --learn-beta --dropout --sre --l1-reg 0.00001
run_combo "rhythm_l1_1e4" --rhythm --learn-beta --dropout --sre --l1-reg 0.0001
run_combo "rhythm_l1_1e3" --rhythm --learn-beta --dropout --sre --l1-reg 0.001
run_combo "rhythm_l1_1e2" --rhythm --learn-beta --dropout --sre --l1-reg 0.01

# Energy + accuracy (rhythm + KD + L1)
run_combo "rhythm_kd_l1_1e4" --rhythm --kd --learn-beta --dropout --sre --l1-reg 0.0001
run_combo "rhythm_kd_l1_1e3" --rhythm --kd --learn-beta --dropout --sre --l1-reg 0.001

# KD alpha/temp sweeps
run_combo "rhythm_kd_T10" --rhythm --kd --temperature 10.0 --learn-beta --dropout --sre
run_combo "rhythm_kd_a03" --rhythm --kd --alpha 0.3 --learn-beta --dropout --sre
run_combo "rhythm_kd_a07" --rhythm --kd --alpha 0.7 --learn-beta --dropout --sre

# Cochleagram combos
run_combo "cochleagram_rhythm_kd" --cochleagram --rhythm --kd --learn-beta --dropout --sre
run_combo "cochleagram_hybrid" --cochleagram --hybrid-init --learn-beta --learn-threshold --dropout --sre --epochs 30

echo ""; echo "BATCH B COMPLETE: $(date)"
echo "=== RESULTS ==="
for dir in results/experiments/combo_*/; do
    s="$dir/summary.json"
    [ -f "$s" ] && python -c "import json; d=json.load(open('$s')); print(f'  {d[\"experiment\"]: <45} {d[\"mean_accuracy\"]*100:.2f}% ± {d[\"std_accuracy\"]*100:.2f}%')"
done | sort -t'±' -k1 -rn
