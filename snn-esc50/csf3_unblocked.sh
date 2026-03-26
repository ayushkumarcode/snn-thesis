#!/bin/bash
#SBATCH --job-name=snn_unblocked
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/unblocked_%j.out
#SBATCH --error=logs/unblocked_%j.err

# Fast experiments that were blocked behind slow delay jobs
# These don't use delays — each takes ~30 min for 5 folds

module load cuda/12.6.2 libs/cuda/12.8.1 python/3.13.1
source ~/scratch/snn-esc50/.venv/bin/activate
cd ~/scratch/snn-esc50
export PYTHONUNBUFFERED=1

echo "============================================"
echo "  UNBLOCKED FAST EXPERIMENTS"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "  Start: $(date)"
echo "============================================"

run_combo() {
    local name="$1"; shift
    # Skip if already has 5-fold summary
    for d in results/experiments/combo_*; do
        if echo "$d" | grep -q "$name" && [ -f "$d/summary.json" ]; then
            echo ">>>>>>>>>> $name ALREADY DONE, skipping <<<<<<<<<<"
            return
        fi
    done
    echo ""; echo ">>>>>>>>>> $name ($(date)) <<<<<<<<<<"
    python -m experiments.combo_experiment "$@" --device cuda 2>&1
    echo ">>>>>>>>>> $name DONE ($(date)) <<<<<<<<<<"
}

# From combo_a that were blocked behind rhythm_delays
run_combo "all_rhythm" --rhythm --delays --kd --tet --learn-beta --dropout --sre
run_combo "all_dendritic" --dendritic --delays --kd --tet --dropout --sre
run_combo "rhythm_kd_T1" --rhythm --kd --temperature 1.0 --learn-beta --dropout --sre
run_combo "rhythm_kd_T5" --rhythm --kd --temperature 5.0 --learn-beta --dropout --sre

# From combo_b that were blocked behind dendritic_delays
run_combo "dendritic_cochleagram" --dendritic --cochleagram --dropout --sre
run_combo "rhythm_l1_1e5" --rhythm --learn-beta --dropout --sre --l1-reg 0.00001
run_combo "rhythm_l1_1e4" --rhythm --learn-beta --dropout --sre --l1-reg 0.0001
run_combo "rhythm_l1_1e3" --rhythm --learn-beta --dropout --sre --l1-reg 0.001
run_combo "rhythm_l1_1e2" --rhythm --learn-beta --dropout --sre --l1-reg 0.01
run_combo "rhythm_kd_l1_1e4" --rhythm --kd --learn-beta --dropout --sre --l1-reg 0.0001
run_combo "rhythm_kd_l1_1e3" --rhythm --kd --learn-beta --dropout --sre --l1-reg 0.001
run_combo "rhythm_kd_T10" --rhythm --kd --temperature 10.0 --learn-beta --dropout --sre
run_combo "rhythm_kd_a03" --rhythm --kd --alpha 0.3 --learn-beta --dropout --sre
run_combo "rhythm_kd_a07" --rhythm --kd --alpha 0.7 --learn-beta --dropout --sre
run_combo "cochleagram_rhythm_kd" --cochleagram --rhythm --kd --learn-beta --dropout --sre
run_combo "cochleagram_hybrid" --cochleagram --hybrid-init --learn-beta --learn-threshold --dropout --sre --epochs 30

echo ""
echo "============================================"
echo "  UNBLOCKED BATCH COMPLETE: $(date)"
echo "============================================"

echo "=== RESULTS ==="
for dir in results/experiments/combo_*/; do
    s="$dir/summary.json"
    [ -f "$s" ] && python -c "import json; d=json.load(open('$s')); print(f'  {d[\"experiment\"]:<45} {d[\"mean_accuracy\"]*100:.2f}% ± {d[\"std_accuracy\"]*100:.2f}%')"
done | sort -t'±' -k1 -rn
