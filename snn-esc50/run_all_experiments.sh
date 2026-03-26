#!/bin/bash
# Run all 9 experiments, fold 1 each, sequentially on specified device.
# Usage: ./run_all_experiments.sh [device]   (default: mps)
# For CSF3: ./run_all_experiments.sh cuda

set -e
cd "$(dirname "$0")"
DEVICE="${1:-mps}"
export PYTHONUNBUFFERED=1

echo "============================================"
echo "  Running all experiments on device: $DEVICE"
echo "  Start time: $(date)"
echo "============================================"

EXPERIMENTS=(
    "learnable_beta"
    "enhanced_snn"
    "tet_training"
    "rhythm_snn"
    "hybrid_ann_snn"
    "knowledge_distillation"
    "dendritic_snn"
    "learnable_delays"
    "cochleagram_experiment"
)

for exp in "${EXPERIMENTS[@]}"; do
    echo ""
    echo ">>>>>>>>>> Starting: $exp ($(date)) <<<<<<<<<<"
    python -m experiments.$exp --fold 1 --device $DEVICE 2>&1
    echo ">>>>>>>>>> Finished: $exp ($(date)) <<<<<<<<<<"
    echo ""
done

echo "============================================"
echo "  All experiments complete: $(date)"
echo "============================================"

# Show results summary
echo ""
echo "Results:"
for exp in "${EXPERIMENTS[@]}"; do
    result_file="results/experiments/$exp/result_fold1.json"
    if [ -f "$result_file" ]; then
        acc=$(python -c "import json; d=json.load(open('$result_file')); print(f'{d[\"best_acc\"]*100:.2f}%')")
        echo "  $exp: $acc"
    else
        echo "  $exp: NO RESULT FILE"
    fi
done
