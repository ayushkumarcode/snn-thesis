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
