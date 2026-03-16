#!/bin/bash
# run_5fold_spinnaker.sh -- Automated 5-fold SpiNNaker FC2-only inference
#
# Prerequisites:
#   1. extract_hidden_features.py run for all 5 folds (400 samples each)
#      Files expected: results/spinnaker_weights/fold{1..5}/hidden_spike_features.npy
#   2. convert_weights.py run for all 5 folds (--fc-only)
#      Files expected: results/spinnaker_weights/fold{1..5}/fc2_connections.npy
#   3. SpiNNaker .venv-spinnaker active with sPyNNaker
#   4. SpiNNaker board accessible at spinnaker.cs.man.ac.uk
#
# Usage:
#   source .venv-spinnaker/bin/activate
#   bash spinnaker/run_5fold_spinnaker.sh
#
# Output:
#   results/spinnaker_results/fc2_results_fold{1..5}.json (one per fold)
#   results/spinnaker_results/5fold_summary.json (aggregated)
#
# Runtime: ~400 samples * ~3s/sample = ~20min per fold = ~100min total
#          (can run overnight or in parallel if multiple boards available)

set -e

REPO_ROOT="$(dirname "$(realpath "$0")")/.."
RESULTS_DIR="${REPO_ROOT}/results/spinnaker_results"
mkdir -p "${RESULTS_DIR}"

echo "============================================================"
echo "  5-Fold SpiNNaker FC2-Only Inference"
echo "  $(date)"
echo "============================================================"

# Calibrated from Run 6 (fold 4, 400 samples): scale=5.0 gave 43.0% SpiNNaker
# Skip scale sweep for folds 1-3,5 to save time (use the calibrated scale)
WEIGHT_SCALE=5.0

for FOLD in 1 2 3 4 5; do
    echo ""
    echo "============================================================"
    echo "  Starting Fold ${FOLD}"
    echo "  $(date)"
    echo "============================================================"

    INPUT_DIR="${REPO_ROOT}/results/spinnaker_weights/fold${FOLD}"
    OUTPUT_DIR="${RESULTS_DIR}"

    if [ ! -f "${INPUT_DIR}/hidden_spike_features.npy" ]; then
        echo "ERROR: Missing hidden_spike_features.npy for fold ${FOLD}"
        echo "Run: python spinnaker/extract_hidden_features.py --model-path results/snn/direct/best_fold${FOLD}.pt --fold ${FOLD} --num-samples 400 --output-dir results/spinnaker_weights/fold${FOLD}"
        exit 1
    fi

    if [ ! -f "${INPUT_DIR}/fc2_connections.npy" ]; then
        echo "ERROR: Missing fc2_connections.npy for fold ${FOLD}"
        echo "Run: python -m spinnaker.convert_weights --model-path results/snn/direct/best_fold${FOLD}.pt --output-dir results/spinnaker_weights/fold${FOLD} --fc-only"
        exit 1
    fi

    echo "Running fold ${FOLD} with ${WEIGHT_SCALE}x weight scale..."
    python spinnaker/run_fc2_spinnaker.py \
        --num-samples 400 \
        --weight-scale ${WEIGHT_SCALE} \
        --skip-to-inference \
        --fold ${FOLD} \
        --input-dir "${INPUT_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        2>&1 | tee "${RESULTS_DIR}/fold${FOLD}_spinnaker.log"

    echo "Fold ${FOLD} complete: $(date)"
done

echo ""
echo "============================================================"
echo "  All 5 folds complete. Aggregating results..."
echo "============================================================"

# Aggregate results into summary
python3 - << 'PYEOF'
import json
import os
from pathlib import Path

results_dir = Path("results/spinnaker_results")
folds = [1, 2, 3, 4, 5]
fold_accs = []
fold_snn_accs = []

print(f"\n{'Fold':>4}  {'SpiNNaker':>10}  {'snnTorch':>10}  {'Gap':>8}")
print("-" * 40)

for fold in folds:
    result_file = results_dir / f"fc2_results_fold{fold}.json"
    if not result_file.exists():
        print(f"  {fold}: MISSING")
        continue
    with open(result_file) as f:
        r = json.load(f)
    spk_acc = r.get("accuracy", 0)
    snn_acc = r.get("snn_reference_accuracy", 0)
    gap = snn_acc - spk_acc
    fold_accs.append(spk_acc)
    fold_snn_accs.append(snn_acc)
    print(f"  {fold:>4}  {spk_acc:>10.1%}  {snn_acc:>10.1%}  {gap:>+8.1%}")

import statistics
if fold_accs:
    mean_spk = statistics.mean(fold_accs)
    std_spk = statistics.stdev(fold_accs) if len(fold_accs) > 1 else 0
    mean_snn = statistics.mean(fold_snn_accs)
    std_snn = statistics.stdev(fold_snn_accs) if len(fold_snn_accs) > 1 else 0
    print("-" * 40)
    print(f"{'Mean':>4}  {mean_spk:>10.1%}  {mean_snn:>10.1%}  {(mean_snn-mean_spk):>+8.1%}")
    print(f"{'Std':>4}  {std_spk:>10.1%}  {std_snn:>10.1%}")

    summary = {
        "platform": "SpiNNaker",
        "approach": "FC2-only hybrid (snnTorch conv+FC1 on CPU, FC2 on SpiNNaker)",
        "weight_scale": 5.0,
        "num_samples_per_fold": 400,
        "fold_accuracies_spinnaker": fold_accs,
        "fold_accuracies_snnTorch": fold_snn_accs,
        "mean_spinnaker": mean_spk,
        "std_spinnaker": std_spk,
        "mean_snnTorch": mean_snn,
        "std_snnTorch": std_snn,
        "hardware_gap_pp": (mean_snn - mean_spk) * 100,
    }
    with open(results_dir / "5fold_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n5-fold summary saved to: {results_dir}/5fold_summary.json")
PYEOF

echo "Done. $(date)"
