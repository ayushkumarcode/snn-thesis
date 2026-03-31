#!/bin/bash
# Deploy ALL T=1 pruned models on SpiNNaker (10 prune levels x 5 folds = 50 runs).
#
# SpiNNaker is a single board — only ONE simulation at a time.
# This script runs all 50 deployments sequentially with progress tracking.
#
# Usage:
#   source .venv-spinnaker/bin/activate
#   bash spinnaker/deploy_all_pruned_t1.sh
#
# Estimated time: ~50 x ~8min = ~6.5 hours (T=1 is faster than T=3/T=10)

set -e
cd /Users/kumar/Documents/University/Year3/thesisproject/snn-esc50

TOTAL_JOBS=50
COMPLETED=0
FAILED=0
SKIPPED=0
START_TIME=$(date +%s)
LOGFILE="results/spinnaker_results/pruned_t1/deploy_all_log.txt"

mkdir -p results/spinnaker_results/pruned_t1

echo "============================================================" | tee "$LOGFILE"
echo "  T=1 Pruned SpiNNaker Deployment — ${TOTAL_JOBS} jobs"      | tee -a "$LOGFILE"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"                     | tee -a "$LOGFILE"
echo "============================================================" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

for PCT in 50 55 60 65 70 75 80 85 90 95; do
  for FOLD in 1 2 3 4 5; do
    JOB_NUM=$((COMPLETED + FAILED + SKIPPED + 1))
    TAG="pruned${PCT}_fold${FOLD}"
    DATA_DIR="results/spinnaker_weights/pruned_t1_${PCT}pct_fold${FOLD}"
    RESULT_FILE="results/spinnaker_results/pruned_t1/fast_${TAG}_400_N256.json"
