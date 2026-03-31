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

    # Skip if result already exists
    if [ -f "$RESULT_FILE" ]; then
      SKIPPED=$((SKIPPED + 1))
      echo "[${JOB_NUM}/${TOTAL_JOBS}] SKIP ${TAG} — result exists" | tee -a "$LOGFILE"
      continue
    fi

    # Skip if weight dir missing
    if [ ! -d "$DATA_DIR" ]; then
      FAILED=$((FAILED + 1))
      echo "[${JOB_NUM}/${TOTAL_JOBS}] MISS ${TAG} — ${DATA_DIR} not found" | tee -a "$LOGFILE"
      continue
    fi

    # Timing
    NOW=$(date +%s)
    ELAPSED=$((NOW - START_TIME))
    DONE=$((COMPLETED + FAILED + SKIPPED))
    if [ "$DONE" -gt 0 ]; then
      AVG_PER_JOB=$((ELAPSED / DONE))
      REMAINING=$(( (TOTAL_JOBS - DONE) * AVG_PER_JOB ))
      ETA_EPOCH=$((NOW + REMAINING))
      ETA=$(date -r "$ETA_EPOCH" '+%H:%M:%S' 2>/dev/null || date -d "@$ETA_EPOCH" '+%H:%M:%S' 2>/dev/null || echo "??:??:??")
      ELAPSED_FMT="$(printf '%02d:%02d:%02d' $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)))"
      REMAIN_FMT="$(printf '%02d:%02d:%02d' $((REMAINING/3600)) $(((REMAINING%3600)/60)) $((REMAINING%60)))"
    else
      ELAPSED_FMT="00:00:00"
      REMAIN_FMT="estimating..."
      ETA="estimating..."
    fi

    echo "" | tee -a "$LOGFILE"
    echo "=== [${JOB_NUM}/${TOTAL_JOBS}] ${TAG} ===" | tee -a "$LOGFILE"
    echo "  Elapsed: ${ELAPSED_FMT} | Remaining: ~${REMAIN_FMT} | ETA: ${ETA}" | tee -a "$LOGFILE"
    echo "  Completed: ${COMPLETED} | Failed: ${FAILED} | Skipped: ${SKIPPED}" | tee -a "$LOGFILE"

    JOB_START=$(date +%s)

    if python -u spinnaker/deploy_pruned_t1.py \
        --prune-pct "$PCT" \
        --fold "$FOLD" \
        --n-samples 400 \
        --batch-size 50 2>&1 | tee -a "$LOGFILE"; then
      COMPLETED=$((COMPLETED + 1))
      JOB_END=$(date +%s)
      JOB_TIME=$((JOB_END - JOB_START))
      echo "  [DONE] ${TAG} in ${JOB_TIME}s" | tee -a "$LOGFILE"
    else
      FAILED=$((FAILED + 1))
      echo "  [FAIL] ${TAG}" | tee -a "$LOGFILE"
    fi

  done
done

# ── Final summary ──
END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))
TOTAL_FMT="$(printf '%02d:%02d:%02d' $((TOTAL_ELAPSED/3600)) $(((TOTAL_ELAPSED%3600)/60)) $((TOTAL_ELAPSED%60)))"

echo "" | tee -a "$LOGFILE"
echo "============================================================" | tee -a "$LOGFILE"
echo "  ALL T=1 PRUNED DEPLOYMENTS COMPLETE" | tee -a "$LOGFILE"
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOGFILE"
echo "  Total time: ${TOTAL_FMT}" | tee -a "$LOGFILE"
echo "  Completed: ${COMPLETED} | Failed: ${FAILED} | Skipped: ${SKIPPED}" | tee -a "$LOGFILE"
echo "============================================================" | tee -a "$LOGFILE"

# Quick accuracy summary from results
echo "" | tee -a "$LOGFILE"
echo "=== Accuracy Summary ===" | tee -a "$LOGFILE"
