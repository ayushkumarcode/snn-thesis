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
