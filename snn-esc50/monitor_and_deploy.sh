#!/bin/bash
# Monitor CSF3 jobs and auto-trigger SpiNNaker deployment when ready
# Run this in background: nohup bash monitor_and_deploy.sh &

cd /Users/kumar/Documents/University/Year3/thesisproject/snn-esc50
LOG="MONITOR_LOG.md"

echo "# Experiment Monitor Log" > $LOG
echo "Started: $(date)" >> $LOG
echo "" >> $LOG

ITERATION=0
CSF3_RHYTHM_DONE=false
CSF3_PRUNE_DONE=false
SPINNAKER_STARTED=false

while true; do
    ITERATION=$((ITERATION + 1))
