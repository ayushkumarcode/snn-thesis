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
    echo "--- Iteration $ITERATION: $(date) ---" >> $LOG

    # Check CSF3 jobs
    CSF3_STATUS=$(ssh -o ControlPath=~/.ssh/csf3-socket -o ConnectTimeout=5 csf3 "squeue -u r36859ak -h 2>/dev/null" 2>/dev/null)

    if [ $? -ne 0 ]; then
        echo "  CSF3: SSH tunnel DOWN" >> $LOG
        echo "$(date): CSF3 tunnel down, waiting..." >> $LOG
        sleep 1200
        continue
    fi

    # Check rhythm_eval job
    if echo "$CSF3_STATUS" | grep -q "rhythm_eval"; then
        echo "  CSF3 rhythm_eval: RUNNING" >> $LOG
    elif [ "$CSF3_RHYTHM_DONE" = false ]; then
        echo "  CSF3 rhythm_eval: COMPLETED" >> $LOG
        CSF3_RHYTHM_DONE=true
        # Pull results
        echo "  Pulling rhythm eval results..." >> $LOG
        scp -o ControlPath=~/.ssh/csf3-socket csf3:~/scratch/snn-esc50/results/adversarial/rhythm_summary.json results/adversarial/ 2>/dev/null
        scp -o ControlPath=~/.ssh/csf3-socket csf3:~/scratch/snn-esc50/results/adversarial/rhythm_robustness_fold*.json results/adversarial/ 2>/dev/null
        scp -o ControlPath=~/.ssh/csf3-socket csf3:~/scratch/snn-esc50/results/energy/rhythm_neurobench_t25.json results/energy/ 2>/dev/null
        scp -o ControlPath=~/.ssh/csf3-socket csf3:~/scratch/snn-esc50/results/continual_learning/rhythm_summary_5fold.json results/continual_learning/ 2>/dev/null
        scp -o ControlPath=~/.ssh/csf3-socket csf3:~/scratch/snn-esc50/results/noise_robustness/rhythm_noise_5fold.json results/noise_robustness/ 2>/dev/null
        echo "  Results pulled." >> $LOG
    fi

    # Check prune_t1 job
    if echo "$CSF3_STATUS" | grep -q "prune_t1"; then
        echo "  CSF3 prune_t1: RUNNING" >> $LOG
    elif [ "$CSF3_PRUNE_DONE" = false ]; then
        echo "  CSF3 prune_t1: COMPLETED" >> $LOG
        CSF3_PRUNE_DONE=true
        # Pull pruned weights
        echo "  Pulling T=1 pruned weights..." >> $LOG
