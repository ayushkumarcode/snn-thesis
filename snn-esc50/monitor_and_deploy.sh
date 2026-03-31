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
        for pct in 50 55 60 65 70 75 80 85 90 95; do
            for fold in 1 2 3 4 5; do
                DIR="results/spinnaker_weights/pruned_t1_${pct}pct_fold${fold}"
                mkdir -p "$DIR"
                scp -o ControlPath=~/.ssh/csf3-socket csf3:~/scratch/snn-esc50/${DIR}/*.npy "$DIR/" 2>/dev/null
            done
        done
        # Pull energy results
        scp -o ControlPath=~/.ssh/csf3-socket csf3:~/scratch/snn-esc50/results/energy/neurobench_t1_pruned_sweep.json results/energy/ 2>/dev/null
        echo "  T=1 pruned weights pulled." >> $LOG
    fi

    # Start SpiNNaker if prune weights ready and not started yet
    if [ "$CSF3_PRUNE_DONE" = true ] && [ "$SPINNAKER_STARTED" = false ]; then
        if [ -f "results/spinnaker_weights/pruned_t1_50pct_fold1/fc2_weight.npy" ]; then
            echo "  Starting SpiNNaker T=1 pruned deployment..." >> $LOG
            SPINNAKER_STARTED=true
            # Run SpiNNaker deployment in background
            source .venv-spinnaker/bin/activate
            nohup bash spinnaker/deploy_all_pruned_t1.sh >> SPINNAKER_T1_LOG.txt 2>&1 &
            echo "  SpiNNaker deployment launched (PID: $!)" >> $LOG
        else
            echo "  Weights not yet available locally, retrying..." >> $LOG
        fi
    fi

    # Check if both CSF3 jobs done
    if [ "$CSF3_RHYTHM_DONE" = true ] && [ "$CSF3_PRUNE_DONE" = true ] && [ "$SPINNAKER_STARTED" = true ]; then
        # Check if SpiNNaker is still running
        if pgrep -f "deploy_all_pruned_t1" > /dev/null; then
            SPINN_COUNT=$(ls results/spinnaker_results/pruned_t1/fast_pruned*_400_N256.json 2>/dev/null | wc -l | tr -d ' ')
            echo "  SpiNNaker: running ($SPINN_COUNT/50 complete)" >> $LOG
        else
            echo "  ALL EXPERIMENTS COMPLETE" >> $LOG
            echo "Finished: $(date)" >> $LOG
            break
        fi
    fi

    # Also pull CSF3 SLURM output logs for debugging
    scp -o ControlPath=~/.ssh/csf3-socket csf3:~/scratch/snn-esc50/rhythm_eval_*.out /tmp/ 2>/dev/null
    scp -o ControlPath=~/.ssh/csf3-socket csf3:~/scratch/snn-esc50/prune_t1_*.out /tmp/ 2>/dev/null

    # Wait 20 minutes
    sleep 1200
done
