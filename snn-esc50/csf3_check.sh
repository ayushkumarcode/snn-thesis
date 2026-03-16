#!/bin/bash
# Quick CSF3 job monitor
# Usage: bash csf3_check.sh [job_id]

JOB_ID=${1:-11782913}

expect -c "
set timeout 120
spawn ssh -o StrictHostKeyChecking=no r36859ak@csf3.itservices.manchester.ac.uk {echo '=== JOB STATUS ===' && squeue -u r36859ak 2>&1 && echo '=== LAST 30 LINES OF OUTPUT ===' && tail -30 ~/scratch/snn-esc50/logs/train_all_${JOB_ID}.out 2>/dev/null && echo '=== RESULTS SO FAR ===' && ls ~/scratch/snn-esc50/results/ 2>/dev/null && for f in ~/scratch/snn-esc50/results/*/summary.json ~/scratch/snn-esc50/results/*/*/summary.json; do echo \\\$f; cat \\\$f 2>/dev/null; done}
expect {
    \"Password:\" { send \"Ayushkumar040612!\r\"; exp_continue }
    \"password:\" { send \"Ayushkumar040612!\r\"; exp_continue }
    \"Passcode or option\" { send \"1\r\"; exp_continue }
    eof {}
    timeout { puts \"TIMEOUT\" }
}
"
