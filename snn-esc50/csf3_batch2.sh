#!/bin/bash
#SBATCH --job-name=snn_exp_batch2
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/batch2_%j.out
#SBATCH --error=logs/batch2_%j.err

# Batch 2: Experiments 6-9, all 5 folds each
# knowledge_distillation, dendritic_snn, learnable_delays, cochleagram_experiment

module load cuda/12.6.2 libs/cuda/12.8.1 python/3.13.1
source ~/scratch/snn-esc50/.venv/bin/activate
cd ~/scratch/snn-esc50

export PYTHONUNBUFFERED=1

echo "============================================"
echo "  BATCH 2: Experiments 6-9 (all folds)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "  Start: $(date)"
echo "============================================"

EXPERIMENTS=("knowledge_distillation" "dendritic_snn" "learnable_delays" "cochleagram_experiment")

for exp in "${EXPERIMENTS[@]}"; do
    echo ""
    echo ">>>>>>>>>> $exp: ALL 5 FOLDS ($(date)) <<<<<<<<<<"
    python -m experiments.$exp --device cuda 2>&1
    echo ">>>>>>>>>> $exp: DONE ($(date)) <<<<<<<<<<"
done

echo ""
echo "============================================"
echo "  BATCH 2 COMPLETE: $(date)"
echo "============================================"

# Summary
echo ""
echo "=== RESULTS SUMMARY ==="
for exp in "${EXPERIMENTS[@]}"; do
    summary="results/experiments/$exp/summary.json"
    if [ -f "$summary" ]; then
        python -c "import json; d=json.load(open('$summary')); print(f'  {d.get(\"experiment\",\"?\"): <30} {d[\"mean_accuracy\"]*100:.2f}% ± {d[\"std_accuracy\"]*100:.2f}%')"
    else
        echo "  $exp: NO SUMMARY (check individual folds)"
    fi
done
