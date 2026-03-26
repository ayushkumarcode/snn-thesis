#!/bin/bash
#SBATCH --job-name=snn_batch3a
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/batch3a_%j.out
#SBATCH --error=logs/batch3a_%j.err

# Batch 3a: Novel technique experiments (5-fold each)
# Spiking-LEAF, ANN-to-SNN conversion, Stochastic Resonance (+Rhythm variant),
# Predictive Coding, Astrocyte

module load cuda/12.6.2 libs/cuda/12.8.1 python/3.13.1
source ~/scratch/snn-esc50/.venv/bin/activate
cd ~/scratch/snn-esc50
export PYTHONUNBUFFERED=1

echo "============================================"
echo "  BATCH 3a: Novel techniques"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "  Start: $(date)"
echo "============================================"

# 1. ANN-to-SNN conversion (no training, just inference — fast)
echo ""; echo ">>>>>>>>>> ann_to_snn_conversion ($(date)) <<<<<<<<<<"
python -m experiments.ann_to_snn_conversion --device cuda 2>&1
echo ">>>>>>>>>> ann_to_snn_conversion DONE ($(date)) <<<<<<<<<<"

# 2. Stochastic Resonance (standalone)
echo ""; echo ">>>>>>>>>> stochastic_resonance_training ($(date)) <<<<<<<<<<"
python -m experiments.stochastic_resonance_training --device cuda 2>&1
echo ">>>>>>>>>> stochastic_resonance_training DONE ($(date)) <<<<<<<<<<"

# 3. Stochastic Resonance + Rhythm (combined)
echo ""; echo ">>>>>>>>>> sr_rhythm ($(date)) <<<<<<<<<<"
python -m experiments.stochastic_resonance_training --with-rhythm --device cuda 2>&1
echo ">>>>>>>>>> sr_rhythm DONE ($(date)) <<<<<<<<<<"

# 4. Predictive Coding SNN
echo ""; echo ">>>>>>>>>> predictive_coding_snn ($(date)) <<<<<<<<<<"
python -m experiments.predictive_coding_snn --device cuda 2>&1
echo ">>>>>>>>>> predictive_coding_snn DONE ($(date)) <<<<<<<<<<"

# 5. Astrocyte SNN
echo ""; echo ">>>>>>>>>> astrocyte_snn ($(date)) <<<<<<<<<<"
python -m experiments.astrocyte_snn --device cuda 2>&1
echo ">>>>>>>>>> astrocyte_snn DONE ($(date)) <<<<<<<<<<"

# 6. Predictive Coding + Rhythm combo (modify pred coding to use rhythm neurons)
# Run via combo_experiment with predictive coding flag if available, else skip

echo ""
echo "============================================"
echo "  BATCH 3a COMPLETE: $(date)"
echo "============================================"

echo ""
echo "=== RESULTS ==="
for dir in results/experiments/ann_to_snn_conversion results/experiments/stochastic_resonance_training results/experiments/sr_rhythm results/experiments/predictive_coding_snn results/experiments/astrocyte_snn; do
    s="$dir/summary.json"
    if [ -f "$s" ]; then
        python -c "import json; d=json.load(open('$s')); print(f'  {d.get(\"experiment\",\"?\"):<40} {d[\"mean_accuracy\"]*100:.2f}% ± {d[\"std_accuracy\"]*100:.2f}%')"
    else
        name=$(basename $dir)
        echo "  $name: check individual fold results"
    fi
done
