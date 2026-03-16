# SNN for Environmental Sound Classification (ESC-50)

Spiking Neural Network applied to the ESC-50 environmental sound dataset.
First SNN implementation on ESC-50 (no prior peer-reviewed SNN work exists for this dataset, confirmed arXiv 2503.11206).

**COMP30040 Undergraduate Thesis, University of Manchester.**
**Author:** Ayush Kumar | **Date:** March 2026

## Key Results

| Model | Encoding | Mean Accuracy | Std |
|-------|----------|---------------|-----|
| ANN (baseline) | - | 63.85% | ±3.07% |
| SNN | Direct | **47.15%** | ±4.50% |
| SNN | Rate | 24.00% | ±1.90% |
| SNN | Latency | 16.30% | ±1.62% |
| SNN | Delta | 7.25% | ±0.94% |

All results from 5-fold cross-validation on ESC-50. Random chance = 2%.
Training performed on NVIDIA A100-SXM4-80GB GPUs (University of Manchester CSF3 cluster).

**Energy Analysis:** SNN uses 1,358M pJ vs ANN 314M pJ per sample in software simulation (4.3x more). Neuromorphic hardware deployment (SpiNNaker) in progress to measure real event-driven efficiency.

For detailed experiment documentation, see [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

## Setup

```bash
cd snn-esc50
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Download ESC-50

```python
from src.dataset import download_esc50
download_esc50()
```

### Train ANN Baseline

```bash
python -m src.train --model ann
```

### Train SNN (with encoding)

```bash
python -m src.train --model snn --encoding direct   # Best performing
python -m src.train --model snn --encoding rate
python -m src.train --model snn --encoding latency
python -m src.train --model snn --encoding delta
```

### Evaluate

```bash
python -m src.evaluate --model snn --encoding direct
python -m src.evaluate --model ann
```

### Energy Comparison

```bash
python -c "from src.energy import save_energy_report; save_energy_report()"
```

### SpiNNaker Deployment (Optional)

Requires separate Python 3.11 venv with sPyNNaker and access to SpiNNaker hardware.

```bash
python -m venv .venv-spinnaker --python=python3.11
source .venv-spinnaker/bin/activate
pip install spynnaker
python spinnaker/convert_weights.py    # Convert snnTorch weights
python spinnaker/extract_features.py   # Extract conv features
python spinnaker/run_on_spinnaker.py   # Run FC classifier on SpiNNaker
```

## Architecture

```
Input: Mel Spectrogram (1, 64, 216)

Conv2d(1, 32, 3x3) → BatchNorm → MaxPool(2) → LIF neuron
Conv2d(32, 64, 3x3) → BatchNorm → MaxPool(2) → LIF neuron
AvgPool(4, 6) → Flatten (2304)
Linear(2304, 256) → LIF neuron
Linear(256, 50) → LIF neuron → Output

ANN baseline: same architecture with ReLU instead of LIF
Total parameters: ~622K
```

## Project Structure

```
snn-esc50/
├── src/
│   ├── config.py          # All hyperparameters and paths
│   ├── dataset.py         # ESC-50 loader + mel-spectrogram pipeline
│   ├── encoding.py        # Spike encoding (rate, delta, latency, direct)
│   ├── train.py           # Training loop (SNN + ANN, 5-fold CV)
│   ├── evaluate.py        # Metrics, confusion matrices, per-class analysis
│   ├── energy.py          # SynOps/MAC energy comparison
│   └── models/
│       ├── snn_model.py   # Convolutional SNN (snnTorch)
│       └── ann_model.py   # Convolutional ANN baseline
├── spinnaker/
│   ├── convert_weights.py     # snnTorch → sPyNNaker weight conversion
│   ├── extract_features.py    # Conv feature extraction for hybrid deployment
│   └── run_on_spinnaker.py    # SpiNNaker inference script
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_encoding_visualisation.ipynb
│   └── 03_results_analysis.ipynb
├── results/                   # All saved metrics, plots, models
├── csf3_results/              # Raw results from CSF3 cluster
├── EXPERIMENT_LOG.md          # Detailed experiment documentation
├── requirements.txt
└── README.md
```

## Tools & Dependencies

- **snnTorch 0.9.4** -- Spiking neural network framework (built on PyTorch)
- **PyTorch 2.6+** -- Deep learning framework
- **librosa** -- Audio processing (mel spectrograms)
- **sPyNNaker** -- SpiNNaker interface (optional, for hardware deployment)
- **Training hardware:** NVIDIA A100-SXM4-80GB (CSF3 cluster)
- **Neuromorphic hardware:** SpiNNaker1 (spinnaker.cs.man.ac.uk)
