# snn-esc50

spiking neural network for environmental sound classification on ESC-50.
this is (as far as i can tell) the first SNN implementation on the full ESC-50 dataset -- no prior peer-reviewed SNN work exists for it (confirmed via arXiv 2503.11206).

COMP30040 undergrad thesis, university of manchester. ayush kumar, march 2026.

## results

| model | encoding | mean acc | std |
|-------|----------|----------|-----|
| ANN (baseline) | - | 63.85% | 3.07% |
| SNN | direct | **47.15%** | 4.50% |
| SNN | rate | 24.00% | 1.90% |
| SNN | latency | 16.30% | 1.62% |
| SNN | delta | 7.25% | 0.94% |

all 5-fold cross-validation on ESC-50. random chance = 2%.
trained on NVIDIA A100-SXM4-80GB (CSF3 cluster).

energy: SNN uses 1,358M pJ vs ANN 314M pJ per sample in software sim (4.3x more). SpiNNaker neuromorphic deployment done -- see EXPERIMENT_LOG.md for the whole story.

## setup

```bash
cd snn-esc50
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## usage

download ESC-50:
```python
from src.dataset import download_esc50
download_esc50()
```

train ANN baseline:
```bash
python -m src.train --model ann
```

train SNN with different encodings:
```bash
python -m src.train --model snn --encoding direct   # best one
python -m src.train --model snn --encoding rate
python -m src.train --model snn --encoding latency
python -m src.train --model snn --encoding delta
```

evaluate:
```bash
python -m src.evaluate --model snn --encoding direct
python -m src.evaluate --model ann
```

energy comparison:
```bash
python -c "from src.energy import save_energy_report; save_energy_report()"
```

### SpiNNaker deployment (optional)

needs a separate Python 3.11 venv with sPyNNaker and access to SpiNNaker hardware.

```bash
python -m venv .venv-spinnaker --python=python3.11
source .venv-spinnaker/bin/activate

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
