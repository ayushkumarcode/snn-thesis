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
pip install spynnaker
python spinnaker/convert_weights.py
python spinnaker/extract_features.py
python spinnaker/run_on_spinnaker.py
```

## architecture

```
input: mel spectrogram (1, 64, 216)

Conv2d(1, 32, 3x3) -> BatchNorm -> MaxPool(2) -> LIF
Conv2d(32, 64, 3x3) -> BatchNorm -> MaxPool(2) -> LIF
AvgPool(4, 6) -> Flatten (2304)
Linear(2304, 256) -> LIF
Linear(256, 50) -> LIF -> output

ANN baseline: same thing but ReLU instead of LIF
~622K params total
```

## project structure

```
snn-esc50/
├── src/
│   ├── config.py          # hyperparameters and paths
│   ├── dataset.py         # ESC-50 loader + mel spectrograms
│   ├── encoding.py        # spike encodings (rate, delta, latency, direct)
│   ├── train.py           # training loop, 5-fold CV
│   ├── evaluate.py        # metrics, confusion matrices
│   ├── energy.py          # synops/MAC energy stuff
│   └── models/
│       ├── snn_model.py   # convolutional SNN (snnTorch)
│       └── ann_model.py   # convolutional ANN baseline
├── spinnaker/
│   ├── convert_weights.py
│   ├── extract_features.py
│   └── run_on_spinnaker.py
├── experiments/           # all the advanced experiment scripts
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_encoding_visualisation.ipynb
│   └── 03_results_analysis.ipynb
├── results/               # saved metrics, plots, models
├── csf3_results/          # raw results from CSF3
├── EXPERIMENT_LOG.md      # what happened (detailed)
├── DECISIONS.md           # why we did things
├── requirements.txt
└── README.md
```

## dependencies

- snnTorch 0.9.4 -- spiking neural network framework (on top of PyTorch)
- PyTorch 2.6+
- librosa -- audio processing, mel spectrograms
- sPyNNaker -- SpiNNaker interface (optional, hardware deployment)
- training hardware: NVIDIA A100-SXM4-80GB (CSF3)
- neuromorphic hardware: SpiNNaker1 (spinnaker.cs.man.ac.uk)
