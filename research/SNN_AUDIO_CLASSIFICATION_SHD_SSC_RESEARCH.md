# SNNs for Audio Classification: SHD & SSC Benchmarks

so i've been looking into whether SNNs are actually competitive for audio classification, specifically on the Spiking Heidelberg Digits (SHD) and Spiking Speech Commands (SSC) benchmarks. turns out they are -- like genuinely beating ANNs, which is kind of surprising honestly.

the best SNN methods on SHD are hitting around 96.4% accuracy, which blows past the best ANN baselines (92.4% CNN, 90.4% GRU). on the larger SSC benchmark, best SNNs get 83.5-86% vs the GRU baseline of 79%. the field moved fast since 2022 -- key innovations include learnable synaptic delays (DCLS-Delays, ICLR 2024), adaptive neuron models (adLIF, RadLIF, SE-adLIF), parameter-free attention (Pfa-SNN), and spiking transformers (SpikCommander). most of these have open-source code and train on a single GPU in minutes to hours. this seems really feasible for a 3rd year thesis.

---

## 1. The Datasets

### 1.1 Spiking Heidelberg Digits (SHD)

| Property | Value |
|----------|-------|
| Task | Spoken digit classification (0-9 in English and German) |
| Classes | 20 |
| Training samples | 8,156 |
| Test samples | 2,264 |
| Input channels | 700 (artificial cochlea model) |
| Data format | Spike trains with timestamps |
| Created by | Cramer et al. (2020), Zenke Lab, Heidelberg |
| Reference | [arXiv:1910.07407](https://arxiv.org/abs/1910.07407) |

SHD encodes spoken digit recordings into spike trains using "Lauscher," an artificial cochlea model that mimics the human inner ear. each sample is spike events across 700 frequency channels with precise temporal information.

### 1.2 Spiking Speech Commands (SSC)

| Property | Value |
|----------|-------|
| Task | Speech command classification |
| Classes | 35 |
| Total samples | ~100,000 |
| Base dataset | Google Speech Commands v0.2 |
| Input channels | 700 (same cochlea model) |
| Data format | Spike trains with timestamps |
| Created by | Same group (Cramer et al.) |

SSC is the bigger, harder sibling of SHD. 35 speech command classes, lots of speakers, same cochlea encoding.

### 1.3 Dataset Access

both datasets are available through multiple loaders:
- **Tonic** library: `pip install tonic` then `tonic.datasets.SHD('./data', train=True)` or `tonic.datasets.SSC('./data', split='train')`
- **snnTorch**: built-in `snntorch.spikevision.spikedata.SHD()`
- **Norse**: `norse.dataset.spiking_heidelberg`
- **sparch toolkit**: auto download via config
- **SNN-delays repo**: auto download and preprocessing
- **Direct download**: [Zenke Lab resources page](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/)

---

## 2. State-of-the-Art Results

### 2.1 SHD Leaderboard (as of February 2026)

| Rank | Model | Accuracy | Params | Time Steps | Recurrent | Year | Code Available |
|------|-------|----------|--------|------------|-----------|------|----------------|
| 1 | MCRE (Multi-Scale Chunked Residual Encoding) | **96.44%** | -- | -- | -- | 2025 | -- |
| 2 | SpikCommander (1L-8-128) | **96.41%** | 0.19M | 100 | No | 2025 | -- |
| 3 | Pfa-SNN (Parameter-free Attention) | **96.26%** | 0.20M | 100 | -- | 2025 | -- |
| 4 | Event-SSMA (ANN model) | 95.90% | 0.40M | -- | -- | -- | -- |
| 5 | SE-adLIF (2L) | 95.81% | 0.45M | 250 | Yes | 2024 | [GitHub](https://github.com/IGITUGraz/SE-adlif) |
| 6 | SpikeSCR (1L) | 95.60% | 0.26M | 100 | -- | 2025 | -- |
| 7 | DCLS-Delays (2L) | **95.07%** | 0.20M | 100 | **No** | 2024 | [GitHub](https://github.com/Thvnvtos/SNN-delays) |
| 8 | d-cAdLIF (2L) | 94.85% | 0.08M | 100 | -- | 2024 | -- |
| 9 | RadLIF (3x1024) | 94.62% | -- | -- | Yes | 2022 | [sparch](https://github.com/idiap/sparch) |
| 10 | adLIF (3x128) | 93.06% | -- | -- | No | 2022 | [sparch](https://github.com/idiap/sparch) |
| 11 | DH-SNN (2L) | 92.10% | 0.05M | 1000 | -- | -- | -- |
| 12 | Spikformer (1L) | 90.10% | 1.77M | 100 | No | -- | -- |
| 13 | SDT (1L) | 89.61% | 1.77M | 100 | No | -- | -- |

### 2.2 SSC Leaderboard (as of February 2026)

| Rank | Model | Accuracy | Params | Time Steps | Year | Code Available |
|------|-------|----------|--------|------------|------|----------------|
| 1 | SpikCommander (2L, T=250) | **85.98%** | 2.13M | 250 | 2025 | -- |
| 2 | SpikCommander (2L, T=200) | 85.52% | 2.13M | 200 | 2025 | -- |
| 3 | SpikCommander (2L, T=100) | 83.49% | 2.13M | 100 | 2025 | -- |
| 4 | SpikCommander (1L, T=100) | 83.26% | 1.12M | 100 | 2025 | -- |
| 5 | SpikeSCR (2L) | 82.79% | 3.30M | 100 | 2025 | -- |
| 6 | SpikeSCR (1L) | 82.54% | 1.71M | 100 | 2025 | -- |
| 7 | DH-SNN (3L) | 82.46% | 0.35M | 1000 | -- | -- |
| 8 | MCRE | 80.92% | -- | -- | 2025 | -- |
| 9 | DCLS-Delays (3L) | 80.69% | 2.50M | 100 | 2024 | [GitHub](https://github.com/Thvnvtos/SNN-delays) |
