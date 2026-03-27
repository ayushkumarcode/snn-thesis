# Spiking Neural Networks for Audio Processing: Keyword Spotting & Speech Command Recognition

## Comprehensive Research Report -- February 2025

---

## Executive Summary

Spiking Neural Networks (SNNs) for audio keyword spotting and speech command recognition have matured significantly in 2024-2025, reaching a point where they are a viable and compelling undergraduate thesis topic. The accuracy gap between SNNs and conventional ANNs has narrowed dramatically: state-of-the-art SNNs now achieve 96.9% on Google Speech Commands V2 (35-class), approaching the ANN ceiling of ~97-98%. Multiple open-source frameworks (snnTorch, SpikingJelly, sparch) provide well-documented starting points, and several complete implementations exist on GitHub with 300-600 lines of core Python code. The energy efficiency argument is substantiated by hardware benchmarks showing 10-200x lower energy per inference on neuromorphic hardware (Intel Loihi) versus conventional processors. This is a well-scoped, feasible thesis project with clear benchmarks, available code, and a strong research narrative around energy-efficient edge AI.

---

## 1. SNN vs ANN Accuracy on Google Speech Commands Dataset

### 1.1 Current State of the Art (as of early 2025)

| Model | Type | Dataset (Task) | Accuracy | Parameters | Year | Code Available |
|-------|------|----------------|----------|------------|------|----------------|
| **SpikCommander** | SNN (Spiking Transformer) | GSC V2 (35-class) | **96.92%** | 2.13M | 2025 | Yes |
| **SpikCommander** | SNN (Spiking Transformer) | GSC V2 (35-class) | **97.08%** (T=200) | 2.13M | 2025 | Yes |
| **SpikeSCR** | SNN (Hybrid Attention) | GSC V2 (35-class) | **95.60%** | ~3.3M | 2024 | Pending |
| **SIDC-KWS** | SNN (Conformer) | GSC V2 (12-class) | **96.8%** | -- | 2025 | -- |
| **Spiking LMUFormer** | SNN | GSC V2 (35-class) | **96.12%** | -- | 2024 | -- |
| **RadLIF (sparch)** | SNN (Recurrent) | GSC V2 (35-class) | **96.60%** | ~1M | 2022 | Yes |
| **adLIF (sparch)** | SNN (Non-recurrent) | GSC V2 (35-class) | **95.50%** | ~1M | 2022 | Yes |
| **LSNN** | SNN (Spiking RNN) | GSC V1 (12-class) | **91.2%** | -- | 2020 | Yes |
| **ED-sKWS** | SNN (Early Decision) | GSC V2 (35-class) | **93.04%** | 27.6K | 2024 | No |
| LMUFormer | ANN | GSC V2 (35-class) | 96.53% | -- | 2024 | -- |
| Attention-RNN | ANN | GSC V2 (20-class) | 94.5% | 202K | 2019 | -- |
| LSTM | ANN (Baseline) | GSC V1 (12-class) | 94.4% | -- | 2020 | Yes |
| CNN (Baseline) | ANN | GSC V1 (12-class) | 87.6% | -- | 2020 | Yes |

### 1.2 Key Accuracy Takeaways

- **The gap is nearly closed.** In 2020, the best SNN (LSNN at 91.2%) trailed the best ANN (LSTM at 94.4%) by ~3.2 percentage points on GSC 12-class. By 2025, SpikCommander achieves 96.92% (35-class), which surpasses many ANN baselines.
- **12-class task (simpler):** SNNs now routinely achieve 95-97% accuracy, matching or exceeding ANN baselines.
- **35-class task (harder):** Best SNNs achieve ~96.9%, within 1-2 points of the ANN ceiling.
- **Parameter efficiency:** SpikCommander achieves 96.71% with only 1.12M parameters. ED-sKWS achieves 93% with only 27.6K parameters -- orders of magnitude fewer than typical ANNs.

### 1.3 SHD (Spiking Heidelberg Digits) Benchmark

| Model | Type | SHD Accuracy | Parameters | Year |
|-------|------|-------------|------------|------|
| **SpikCommander** | SNN | **96.41%** | 0.19M | 2025 |
| **SpikeSCR** | SNN | **95.70%** | -- | 2024 |
| **SE-adLIF** | SNN | **95.81%** | 0.45M | 2024 |
| **RadLIF (sparch)** | SNN | **97.60%** | ~1M | 2022 |
| **adLIF (sparch)** | SNN | **97.40%** | ~1M | 2022 |
| Hardware deployment | SNN | **93.4%** | -- | 2024 |

### 1.4 SSC (Spiking Speech Commands) Benchmark

| Model | Type | SSC Accuracy | Parameters | Year |
|-------|------|-------------|------------|------|
| **SpikCommander** | SNN | **83.49%** | 2.13M | 2025 |
| **SpikeSCR** | SNN | **82.79%** | -- | 2024 |
| **RadLIF (sparch)** | SNN | **93.40%** | ~1M | 2022 |
| CNN (Cramer et al.) | ANN | 77.7% | -- | 2020 |
| GRU | ANN | 79.05% | -- | 2020 |

---

## 2. Frameworks and Tools Available

### 2.1 Framework Comparison

| Framework | Maintainer | Language | Backend | Audio Support | Tutorials | Difficulty | PyPI |
|-----------|-----------|----------|---------|--------------|-----------|------------|------|
| **snnTorch** | UCSC (Eshraghian) | Python | PyTorch | SHD loader built-in | 18 tutorials | Beginner-friendly | Yes |
| **SpikingJelly** | Peking Univ. | Python | PyTorch | Speech Commands example (594 LOC) | Extensive docs | Intermediate | Yes |
| **sparch** | Idiap Research | Python | PyTorch | SHD, SSC, GSC, HD | Minimal (research code) | Intermediate | No |
| **Norse** | Community | Python | PyTorch | No dedicated audio | Intro notebooks | Intermediate | Yes |
| **Lava** | Intel | Python | Custom | Loihi deployment | Good docs | Advanced | Yes |
| **BindsNET** | UMass | Python | PyTorch | No dedicated audio | Examples | Intermediate | Yes |
| **Tonic** | Community | Python | PyTorch | SHD, SSC loaders | Data loading tutorials | Beginner-friendly | Yes |
| **Rockpool** | SynSense | Python | PyTorch | WaveSense tutorial | Good docs | Intermediate | Yes |

### 2.2 snnTorch (Recommended for Beginners)

- **Website:** https://snntorch.readthedocs.io/
- **GitHub:** https://github.com/jeshraghian/snntorch
- **Key features:**
  - 18 tutorials covering neuron models, feedforward SNNs, training, surrogate gradients, neuromorphic datasets
  - Built-in SHD dataset loader via `snntorch.spikevision.spikedata.SHD`
  - Google Colab notebook support (no local GPU needed)
  - Active maintenance, good community
- **Audio-specific:** Has SHD dataset example, but no dedicated audio classification tutorial. The general tutorials are directly applicable.
- **Install:** `pip install snntorch`

### 2.3 SpikingJelly

- **GitHub:** https://github.com/fangwei123456/spikingjelly
- **Published in Science Advances** (high-quality, peer-reviewed framework)
- **Key features:**
  - Includes a complete 594-line Speech Commands audio recognition example
  - Supports both activation-based and timestep-based training
  - CuPy acceleration for faster training
  - Internal MelScale implementation
- **Audio-specific:** `spikingjelly/activation_based/examples/speechcommands.py` -- a complete convolutional SNN for 12-class GSC
- **Install:** `pip install spikingjelly` or build from source

### 2.4 sparch (Purpose-Built for Audio)

- **GitHub:** https://github.com/idiap/sparch
- **Paper:** "A Surrogate Gradient Spiking Baseline for Speech Command Recognition" (Frontiers in Neuroscience, 2022)
- **Key features:**
  - Purpose-built for SNN speech command recognition
  - Supports 4 datasets: SHD, SSC, HD, GSC
  - Implements 4 neuron types: LIF, RLIF, adLIF, RadLIF
  - Clean PyTorch module design
  - Command-line experiment runner
- **Best for:** Reproducing published results and running comparative experiments
- **Install:** Clone from GitHub

### 2.5 Tonic (Data Loading)

- **Website:** https://tonic.readthedocs.io/
- **Purpose:** PyTorch-compatible loader for neuromorphic datasets (analogous to torchvision)
- **Key features:**
  - SHD and SSC dataset support built-in
  - Transform pipeline for event-based data
  - Works seamlessly with snnTorch and SpikingJelly
- **Install:** `pip install tonic`

---

## 3. Undergraduate-Level Implementations

### 3.1 Available Implementations Ranked by Accessibility

| Repository | Accessibility | Framework | Accuracy | LOC (core) | Dataset |
|-----------|-------------|-----------|----------|------------|---------|
| **SpikingJelly speechcommands.py** | Good | SpikingJelly/PyTorch | Competitive | ~494 | GSC V1 (12-class) |
| **sparch** | Good | PyTorch | SOTA | ~500-800 | SHD, SSC, GSC |
| **GoogleSpeechCommandsRNN** | Moderate | TensorFlow 2 | 91.2% (SNN) | ~1000+ | GSC V1 (12-class) |
| **SCommander** | Moderate | SpikingJelly | 96.9% | ~800+ | SHD, SSC, GSC |
| **RSNN** | Difficult | TensorFlow 1.2 | -- | ~500 | Custom |

### 3.2 Recommended Path for an Undergraduate

**Phase 1: Learning (Weeks 1-4)**
1. Complete snnTorch tutorials 1-5 (neuron models, feedforward SNNs, training)
2. Complete snnTorch tutorial 7 (neuromorphic datasets with Tonic)
3. Load and explore the SHD dataset using Tonic

**Phase 2: Baseline Implementation (Weeks 5-8)**
4. Implement a basic LIF-based SNN on SHD using snnTorch (~200-300 lines)
5. Implement the same architecture using SpikingJelly for comparison
6. Train and evaluate, achieving ~90% on SHD as a baseline

**Phase 3: Speech Commands (Weeks 9-14)**
7. Adapt to Google Speech Commands V2 (12-class first, then 35-class)
8. Implement Mel-spectrogram preprocessing pipeline
9. Build convolutional SNN architecture
10. Compare with an equivalent ANN baseline

**Phase 4: Analysis & Writing (Weeks 15-20)**
11. Energy consumption estimation (synaptic operations counting)
12. Accuracy vs. energy tradeoff analysis
13. Parameter sensitivity study
14. Thesis writing

### 3.3 Estimated Code Complexity

| Component | Estimated Lines | Difficulty |
|-----------|----------------|------------|
| Data loading + preprocessing | 50-100 | Easy |
| SNN model definition | 50-100 | Moderate |
| Training loop | 80-150 | Moderate |
| Evaluation + metrics | 50-80 | Easy |
| Visualization + analysis | 50-100 | Easy |
| **Total core implementation** | **280-530** | -- |
| ANN baseline for comparison | 100-200 | Easy |
| Full project with utilities | 500-1000 | -- |

A minimal working SNN for SHD classification can be achieved in approximately **200-300 lines** of Python using snnTorch. A full thesis-quality implementation with preprocessing, training, evaluation, comparison baselines, and visualization would typically be **500-1000 lines**.

---

## 4. Dataset Comparison and Recommendations

### 4.1 Dataset Overview

| Dataset | Classes | Samples | Format | Pre-spiked | Size | Task | Availability |
|---------|---------|---------|--------|------------|------|------|-------------|
| **SHD** | 20 (digits 0-9, EN+DE) | ~10,420 | Spike trains | Yes | ~700 MB | Digit recognition | Free (Zenke Lab) |
| **SSC** | 35 (speech commands) | ~105,829 | Spike trains | Yes | ~6 GB | Command recognition | Free (Zenke Lab) |
| **GSC V2** | 35 (or 12 subset) | ~105,829 | Raw audio (16kHz) | No | ~2.3 GB | Command recognition | Free (TensorFlow) |
| **TIDIGITS** | 11 (digits 0-9 + "oh") | ~25,104 | Raw audio | No | ~500 MB | Digit recognition | Licensed (LDC) |

### 4.2 Recommendation for Thesis

**Primary dataset: SHD (Spiking Heidelberg Digits)**
- Already in spike format (no encoding pipeline needed)
- Small enough for rapid iteration (~10K samples)
- Well-established benchmarks for comparison
- 20 classes -- enough complexity for a thesis
- Built-in loader in snnTorch and Tonic
- Published state-of-the-art: 96.41% (SpikCommander) to 97.60% (RadLIF)

**Secondary dataset: GSC V2 (Google Speech Commands, 12-class subset)**
- Industry-standard benchmark
- Requires audio-to-spike encoding (adds thesis content)
- Large community with many baselines
- 12-class subset is manageable; 35-class is stretch goal
- Published SNN SOTA: ~96.9%

**Why NOT TIDIGITS:**
- Requires LDC license (may cost money or institutional access)
- Fewer published SNN benchmarks
- Less active research community

**Why NOT SSC alone:**
- Very large dataset (6GB, long training times)
- 35 classes is challenging for a first SNN project
- Better as a stretch goal after SHD
