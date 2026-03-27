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
