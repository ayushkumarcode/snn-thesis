# Spiking Neural Networks for Audio Classification: SHD & SSC Benchmarks

**Research Date:** 2026-02-25
**Scope:** State-of-the-art survey of SNN audio classification on the Spiking Heidelberg Digits (SHD) and Spiking Speech Commands (SSC) benchmarks, with feasibility assessment for a 3rd-year undergraduate thesis project.

---

## Executive Summary

Spiking Neural Networks (SNNs) for audio classification have reached a mature and competitive state. On the Spiking Heidelberg Digits (SHD) benchmark, the best SNN methods now achieve **96.41-96.44% accuracy**, significantly surpassing the best non-spiking ANN baselines (92.4% CNN, 90.4% GRU). On the larger Spiking Speech Commands (SSC) benchmark, the best SNNs reach **83.49-85.98% accuracy**, also surpassing the non-spiking GRU baseline of 79.05%. The field has progressed rapidly since 2022, with key innovations including learnable synaptic delays (DCLS-Delays, ICLR 2024), adaptive neuron models (adLIF, RadLIF, SE-adLIF), parameter-free attention (Pfa-SNN), and spiking transformers (SpikCommander). Most state-of-the-art methods have open-source code, use PyTorch-based frameworks, and can be trained on a single GPU in minutes to hours. This makes SNN audio classification on SHD/SSC highly feasible and well-scoped for a 3rd-year undergraduate thesis project, with multiple clear research angles available.

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

The SHD dataset encodes spoken digit recordings into spike trains using "Lauscher," an artificial cochlea model that mimics the human inner ear and ascending auditory pathway. Each sample consists of spike events across 700 frequency channels with precise temporal information.

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

SSC is the larger, more challenging counterpart to SHD. It contains 35 speech command classes from a large number of speakers, converted to spike trains using the same cochlea model.

### 1.3 Dataset Access

Both datasets are available through multiple loaders:
- **Tonic** library: `pip install tonic` then `tonic.datasets.SHD('./data', train=True)` or `tonic.datasets.SSC('./data', split='train')`
- **snnTorch**: Built-in `snntorch.spikevision.spikedata.SHD()`
- **Norse**: `norse.dataset.spiking_heidelberg`
- **sparch toolkit**: Automatic download via config
- **SNN-delays repo**: Automatic download and preprocessing
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
| 10 | SE-adLIF (2L) | 80.44% | 1.60M | 250 | 2024 | [GitHub](https://github.com/IGITUGraz/SE-adlif) |
| 11 | d-cAdLIF (2L) | 80.23% | 0.70M | 100 | 2024 | -- |
| 12 | Pfa-SNN | 80.18% | 0.71M | 100 | 2025 | -- |
| 13 | DCLS-Delays (2L) | 80.16% | 1.40M | 100 | 2024 | [GitHub](https://github.com/Thvnvtos/SNN-delays) |
| 14 | Spikformer (2L) | 80.18% | 2.57M | 100 | -- | -- |
| 15 | SDT (2L) | 79.82% | 2.57M | 100 | -- | -- |
| 16 | RadLIF (3x1024) | 77.40% | -- | -- | 2022 | [sparch](https://github.com/idiap/sparch) |

### 2.3 Key Methods Explained

**DCLS-Delays (ICLR 2024)** -- The breakthrough paper. Uses Dilated Convolutions with Learnable Spacings to learn synaptic delays in feedforward SNNs. Each synapse gets a 1D Gaussian kernel whose position (representing the delay) is learned during training. Achieves 95.07% on SHD with only 2 feedforward layers of 256 LIF neurons each, **without recurrent connections**. The Gaussians narrow during training to produce discrete delays compatible with neuromorphic hardware. Open-source, clean implementation, and easy to reproduce.

**SE-adLIF (2024)** -- Uses an improved discretization scheme (Symplectic Euler) for adaptive Leaky Integrate-and-Fire neurons. The standard Euler-forward discretization introduces systematic errors; the SE scheme corrects this. Achieves 95.81% on SHD with recurrent connections.

**SpikCommander (2025)** -- A fully spike-driven transformer architecture using Multi-view Spiking Temporal-Aware Self-Attention (MSTASA) and Spiking Contextual Refinement Channel MLP (SCR-MLP). Achieves 96.41% on SHD with just 0.19M parameters and 83.49-85.98% on SSC.

**Pfa-SNN (2025)** -- Adds a parameter-free attention mechanism directly into the spiking neuron. No additional parameters required. Achieves 96.26% on SHD.

**MCRE (2025)** -- Multi-Scale Chunked Residual Encoding inspired by hippocampus-cortex information reorganization. Achieves 96.44% on SHD (current best) and 80.92% on SSC, while reducing energy consumption by up to 55%.

**RadLIF / adLIF (2022)** -- The surrogate gradient baseline from Bittar and Garner. RadLIF = Recurrent Adaptive LIF. adLIF = Adaptive LIF (non-recurrent). These established the competitive SNN baselines for SHD/SSC. Open-source via the sparch toolkit.

---

## 3. SNN vs. Traditional ANN Comparison

### 3.1 Accuracy Comparison on SHD

| Architecture | Type | Best SHD Accuracy | Notes |
|-------------|------|-------------------|-------|
| SpikCommander | SNN (Transformer) | 96.41% | 0.19M params |
| DCLS-Delays | SNN (Feedforward) | 95.07% | 0.20M params, no recurrence |
| RadLIF | SNN (Recurrent) | 94.62% | Surrogate gradient |
| **CNN (Cramer 2020)** | **ANN** | **92.4%** | **Best ANN baseline** |
| GRU (3x128) | ANN | 90.40% | Gated recurrent unit |
| liBRU (3x128) | ANN | 89.61% | Lightweight bistable RNN |
| LSTM | ANN | ~89% | Standard LSTM |

**Key finding: SNNs now SURPASS ANNs on SHD by a significant margin (96.4% vs 92.4%).** This is one of the few benchmarks where SNNs definitively outperform traditional deep learning approaches.

### 3.2 Accuracy Comparison on SSC

| Architecture | Type | Best SSC Accuracy | Notes |
|-------------|------|-------------------|-------|
| SpikCommander | SNN (Transformer) | 85.98% (T=250) | Current SOTA |
| SpikeSCR | SNN | 82.54% | Curriculum distillation |
| DCLS-Delays | SNN (Feedforward) | 80.69% | 3-layer, no recurrence |
| **GRU (3x512)** | **ANN** | **79.05%** | **Best ANN baseline** |
| liBRU (3x512) | ANN | 78.70% | Lightweight bistable RNN |
| CNN | ANN | 77.7% | Convolutional |
| RadLIF (3x1024) | SNN (Recurrent) | 77.40% | 2022 baseline |
| LSTM | ANN | ~73% | Standard LSTM |

**Key finding: On SSC, SNNs also surpass ANNs (85.98% vs 79.05%), but the gap took longer to open.** The 2022 SNN baseline (77.4%) was below the GRU (79.05%), but by 2025, SNNs lead convincingly.

### 3.3 Energy Efficiency

| Metric | SNN | ANN |
|--------|-----|-----|
| Inference energy (general) | 5-15 mJ | ~200 mJ |
| SynOps ratio vs ANN | 0.68x | 1.0x (baseline) |
| Energy-delay product | Up to 8.2x better | Baseline |
| Neuromorphic hardware (SpiNNaker) | ~0.3W | N/A |

SNNs achieve energy efficiency through sparse, event-driven computation. The advantage is most pronounced on neuromorphic hardware (Loihi, SpiNNaker, BrainScaleS), where SNNs can be 10-100x more energy efficient than equivalent ANNs on GPU. Even when both run on GPU, SNNs use fewer multiply-accumulate operations due to spike sparsity.

### 3.4 Why SNNs Win on These Audio Benchmarks

1. **Natural temporal affinity**: Audio is inherently temporal. SNN spike trains naturally encode timing information, unlike frame-based ANNs that must learn temporal structure.
2. **The data is already spike-encoded**: SHD/SSC data comes from an artificial cochlea model that outputs spike trains. SNNs process this natively; ANNs must convert it to dense tensors, losing temporal precision.
3. **Learnable delays**: Methods like DCLS-Delays exploit the temporal structure of spike trains by learning optimal signal propagation delays, something ANNs cannot do naturally.
4. **Biological plausibility**: The cochlea-to-spike encoding mirrors biological auditory processing, and SNNs process these signals in a biologically plausible manner.

---
