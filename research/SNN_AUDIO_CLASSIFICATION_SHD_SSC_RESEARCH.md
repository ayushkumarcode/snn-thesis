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
