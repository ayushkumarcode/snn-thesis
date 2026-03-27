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
