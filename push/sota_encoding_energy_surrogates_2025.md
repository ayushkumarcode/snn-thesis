# State-of-the-Art: Spike Encoding, Energy Efficiency, and Surrogate Gradients (2024--2026)

**Deep Research Report for COMP30040 Thesis**
**Date: 5 March 2026**

---

## Executive Summary

This report synthesizes the state-of-the-art literature (2024--2026) across three critical axes of spiking neural network (SNN) research: spike encoding methods, energy efficiency comparisons between SNNs and ANNs, and surrogate gradient functions. The investigation spans over 40 recent publications and establishes the context for the thesis findings on ESC-50 environmental sound classification with 7 encoding schemes.

**Key findings from the literature:**

1. **Encoding:** Direct/current encoding consistently outperforms rate coding at low timesteps across multiple benchmarks (Kim et al. ICASSP 2022; Practical Tutorial 2025). No prior work has benchmarked 7 encoding schemes on ESC-50---the thesis result is novel. The emerging consensus is that no single encoding dominates: the optimal choice depends on the application's accuracy, latency, energy, and robustness requirements.

2. **Energy:** The widely cited claim that "SNNs are inherently more efficient" has been substantially challenged. Dampfhoffer et al. (IEEE TECI 2023) show SNNs need spike sparsity between 0.15--1.38 spikes per synapse per inference to compete with efficient ANN implementations. Yang et al. (arXiv 2024) show spike rates must be below 6.4% to outperform quantized ANNs. The thesis finding that the SNN is 2.1x MORE expensive than the ANN in software simulation (976 vs 463 nJ) is consistent with these critical reassessments.

3. **Surrogate gradients:** Zenke & Vogels (Neural Computation 2021) established that surrogate gradient shape matters less than scale. The thesis's bimodal ablation result (spike_rate_escape/fast_sigmoid/atan succeed; STE/sigmoid/sfs/triangular fail) adds a novel empirical data point that challenges this claim---some functions categorically fail for audio classification tasks.

---

## Part 1: Spike Encoding Methods

### 1.1 Comprehensive Benchmark Papers (2024--2026)

#### (A) Practical Tutorial on Spiking Neural Networks (2025)

**Citation:** A Practical Tutorial on Spiking Neural Networks: Comprehensive Review, Models, Experiments, Software Tools, and Implementation Guidelines. Preprints/MDPI, 2025.

This is the most comprehensive recent benchmark. It evaluates multiple neuron models (LIF, sigma-delta) with multiple input encodings (direct, rate, temporal) across two datasets and five SNN frameworks (Intel Lava, SLAYER, SpikingJelly, Norse, PyTorch).

**Key results:**
