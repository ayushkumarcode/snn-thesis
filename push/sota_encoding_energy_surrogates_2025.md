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
- MNIST: Sigma-delta neurons with rate/sigma-delta encoding = 98.1% (ANN baseline: 98.23%)
- CIFAR-10: Sigma-delta neurons with direct input = 83.0% at just 2 timesteps (ANN baseline: 83.6%)
- Design rule: "intermediate thresholds and the minimal time window that still meets accuracy targets typically maximize efficiency"
- Many SNN configurations yield up to 3-fold energy efficiency vs matched ANNs

**Relevance to thesis:** Confirms the general pattern that direct encoding achieves highest accuracy, especially at low timesteps. The 2-timestep result on CIFAR-10 is remarkable.

#### (B) Kim et al., "Rate Coding or Direct Coding" (ICASSP 2022)

**Citation:** Y. Kim, H. Park, A. Moitra, A. Bhattacharjee, Y. Venkatesha, P. Panda. "Rate Coding or Direct Coding: Which One is Better for Accurate, Robust, and Energy-efficient Spiking Neural Networks?" ICASSP 2022. arXiv:2202.03133.

**Key findings---three-dimensional comparison:**
- **Accuracy:** Direct coding achieves better accuracy, especially for small timesteps. As timesteps increase, the gap narrows. As dataset complexity increases, the gap widens.
- **Robustness:** Rate coding shows better adversarial robustness due to the non-differentiable spike generation process.
- **Energy:** Rate coding yields higher energy-efficiency because direct coding requires multi-bit precision for the first layer (continuous inputs, not binary spikes).

**Relevance to thesis:** The thesis finding that direct encoding (47.15%) massively outperforms rate (24.00%) is consistent with Kim et al. The magnitude of the gap (23.15 pp) likely reflects the small dataset size (1,600 training samples) and complex audio features, which amplify the advantage of continuous direct input. The adversarial robustness finding (SNN more robust at eps=0.1: 26% vs 1.75%) is consistent with the rate-coding robustness result.

#### (C) Guo et al., "Neural Coding in Spiking Neural Networks" (Frontiers in Neuroscience, 2021)

**Citation:** W. Guo, M. E. Fouda, A. M. Eltawil, K. N. Salama. "Neural Coding in Spiking Neural Networks: A Comparative Study for Robust Neuromorphic Systems." Frontiers in Neuroscience 15:638474, 2021.

This paper compares 4 encoding schemes (rate, TTFS, phase, burst) using STDP-trained 2-layer SNNs on MNIST and Fashion-MNIST.

**Key rankings:**
- **Speed/efficiency:** TTFS best (4x/7.5x lower latency and 3.5x/6.5x fewer SOPs than rate coding)
- **Noise robustness:** Phase coding most resilient to input noise
- **Compression/hardware robustness:** Burst coding best for network compression and hardware non-idealities
- **Rate coding:** Worst accuracy loss under quantization

**Relevance to thesis:** The finding that phase coding is the most noise-resilient parallels the thesis observation that phase coding ties with rate coding (24.15% vs 24.00%)---deterministic single-spike-per-neuron achieves the same accuracy as stochastic multi-spike rate coding. Burst coding's advantage for hardware robustness is interesting given the thesis's burst coding failure (6.50%), though the failure mechanism (temporal front-loading) is specific to the architecture.

### 1.2 Papers Comparing 5+ Encoding Schemes

#### (D) Bian et al., "Evaluation of Encoding Schemes on Ubiquitous Sensor Signal for SNN" (2024)

**Citation:** S. Bian et al. "Evaluation of Encoding Schemes on Ubiquitous Sensor Signal for Spiking Neural Network." arXiv:2407.09260, July 2024. Also IEEE (10675361).

Compares 4 encoding families with multiple variants on the RecGym IMU dataset, with **Loihi 2 deployment**:

| Encoding | Accuracy | Avg Fire Rate | Loihi 2 Energy (mJ) | Robustness (acc drop at 0.1 noise) |
|----------|----------|---------------|----------------------|-------------------------------------|
| Rate (Beta) | **91.7%** | 49.9% | 250.15 | -9.5% |
| Rate (Normal) | 90.9% | 49.9% | 402.14 | -10.6% |
| Delta Modulation | 89.8% | 38.5% | 24.47 | **-0.7%** |
| Binary (10-bit) | 89.6% | 46.9% | **6.31** | -1.0% |
| TTFS (Logarithmic) | 89.2% | **2%** | 144.39 | -37.3% |
| TTFS (Linear) | 89.1% | **2%** | 144.39 | -37.3% |
| Binary (6-bit) | 86.5% | 33.3% | 8.87 | -2.5% |
| Rate (Uniform) | 85.4% | 49.9% | 436.51 | -11.1% |

**Key insight:** No single encoding wins across all metrics. Rate (Beta) wins accuracy, delta modulation wins robustness, binary wins energy, TTFS wins sparsity but worst robustness.

**Relevance to thesis:** This multi-dimensional tradeoff mirrors the thesis finding. The delta modulation result (best robustness) is interesting given the thesis's delta encoding performed poorly (7.25%)---likely because the thesis uses a simple threshold-based delta encoder rather than multi-threshold adaptive delta modulation. The TTFS fragility (37.3% accuracy drop under noise) is consistent with the thesis's latency encoding weakness (16.30%).

#### (E) Petro et al., "Spike Encoding Techniques for IoT Time-Varying Signals" (Frontiers in Neuroscience, 2022)
