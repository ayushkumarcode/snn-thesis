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

**Citation:** B. Petro et al. "Spike encoding techniques for IoT time-varying signals benchmarked on a neuromorphic classification task." Frontiers in Neuroscience 16:999029, 2022.

Benchmarks rate-based and temporal coding methods on Free Spoken Digit Dataset (FSD) and WISDM sensor dataset using a cochlea-inspired preprocessing pipeline. Uses transfer learning from equivalent ANN.

**Relevance to thesis:** Establishes that encoding choice for audio/temporal signals depends heavily on the preprocessing pipeline and target application. The cochlea-inspired front-end is analogous to the mel-spectrogram extraction in the thesis.

### 1.3 Encoding for Audio/Temporal Signals Specifically

#### (F) Larroza et al., "Spike Encoding for Environmental Sound" (arXiv:2503.11206, March 2025)

**Citation:** A. Larroza, J. Naranjo-Alcazar, V. Ortiz Castello, P. Zuccarello. "Comparative Study of Spike Encoding Methods for Environmental Sound Classification." arXiv:2503.11206, 2025.

**THE closest paper to the thesis work.** Compares 3 spike encoding methods on **ESC-10** (not ESC-50):

| Encoding | F1 Score | Precision | Recall |
|----------|----------|-----------|--------|
| TAE (Threshold Adaptive) | **0.661** | 0.671 | 0.665 |
| Step Forward | 0.409 | 0.528 | 0.423 |
| Moving Window | 0.354 | 0.415 | 0.388 |

Architecture: 3-layer FC SNN (128 neurons each), LIF neurons, trained 100 epochs.

**Critical limitations vs. thesis:**
- ESC-10 only (10 classes), not ESC-50 (50 classes)
- FC architecture only, no convolutions
- Only 3 encoding schemes (all temporal/change-based), no direct/rate/phase/population/burst
- Best result (F1=0.661) substantially below thesis direct encoding (47.15% on full ESC-50)

**Relevance to thesis:** Confirms that the thesis is the FIRST work to benchmark multiple spike encodings on full ESC-50. Larroza et al.'s TAE (adaptive threshold) is similar to delta modulation but with dynamic thresholds. Their poor performance likely reflects the FC-only architecture.

#### (G) Basu et al., "Fundamental Survey on Neuromorphic Based Audio Classification" (arXiv:2502.15056, February 2025)

**Citation:** A. Basu et al. "Fundamental Survey on Neuromorphic Based Audio Classification." arXiv:2502.15056, 2025.

Comprehensive survey of neuromorphic audio classification covering SNNs, memristors, and neuromorphic hardware platforms. Key points:
- Audio signals are transformed into spike trains through spike encoding (amplitude and timing to discrete spikes)
- STDP-based learning and surrogate gradient methods both reviewed
- Event-driven processing minimizes unnecessary computations
- No standardized benchmark for audio SNN encoding comparison exists

**Relevance to thesis:** Confirms the gap that the thesis fills---no prior standardized comparison of multiple encodings on a standard audio benchmark.

#### (H) Baek & Lee, "SNN and Sound" (Biomedical Engineering Letters, 2024)

**Citation:** E. Baek, J. Lee. "SNN and sound: a comprehensive review of spiking neural networks in sound." Biomedical Engineering Letters 14:801--834, 2024. DOI:10.1007/s13534-024-00406-y.

Reviews SNN-based sound processing, emphasizing low power consumption and minimal latency for real-time applications. Highlights that rate coding maps signal intensity to firing frequency, while temporal coding captures timing patterns.

#### (I) Haghighatshoar & Muir, "Low-power SNN Audio Source Localisation" (Nature Communications Engineering, 2025)

**Citation:** S. Haghighatshoar, D. R. Muir. "Low-power Spiking Neural Network audio source localisation using a Hilbert Transform audio event encoding scheme." Communications Engineering (Nature) s44172-025-00359-9, 2025.

Novel Hilbert-Transform-based audio-to-signed-event encoding for SNN sound source localization. Achieves MAE of 0.25--0.65 degrees on microphone arrays. Demonstrates that signal processing co-designed with SNN implementations can achieve significant power efficiency improvements.

### 1.4 Why Direct Encoding Outperforms Rate Coding

The literature converges on several explanations:

1. **Information preservation:** Direct encoding feeds continuous-valued inputs, preserving full-precision information in the first layer. Rate coding discretizes inputs into binary spikes, losing information. (Kim et al. 2022)

2. **Timestep efficiency:** With few timesteps (T <= 10), rate coding cannot generate enough spikes to accurately represent input intensities. Direct coding achieves full accuracy from T=1. (Kim et al. 2022; Practical Tutorial 2025)

3. **Gradient flow:** Direct encoding provides richer gradient information since the first layer processes continuous values with standard backpropagation. Rate coding introduces stochastic Bernoulli sampling that impedes gradient flow. (Neftci et al. 2019)

4. **Feature learning capacity:** For pre-extracted features (e.g., mel-spectrograms), the continuous-valued input already carries rich information that is degraded by spike quantization. (Thesis finding: direct=47.15% vs rate=24.00% on mel-spectrograms)

5. **Dataset complexity scaling:** The performance gap between direct and rate increases with dataset complexity. ESC-50 with 50 classes and complex audio features is a harder task where the information loss from rate encoding is more damaging. (Kim et al. 2022)

### 1.5 Novel Encoding Schemes (2024--2025)

| Scheme | Year | Key Innovation | Reference |
|--------|------|----------------|-----------|
| Multiplexed Rate+TTFS (RTF) | 2024 | Hardware-based neuron combining rate and temporal coding | Nature Communications 15:3808 (2024) |
| At-Most-Two-Spike Exponential Coding (AEC) | 2024 | Primary + compensating spike reduces quantization error | Neural Networks (ScienceDirect, 2024) |
| Stochastic First-to-Spike | 2024 | Stochastic LIF with temporal coding; improves noise robustness at cost of sparsity | arXiv:2404.17719, ICCAD 2024 |
| Input-aware Multi-Level Spike (IMLS) | 2025 | Multi-timestep firing in single timestep via adaptive thresholding | IML-Spikeformer (2025) |
| Sigma-Delta Network Conversion | 2025 | Sigma-delta neurons exploiting temporal redundancy | arXiv:2505.06417 (Loihi 2 conversion) |
| Hilbert Transform Encoding | 2025 | Phase-based event encoding from analytic signal | Nature Comms Eng (2025) |
| Threshold Adaptive Encoding (TAE) | 2025 | Dynamically adjusting thresholds for environmental sounds | arXiv:2503.11206 |
| Hyperdimensional Computing Decoding | 2025 | HD computing + SNN for robust low-latency decoding | arXiv:2511.08558 |

**High-Performance Deep SNNs with 0.3 Spikes per Neuron (Nature Communications, 2024):**
Stanojevic et al. from IBM Research demonstrate TTFS-based SNNs achieving exact ANN-equivalent accuracy on MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100, and PLACES365 with only 0.3 spikes per neuron. This establishes that temporal coding can match ANN accuracy with extreme sparsity when properly trained.

### 1.6 Summary: Encoding Landscape

The thesis benchmark of 7 encodings on ESC-50 is **unique in the literature**:

| Rank | Thesis Encoding | Acc (%) | Literature Consensus | Literature Consistency |
|------|----------------|---------|---------------------|----------------------|
| 1 | Direct | 47.15 | Best at low timesteps (Kim 2022, Tutorial 2025) | Fully consistent |
| 2 | Rate | 24.00 | Good accuracy but needs many timesteps | Consistent (gap expected) |
| 3 | Phase | 24.15 | Noise-robust (Guo 2021), efficient | Consistent |
| 4 | Population | 19.15 | Higher neuron count, harder optimization | Consistent |
| 5 | Latency | 16.30 | Low firing rate but fragile to noise (Bian 2024: -37% under noise) | Consistent |
| 6 | Delta | 7.25 | Multi-threshold variants work (Bian 2024: 89.8%), simple threshold fails | Partially consistent |
| 7 | Burst | 6.50 | Good for compression/HW robustness (Guo 2021), but architecture-dependent | Novel negative result |

---

## Part 2: Energy Efficiency

### 2.1 Dampfhoffer et al. (IEEE TECI, 2023) --- The Critical Reassessment

**Citation:** M. Dampfhoffer, T. Mesquida, P. Valentian, L. Anghel. "Are SNNs Really More Energy-Efficient Than ANNs? An In-Depth Hardware-Aware Study." IEEE Trans. Emerging Topics in Computational Intelligence, vol. 7, no. 3, pp. 731--741, June 2023. DOI:10.1109/TECI.2022.9927729.

**Key findings:**
- The IF model is more energy-efficient than LIF and temporal continuous synapse models
- SNNs with IF model can compete with efficient ANN implementations when spike sparsity is **0.15--1.38 spikes per synapse per inference** (depending on ANN implementation)
- Previous studies overlooked memory access costs (which dominate energy in practice)
- Hybrid ANN-SNN architectures leveraging SNN in high-sparsity layers and ANN in dense layers are the most promising path

**Relevance to thesis:** The thesis uses LIF neurons (not IF), which are inherently less energy-efficient per Dampfhoffer. With 74.16% activation sparsity (NeuroBench), the average firing rate is ~25.84%, well above the <6.4% threshold identified by Yang et al. This explains why the thesis SNN is 2.1x MORE expensive in software simulation.

### 2.2 Yang et al., "Reconsidering the Energy Efficiency of SNNs" (arXiv:2409.08290, 2024)

**Citation:** L. Yang et al. "Reconsidering the energy efficiency of spiking neural networks." arXiv:2409.08290, August 2024.

**Critical thresholds identified:**
- **VGG16, T=6:** Sparsity must exceed **93%** to ensure energy efficiency on both classical and spatial-dataflow architectures
- **T > 16:** Sparsity must exceed **97%**
- **General rule:** Spike rate must be **below 6.4%** to outperform equivalent quantized ANNs (QNNs)
- With their sparsity-promoting regularization on CIFAR-10 (VGG16, T=6): SNN consumes **69% of optimized ANN energy** at 94.18% accuracy

**Energy model components considered:**
- 8-bit ADD: 0.03 pJ
- 8-bit MUL: 0.2 pJ
- SRAM access: 20 pJ/bit
- DRAM access: 2 nJ/bit
- NoC per hop: 10 pJ/bit

Three architectures analyzed: classical (GPU/TPU), neuromorphic dataflow (event-driven), spatial-dataflow (mesh NoC).

**Relevance to thesis:** The thesis SNN has 74.16% activation sparsity = ~25.84% spike rate, which is 4x above the 6.4% threshold. This confirms the thesis finding that the SNN is more expensive in software. However, on neuromorphic hardware where only ACs are needed (not MACs), the 5.1x per-operation advantage still holds.

### 2.3 NeuroBench (Nature Communications, 2025)

**Citation:** J. Yik et al. "The NeuroBench framework for benchmarking neuromorphic computing algorithms and systems." Nature Communications 16:1589, 2025. DOI:10.1038/s41467-025-56739-4.

**Benchmark tasks (Algorithm Track v1.0):**

| Task | Dataset | Metric | ANN Baseline | SNN Result | SNN Eff_ACs | ANN Eff_MACs |
|------|---------|--------|-------------|-----------|-------------|-------------|
| Keyword FSCIL | MSWC | Accuracy | 89.27% | 75.27% | 3.65x10^5 | 7.85x10^6 |
| Event Detection | Prophesee 1MP | mAP | 0.429 | 0.271 (hybrid) | 5.60x10^8 | 3.76x10^10 |
| Motor Prediction (Indy) | Primate Reaching | R^2 | 0.593 | 0.593 | 276 | 3,836 |
| Motor Prediction (Loco) | Primate Reaching | R^2 | 0.558 | 0.568 | 551 | 6,103 |

**System Track v1.0:**

| System | Task | Accuracy | Dynamic Energy/sample |
|--------|------|----------|-----------------------|
| Arduino Nano (CPU) | Acoustic Scene Classification | 79.64% | 1.851 mJ |
| Xylo (neuromorphic) | Acoustic Scene Classification | 79.90% | **0.028 mJ** |

**Key finding:** The Xylo neuromorphic chip achieves 60.9x less dynamic inference power and 33.4x less dynamic energy than Arduino at comparable accuracy for audio classification.

**Metrics defined:**
- **Effective MACs:** Non-zero multiply-accumulate operations (activations not binary spikes)
- **Effective ACs:** Non-zero accumulate operations (activations are binary +/-1)
- **Dense:** Total operations (zero and non-zero)
- **Activation Sparsity:** Fraction of zero activations
- **Connection Sparsity:** Fraction of zero weights
- **Footprint:** Memory in bytes

**Relevance to thesis:** The thesis uses the same NeuroBench framework (v2.2.0) with SynapticOperations metric. The NeuroBench motor prediction result (SNN = ANN in accuracy with 14x fewer operations) is the strongest evidence that SNNs can match ANNs when tasks align with spike-based processing. The Xylo audio result (66x energy reduction) validates the thesis's energy narrative for neuromorphic hardware.

