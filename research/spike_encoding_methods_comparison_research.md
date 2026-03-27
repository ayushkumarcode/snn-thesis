# Spike Encoding Methods: Systematic Comparison as a Thesis Topic

**Research Date:** 2026-02-25
**Scope:** Comprehensive investigation of spike encoding methods for SNNs, assessment of existing comparison studies, and viability as an undergraduate thesis topic

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Complete Taxonomy of Spike Encoding Methods](#2-complete-taxonomy-of-spike-encoding-methods)
3. [Impact of Encoding Choice on SNN Performance](#3-impact-of-encoding-choice-on-snn-performance)
4. [Existing Systematic Comparison Studies](#4-existing-systematic-comparison-studies)
5. [Which Encoding Works Best for Which Data Type](#5-which-encoding-works-best-for-which-data-type)
6. [Implementation in snnTorch](#6-implementation-in-snntorch)
7. [Thesis Viability Assessment](#7-thesis-viability-assessment)
8. [Research Gaps and Novel Contribution Opportunities](#8-research-gaps-and-novel-contribution-opportunities)
9. [Proposed Thesis Structure](#9-proposed-thesis-structure)
10. [Key Papers Reference Table](#10-key-papers-reference-table)
11. [Sources](#11-sources)

---

## 1. Executive Summary

Spike encoding -- the process of converting real-valued data into spike trains for processing by spiking neural networks -- is a fundamental and still actively researched problem in neuromorphic computing. There are at least 6-8 major encoding families (rate, latency/TTFS, delta/temporal contrast, phase, burst, population/Gaussian receptive field, direct/learned, and binary), each with distinct trade-offs in accuracy, latency, energy efficiency, noise robustness, and hardware suitability.

**The critical finding from this research: several comparison studies already exist, but none are truly comprehensive.** Each existing study compares a subset of encodings on a narrow set of tasks (usually just MNIST/Fashion-MNIST, or just one sensor modality). No single study has systematically compared all major encoding methods across multiple data modalities (images, audio, time-series, event-driven) using a unified framework and consistent evaluation metrics. This gap represents a genuine and achievable undergraduate thesis contribution.

The encoding choice demonstrably matters -- accuracy differences of 3-5% between methods on the same task are common, while latency and energy consumption can differ by 4-7.5x. This is not a trivial question with a known answer; it is a live research area where a well-executed systematic study would be valued.

**Verdict: "Systematic Evaluation of Spike Encoding Methods for Spiking Neural Networks" is a strong, feasible undergraduate thesis topic** with clear novelty potential if scoped correctly (more data modalities, more encoding methods, unified framework, consistent metrics).

---

## 2. Complete Taxonomy of Spike Encoding Methods

Based on the comprehensive survey by Auge, Hille, Mueller, and Knoll (2021) in Neural Processing Letters, and supplemented by multiple other sources, here is the complete taxonomy.

### 2.1 Rate-Based Encoding

Information is embedded in the firing frequency of neurons. Robust against noise, simple to implement, but requires many timesteps and many spikes (energy-expensive).

| Method | Description | Key Property |
|--------|-------------|--------------|
| **Poisson Rate Coding** | Each input value is treated as the probability of a spike at each timestep (Bernoulli process). Higher values = more spikes on average. | Most common baseline; stochastic; high spike count |
| **Regular Rate Coding** | Deterministic variant where spikes are evenly spaced with frequency proportional to input value. | Lower variance than Poisson; easier to analyse |
| **Population Rate Coding** | A group of neurons collectively encodes a value through their combined firing rate. | Higher information capacity; uses more neurons |

### 2.2 Temporal/Latency-Based Encoding

Information is in the precise timing of spikes. A single spike carries much more meaning than in rate codes. Much fewer spikes needed, but more susceptible to noise.

| Method | Description | Key Property |
|--------|-------------|--------------|
| **Time-to-First-Spike (TTFS)** | Each neuron fires exactly once. Stronger inputs fire earlier, weaker inputs fire later. Based on LIF neuron RC model. | Very low spike count; fast inference; ~4x lower latency than rate coding |
| **Rank-Order Coding** | Only the relative ordering of spike times matters, not absolute times. | Robust to time distortions; loses amplitude info |
| **Inter-Spike Interval (ISI)** | Information encoded in the time gap between consecutive spikes from the same neuron. | Compact encoding; good for periodic signals |

### 2.3 Delta Modulation / Temporal Contrast

Event-driven encoding that generates spikes only when the input signal changes by more than a threshold. Directly inspired by how biological retinas and DVS cameras work.

| Method | Description | Key Property |
|--------|-------------|--------------|
| **Simple Delta** | Spike when difference between consecutive timesteps exceeds threshold. Can optionally generate "off-spikes" for negative changes. | Natural for time-series; very sparse; event-driven |
| **Multi-Threshold Delta** | Multiple threshold levels for finer-grained encoding of change magnitude. | Better signal reconstruction; more spikes |
| **Sigma-Delta Modulation** | Accumulates error (sigma) and spikes when accumulated error exceeds threshold (delta). | Lower quantisation error; hardware-efficient |

### 2.4 Phase Coding

Information is encoded in spike patterns whose phases are correlated with internally generated background oscillations (inspired by theta oscillations in the hippocampus).

| Method | Description | Key Property |
|--------|-------------|--------------|
| **Phase Coding** | Input features determine the phase offset of spikes relative to a global oscillator. Higher values produce spikes at earlier phases. | Best noise resilience of all methods; periodic encoding; highest SOP cost |

### 2.5 Burst Coding

Information transmitted through rapid successive bursts of spikes within a short time window.

| Method | Description | Key Property |
|--------|-------------|--------------|
| **Burst Coding** | Number of spikes in a burst proportional to input strength. More reliable synaptic communication than single spikes. | Best fault tolerance; best compression efficacy; higher spike count than TTFS |

### 2.6 Population Coding with Gaussian Receptive Fields (GRF)

Each scalar input value is projected onto a population of neurons, each with a different Gaussian receptive field centre. The neuron whose centre is closest to the input fires earliest/most.

| Method | Description | Key Property |
|--------|-------------|--------------|
| **GRF Population Coding** | N neurons cover the input range with overlapping Gaussians. Activation level determines spike timing within each neuron. | High information capacity; requires multiple neurons per input feature |

### 2.7 Direct / Learned Encoding

A trainable neural network layer converts raw input into spike trains. The encoding is learned jointly with the rest of the network during training.

| Method | Description | Key Property |
|--------|-------------|--------------|
| **Direct Coding** | A trainable linear layer converts input pixels to floating-point values at each timestep; thresholding produces spikes. | Best accuracy with few timesteps; requires multi-bit first layer; less robust to adversarial attacks |
| **H-Direct (Homeostatic Direct)** | Improved direct coding with homeostasis mechanism to prevent encoding collapse. | Addresses training efficiency limitations of vanilla direct coding |

### 2.8 Signal-Reconstruction-Oriented Encodings (for FPGA/hardware)

These focus on accurate reconstruction of the original signal from the spike train, important for signal processing and hardware implementations.

| Method | Description | Key Property |
|--------|-------------|--------------|
| **Step Forward (SF)** | Adjusts a baseline threshold when signal crosses it. | Fastest encoding speed; lowest energy; unstable with abrupt transitions |
| **Ben's Spiker Algorithm (BSA)** | FIR filter deconvolution approach. | Good for square waves; very slow encoding speed |
| **Pulse Width Modulation (PWM)** | Compares signal against sawtooth carrier wave. | Poor reconstruction accuracy |
| **Binary Encoding** | Multi-bit binary representation of input value. | Best SNR (139dB with 10 bits); balanced noise resistance |

---

## 3. Impact of Encoding Choice on SNN Performance

### 3.1 The Impact Is Significant and Well-Documented

The choice of encoding method has a demonstrable and meaningful impact on SNN performance across every metric measured. This is not a marginal effect.

### 3.2 Accuracy Impact

From Guo et al. (2021), on a 2-layer STDP-trained SNN:

| Encoding | MNIST Accuracy | Fashion-MNIST Accuracy |
|----------|---------------|----------------------|
| Rate Coding | 87.46% | 68.29% |
| TTFS Coding | 88.57% | 71.31% |
| Phase Coding | 88.18% | 71.36% |
| Burst Coding | 88.39% | 71.27% |

Accuracy differences of ~1-3% on MNIST and ~3% on Fashion-MNIST between rate coding and temporal methods. On more complex datasets with deeper networks, Kim et al. (2022) found that direct coding achieves better accuracy than rate coding, especially with smaller numbers of timesteps (T=5-10).

From Bian et al. (2024), on IMU-based activity recognition:

| Encoding | Accuracy |
|----------|---------|
| Rate (Beta mapping) | 91.7% |
| TTFS (Log) | 89.2% |
| Binary (10-bit) | 89.6% |
| Multi-threshold Delta | 89.8% |

### 3.3 Latency Impact

From Guo et al. (2021), processing latency in milliseconds:

| Encoding | Training Latency (ms) | Inference Latency (ms) |
|----------|----------------------|----------------------|
| Rate Coding | 320 | 150 |
| TTFS Coding | 80 | 20 |
| Phase Coding | 90 | 30 |
| Burst Coding | 60 | 30 |

TTFS coding requires **4x lower training latency and 7.5x lower inference latency** compared to rate coding.

### 3.4 Synaptic Operations (Energy Proxy)

From Guo et al. (2021), SOPs x 10^8:

| Encoding | Training SOPs | Inference SOPs |
|----------|-------------|---------------|
| Rate Coding | 130.785 | 9.932 |
| TTFS Coding | 37.300 | 1.506 |
| Phase Coding | 690.072 | 57.798 |
| Burst Coding | 104.947 | 5.679 |

TTFS achieves **3.5x fewer SOPs in training and 6.5x fewer in inference** compared to rate coding. Phase coding is the worst performer at ~5x more SOPs than rate coding.

### 3.5 Noise Resilience

| Encoding | Input Noise Resilience | Synaptic Noise Tolerance |
|----------|----------------------|------------------------|
| Rate Coding | Moderate | Poor (worst at training) |
| TTFS Coding | Poor (worst) | Moderate |
| Phase Coding | **Best** (highest resilience) | Good |
| Burst Coding | Poor | **Best** (at 20% fault rate) |

### 3.6 Hardware Implementation Cost (NAND gates per module)

| Encoding | Hardware Cost |
|----------|-------------|
| Rate Coding | 316N |
| TTFS Coding | 340N + 1,703 (shared overhead) |
| Phase Coding | 76N (simplest -- just multiplexers and 8-bit registers) |
| Burst Coding | 544N (most expensive) |

### 3.7 Summary: No Single Best Encoding

Each encoding creates distinct trade-offs:
- **TTFS**: Best computational efficiency (latency + SOPs), worst noise resilience
- **Phase**: Best noise resilience, simplest hardware, worst SOPs
- **Burst**: Best fault tolerance and compression, most expensive hardware
- **Rate**: Robust baseline, best adversarial robustness, highest latency/SOPs

This multi-dimensional trade-off space is precisely what makes a systematic comparison thesis valuable.

---

## 4. Existing Systematic Comparison Studies

### 4.1 Paper-by-Paper Analysis of Existing Comparisons

This section catalogues every significant comparison study found, identifying what each covers and, critically, what each leaves out.

---

**Study 1: Guo, Fouda, Eltawil, Salama (2021)**
"Neural Coding in Spiking Neural Networks: A Comparative Study for Robust Neuromorphic Systems"
*Frontiers in Neuroscience, Vol. 15*

- **Encodings compared:** Rate, TTFS, Phase, Burst
- **Network:** 2-layer SNN with STDP (unsupervised)
- **Datasets:** MNIST, Fashion-MNIST
- **Metrics:** Accuracy, latency, SOPs, hardware cost, compression, noise resilience, fault tolerance
- **Strengths:** Most comprehensive multi-metric comparison found; includes hardware analysis
- **Limitations:** Only MNIST/Fashion-MNIST (image only); only STDP training (no surrogate gradient); no delta/direct/population encoding; only 2-layer shallow network
- **Source:** https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.638474/full

---

**Study 2: Kim, Park, Moitra, Bhattacharjee, Venkatesha, Panda (2022)**
"Rate Coding or Direct Coding: Which One is Better for Accurate, Robust, and Energy-efficient Spiking Neural Networks?"
*ICASSP 2022*

- **Encodings compared:** Rate (Poisson) vs. Direct (trainable layer)
- **Networks:** MLP, VGG5, VGG9
- **Datasets:** MNIST, CIFAR-10, CIFAR-100
- **Metrics:** Accuracy, adversarial robustness (FGSM, PGD), energy consumption
