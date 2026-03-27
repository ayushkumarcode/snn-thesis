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
- **Strengths:** Larger datasets (CIFAR-10/100); deeper architectures; adversarial robustness analysis; code available on GitHub
- **Limitations:** Only 2 encoding methods; no temporal, phase, burst, delta, or population coding
- **Code:** https://github.com/Intelligent-Computing-Lab-Panda/Rate-vs-Direct
- **Source:** https://arxiv.org/abs/2202.03133

---

**Study 3: Forno, Fra, Pignari, Macii, Urgese (2022)**
"Spike encoding techniques for IoT time-varying signals benchmarked on a neuromorphic classification task"
*Frontiers in Neuroscience, Vol. 16*

- **Encodings compared:** Rate-based variants, temporal coding variants
- **Datasets:** Free Spoken Digit Dataset (audio, 8kHz), WISDM (IMU sensors, 20Hz)
- **Metrics:** Classification accuracy, spike density
- **Strengths:** Multi-modal (audio + sensor); IoT-focused; practical benchmarking
- **Limitations:** Specific to IoT signals; limited encoding method coverage
- **Source:** https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.999029/full

---

**Study 4: Bian, Donati, Magno (2024)**
"Evaluation of Encoding Schemes on Ubiquitous Sensor Signal for Spiking Neural Network"
*arXiv: 2407.09260*

- **Encodings compared:** Rate (uniform/normal/beta), TTFS (linear/log), Binary (6-bit/10-bit), Multi-threshold Delta
- **Dataset:** RecGym (IMU-based gym activity recognition)
- **Metrics:** Average firing rate, SNR, classification accuracy, robustness, energy, execution time
- **Strengths:** Most diverse metric set; includes deployment metrics; includes binary encoding
- **Limitations:** Single dataset (IMU only); no phase, burst, or population coding
- **Source:** https://arxiv.org/html/2407.09260v1

---

**Study 5: Plank et al. (2022)**
"Evaluating Encoding and Decoding Approaches for Spiking Neuromorphic Systems"
*ICONS 2022 (ACM)*

- **Encodings compared:** Rate, Temporal, Population, Spike encoding
- **Decoding approaches:** Also compared (voting, first-to-spike, etc.)
- **Tasks:** Classification, regression, control
- **Hardware:** Caspian neuromorphic processor (TENNLab)
- **Strengths:** Includes decoding (not just encoding); multi-task; actual neuromorphic hardware
- **Limitations:** Specific to Caspian architecture; limited encoding detail
- **Source:** https://dl.acm.org/doi/fullHtml/10.1145/3546790.3546792

---

**Study 6: Vasilache et al. (2025)**
"A PyTorch-Compatible Spike Encoding Framework for Energy-Efficient Neuromorphic Applications"
*arXiv: 2504.11026*

- **Encodings compared:** LIF, Step Forward, PWM, Ben's Spiker Algorithm
- **Test signals:** Vibration, trended, rectangular, sinusoidal (synthetic)
- **Metrics:** MSE reconstruction, energy efficiency, spike sparsity, encoding speed
- **Strengths:** Open-source PyTorch framework; hardware-oriented; includes parameter optimisation
- **Limitations:** Signal reconstruction only (no classification); synthetic signals only; no rate/TTFS/phase/burst
- **Source:** https://ar5iv.labs.arxiv.org/html/2504.11026

---

**Study 7: IEEE Sensors Journal (2023)**
"Comparison and Selection of Spike Encoding Algorithms for SNN on FPGA"

- **Encodings compared:** Sliding window, PWM-based, Step-forward, Ben's Spiker Algorithm
- **Focus:** FPGA implementation: calculation speed, resource consumption, accuracy, anti-noise ability
- **Strengths:** Practical FPGA selection criteria; scoring method for algorithm selection
- **Limitations:** Hardware-focused; no classification task evaluation
- **Source:** https://ieeexplore.ieee.org/document/10021878/

---

### 4.2 Gap Analysis: What Has NOT Been Done

| Gap | Description |
|-----|-------------|
| **No single study compares ALL major encoding methods** | Each study compares 2-4 methods; no study includes rate + TTFS + delta + phase + burst + population + direct |
| **No cross-modality study** | No study tests encodings across images AND audio AND time-series AND event-driven data |
| **No modern deep SNN architectures** | Guo et al. used 2-layer STDP; Kim et al. used VGG. No study compares encodings on modern architectures (ResNet-based SNNs, Transformer-based SNNs) |
| **No unified framework** | Each study uses different frameworks, neuron models, and hyperparameters, making cross-study comparison impossible |
| **No snnTorch-based comprehensive comparison** | snnTorch is the most popular educational SNN framework, but no study benchmarks all its encoding options |
| **No analysis of interaction between encoding and decoding** | Only Plank et al. touched this, but on limited hardware |
| **No population coding (GRF) vs. other methods comparison** | GRF is well-documented but rarely compared head-to-head with simpler encodings |

---

## 5. Which Encoding Works Best for Which Data Type

### 5.1 Summary Table

| Data Type | Best Encoding(s) | Why | Evidence |
|-----------|-----------------|-----|----------|
| **Static images (MNIST, CIFAR)** | Rate coding (baseline), Direct coding (best accuracy) | Pixel intensities map naturally to firing rates; direct coding learns optimal conversion | Kim et al. 2022; Guo et al. 2021 |
| **Time-series sensor data (IMU, IoT)** | Delta modulation, Rate with beta mapping | Delta naturally captures changes; rate captures magnitude | Bian et al. 2024; Forno et al. 2022 |
| **Audio / speech** | Temporal contrast, cochlea-inspired encoding | Audio is inherently temporal; cochlea model produces sparse spike trains | Forno et al. 2022; SHD dataset papers |
| **Event-driven data (DVS cameras)** | Already in spikes (no encoding needed), Delta for frame-based conversion | DVS data is natively event-driven | CIFAR10-DVS, DVS128 literature |
| **Noisy environments** | Phase coding | Highest resilience to input noise | Guo et al. 2021 |
| **Low-power / edge deployment** | TTFS, Delta modulation | Fewest spikes = lowest energy | Guo et al. 2021; Bian et al. 2024 |
| **Hardware with faults** | Burst coding | Best fault tolerance at 20% fault rate | Guo et al. 2021 |
| **Real-time / low-latency** | TTFS, Direct coding (few timesteps) | TTFS fires once; direct coding works with T=5-10 | Guo et al. 2021; Kim et al. 2022 |

### 5.2 The "No Free Lunch" Principle

No single encoding is optimal across all dimensions. The choice must be guided by the specific application priorities:
- If accuracy is paramount: **Direct coding** or **TTFS**
- If energy efficiency is paramount: **TTFS** or **Delta modulation**
- If noise robustness is paramount: **Phase coding**
- If hardware reliability is paramount: **Burst coding**
- If implementation simplicity is paramount: **Rate coding**

This inherent trade-off space is what makes the comparison thesis valuable -- practitioners need guidance on which encoding to use for their specific use case.

---

## 6. Implementation in snnTorch

### 6.1 Built-in Encodings in snnTorch (snntorch.spikegen)

snnTorch provides three encoding methods natively. Phase coding, burst coding, population coding (GRF), and direct coding must be implemented manually.

#### Rate Coding: `spikegen.rate()`

```python
import snntorch as snn
from snntorch import spikegen

# data_it: shape [batch x input_size], values in [0, 1]
# num_steps: number of simulation timesteps
# gain: multiplier to scale spike probability

spike_data = spikegen.rate(data_it, num_steps=100, gain=1.0)
# Output shape: [num_steps x batch x input_size]
# Each element is 0 or 1 (Bernoulli trial per timestep)
```

**Key parameters:**
- `num_steps` (int): Sequence length / number of timesteps
- `gain` (float, default=1.0): Scale factor for spike probability
- `offset` (float, default=0): Shift factor
- `first_spike_time` (int, default=0): Delay before first possible spike
- `time_var_input` (bool, default=False): Set True for time-varying inputs

#### Latency/TTFS Coding: `spikegen.latency()`

```python
spike_data = spikegen.latency(
    data_it,
    num_steps=100,
    tau=5,              # RC time constant (higher = slower firing)
    threshold=0.01,     # Below this, input is clipped to final timestep
    normalize=True,     # Span full time range
    linear=True,        # Linear (vs logarithmic) time mapping
    clip=True           # Remove sub-threshold spikes
)
# Output shape: [num_steps x batch x input_size]
# Each neuron fires AT MOST once
```

**Key parameters:**
- `tau` (float, default=1): RC time constant
- `threshold` (float, default=0.01): Minimum input to generate a spike
- `normalize` (bool): Normalize spike times to fill num_steps
- `linear` (bool): Linear vs. logarithmic encoding
- `clip` (bool): Remove sub-threshold spikes entirely

#### Delta Modulation: `spikegen.delta()`

```python
# data: shape [num_steps x batch x input_size] (time-series input)
spike_data = spikegen.delta(
    data,
    threshold=0.1,      # Change threshold for spike generation
    padding=False,       # First timestep handling
    off_spike=True       # Enable negative spikes (-1) for decreases
)
# Output shape: [num_steps x batch x input_size]
# Values are +1 (increase), -1 (decrease), or 0 (below threshold)
```

**Key parameters:**
- `threshold` (float, default=0.1): Magnitude of change required
- `padding` (bool): How to handle the first timestep
- `off_spike` (bool): Generate -1 for negative changes

#### Target Encoding: `spikegen.targets_convert()`

```python
# Encode target labels as spike trains for supervised learning
spike_targets = spikegen.targets_convert(
    targets,             # Class indices [0, C-1]
    num_classes=10,
    code='rate',         # 'rate' or 'latency'
    num_steps=100,
    correct_rate=0.8,    # Firing rate for correct class
    incorrect_rate=0.2   # Firing rate for incorrect classes
)
```

### 6.2 Custom Implementations Needed for Thesis

The following encodings are NOT in snnTorch and must be implemented as custom PyTorch functions. Here are implementation sketches.

#### Phase Coding (Custom Implementation)

```python
import torch
import numpy as np

def phase_encode(data, num_steps, num_phases=8):
    """
    Phase coding: encode input values as phase offsets
    relative to a global oscillator.

    Args:
        data: [batch x input_size], values in [0, 1]
        num_steps: number of timesteps
        num_phases: number of phase levels (resolution)

    Returns:
        spike_train: [num_steps x batch x input_size]
    """
    batch_size, input_size = data.shape
    spike_train = torch.zeros(num_steps, batch_size, input_size)

    # Create global oscillator (theta rhythm)
    period = num_steps // num_phases
    oscillator = torch.arange(num_steps) % period

    # Map input values to phase offsets
    # Higher values -> earlier phase (smaller offset)
    phase_offsets = ((1 - data) * (period - 1)).long()  # [batch x input]

    for t in range(num_steps):
        current_phase = t % period
        # Spike when current phase matches the neuron's phase offset
        spike_train[t] = (current_phase == phase_offsets).float()

    return spike_train
```

#### Burst Coding (Custom Implementation)

```python
def burst_encode(data, num_steps, max_burst_length=5, burst_gap=10):
    """
    Burst coding: encode input values as bursts of rapid spikes.

    Args:
        data: [batch x input_size], values in [0, 1]
        num_steps: number of timesteps
        max_burst_length: maximum spikes per burst
        burst_gap: minimum gap between burst windows

    Returns:
        spike_train: [num_steps x batch x input_size]
    """
    batch_size, input_size = data.shape
    spike_train = torch.zeros(num_steps, batch_size, input_size)

    # Number of spikes in burst proportional to input value
    burst_lengths = (data * max_burst_length).long().clamp(0, max_burst_length)

    # Place burst at beginning of each burst window
    num_windows = num_steps // burst_gap
    for w in range(num_windows):
        start = w * burst_gap
        for b in range(max_burst_length):
            t = start + b
            if t < num_steps:
                spike_train[t] = (b < burst_lengths).float()

    return spike_train
```

#### Population Coding with Gaussian Receptive Fields (Custom Implementation)

```python
def grf_population_encode(data, num_steps, num_neurons_per_feature=10,
                           tau=5, threshold=0.01):
    """
    Gaussian Receptive Field population coding.
