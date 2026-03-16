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

    Args:
        data: [batch x input_size], values in [0, 1]
        num_steps: number of timesteps
        num_neurons_per_feature: number of GRF neurons per input
        tau: time constant for latency conversion

    Returns:
        spike_train: [num_steps x batch x input_size * num_neurons_per_feature]
    """
    from snntorch import spikegen

    batch_size, input_size = data.shape
    n = num_neurons_per_feature

    # GRF centres evenly spaced in [0, 1]
    centres = torch.linspace(0, 1, n)  # [n]
    # GRF width (sigma) based on spacing
    sigma = 1.0 / (2 * (n - 1))

    # Compute activation for each neuron
    # data: [batch x input_size] -> [batch x input_size x 1]
    # centres: [n] -> [1 x 1 x n]
    data_exp = data.unsqueeze(-1)  # [batch x input x 1]
    centres_exp = centres.unsqueeze(0).unsqueeze(0)  # [1 x 1 x n]

    # Gaussian activation
    activations = torch.exp(-0.5 * ((data_exp - centres_exp) / sigma) ** 2)
    # Shape: [batch x input_size x n]

    # Reshape to [batch x (input_size * n)]
    activations = activations.reshape(batch_size, input_size * n)

    # Convert activations to latency-coded spikes
    spike_train = spikegen.latency(
        activations, num_steps=num_steps, tau=tau,
        threshold=threshold, normalize=True, linear=True
    )

    return spike_train
```

#### Direct Coding (Trainable Layer)

```python
import torch.nn as nn

class DirectEncoder(nn.Module):
    """
    Direct coding: trainable layer that learns to generate spikes.
    Jointly trained with the SNN via surrogate gradients.
    """
    def __init__(self, input_size, num_steps):
        super().__init__()
        self.num_steps = num_steps
        # Trainable projection: input -> num_steps copies
        self.encoder = nn.Linear(input_size, input_size * num_steps, bias=True)

    def forward(self, x):
        # x: [batch x input_size]
        batch_size = x.shape[0]
        input_size = x.shape[1]

        encoded = self.encoder(x)  # [batch x input_size * num_steps]
        encoded = encoded.reshape(batch_size, self.num_steps, input_size)
        encoded = encoded.permute(1, 0, 2)  # [num_steps x batch x input_size]

        # Apply threshold to generate binary spikes
        # (using surrogate gradient for backprop)
        spikes = (encoded > 0.5).float()
        # Straight-through estimator for gradient
        spikes = spikes + encoded - encoded.detach()

        return spikes
```

### 6.3 Alternative Frameworks with More Built-in Encodings

| Framework | Built-in Encodings | Notes |
|-----------|-------------------|-------|
| **snnTorch** | Rate, Latency, Delta | Most tutorials; best for learning; PyTorch-based |
| **BindsNET** | Rate, Poisson, Rank-order, GRF/Binning | More biologically-oriented; STDP focus |
| **SpikingJelly** | Rate, Latency, Direct, Poisson | More complete; better for deep SNNs; Chinese documentation |
| **Norse** | Current-based, LIF-based, custom | Lower-level; maximum flexibility |
| **Lava (Intel)** | Custom (hardware-oriented) | For Loihi hardware; production focus |

For the thesis, **snnTorch is recommended** as the primary framework due to its tutorial ecosystem, PyTorch integration, and the fact that implementing the missing encodings constitutes part of the contribution.

---

## 7. Thesis Viability Assessment

### 7.1 Is This a Valid Thesis Topic?

**Yes, unambiguously.** Here is the evidence:

1. **Active research area:** Papers are still being published on encoding comparison in 2024-2025 (Bian et al. 2024, Vasilache et al. 2025), demonstrating that the question is not settled.

2. **Acknowledged research gap:** Plank et al. (2022) explicitly state: "There are a variety of commonly used input encoding approaches... but it is not clear which is the most appropriate approach or whether the choice has a significant impact on performance." Bian et al. (2024) note that "a systematic approach to quantitatively evaluate spike encoding performance is currently lacking."

3. **Practical relevance:** Every SNN practitioner must choose an encoding method, and there is no definitive guide. A well-organised comparison would be immediately useful to the community.

4. **Clear methodology:** The thesis follows an established pattern (systematic evaluation / benchmarking study) that is well-understood in computer science.

5. **Publishable potential:** If the comparison is broader than existing studies (more methods, more datasets, more metrics), it could be submitted to a workshop or conference (e.g., ICONS, NICE, or SNN workshops at NeurIPS/ICML).

### 7.2 What Would Make It Novel vs. Existing Work?

| Existing Work Covers | Your Thesis Could Add |
|----------------------|----------------------|
| 2-4 encoding methods per study | 6-8 encoding methods in one unified study |
| Single data modality per study | Multiple modalities (image + audio + time-series) |
| MNIST/Fashion-MNIST only | MNIST + CIFAR-10 + SHD (audio) + sensor data |
| 2-layer STDP networks | Modern deep SNNs with surrogate gradient training |
| Accuracy only, or accuracy + 1-2 metrics | Comprehensive metrics: accuracy, latency, spike count, energy proxy, noise robustness |
| Different frameworks per study | Unified snnTorch framework for all experiments |
| No practitioner guidelines | Decision framework / recommendation guide |

### 7.3 Feasibility for Undergraduate Level

| Aspect | Assessment |
|--------|-----------|
| **Technical difficulty** | Moderate. Implementing rate, latency, delta is trivial (snnTorch built-in). Phase, burst, GRF require custom code but are straightforward algorithms. Direct coding requires basic neural network knowledge. |
| **Computational requirements** | Low-Moderate. MNIST/Fashion-MNIST run on a laptop in minutes. CIFAR-10 needs a GPU but runs in hours. SHD is manageable. No HPC needed. |
| **Background knowledge** | Requires understanding of: LIF neuron model, surrogate gradients, basic signal processing. All learnable within a few weeks from snnTorch tutorials. |
| **Risk** | Low. The experiments are well-defined, reproducible, and not dependent on external resources. The worst case is that you confirm existing results (still valid as a replication + extension study). |
| **Timeline** | Achievable. Core experiments (6 encodings x 3 datasets x 3 metrics) could be completed in 8-12 weeks of coding + running experiments. |

### 7.4 Potential Concerns and Mitigations

| Concern | Mitigation |
|---------|-----------|
| "It's just a comparison, not original research" | Framing matters: "systematic evaluation" is a recognised contribution type in CS. You are creating new knowledge about trade-offs that do not exist in one place. Adding a novel recommendation framework or decision tree adds originality. |
| "Guo et al. already did this" | Guo et al. used only STDP on MNIST with 4 methods. You would use modern training (surrogate gradients), more datasets, and more methods. The overlap is partial, not total. |
| "The results might be obvious" | Current literature shows they are NOT obvious -- phase coding has best noise resilience but worst SOPs, TTFS has best efficiency but worst noise tolerance. These trade-offs are complex and data-dependent. |
| "Not enough novelty for a UK undergraduate thesis" | UK undergrad theses require demonstration of research methodology and critical analysis, not necessarily breakthrough discoveries. A well-executed systematic study with clear methodology, reproducible results, and a practical decision framework is strong. |

---

## 8. Research Gaps and Novel Contribution Opportunities

### 8.1 Primary Gaps (Highest Value Contributions)

1. **Unified cross-modality comparison:** Test the same 6+ encodings on image data (MNIST/CIFAR-10), audio data (SHD/Speech Commands), and time-series data (a sensor dataset) using the same network architecture and training procedure. No existing study does this.

2. **Encoding-architecture interaction:** Does the best encoding change when you change the network architecture (e.g., feedforward vs. recurrent vs. convolutional SNN)? No systematic study addresses this.

3. **Practitioner decision framework:** Create a flowchart or decision matrix that helps practitioners choose an encoding based on their data type, hardware constraints, and performance priorities. This does not exist as a standalone contribution.

### 8.2 Secondary Gaps (Nice-to-Have Extensions)

4. **Encoding + decoding interaction:** Which encoding-decoding combination works best? Only Plank et al. partially addressed this.

5. **Encoding impact on learning dynamics:** How does encoding choice affect training convergence speed, loss curves, and gradient flow?

6. **Sensitivity analysis:** How sensitive is each encoding to its hyperparameters (e.g., tau for latency, threshold for delta)?

7. **Information-theoretic analysis:** Measure mutual information between input signal and spike train for each encoding to quantify information loss.

### 8.3 Stretch Goals

8. **Neuromorphic hardware deployment:** If access to Intel Loihi or SpiNNaker is available, compare encodings on actual hardware.

9. **Hybrid encodings:** Propose and evaluate a hybrid that combines strengths of multiple methods.

10. **Adaptive encoding:** An encoding that switches strategy based on input characteristics.

---

## 9. Proposed Thesis Structure

### 9.1 Title Options

- "A Systematic Evaluation of Spike Encoding Methods for Spiking Neural Networks Across Data Modalities"
- "Comparing Spike Encoding Strategies: Performance Trade-offs in Modern Spiking Neural Networks"
- "Which Spike Encoding? A Benchmark Study Across Tasks, Architectures, and Metrics"

### 9.2 Chapter Outline

**Chapter 1: Introduction** (2,000 words)
- Motivation: why SNNs, why encoding matters
- Research questions (the 5 questions from above)
- Scope and contributions

**Chapter 2: Background and Literature Review** (4,000 words)
- SNN fundamentals (neuron models, learning rules)
- Taxonomy of encoding methods (Section 2 of this report)
- Review of existing comparison studies (Section 4 of this report)
- Gap identification

**Chapter 3: Methodology** (3,000 words)
- Encoding implementations (built-in + custom)
- Datasets: MNIST, CIFAR-10, SHD (Spiking Heidelberg Digits), one sensor dataset
- Network architecture: Convolutional SNN (same for all experiments)
- Training procedure: Surrogate gradient descent
- Evaluation metrics: accuracy, spike count, inference timesteps, estimated energy (SynOps), noise robustness
- Experimental design: controlled variables, repetitions, statistical significance

**Chapter 4: Results** (4,000 words)
- Per-dataset results tables
- Cross-dataset comparison figures
- Latency-accuracy trade-off curves
- Energy-accuracy trade-off curves
- Noise robustness analysis
- Statistical significance tests

**Chapter 5: Analysis and Discussion** (3,000 words)
- Which encoding wins on which metric and why?
- Data-type-specific recommendations
- Comparison with prior studies (Guo et al., Kim et al., Bian et al.)
- Limitations of this study
- Practitioner decision framework

**Chapter 6: Conclusion** (1,500 words)
- Summary of findings
- Contributions
- Future work (encoding-architecture interaction, hardware deployment)

**Total: ~17,500 words** (appropriate for UK undergraduate thesis)

### 9.3 Experimental Matrix

| Encoding | MNIST | CIFAR-10 | SHD (Audio) | Sensor (TBD) |
|----------|-------|----------|-------------|---------------|
| Rate (Poisson) | X | X | X | X |
| Latency (TTFS) | X | X | X | X |
| Delta Modulation | X | X | X | X |
| Phase Coding | X | X | X | X |
| Burst Coding | X | X | X | X |
| GRF Population | X | X | X | X |
| Direct (Learned) | X | X | X | X |

= 7 encodings x 4 datasets x 5+ metrics x 3 repetitions = 420+ experiment runs

Each run on MNIST/Fashion-MNIST takes ~5-15 minutes. CIFAR-10 takes ~30-60 minutes. SHD takes ~15-30 minutes. Total compute: approximately 50-100 GPU-hours, well within a personal GPU or free Google Colab tier over several weeks.

---

## 10. Key Papers Reference Table

| # | Authors | Year | Title | Venue | Key Contribution | Encodings Covered |
|---|---------|------|-------|-------|-----------------|-------------------|
| 1 | Guo, Fouda, Eltawil, Salama | 2021 | Neural Coding in SNNs: A Comparative Study for Robust Neuromorphic Systems | Frontiers in Neuroscience | Most comprehensive multi-metric comparison | Rate, TTFS, Phase, Burst |
| 2 | Kim, Park, Moitra et al. | 2022 | Rate Coding or Direct Coding: Which One is Better? | ICASSP 2022 | Rate vs. Direct on larger datasets | Rate, Direct |
| 3 | Forno, Fra, Pignari et al. | 2022 | Spike encoding techniques for IoT time-varying signals | Frontiers in Neuroscience | Multi-modal IoT benchmarking | Rate variants, Temporal variants |
| 4 | Bian, Donati, Magno | 2024 | Evaluation of Encoding Schemes on Ubiquitous Sensor Signal | arXiv | Most diverse metrics including deployment | Rate, TTFS, Binary, Delta |
| 5 | Plank et al. | 2022 | Evaluating Encoding and Decoding Approaches for Spiking Neuromorphic Systems | ICONS (ACM) | Encoding + decoding on neuromorphic hardware | Rate, Temporal, Population |
| 6 | Vasilache et al. | 2025 | A PyTorch-Compatible Spike Encoding Framework | arXiv | Open-source encoding framework | LIF, SF, PWM, BSA |
| 7 | Auge, Hille, Mueller, Knoll | 2021 | A Survey of Encoding Techniques for Signal Processing in SNNs | Neural Processing Letters | Definitive survey/taxonomy of encoding methods | All major categories |
| 8 | Petro, Kasabov, Kiss | 2019 | Selection and Optimization of Temporal Spike Encoding Methods | IEEE Trans Neural Networks | Temporal encoding optimisation | BSA, SF, SW, other temporal |

---

## 11. Sources

### Survey Papers
- [A Survey of Encoding Techniques for Signal Processing in Spiking Neural Networks (Auge et al., 2021)](https://link.springer.com/article/10.1007/s11063-021-10562-2)
- [Neural Coding in Spiking Neural Networks: A Comparative Study (Guo et al., 2021)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.638474/full)
- [Spiking Neural Networks in Imaging: A Review and Case Study (2025)](https://www.mdpi.com/1424-8220/25/21/6747)

### Encoding Comparison Studies
- [Rate Coding or Direct Coding: Which One is Better? (Kim et al., 2022)](https://arxiv.org/abs/2202.03133)
- [Spike encoding techniques for IoT time-varying signals (Forno et al., 2022)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.999029/full)
- [Evaluation of Encoding Schemes on Ubiquitous Sensor Signal (Bian et al., 2024)](https://arxiv.org/html/2407.09260v1)
- [Evaluating Encoding and Decoding Approaches for Spiking Neuromorphic Systems (Plank et al., 2022)](https://dl.acm.org/doi/fullHtml/10.1145/3546790.3546792)
- [Comparison and Selection of Spike Encoding Algorithms for SNN on FPGA (2023)](https://ieeexplore.ieee.org/document/10021878/)
- [A PyTorch-Compatible Spike Encoding Framework (Vasilache et al., 2025)](https://ar5iv.labs.arxiv.org/html/2504.11026)
- [Selection and Optimization of Temporal Spike Encoding Methods (Petro et al., 2019)](https://ieeexplore.ieee.org/document/8689349/)

### Rate vs. Temporal / Direct Coding
- [Rate Coding or Direct Coding GitHub Repository](https://github.com/Intelligent-Computing-Lab-Panda/Rate-vs-Direct)
- [First-spike coding promotes accurate and efficient SNNs (2023)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1266003/full)
- [Stochastic Spiking Neural Networks with First-to-Spike Coding (2024)](https://arxiv.org/html/2404.17719v2)
- [H-Direct: Homeostasis-aware Direct Spike Encoding (2024)](https://openreview.net/forum?id=QkDUdPRcma)

### snnTorch Implementation Resources
- [snnTorch Tutorial 1: Spike Encoding](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html)
- [snnTorch spikegen API Documentation](https://snntorch.readthedocs.io/en/latest/snntorch.spikegen.html)
- [snnTorch Population Coding Tutorial](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_pop.html)
- [snnTorch GitHub Repository](https://github.com/jeshraghian/snntorch)

### Datasets
- [Spiking Heidelberg Digits (SHD) Dataset](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/)
- [CIFAR10-DVS Dataset](https://www.semanticscholar.org/paper/CIFAR10-DVS:-An-Event-Stream-Dataset-for-Object-Li-Liu/e72b7962133921fa3e84299cd6a4a2aeb60bab19)

### Frameworks and Tools
- [Open Neuromorphic Software Guide](https://open-neuromorphic.org/neuromorphic-computing/software/)
- [SNN Library Benchmarks (Open Neuromorphic)](https://open-neuromorphic.org/blog/spiking-neural-network-framework-benchmarking/)
- [BindsNET GitHub Repository](https://github.com/BindsNET/bindsnet)

### General SNN Reviews
- [Spiking Neural Networks: Comprehensive Review (2025)](https://link.springer.com/article/10.1007/s12530-025-09755-0)
- [The Promise of SNNs for Ubiquitous Computing (2025)](https://arxiv.org/html/2506.01737v1)
- [Non-Traditional Input Encoding Schemes for Spiking Neuromorphic Systems (Oak Ridge)](https://www.osti.gov/servlets/purl/1607189)

---

## Appendix: Quick-Start Experimental Code Skeleton

```python
"""
Thesis Experiment Runner: Spike Encoding Comparison
Runs all encoding methods on a given dataset and collects metrics.
"""

import torch
import snntorch as snn
from snntorch import spikegen, surrogate, functional
import time

# ---- Encoding Functions ----

def encode_rate(data, num_steps):
    return spikegen.rate(data, num_steps=num_steps)

def encode_latency(data, num_steps):
    return spikegen.latency(data, num_steps=num_steps, tau=5,
                            threshold=0.01, normalize=True, linear=True)

def encode_delta(data, num_steps):
    # Repeat static image across timesteps, then apply delta
    repeated = data.unsqueeze(0).repeat(num_steps, 1, 1)
    return spikegen.delta(repeated, threshold=0.1, off_spike=True)

# Phase, Burst, GRF, Direct: use custom implementations from Section 6.2

# ---- Metrics Collection ----

def evaluate_encoding(encode_fn, data_loader, model, num_steps, device):
    """Evaluate a model with a specific encoding on a dataset."""
    correct = 0
    total = 0
    total_spikes = 0
    total_time = 0.0

    model.eval()
    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)

            # Encode
            t_start = time.time()
            spike_data = encode_fn(data, num_steps)
            t_encode = time.time() - t_start

            # Forward pass
            t_start = time.time()
            spk_rec, mem_rec = model(spike_data)
            t_infer = time.time() - t_start

            # Accuracy
            _, predicted = spk_rec.sum(0).max(1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

            # Spike count
            total_spikes += spike_data.sum().item()
            total_time += t_encode + t_infer

    accuracy = correct / total
    avg_spikes = total_spikes / total
    return {
        'accuracy': accuracy,
        'avg_spikes_per_sample': avg_spikes,
        'total_time': total_time,
    }

# ---- Main Experiment Loop ----

ENCODINGS = {
    'rate': encode_rate,
    'latency': encode_latency,
    'delta': encode_delta,
    # 'phase': encode_phase,
    # 'burst': encode_burst,
    # 'grf': encode_grf,
    # 'direct': encode_direct,  # (uses DirectEncoder module)
}

DATASETS = ['mnist', 'cifar10', 'shd']

results = {}
for dataset_name in DATASETS:
    results[dataset_name] = {}
    # data_loader, model = setup_dataset_and_model(dataset_name)
    for enc_name, enc_fn in ENCODINGS.items():
        # metrics = evaluate_encoding(enc_fn, data_loader, model, num_steps=100)
        # results[dataset_name][enc_name] = metrics
        pass

# Save results to CSV/JSON for analysis
```
