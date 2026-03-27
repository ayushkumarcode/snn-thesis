# Spike Encoding Methods: Could a Comparison Study Work as a Thesis?

i went deep into spike encoding methods for SNNs -- how you convert real-valued data into spike trains -- and whether doing a comparison study would be a viable thesis topic.

the key finding: several comparison studies exist, but none of them are truly complete. each one compares a subset of encodings on a narrow set of tasks (usually just MNIST/Fashion-MNIST, or one sensor modality). nobody has done a proper comparison of all major encoding methods across multiple data types (images, audio, time-series, event-driven) using a unified framework with consistent evaluation metrics. that gap is genuine and achievable for an undergrad thesis.

encoding choice demonstrably matters -- accuracy differences of 3-5% between methods on the same task are common, while latency and energy can differ by 4-7.5x. this is not a trivial question with a known answer.

---

## Complete Taxonomy of Spike Encoding Methods

based on Auge, Hille, Mueller, and Knoll (2021) in Neural Processing Letters plus other sources.

### Rate-Based Encoding

information is in firing frequency. robust against noise, simple to implement, but needs many timesteps and lots of spikes (energy-expensive).

| Method | Description | Key Property |
|--------|-------------|--------------|
| **Poisson Rate Coding** | Each input value = probability of spike at each timestep (Bernoulli process). Higher value = more spikes on average. | Most common baseline; stochastic; high spike count |
| **Regular Rate Coding** | Deterministic variant, spikes evenly spaced with frequency proportional to input. | Lower variance; easier to analyze |
| **Population Rate Coding** | Group of neurons collectively encodes a value through combined firing rate. | Higher info capacity; uses more neurons |

### Temporal/Latency-Based Encoding

information is in precise timing of spikes. a single spike carries way more meaning than in rate codes. much fewer spikes needed, but more susceptible to noise.

| Method | Description | Key Property |
|--------|-------------|--------------|
| **Time-to-First-Spike (TTFS)** | Each neuron fires exactly once. Stronger inputs fire earlier. Based on LIF RC model. | Very low spike count; fast inference; ~4x lower latency than rate |
| **Rank-Order Coding** | Only relative ordering of spike times matters, not absolute times. | Robust to time distortions; loses amplitude info |
| **Inter-Spike Interval (ISI)** | Information in time gap between consecutive spikes from same neuron. | Compact; good for periodic signals |

### Delta Modulation / Temporal Contrast

event-driven encoding that generates spikes only when input changes by more than a threshold. inspired by how biological retinas and DVS cameras work.

| Method | Description | Key Property |
|--------|-------------|--------------|
| **Simple Delta** | Spike when diff between consecutive timesteps exceeds threshold. Optional "off-spikes" for decreases. | Natural for time-series; very sparse; event-driven |
| **Multi-Threshold Delta** | Multiple threshold levels for finer-grained encoding. | Better reconstruction; more spikes |
| **Sigma-Delta Modulation** | Accumulates error (sigma), spikes when accumulated error exceeds threshold (delta). | Lower quantization error; hardware-efficient |

### Phase Coding

input encoded in spike patterns whose phases correlate with internal oscillations (inspired by hippocampal theta oscillations).

| Method | Description | Key Property |
|--------|-------------|--------------|
| **Phase Coding** | Input features determine phase offset relative to global oscillator. Higher values = earlier phase spikes. | Best noise resilience; periodic; highest SOP cost |

### Burst Coding

information through rapid successive bursts within short time windows.

| Method | Description | Key Property |
|--------|-------------|--------------|
| **Burst Coding** | Number of spikes in burst proportional to input strength. More reliable synaptic communication than single spikes. | Best fault tolerance; best compression; higher spike count than TTFS |

### Population Coding with Gaussian Receptive Fields (GRF)

each scalar input projected onto a population of neurons with different Gaussian receptive field centres. neuron closest to input fires earliest/most.

| Method | Description | Key Property |
|--------|-------------|--------------|
| **GRF Population Coding** | N neurons cover input range with overlapping Gaussians. Activation determines spike timing. | High info capacity; requires multiple neurons per input feature |

### Direct / Learned Encoding

trainable layer converts raw input into spike trains. encoding learned jointly with network during training.

| Method | Description | Key Property |
|--------|-------------|--------------|
| **Direct Coding** | Trainable linear layer converts pixels to float values at each timestep; thresholding produces spikes. | Best accuracy with few timesteps; requires multi-bit first layer; less robust to adversarial attacks |
| **H-Direct (Homeostatic)** | Improved direct coding with homeostasis to prevent encoding collapse. | Addresses training efficiency limitations |

### Signal-Reconstruction-Oriented (for FPGA/hardware)

focused on accurate reconstruction from spike train, important for signal processing and hardware.

| Method | Description | Key Property |
|--------|-------------|--------------|
| **Step Forward (SF)** | Adjusts baseline threshold when signal crosses it. | Fastest encoding speed; lowest energy; unstable with abrupt transitions |
| **Ben's Spiker Algorithm (BSA)** | FIR filter deconvolution approach. | Good for square waves; very slow |
| **PWM** | Compare signal against sawtooth carrier wave. | Poor reconstruction accuracy |
| **Binary Encoding** | Multi-bit binary representation. | Best SNR (139dB with 10 bits); balanced noise resistance |

---

## Impact of Encoding Choice on Performance

### the impact is real and well-documented

the choice of encoding has a meaningful, measurable impact across every metric. this is not marginal.

### Accuracy

from Guo et al. (2021), on a 2-layer STDP-trained SNN:

| Encoding | MNIST Accuracy | Fashion-MNIST Accuracy |
|----------|---------------|----------------------|
| Rate | 87.46% | 68.29% |
| TTFS | 88.57% | 71.31% |
| Phase | 88.18% | 71.36% |
| Burst | 88.39% | 71.27% |

1-3% accuracy difference on MNIST, ~3% on Fashion-MNIST between rate and temporal methods. on deeper networks, Kim et al. (2022) found direct coding beats rate coding, especially with fewer timesteps (T=5-10).

from Bian et al. (2024), on IMU activity recognition:

| Encoding | Accuracy |
|----------|---------|
| Rate (Beta mapping) | 91.7% |
| TTFS (Log) | 89.2% |
| Binary (10-bit) | 89.6% |
| Multi-threshold Delta | 89.8% |

### Latency

from Guo et al. (2021):

| Encoding | Training Latency (ms) | Inference Latency (ms) |
|----------|----------------------|----------------------|
| Rate | 320 | 150 |
| TTFS | 80 | 20 |
| Phase | 90 | 30 |
| Burst | 60 | 30 |

TTFS needs **4x lower training latency and 7.5x lower inference latency** vs rate coding. that's a big deal.

### Synaptic Operations (Energy Proxy)

from Guo et al. (2021), SOPs x 10^8:

| Encoding | Training SOPs | Inference SOPs |
|----------|-------------|---------------|
| Rate | 130.785 | 9.932 |
| TTFS | 37.300 | 1.506 |
| Phase | 690.072 | 57.798 |
| Burst | 104.947 | 5.679 |

TTFS gets **3.5x fewer SOPs in training and 6.5x fewer in inference** vs rate. phase coding is the worst at ~5x MORE SOPs than rate. interesting trade-off.

### Noise Resilience

| Encoding | Input Noise Resilience | Synaptic Noise Tolerance |
|----------|----------------------|------------------------|
| Rate | Moderate | Poor (worst at training) |
| TTFS | Poor (worst) | Moderate |
| Phase | **Best** | Good |
| Burst | Poor | **Best** (at 20% fault rate) |

### Hardware Cost (NAND gates per module)

| Encoding | Hardware Cost |
|----------|-------------|
| Rate | 316N |
| TTFS | 340N + 1,703 (shared overhead) |
| Phase | 76N (simplest -- just muxes and 8-bit registers) |
| Burst | 544N (most expensive) |

### Bottom Line: No Single Best Encoding

each encoding creates different trade-offs:
- **TTFS**: best efficiency (latency + SOPs), worst noise resilience
- **Phase**: best noise resilience, simplest hardware, worst SOPs
- **Burst**: best fault tolerance and compression, most expensive hardware
- **Rate**: robust baseline, best adversarial robustness, highest latency/SOPs

this multi-dimensional trade-off space is exactly what makes a comparison thesis worth doing.

---

## Existing Comparison Studies

i went through every significant comparison study i could find and noted what each covers and what it leaves out.

**Study 1: Guo, Fouda, Eltawil, Salama (2021)** -- "Neural Coding in SNNs: A Comparative Study for Robust Neuromorphic Systems" (Frontiers in Neuroscience)
- Compared: Rate, TTFS, Phase, Burst
- Network: 2-layer SNN with STDP
- Datasets: MNIST, Fashion-MNIST
- Strengths: most complete multi-metric comparison i found, includes hardware analysis
- Gaps: only MNIST/Fashion-MNIST (image only), only STDP, no delta/direct/population, shallow network only
- Source: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.638474/full

**Study 2: Kim, Park, Moitra et al. (2022)** -- "Rate Coding or Direct Coding: Which One is Better?" (ICASSP 2022)
- Compared: Rate (Poisson) vs Direct (trainable layer)
- Networks: MLP, VGG5, VGG9
- Datasets: MNIST, CIFAR-10, CIFAR-100
- Strengths: larger datasets, deeper architectures, adversarial robustness, code on GitHub
- Gaps: only 2 encoding methods, no temporal/phase/burst/delta/population
- Code: https://github.com/Intelligent-Computing-Lab-Panda/Rate-vs-Direct

**Study 3: Forno, Fra, Pignari, Urgese (2022)** -- "Spike encoding techniques for IoT time-varying signals" (Frontiers in Neuroscience)
- Compared: rate-based variants, temporal coding variants
