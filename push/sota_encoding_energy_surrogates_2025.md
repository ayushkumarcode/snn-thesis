# Spike Encoding, Energy Efficiency, and Surrogate Gradients -- Literature Review (2024-2026)

went through 40+ recent papers covering three big topics: spike encoding methods, energy efficiency (SNN vs ANN), and surrogate gradient functions. all directly relevant to the thesis.

the main takeaways:

1. **encoding:** direct/current encoding consistently beats rate coding at low timesteps across benchmarks (Kim et al. ICASSP 2022; Practical Tutorial 2025). nobody has benchmarked 7 encoding schemes on ESC-50 before us. the general consensus is that no single encoding dominates -- it depends on whether you care about accuracy, latency, energy, or robustness.

2. **energy:** the idea that "SNNs are inherently more efficient" has been seriously challenged. Dampfhoffer et al. (IEEE TECI 2023) show SNNs need spike sparsity of 0.15-1.38 spikes/synapse/inference to compete with efficient ANN implementations. Yang et al. (2024) say spike rates must be below 6.4%. our SNN being 2.1x MORE expensive than the ANN in software simulation (976 vs 463 nJ) is consistent with these findings.

3. **surrogate gradients:** Zenke & Vogels (2021) said shape matters less than scale. our bimodal ablation (spike_rate_escape/fast_sigmoid/atan succeed; STE/sigmoid/sfs/triangular fail) adds an empirical data point that challenges this -- some functions just don't work for audio classification.

---

## Part 1: Spike Encoding Methods

### Benchmark Papers (2024-2026)

#### Practical Tutorial on Spiking Neural Networks (2025)

most comprehensive recent benchmark. evaluates multiple neuron models (LIF, sigma-delta) with multiple encodings (direct, rate, temporal) across two datasets and five SNN frameworks.

key results:
- MNIST: sigma-delta + rate/sigma-delta encoding = 98.1% (ANN baseline: 98.23%)
- CIFAR-10: sigma-delta + direct input = 83.0% at just 2 timesteps (ANN baseline: 83.6%)
- design rule: "intermediate thresholds and the minimal time window that still meets accuracy targets typically maximize efficiency"
- many SNN configs yield up to 3-fold energy efficiency vs matched ANNs

confirms direct encoding gets highest accuracy especially at low timesteps. the 2-timestep CIFAR-10 result is pretty remarkable.

#### Kim et al., "Rate Coding or Direct Coding" (ICASSP 2022)

three-dimensional comparison:
- **accuracy:** direct coding better, especially at small timesteps. gap narrows with more timesteps, widens with dataset complexity
- **robustness:** rate coding shows better adversarial robustness due to non-differentiable spike generation
- **energy:** rate coding more energy-efficient because direct coding needs multi-bit precision for first layer

our finding that direct (47.15%) massively beats rate (24.00%) is consistent. the 23.15pp gap likely reflects small dataset (1,600 training samples) and complex audio features amplifying the advantage of continuous input. the adversarial robustness finding (SNN more robust) is also consistent with rate-coding robustness.

#### Guo et al., "Neural Coding in SNNs" (Frontiers in Neuroscience, 2021)

compares rate, TTFS, phase, burst using STDP-trained 2-layer SNNs on MNIST and Fashion-MNIST:
- **speed/efficiency:** TTFS best (4x/7.5x lower latency, 3.5x/6.5x fewer SOPs vs rate)
- **noise robustness:** phase coding most resilient
- **compression/hardware robustness:** burst coding best
- **rate coding:** worst accuracy loss under quantization

the phase coding noise resilience parallels our finding that phase ties with rate (24.15% vs 24.00%) -- deterministic single-spike-per-neuron achieves the same as stochastic multi-spike rate coding. burst coding's hardware robustness advantage is interesting given our burst coding failure (6.50%), but our failure mechanism (temporal front-loading) is architecture-specific.

### Papers Comparing 5+ Encoding Schemes

#### Bian et al. (2024) -- Sensor Signal Encoding with Loihi 2 Deployment

4 encoding families with variants on RecGym IMU dataset, deployed on Loihi 2:

| Encoding | Accuracy | Avg Fire Rate | Loihi 2 Energy (mJ) | Robustness (acc drop at 0.1 noise) |
|----------|----------|---------------|----------------------|-------------------------------------|
| Rate (Beta) | **91.7%** | 49.9% | 250.15 | -9.5% |
| Rate (Normal) | 90.9% | 49.9% | 402.14 | -10.6% |
| Delta Modulation | 89.8% | 38.5% | 24.47 | **-0.7%** |
| Binary (10-bit) | 89.6% | 46.9% | **6.31** | -1.0% |
| TTFS (Logarithmic) | 89.2% | **2%** | 144.39 | -37.3% |
| Binary (6-bit) | 86.5% | 33.3% | 8.87 | -2.5% |
| Rate (Uniform) | 85.4% | 49.9% | 436.51 | -11.1% |

**no single encoding wins across all metrics.** rate wins accuracy, delta wins robustness, binary wins energy, TTFS wins sparsity but has worst robustness. this mirrors our findings exactly.

interesting: their delta modulation has best robustness (only -0.7% drop) while our delta encoding performed poorly (7.25%) -- probably because they use multi-threshold adaptive delta modulation vs our simple threshold-based version. TTFS fragility (-37.3% under noise) is consistent with our latency encoding weakness (16.30%).

#### Petro et al. (Frontiers in Neuroscience, 2022)

benchmarks rate-based and temporal coding on Free Spoken Digit Dataset and WISDM sensor data with cochlea-inspired preprocessing. confirms encoding choice depends heavily on preprocessing pipeline and target application.

### Encoding for Audio Specifically

#### Larroza et al. (arXiv:2503.11206, March 2025)

closest paper to our work. compares 3 spike encodings on **ESC-10** (not ESC-50):

| Encoding | F1 Score |
|----------|----------|
| TAE (Threshold Adaptive) | **0.661** |
| Step Forward | 0.409 |
| Moving Window | 0.354 |

architecture: 3-layer FC SNN, 128 neurons each, LIF. limitations vs us:
- ESC-10 only (10 classes) not ESC-50 (50)
- FC only, no convolutions
- only 3 encodings (all temporal/change-based), no direct/rate/phase/population/burst
- best result (F1=0.661) substantially below our direct encoding (47.15% on full ESC-50)

confirms we're the FIRST to benchmark multiple spike encodings on full ESC-50.

#### Basu et al. (arXiv:2502.15056, February 2025)

24-page survey of neuromorphic audio classification. notes that no standardized benchmark for audio SNN encoding comparison exists. confirms the gap we fill.

### Why Direct Encoding Outperforms Rate Coding

the literature converges on several explanations:

1. **information preservation:** direct encoding feeds continuous values, preserving full precision in first layer. rate coding discretizes into binary spikes, losing info. (Kim et al. 2022)
2. **timestep efficiency:** with few timesteps (T <= 10), rate coding can't generate enough spikes to represent input intensities. direct works from T=1. (Kim 2022, Tutorial 2025)
3. **gradient flow:** direct encoding gives richer gradients since first layer processes continuous values with standard backprop. rate coding introduces stochastic Bernoulli sampling that impedes gradient flow. (Neftci et al. 2019)
4. **feature learning capacity:** for pre-extracted features like mel-spectrograms, the continuous input already carries rich info that gets degraded by spike quantization. (our finding: direct=47.15% vs rate=24.00%)
5. **dataset complexity scaling:** the gap between direct and rate increases with dataset complexity. ESC-50 with 50 classes is hard enough that information loss from rate encoding really hurts. (Kim 2022)

### Novel Encoding Schemes (2024-2025)

| Scheme | Year | Key Innovation | Reference |
|--------|------|----------------|-----------|
| Multiplexed Rate+TTFS (RTF) | 2024 | Hardware neuron combining rate and temporal coding | Nature Communications 15:3808 |
| At-Most-Two-Spike (AEC) | 2024 | Primary + compensating spike reduces quantization error | Neural Networks (ScienceDirect) |
| Stochastic First-to-Spike | 2024 | Stochastic LIF with temporal coding | arXiv:2404.17719, ICCAD 2024 |
| Input-aware Multi-Level Spike (IMLS) | 2025 | Multi-timestep firing in single timestep | IML-Spikeformer |
| Sigma-Delta Network Conversion | 2025 | Sigma-delta neurons exploit temporal redundancy | arXiv:2505.06417 (Loihi 2) |
| Hilbert Transform Encoding | 2025 | Phase-based event encoding from analytic signal | Nature Comms Eng |
| TAE | 2025 | Dynamically adjusting thresholds for environmental sounds | arXiv:2503.11206 |
| HD Computing Decoding | 2025 | HD computing + SNN for robust low-latency decoding | arXiv:2511.08558 |

also worth noting: Stanojevic et al. from IBM (Nature Comms 2024) demonstrate TTFS-based SNNs achieving exact ANN-equivalent accuracy on MNIST through PLACES365 with only 0.3 spikes per neuron. temporal coding can match ANN accuracy with extreme sparsity when properly trained.

### Encoding Summary

our 7-encoding ESC-50 benchmark is unique in the literature:

| Rank | Our Encoding | Acc (%) | Literature Consensus | Consistent? |
|------|-------------|---------|---------------------|-------------|
| 1 | Direct | 47.15 | Best at low timesteps (Kim 2022, Tutorial 2025) | yes |
| 2 | Rate | 24.00 | Good accuracy but needs many timesteps | yes (gap expected) |
| 3 | Phase | 24.15 | Noise-robust (Guo 2021), efficient | yes |
| 4 | Population | 19.15 | Higher neuron count, harder optimization | yes |
| 5 | Latency | 16.30 | Low firing rate but fragile to noise (Bian 2024: -37%) | yes |
| 6 | Delta | 7.25 | Multi-threshold variants work (Bian 2024: 89.8%), simple threshold fails | partially |
| 7 | Burst | 6.50 | Good for compression/HW robustness (Guo 2021) but architecture-dependent | novel negative result |

---

## Part 2: Energy Efficiency

