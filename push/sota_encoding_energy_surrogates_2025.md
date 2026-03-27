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
