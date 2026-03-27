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

