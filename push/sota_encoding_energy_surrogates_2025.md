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

### Dampfhoffer et al. (IEEE TECI, 2023) -- the critical reassessment

"Are SNNs Really More Energy-Efficient Than ANNs? An In-Depth Hardware-Aware Study."

key findings:
- IF model is more energy-efficient than LIF and temporal continuous synapse models
- SNNs with IF can compete with efficient ANNs when spike sparsity is **0.15-1.38 spikes/synapse/inference**
- previous studies overlooked memory access costs (which dominate energy in practice)
- hybrid ANN-SNN architectures leveraging SNN in high-sparsity layers are most promising

we use LIF neurons (not IF), inherently less efficient per Dampfhoffer. with 74.16% activation sparsity (NeuroBench), our average firing rate is ~25.84%, well above the <6.4% threshold. this explains why our SNN is 2.1x MORE expensive in software simulation.

### Yang et al., "Reconsidering the Energy Efficiency of SNNs" (arXiv:2409.08290, 2024)

critical thresholds:
- VGG16, T=6: sparsity must exceed **93%** for energy efficiency
- T > 16: sparsity must exceed **97%**
- general rule: spike rate must be **below 6.4%** to outperform equivalent quantized ANNs
- with their sparsity-promoting regularization on CIFAR-10: SNN uses 69% of optimized ANN energy at 94.18% accuracy

energy model components: 8-bit ADD (0.03 pJ), 8-bit MUL (0.2 pJ), SRAM (20 pJ/bit), DRAM (2 nJ/bit), NoC per hop (10 pJ/bit).

our SNN has ~25.84% spike rate, 4x above the 6.4% threshold. confirms SNN is more expensive in software. but on neuromorphic hardware where only ACs are needed (not MACs), the 5.1x per-operation advantage still holds.

### NeuroBench (Nature Communications, 2025)

benchmark tasks (Algorithm Track v1.0):

| Task | Metric | ANN Baseline | SNN Result | SNN Eff_ACs | ANN Eff_MACs |
|------|--------|-------------|-----------|-------------|-------------|
| Keyword FSCIL | Accuracy | 89.27% | 75.27% | 3.65x10^5 | 7.85x10^6 |
| Event Detection | mAP | 0.429 | 0.271 | 5.60x10^8 | 3.76x10^10 |
| Motor Prediction (Indy) | R^2 | 0.593 | 0.593 | 276 | 3,836 |
| Motor Prediction (Loco) | R^2 | 0.558 | 0.568 | 551 | 6,103 |

System Track: Xylo neuromorphic chip achieves **0.028 mJ** vs 1.851 mJ (Arduino CPU) for acoustic scene classification at comparable accuracy. that's 60.9x less power and 33.4x less energy.

we use the same NeuroBench framework (v2.2.0). the motor prediction result (SNN = ANN with 14x fewer ops) is strong evidence that SNNs can match ANNs when tasks align with spike-based processing. the Xylo audio result (66x energy reduction) validates our energy narrative.

### Horowitz Energy Model (ISSCC 2014)

standard reference for operation energy in 45nm CMOS:

| Operation | Energy (pJ) |
|-----------|-------------|
| 32-bit FP MUL | 3.7 |
| 32-bit FP ADD | 0.9 |
| 32-bit INT MUL | 3.1 |
| 32-bit INT ADD | 0.1 |
| 8-bit INT ADD | 0.03 |
| 8-bit INT MUL | 0.2 |
| SRAM read (32b) | 5.0 |
| DRAM read (32b) | 640 (~128x more than compute) |

commonly cited SNN values:
- MAC (multiply-accumulate) at 32-bit: ~4.6 pJ (3.7 + 0.9)
- AC (accumulate only) at 32-bit: ~0.9 pJ
- ratio: MAC/AC ~ 5.1x

we use AC=0.9 pJ, MAC=4.6 pJ -- standard 32-bit FP from Horowitz.

### SNN vs ANN Energy on Audio Tasks

**Loihi 2:** keyword spotting 10x faster, 200x less energy than Jetson Orin Nano. RF neurons process spectral features directly, eliminating FFT preprocessing. sigma-delta neurons: 47x more efficient encoding.

**SpiNNaker 2:** 0.292 pJ/SOP. 10x neural simulation capacity per watt over SpiNNaker1. 22nm, 0.5V operation.

**Xylo (NeuroBench):** acoustic scene classification at 0.028 mJ/sample vs 1.851 mJ (Arduino CPU). **66x energy reduction.**

### AC vs MAC Energy Ratios

| Source | AC Energy | MAC Energy | Ratio | Technology |
|--------|-----------|------------|-------|------------|
| Horowitz (ISSCC 2014) | 0.9 pJ (32b FP) | 4.6 pJ (32b FP) | 5.1x | 45nm |
| Horowitz (ISSCC 2014) | 0.03 pJ (8b INT) | 0.2 pJ (8b INT) | 6.7x | 45nm |
| Yang et al. (2024) | 0.03 pJ (8b) | 0.2 pJ (8b) | 6.7x | 45nm |
| SpiNNaker 2 | 0.292 pJ/SOP | N/A | N/A | 22nm |
| TrueNorth | ~26 pJ/SOP | N/A | N/A | 28nm |
| Loihi 1 | ~23.6 pJ/SOP | N/A | N/A | 14nm |

important: these ratios only capture compute energy. memory access (SRAM: 5 pJ, DRAM: 640 pJ) often dominates total energy. Dampfhoffer and Yang both emphasize this.

### Sparsity-Energy Relationship

Shafique et al. (arXiv:2408.14437, 2024) -- two types of sparsity:
1. **static:** fixed zero weights (pruning), allows predetermined compression
2. **dynamic:** temporal event-based activations, requires flexible hardware

hardware accelerators exploiting sparsity (2024): MISS framework (36% energy improvement, 23% speedup), ESSA (253.1 GSOP/s at 75% weight sparsity on FPGA), SATA (exploits spike/gradient/membrane potential sparsity).

our SNN has 74.16% activation sparsity. on hardware that skips zero-spike computations, energy cost drops proportionally. the 5.1x per-op advantage * sparsity exploitation could yield big savings on neuromorphic hardware, even though software shows 2.1x disadvantage.

### Energy Summary

| Metric | Our SNN | Our ANN | Literature Context |
|--------|---------|---------|-------------------|
| Energy/sample (software) | 976 nJ | 463 nJ | SNN 2.1x MORE expensive -- consistent with Dampfhoffer/Yang |
| Effective ACs | 1.08M | 0 | Binary spike operations |
| Effective MACs | 0 | 101K | Multiply-accumulate operations |
| Activation sparsity | 74.16% | 59% | Below 93% threshold (Yang 2024) |
| Implied spike rate | ~25.8% | N/A | Above 6.4% threshold (Yang 2024) |
| Per-op cost (32b) | 0.9 pJ/AC | 4.6 pJ/MAC | 5.1x per-op advantage (Horowitz) |
| Hypothetical neuromorphic | ~190 nJ | 463 nJ | 2.4x cheaper IF hardware exploits sparsity |

---

## Part 3: Surrogate Gradients

### Zenke & Vogels (Neural Computation, 2021)

foundational paper. key claim: surrogate gradient learning is **robust to the shape** of the surrogate derivative, but **scale (steepness) substantially affects performance.** tested SuperSpike (fast sigmoid), standard sigmoid, and piecewise linear -- all worked comparably when scale was tuned.

our ablation partially contradicts this. some functions (sigmoid, STE, triangular, SFS) categorically fail. might be because they tested simpler tasks (XOR, MNIST-like) where shape genuinely doesn't matter, while ESC-50 audio classification is harder and more sensitive.

### Gygax & Zenke (Neural Computation, 2025)

theoretical foundation paper. investigates relation of surrogate gradients to: (1) smoothed probabilistic models providing derivatives for single neurons, and (2) stochastic automatic differentiation. key finding: spike_rate_escape is theoretically justified as the derivative of the neuronal **escape noise function** (Boltzmann distribution).

this might explain why SRE outperforms other surrogates (46.00% vs 44.75% for fast_sigmoid) -- it's the most theoretically justified surrogate for stochastic LIF neurons.

### Lian et al., "Learnable Surrogate Gradient" (IJCAI 2023)

key innovation: LSG modulates SG width according to membrane potential distribution. addresses the problem that fixed-width surrogates cause gradient vanishing when membrane potentials are far from threshold.

this might explain our bimodal failure pattern -- surrogates with appropriate effective widths for the audio task's membrane potential distribution succeed; those with mismatched widths catastrophically fail.

### Other Recent Surrogate Work

**Sparse Surrogate Gradients (Neural Networks, 2024):** Masked Surrogate Gradients (MSGs) apply sparsity masks to preserve SNN sparsity during training. also introduces Temporal Weighted Output for decoding.

**Klos et al. (Physical Review Letters, 2025):** eliminates surrogate gradients entirely using "pseudospikes" that provide exact (not approximate) gradients. smooth and exact but only demonstrated on MNIST so far. potential future direction.

**Efficient Surrogate Gradients (NeurIPS 2025):** Chi-based pipeline that adaptively trades off between shape and effective domain. shows keeping fixed surrogate for all layers is suboptimal.

### Surrogate Functions: Literature vs Our Results

| Function | Our Result | Literature Status | Key Reference |
|----------|-----------|-------------------|---------------|
| **Spike Rate Escape** | **46.00% (BEST)** | Theoretically justified via escape noise | Gygax & Zenke 2025 |
| **Fast Sigmoid** | **44.75%** | SuperSpike original; widely used | Zenke & Ganguli 2018 |
| **Arctan (atan)** | **35.75%** | Generally preferred in recent work | 2024 consensus |
| STE (Straight-Through) | 10.25% (failed) | Known to struggle with deeper networks | standard observation |
| Sigmoid | 2.00% (failed) | Vanishing gradient problems | Lian et al. 2023 |
| SFS | 2.00% (failed) | Less commonly used | limited literature |
| Triangular | 2.75% (failed) | Piecewise linear; Zenke 2021 said it works | **contradicts Zenke 2021** |
| LSO (Stochastic) | Crashed | Python 3.14 incompatibility | implementation issue |

### Explaining the Bimodal Pattern

our bimodal result (3 learn, 4 fail) is pretty significant i think. possible explanations:

1. **effective gradient width:** Lian et al. (2023) show membrane potential distribution determines optimal SG width. audio classification may produce distributions incompatible with narrow surrogates (sigmoid, STE, triangular). the three working ones (SRE, fast_sigmoid, atan) all have broader effective domains.

2. **gradient magnitude at threshold:** SRE and fast_sigmoid have larger gradients near threshold vs sigmoid. 50-class task may need stronger gradients near threshold.

3. **task complexity:** Zenke 2021 showed robustness on simple tasks (XOR, MNIST). ESC-50 with mel spectrograms may be complex enough that shape DOES matter -- challenging "shape doesn't matter" for harder tasks.

4. **training dynamics:** failed surrogates collapsed to chance within first 10-15 epochs, suggesting gradient vanishing rather than slow convergence.

5. **escape noise theory:** Gygax & Zenke (2025) show SRE is theoretically grounded for stochastic neurons. others are heuristic approximations.

---

## Part 4: Cross-Cutting Themes

### The Encoding-Energy-Accuracy Trilemma

the literature consistently shows a three-way tradeoff:
- **direct:** highest accuracy, highest firing rate (expensive)
- **temporal (TTFS/latency):** lowest firing rate (efficient), lowest accuracy, fragile
- **rate:** moderate accuracy, moderate energy, best robustness

our results perfectly illustrate this: direct (47.15%, high energy) vs rate (24.00%, moderate) vs latency (16.30%, low energy).

### The Simulation-vs-Hardware Paradox

our SNN being 2.1x more expensive than ANN in software is consistent with Yang 2024 and Dampfhoffer 2023. but on neuromorphic hardware (Loihi 2, Xylo, SpiNNaker 2), 10x-200x energy advantages are measured. the discrepancy: software simulation doesn't benefit from event-driven processing -- it iterates through all timesteps sequentially. real savings need hardware that skips zero-spike operations.

### The Feature Learning Bottleneck

our most important finding (PANNs+SNN=92.5% vs scratch SNN=47.15%) is consistent with:
- NeuroBench motor prediction: SNN matches ANN when features are naturally spike-compatible
- Stanojevic et al. 2024: 0.3 spikes/neuron achieves ANN accuracy when converting from pretrained ANN
- the bottleneck is feature learning, not spiking computation

---

## Research Gaps

things that don't exist in the literature:
1. nobody benchmarks 7+ encoding schemes on a single audio dataset -- we're unique
2. no energy measurements for ESC-50 specifically on neuromorphic hardware
3. nobody explains the bimodal surrogate gradient pattern -- our result is novel
4. limited data on population encoding performance -- most work focuses on rate/temporal/direct

---
