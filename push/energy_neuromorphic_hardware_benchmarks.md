# SNN vs ANN Energy Efficiency & Neuromorphic Hardware Benchmarks
*Generated: 5 March 2026*

## Executive Summary

**SNNs are NOT automatically more energy-efficient than ANNs.** The advantage is conditional on spike sparsity, hardware platform, and whether memory access costs are included. Our SNN (25.8% spike rate, 74.2% sparsity) is 4x above the software break-even threshold. Neuromorphic hardware is required for real energy savings.

---

## 1. When Do SNNs Actually Beat ANNs in Energy?

### The Threshold Problem

| Source | Year | Threshold | Context |
|--------|------|-----------|---------|
| Dampfhoffer et al. | 2023 (IEEE TECI) | >92-93% sparsity at T=6 | vs quantized ANNs |
| Yan, Bai & Wong | 2024 | >92-93% at T=6, >97% at T>16 | Classical architectures |
| Shen et al. | 2024 (CVPR) | <10-15% spike rate | "Bit Budget" framework |
| Hardware-aware analysis | 2025 | N/A — SNNs don't beat CNNs on digital HW | Neuromorphic HW required |

### What This Means for Us
- **Our spike rate: 25.8%** (sparsity: 74.2%)
- **Software break-even: ~6-8% spike rate** (>92% sparsity)
- **We are 4x above break-even in software**
- **On neuromorphic hardware: we win** because AC costs 0.9pJ vs MAC 4.6pJ (5.1x per-op advantage)

### Honest Thesis Framing
In software simulation, our SNN uses MORE energy than the ANN (976 nJ vs 463 nJ). The energy advantage is ONLY realized on neuromorphic hardware where accumulate operations (ACs) are natively cheaper than multiply-accumulates (MACs).

---

## 2. The Canonical Energy Numbers (Horowitz ISSCC 2014, 45nm)

| Operation | Energy (pJ) | Ratio to MAC |
|-----------|-------------|-------------|
| 8-bit Integer Add | 0.03 | 0.007x |
| 8-bit Integer Multiply | 0.2 | 0.04x |
| **32-bit FP Add (AC)** | **0.9** | **0.2x** |
| 32-bit FP Multiply | 3.7 | 0.8x |
| **32-bit FP MAC** | **4.6** | **1.0x** |
| 32KB SRAM Read | 20 | 4.3x |
| **DRAM Read** | **~640** | **139x** |

**Key insight:** DRAM access (640 pJ) costs 139x more than a MAC. **Memory dominates energy**, not computation. This is why hardware-aware analyses give different results from operation-counting analyses.

**Has Horowitz been updated?** No official update. The MAC/AC ratio (~5.1x) remains approximately constant across process nodes because both scale similarly. Absolute values decrease at modern nodes but the relative cost is stable.

---

## 3. NeuroBench Benchmark Details

**Yik et al., Nature Communications 16:1589 (Feb 2025)**. 60+ institutions.

### Metrics Defined
| Metric | Description |
|--------|-------------|
| Eff_MACs | Effective multiply-accumulates (non-binary activations, excluding zeros) |
| Eff_ACs | Effective accumulates (binary/spike activations) |
| Activation Sparsity | Proportion of zero activations (averaged over neurons, timesteps, samples) |
| Connection Sparsity | Ratio of zero weights to total |
| Footprint | Model memory in bytes |

### Baseline Results from Paper
| Task | ANN Eff_MACs | SNN Eff_ACs | SNN Activation Sparsity |
|------|-------------|-------------|------------------------|
| Keyword FSCIL | 7.85e6 | 3.65e5 | 0.916 |
| NHP Motor (Indy) | 3,836 | 276 | 0.997 |

The NHP motor result: SNN uses 276 ACs vs 3,836 MACs (13.9x fewer operations) with identical accuracy. With 5.1x per-operation cost advantage, that's ~71x energy reduction.

### Our NeuroBench Results
- SNN: 1.08M ACs → 976 nJ (at 0.9 pJ/AC)
- ANN: 101K MACs → 463 nJ (at 4.6 pJ/MAC)
- SNN Activation Sparsity: 74.16%
- ANN Activation Sparsity: ~59%

---

## 4. Measured Energy: Real Neuromorphic Hardware

| Platform | Process | Energy/Op | Power | Key Benchmark |
|----------|---------|-----------|-------|---------------|
| **Loihi 1** | 14nm | ~23.6 pJ/synaptic op | ~0.5W | KWS: 5.3-109x better than CPU/GPU |
| **Loihi 2** | Intel 4 | Improved | ~1W | SSM: 1000x less energy vs Jetson |
| **TrueNorth** | 28nm | ~26 pJ/synaptic op | 65-275 mW | 46 GSOPS/W |
| **SpiNNaker 1** | 130nm | ~5.8 μJ/synaptic op | ~1W/chip | Real-time bio simulation |
| **SpiNNaker 2** | 22nm FDSOI | 10x better than S1 | Improved | 18x vs GPUs (claimed) |
| **BrainScaleS-2** | 65nm | Analog (low pJ) | ~1W | 0.2 mJ/inference |
| **DarwinWafer** | Wafer-scale | **4.9 pJ/SOP** | ~100W | New SOTA efficiency |
| **Innatera T1** | 28nm | <200 fJ/spike | <10 mW | Audio scene classification |

### Head-to-Head Benchmark (Ostrau et al. 2022, Frontiers)
The only apples-to-apples comparison across platforms:
- **BrainScaleS-2 (Spikey)**: 0.2 mJ/inference — most efficient
- **Coral Edge TPU**: 0.3 mJ/inference
- **SpiNNaker 1**: 38.2 mJ/inference — relatively expensive (ARM core overhead)

**SpiNNaker 1 is not energy-competitive** with purpose-built chips. Its value is flexibility and programmability, not raw efficiency.

---

## 5. Commercial Neuromorphic Products (Shipping or Near-Production)

### Innatera Pulsar (May 2025)
- **First mass-market neuromorphic microcontroller**
- Audio scene classification, anomaly detection
- <10 mW power, 500x lower energy, 100x lower latency vs conventional
- Demonstrated at CES 2024 and 2025

### BrainChip Akida / AKD1500
- In production, M.2 modules shipping
- Vision, audio, sensor fusion
- Milliwatt range
- $25M funding for Akida 2

### Intel Loihi 2
- Research platform (not commercially sold)
- Deployed at BMW factories for defect detection (20ms → 2ms)

### SpiNNaker 2 (SpiNNcloud Systems)
- Research deployments at Sandia National Labs and Leipzig University
- 175M neuron system at Sandia
- First commercially available neuromorphic supercomputer

---

## 6. Implications for Our Thesis

### What to claim honestly:
1. In **software simulation**, our ANN is 2.1x more energy-efficient (463 nJ vs 976 nJ)
2. On **neuromorphic hardware**, each SNN operation is 5.1x cheaper (0.9 vs 4.6 pJ)
3. The **real energy argument** for SNNs requires dedicated hardware
4. Our spike rate (25.8%) is above the software break-even but the per-operation advantage on hardware compensates
5. SpiNNaker 1 is not the most energy-efficient neuromorphic chip — cite Loihi and Innatera for better numbers

### What NOT to claim:
- Don't claim SNNs are inherently more efficient (they need specific conditions)
- Don't extrapolate SpiNNaker 1 energy numbers to represent all neuromorphic hardware
- Don't ignore memory access costs (they dominate)

---

## Key Sources
- [Dampfhoffer et al. 2023 — IEEE TECI](https://ieeexplore.ieee.org/document/9927729/)
- [Yan et al. 2024](https://arxiv.org/abs/2409.08290)
- [Shen et al. 2024 — CVPR "Bit Budget"](https://openaccess.thecvf.com/content/CVPR2024/html/Shen_Are_Conventional_SNNs_Really_Efficient)
- [Horowitz 2014 — ISSCC](https://ieeexplore.ieee.org/document/6757323/)
- [Yik et al. 2025 — NeuroBench, Nature Communications](https://www.nature.com/articles/s41467-025-56739-4)
- [Ostrau et al. 2022 — Hardware Benchmark, Frontiers](https://www.frontiersin.org/articles/10.3389/fnins.2022.873935/)
- [Blouw et al. 2019 — Loihi KWS](https://arxiv.org/abs/1812.01739)
- [DarwinWafer 2025](https://arxiv.org/abs/2509.16213)
- [Innatera T1](https://innatera.com/products/spiking-neural-processor-t1)
- [Hardware-aware vs Agnostic 2025](https://arxiv.org/abs/2508.19654)
