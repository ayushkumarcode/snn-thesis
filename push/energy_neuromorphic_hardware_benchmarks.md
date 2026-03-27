# SNN vs ANN Energy Efficiency & Neuromorphic Hardware Benchmarks

**SNNs are NOT automatically more energy-efficient than ANNs.** the advantage depends on spike sparsity, hardware platform, and whether you count memory access costs. our SNN (25.8% spike rate, 74.2% sparsity) is 4x above the software break-even threshold. need neuromorphic hardware for actual energy savings.

---

## 1. When Do SNNs Actually Beat ANNs in Energy?

### the threshold problem

| Source | Year | Threshold | Context |
|--------|------|-----------|---------|
| Dampfhoffer et al. | 2023 (IEEE TECI) | >92-93% sparsity at T=6 | vs quantized ANNs |
| Yan, Bai & Wong | 2024 | >92-93% at T=6, >97% at T>16 | classical architectures |
| Shen et al. | 2024 (CVPR) | <10-15% spike rate | "Bit Budget" framework |
| Hardware-aware analysis | 2025 | N/A -- SNNs don't beat CNNs on digital HW | neuromorphic HW required |

### what this means for us
- **our spike rate: 25.8%** (sparsity: 74.2%)
- **software break-even: ~6-8% spike rate** (>92% sparsity)
- **we are 4x above break-even in software**
- **on neuromorphic hardware: we win** because AC costs 0.9pJ vs MAC 4.6pJ (5.1x per-op advantage)

### honest framing
in software simulation, our SNN uses MORE energy than the ANN (976 nJ vs 463 nJ). the energy advantage is ONLY realized on neuromorphic hardware where accumulate operations are natively cheaper than multiply-accumulates.

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

the key insight here: DRAM access (640 pJ) costs 139x more than a MAC. **memory dominates energy**, not computation. this is why hardware-aware analyses give different results from simple operation-counting.

has Horowitz been updated? no official update. the MAC/AC ratio (~5.1x) stays approximately constant across process nodes because both scale similarly. absolute values decrease at modern nodes but the relative cost is stable.

---

## 3. NeuroBench Details

**Yik et al., Nature Communications 16:1589 (Feb 2025)**. 60+ institutions.

### metrics
| Metric | Description |
|--------|-------------|
| Eff_MACs | effective multiply-accumulates (non-binary activations, excluding zeros) |
| Eff_ACs | effective accumulates (binary/spike activations) |
| Activation Sparsity | proportion of zero activations (averaged over neurons, timesteps, samples) |
| Connection Sparsity | ratio of zero weights to total |
| Footprint | model memory in bytes |

### baseline results from their paper
| Task | ANN Eff_MACs | SNN Eff_ACs | SNN Activation Sparsity |
|------|-------------|-------------|------------------------|
| Keyword FSCIL | 7.85e6 | 3.65e5 | 0.916 |
| NHP Motor (Indy) | 3,836 | 276 | 0.997 |

the NHP motor result is pretty compelling: SNN uses 276 ACs vs 3,836 MACs (13.9x fewer operations) with identical accuracy. with 5.1x per-op cost advantage, that's ~71x energy reduction.

### our NeuroBench results
- SNN: 1.08M ACs -> 976 nJ (at 0.9 pJ/AC)
- ANN: 101K MACs -> 463 nJ (at 4.6 pJ/MAC)
- SNN activation sparsity: 74.16%
- ANN activation sparsity: ~59%

---

## 4. Measured Energy on Real Neuromorphic Hardware

| Platform | Process | Energy/Op | Power | Key Benchmark |
|----------|---------|-----------|-------|---------------|
| **Loihi 1** | 14nm | ~23.6 pJ/synaptic op | ~0.5W | KWS: 5.3-109x better than CPU/GPU |
| **Loihi 2** | Intel 4 | improved | ~1W | SSM: 1000x less energy vs Jetson |
| **TrueNorth** | 28nm | ~26 pJ/synaptic op | 65-275 mW | 46 GSOPS/W |
| **SpiNNaker 1** | 130nm | ~5.8 uJ/synaptic op | ~1W/chip | real-time bio simulation |
| **SpiNNaker 2** | 22nm FDSOI | 10x better than S1 | improved | 18x vs GPUs (claimed) |
| **BrainScaleS-2** | 65nm | analog (low pJ) | ~1W | 0.2 mJ/inference |
| **DarwinWafer** | wafer-scale | **4.9 pJ/SOP** | ~100W | new SOTA efficiency |
| **Innatera T1** | 28nm | <200 fJ/spike | <10 mW | audio scene classification |

### head-to-head benchmark (Ostrau et al. 2022, Frontiers)

the only real apples-to-apples comparison:
- **BrainScaleS-2 (Spikey)**: 0.2 mJ/inference -- most efficient
- **Coral Edge TPU**: 0.3 mJ/inference
- **SpiNNaker 1**: 38.2 mJ/inference -- relatively expensive (ARM core overhead)

**SpiNNaker 1 is not energy-competitive** with purpose-built chips. its value is flexibility and programmability, not raw efficiency.

---

## 5. Commercial Neuromorphic Products (Shipping or Near-Production)

### Innatera Pulsar (May 2025)
- first mass-market neuromorphic microcontroller
- audio scene classification, anomaly detection
- <10 mW power, 500x lower energy, 100x lower latency vs conventional
- demos at CES 2024 and 2025

### BrainChip Akida / AKD1500
- in production, M.2 modules shipping
- vision, audio, sensor fusion
- milliwatt range
- $25M funding for Akida 2
