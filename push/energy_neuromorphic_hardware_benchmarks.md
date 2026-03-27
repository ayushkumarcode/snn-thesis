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

