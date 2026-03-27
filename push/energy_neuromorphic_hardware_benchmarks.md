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
