# SNN vs ANN Energy Efficiency: Comprehensive Research Report

**Date:** 5 March 2026
**Scope:** Energy comparisons, NeuroBench benchmarks, neuromorphic hardware measurements, AC vs MAC costs, spike sparsity thresholds, edge deployments
**Research depth:** 40+ sources consulted across academic papers, hardware documentation, and industry reports

---

## Executive Summary

The energy efficiency narrative for SNNs is far more nuanced than commonly presented. The literature reveals that **SNNs are NOT automatically more energy-efficient than ANNs** on conventional digital hardware. The advantage depends critically on three factors: (1) spike sparsity rates, which must exceed ~92-93% for moderate time windows; (2) the hardware platform, with neuromorphic hardware being essential for realizing theoretical gains; and (3) whether memory access and data movement costs are included in the analysis. When these are properly accounted for, the bar for SNN energy superiority rises dramatically.

The key threshold from Dampfhoffer et al. (2023) and Yan et al. (2024) is that SNNs need spike rates below approximately 6-8% (i.e., sparsity above 92-94%) at time window T=6 to beat quantized ANNs. Most real-world SNN implementations on vision tasks report spike rates of 20-40%, well above this threshold. This makes the energy claim questionable for many practical deployments on digital hardware, though neuromorphic hardware changes the calculus significantly.

For your thesis, your measured 74.16% activation sparsity (NeuroBench) translates to a ~25.84% spike rate -- well above the 6-8% threshold needed for software-level energy superiority. However, on neuromorphic hardware with native AC operations, the per-operation cost advantage (0.9 pJ/AC vs 4.6 pJ/MAC) does provide a genuine 5.1x per-operation benefit. The honest framing for your thesis is: "SNNs achieve energy efficiency advantages on neuromorphic hardware through sparse AC operations, but require specialized hardware to realize these gains."

---

## 1. Energy Comparisons: SNN vs ANN

### 1.1 The Core Papers

#### Dampfhoffer et al. (2023) -- "Are SNNs Really More Energy-Efficient Than ANNs?"
- **Citation:** IEEE Transactions on Emerging Topics in Computational Intelligence (TECI), Vol. 7, pp. 731+, 2023
- **DOI:** IEEE Xplore 9927729
- **Key finding:** SNNs with the IF model can compete with efficient ANN implementations when there is very high spike sparsity, between **0.15 and 1.38 spikes per synapse per inference**, depending on the ANN implementation.
- **Hardware-aware analysis:** The main advantage of SNNs compared to ANNs (on digital hardware) comes primarily from **exploiting the sparsity of spikes and NOT from the replacement of MAC by AC operations**.
- **Critical insight:** Many studies do not consider memory accesses, which account for an important fraction of the energy consumption.
- **Sparsity requirements:** For T=6 timesteps, SNNs need spike sparsity rates of 0.92-0.93 (i.e., 92-93% of neurons silent) to match optimized quantized ANNs.

#### Yan, Bai & Wong (2024) -- "Reconsidering the Energy Efficiency of Spiking Neural Networks"
- **Citation:** arXiv:2409.08290, submitted August 2024, updated July 2025
- **Key finding:** Establishes a fair baseline by mapping rate-encoded SNNs with T timesteps to functionally equivalent QNNs with ceil(log2(T+1)) bits.
- **Sparsity thresholds by time window:**

| Time Window (T) | Required Sparsity (Classical) | Required Sparsity (Spatial-Dataflow) |
|:---:|:---:|:---:|
| 6 | >92% | >93% |
| >16 | >97% | >97%+ |

- **Experimental validation:** Using VGG16 on CIFAR-10 with their sparsity regularization:
  - Achieved 94.19% sparsity at T=6
  - Energy: 0.85x (classical) and 0.78x (spatial-dataflow) relative to ANNs
  - Accuracy: 92.76%
- **Energy model parameters used (normalized):**

| Operation | Energy Cost |
|:---|:---:|
| 8-bit ADD | 0.03 pJ |
| 8-bit MUL | 0.2 pJ |
| SRAM (per bit) | 20 pJ |
| DRAM (per bit) | 2 nJ |
| NOC per hop | 10 pJ/bit |

- **Critical warning:** Without accounting for memory access and data movement overhead, SNNs appear efficient even at 0% sparsity. Realistic hardware analysis completely changes the picture.

#### Shen et al. (2024) -- "Are Conventional SNNs Really Efficient? A Perspective from Network Quantization" (CVPR 2024)
- **Key contribution:** Introduces the "Bit Budget" concept -- total computational work = bit-width x operations performed.
- **Finding:** SNNs only become more efficient than quantized ANNs when maintaining spike rates below approximately 10-15% of neurons firing per inference step.
- **Reported actual spike rates:** Typical SNN spike rates of 20-40% on ImageNet-scale tasks -- above the efficiency crossover point.
- **Implication:** Conventional SNN implementations sacrifice efficiency gains by tolerating relatively high spiking activity to maintain accuracy.

#### Hardware-Aware vs Hardware-Agnostic (2025) -- arXiv:2508.19654
- **Key finding:** SNNs possess a ~50-60% efficiency advantage over CNNs when evaluated using hardware-agnostic methodology, but hardware-aware results indicate SNNs do NOT surpass CNNs on classical computing architectures.
- **Conclusion:** SNNs require neuromorphic hardware to achieve competitive energy efficiency.

#### Li et al. (2023) -- "Are SNNs Truly Energy-efficient? A Hardware Perspective" (arXiv:2309.03388)
- **Hardware bottlenecks identified:**
  1. Repeated computations and data movements over timesteps
  2. Neuronal module overhead
  3. Vulnerability of SNNs towards crossbar non-idealities
- **Finding:** Actual energy-efficiency improvements differ significantly from estimated values due to various hardware bottlenecks.

### 1.2 Additional Energy Papers

