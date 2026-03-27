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

#### Spike-Thrift (Kundu et al., WACV 2021)
- Attention-guided compression to limit spiking activity
- Achieved up to 33.4x compression with no significant accuracy drop
- Compressed SNN models: up to 12.2x better compute energy-efficiency compared to ANNs with similar parameters

#### All In One Timestep (Castagnetti et al., 2025, arXiv:2510.24637)
- Multi-level spiking neurons: reduces energy consumption by 2-3x compared to binary SNNs
- Reduces network activity by >20% compared to previous spiking ResNets
- Achieves inference in 1 timestep (10x compression factor)

### 1.3 Summary of Energy Threshold Literature

| Paper | Year | Venue | Spike Rate Threshold for SNN Advantage | Notes |
|:---|:---:|:---|:---:|:---|
| Dampfhoffer et al. | 2023 | IEEE TECI | 0.15-1.38 spikes/synapse/inference | Hardware-aware, includes memory |
| Yan et al. | 2024 | arXiv | <7-8% at T=6 | Includes data movement |
| Shen et al. | 2024 | CVPR | <10-15% | Bit budget framework |
| Li et al. | 2023 | arXiv | Varies by hardware | Hardware bottleneck analysis |
| HW-aware vs agnostic | 2025 | arXiv | N/A | 50-60% gap between methods |

**Consensus:** SNNs need spike rates below ~6-10% (sparsity >90-94%) to beat optimized quantized ANNs on digital hardware. On neuromorphic hardware, the threshold is more relaxed due to native AC support.

---

## 2. NeuroBench Benchmark

### 2.1 Overview
- **Citation:** Yik et al., "The NeuroBench Framework for Benchmarking Neuromorphic Computing Algorithms and Systems," Nature Communications 16:1589, February 2025
- **Community:** 60+ institutions across industry and academia
- **Current version:** NeuroBench 2.2.0 (pip install neurobench)
- **Website:** neurobench.ai

### 2.2 Metric Definitions

**Correctness Metrics:**
- Accuracy, mAP, MSE, R-squared, sMAPE (task-dependent)

**Complexity Metrics:**
- **Footprint:** Memory required in bytes for model representation (weights, parameters, buffers)
- **Connection Sparsity:** Ratio of zero weights to total weights (0=fully connected, 1=fully sparse)
- **Activation Sparsity:** Average proportion of zero activations across all neurons, timesteps, and samples (0=all active, 1=all silent)
- **Synaptic Operations:**
  - **Dense:** All operations regardless of zeros
  - **Eff_MACs (Effective MACs):** Multiply-accumulates excluding zero activations/connections. Non-binary activation operations.
  - **Eff_ACs (Effective ACs):** Accumulates with binary activations. Spike-based operations.
- **Model Execution Rate:** Forward pass frequency in Hz

**Key distinction:** Synaptic operations with non-binary activation are MACs; those with binary activation (spikes) are ACs.

### 2.3 Algorithm Track Benchmark Tasks

1. **Keyword Few-Shot Class-Incremental Learning (FSCIL):** Audio keyword classification with continual learning across languages (MSWC dataset)
2. **Event Camera Object Detection:** Neuromorphic event camera video (Prophesee 1MP dataset), COCO mAP metric
3. **Non-human Primate (NHP) Motor Prediction:** Fingertip velocity prediction from cortical recordings, R-squared metric
4. **Chaotic Function Prediction:** Mackey-Glass time series, sMAPE metric

### 2.4 Baseline Results (from Nature Communications paper)

#### Table 2: Keyword FSCIL Task

| Model | Accuracy (Base/Avg) | Footprint (bytes) | Activation Sparsity | Dense SynOps | Eff_MACs | Eff_ACs |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| M5 ANN | 97.09%/89.27% | 6.03e6 | 0.783 | 2.59e7 | 7.85e6 | 0 |
| SNN | 93.48%/75.27% | 1.36e7 | 0.916 | 3.39e6 | 0 | 3.65e5 |

**Key insight:** SNN baseline computes each sample over 200 passes, using an order of magnitude fewer effective AC synaptic operations than the ANN's MACs.

#### Table 3: Event Camera Object Detection

| Model | mAP | Footprint (bytes) | Activation Sparsity | Dense SynOps | Eff_MACs | Eff_ACs |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| RED ANN | 0.429 | 9.13e7 | 0.634 | 2.84e11 | 2.48e11 | 0 |
| Hybrid | 0.271 | 1.21e7 | 0.613 | 9.85e10 | 3.76e10 | 5.60e8 |

#### Table 4: NHP Motor Prediction (96-channel, Indy)

| Model | R-squared | Footprint | Activation Sparsity | Eff_MACs | Eff_ACs |
|:---|:---:|:---:|:---:|:---:|:---:|
| ANN | 0.593 | 20,824 | 0.683 | 3,836 | 0 |
| SNN | 0.593 | 19,648 | 0.997 | 0 | 276 |

**Key insight:** SNN achieves identical R-squared (0.593) with 0.997 activation sparsity, translating to only 276 effective ACs vs 3,836 MACs. This is a 13.9x operation count reduction BEFORE accounting for the AC vs MAC energy difference.

#### Table 5: Chaotic Function Prediction (Mackey-Glass)

| Model | sMAPE | Footprint | Connection Sparsity | Activation Sparsity | Eff_MACs | Eff_ACs |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| ESN | 14.79 | 2.81e5 | 0.876 | 0.0 | 4.37e3 | 0 |
| LSTM | 13.37 | 4.90e5 | 0.0 | 0.530 | 6.03e4 | 0 |

### 2.5 System Track

Two v1.0 system benchmarks:
1. **Acoustic Scene Classification (edge-scale):** DCASE dataset, 41,360/16,240 train/test samples, 4 classes. Measures average power during inference and inference latency.
2. **Optimization tasks (datacenter-scale)**

### 2.6 Energy Estimation Using NeuroBench Metrics

NeuroBench provides the building blocks for energy estimation:
- **Energy_SNN = Eff_ACs x E_AC** (where E_AC ~ 0.9 pJ at 45nm)
- **Energy_ANN = Eff_MACs x E_MAC** (where E_MAC ~ 4.6 pJ at 45nm)

This is the simplified hardware-agnostic approach. The hardware-aware approach adds memory access, data movement, and control overhead.

---

## 3. Neuromorphic Hardware: Measured Energy Numbers

### 3.1 Comprehensive Hardware Comparison Table

| Platform | Process | Neurons | Synapses | Power | Energy/Syn. Op | Key Metric | Year |
|:---|:---:|:---:|:---:|:---:|:---:|:---|:---:|
