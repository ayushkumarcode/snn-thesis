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
| **Intel Loihi 1** | 14nm | 131K | 130M | ~0.5W active | ~23.6 pJ/SynOp | 30 mW idle | 2018 |
| **Intel Loihi 2** | Intel 4 | 1M | 120M | ~1W | Improved over Loihi 1 | 10x faster spike processing | 2021 |
| **IBM TrueNorth** | 28nm | 1M | 256M | 65-275 mW | ~26 pJ/SynEvent | 46 GSOPS/W | 2014 |
| **IBM NorthPole** | 12nm | N/A (DNN) | N/A | N/A | N/A | 5x more efficient than H100 | 2023 |
| **SpiNNaker 1** | 130nm | 18 ARM968 cores/chip | Software-defined | ~1W/chip | ~5.8 uJ/syn event | 100 kW full machine | 2014 |
| **SpiNNaker 2** | 22nm FDSOI | 153 ARM cores/chip | Hardware-defined | 10x efficiency vs S1 | Improved | 18x more efficient than GPUs (claimed) | 2021+ |
| **BrainScaleS-2** | 65nm | 512 | 131K | ~1W | Analog (low pJ range) | 1000x acceleration vs bio | 2022 |
| **DarwinWafer** | N/A | 0.15B (wafer) | 6.4B (wafer) | ~100W (wafer) | **4.9 pJ/SOP** | 0.64 TSOPS/W | 2025 |
| **Innatera T1** | 28nm | SNN-based | Analog SNN | <10 mW total | <200 fJ/spike event | <0.5 mJ keyword spotting | 2024 |

### 3.2 Detailed Platform Data

#### Intel Loihi 1 (Davies et al., IEEE Micro 2018)
- 128 neuromorphic cores, 14nm Intel process
- ~131,072 neurons, up to 130M synapses
- **~23.6 pJ per synaptic operation** (measured)
- Idle power: 30 mW
- Active power: dependent on network activity

**Keyword Spotting Benchmark (Blouw et al., 2019):**
- Loihi outperforms CPU, GPU, Jetson TX1, and Movidius NCS on energy per inference
- 5.3x to 109.1x improvement in energy cost per inference vs conventional hardware
- Loihi advantage improves for larger networks

#### Intel Loihi 2 (2021)
- Intel 4 process, 1M neurons, 120M synapses
- 128 fully asynchronous neuron cores + 6 Lakemont x86 cores
- ~1W power consumption
- 10x faster spike processing vs Loihi 1

**State Space Model Benchmark (arXiv:2409.15022):**
- First-ever SSM implementation on neuromorphic hardware
- On sMNIST, psMNIST, sCIFAR token-by-token inference:
  - **1000x less energy** than NVIDIA Jetson Orin Nano
  - **75x lower latency** and 75x higher throughput
- Caveat: Jetson performs better in offline batched processing mode

#### IBM TrueNorth (Merolla et al., Science 2014)
- 4,096 cores, 1M neurons, 256M synapses, 28nm
- 65 mW typical, up to 275 mW active
- ~26 pJ per synaptic event
- 46 GSOPS and 400 GSOPS/W
- 1,200-2,600 frames/s for CNN inference

#### IBM NorthPole (2023)
- 12nm, 256 cores, 22B transistors
- NOT spiking -- optimized DNN inference chip
- 5x more energy efficient than NVIDIA H100
- ~4,000x faster than TrueNorth
- All memory on-chip (eliminates von Neumann bottleneck)

#### SpiNNaker 1 (Furber et al., 2014)
- 130nm, 18 ARM968 cores per chip
- ~1W per chip when fully loaded
- ~5.8 uJ per synaptic event (minimal energy)
- Full machine: 1M ARM cores, ~100 kW
- A few nanojoules per event and per neuron

#### SpiNNaker 2 (Mayr et al.)
- 22nm FDSOI, 153 ARM cores per chip with 19 MB SRAM, 2 GB DRAM
- 10x increase in neural simulation capacity per watt vs SpiNNaker 1
- Operates down to 0.5V through adaptive biasing
- **Sandia National Labs deployment (March 2025):** 24 boards x 48 chips = 1,152 chips, 175M neurons
- SpiNNcloud claims 18x more efficient than GPUs

**SpiNNaker 2 Benchmarks:**
- MNIST: 96.6% accuracy at ~23 uJ per image classification (prototype, 2018 measurement)
- Google Speech Commands: 91.12% accuracy (12-category), 682 KB memory
- DVS gesture recognition: Lower energy than GPU (A100/GTX1070Ti) at batch-one

#### BrainScaleS-2 (Heidelberg, 2022)
- Mixed-signal, 512 neurons, 131,072 synapses, 6-bit weights
- ~1W power consumption
- 1000x acceleration vs biological real time
- Analog circuits: energy per synaptic transmission "several orders of magnitude lower" than software simulation

**Ostrau et al. Benchmark (Frontiers in Neuroscience, 2022):**
- DNN inference energy per inference:
  - Spikey (BrainScaleS precursor): **0.2 mJ** (most efficient in study)
  - Intel NCS 2: 10.6 mJ
  - Coral Edge TPU: 0.3 mJ
  - GeNN-GPU: 3.7 mJ
  - SpiNNaker: 38.2 mJ

#### DarwinWafer (2025, arXiv:2509.16213)
- Wafer-scale: 64 Darwin3 chiplets on 300mm silicon interposer
- 0.15B neurons, 6.4B synapses per wafer
- **4.9 pJ/SOP at 333 MHz, 0.8V**
- ~100W for entire wafer
- 0.64 TSOPS/W
- State-of-the-art neuromorphic energy efficiency

### 3.3 Single-Neuron Energy Comparison (Ostrau et al. 2022)

From the Frontiers benchmarking paper, energy for simulating 1 second of a single neuron:

| Platform | Total Energy (J) |
|:---|:---:|
| Spikey (BrainScaleS precursor) | 1.49e-6 |
| SpiNNaker | 3.33e-4 |
| GPU (RTX 2070) | 3.18e-5 |
| Human Brain | 2.49e-10 |

**The biological brain is ~4 orders of magnitude more efficient than current neuromorphic hardware.**

---

## 4. AC vs MAC Energy Cost

### 4.1 Horowitz ISSCC 2014 -- The Canonical Reference

**Citation:** M. Horowitz, "1.1 Computing's Energy Problem (and what we can do about it)," ISSCC 2014 Digest of Technical Papers, pp. 10-14, Feb. 2014.

This is THE foundational reference for energy per arithmetic operation. All values are for **45nm CMOS at 0.9V**.

**Reconstructed Energy Table (from multiple citing sources):**

| Operation | Energy (pJ) |
|:---|:---:|
| 8-bit Integer Add | 0.03 |
| 8-bit Integer Multiply | 0.2 |
| 16-bit Integer Add | ~0.05 |
| 16-bit Integer Multiply | ~1.0 |
| 32-bit Integer Add | 0.1 |
| 32-bit Integer Multiply | 3.1 |
| 32-bit FP Add | 0.9 |
| 32-bit FP Multiply | 3.7 |
| **32-bit FP MAC (MUL+ADD)** | **4.6** |
| **32-bit FP AC (ADD only)** | **0.9** |
| 8-bit FP MAC | ~1.1 |
| 8-bit FP AC | ~0.2 |
| 32-bit SRAM Read (8KB) | 5 |
| 32-bit SRAM Read (32KB) | 20 |
| 32-bit DRAM Read | ~640 |

**Key ratios:**
- MAC/AC ratio at 32-bit FP: **4.6/0.9 = 5.1x** (MACs cost 5.1x more than ACs)
- DRAM access vs compute: **640 pJ vs 4.6 pJ = 139x** (memory dominates energy)

### 4.2 Has Horowitz Been Updated?

**Short answer: No official update, but the ratios remain approximately valid.**

- The absolute pJ values scale down with process node (roughly linearly with voltage^2 x capacitance)
- At modern nodes (7nm, 5nm), compute energy has decreased significantly but:
  - DRAM access energy has NOT decreased as fast
  - The ratio of memory-to-compute energy has actually INCREASED
  - At 7nm: ~3,000x energy reduction for compute vs 45nm (but this is for the basic gate, not the full operation)
- **SpiNNaker 2 uses 22nm FDSOI** -- roughly 2-4x more efficient per operation than 45nm
- **Loihi 2 uses Intel 4 (~7nm equivalent)** -- roughly 5-10x more efficient than 45nm
- The MAC/AC ratio remains approximately the same (~5x) because both operations scale similarly with process node

**What has changed:**
- Memory access dominance has increased (not decreased)
- This actually STRENGTHENS the argument that data movement, not compute, dominates energy
- Papers using Horowitz 45nm values for modern hardware comparison are providing UPPER BOUNDS on compute energy, which means the actual sparsity requirements for SNN advantage may be LESS strict on modern hardware (compute costs less relative to memory)

### 4.3 How Different Papers Use AC/MAC Costs

| Paper | E_AC (pJ) | E_MAC (pJ) | Note |
|:---|:---:|:---:|:---|
| NeuroBench (Yik et al. 2025) | 0.9 | 4.6 | Standard Horowitz 45nm |
| Dampfhoffer et al. 2023 | Hardware-specific | Hardware-specific | Full memory model |
| Yan et al. 2024 | 0.03 (8-bit ADD) | 0.2 (8-bit MUL) | 8-bit integer, includes memory model |
| Castagnetti et al. 2025 | 0.9 | 4.6 | Standard Horowitz 45nm |

**IMPORTANT NOTE FOR YOUR THESIS:** The 0.9 pJ/AC and 4.6 pJ/MAC values you use in NeuroBench analysis are the standard reference. They represent 32-bit floating-point operations at 45nm. For your SNN (which uses binary spikes as activations), the AC cost is appropriate. For your ANN (which uses FP32 activations), the MAC cost is appropriate. The 5.1x ratio is the key number.

---

## 5. Spike Sparsity Thresholds

### 5.1 Threshold Summary

The critical question: **At what spike rate do SNNs become energy-competitive with ANNs?**

| Source | Threshold | Conditions | Hardware |
|:---|:---:|:---|:---|
| Dampfhoffer et al. 2023 | >92-93% sparsity | T=6, includes memory | Digital (classical + dataflow) |
| Yan et al. 2024 | >92-93% sparsity (T=6), >97% (T>16) | Full analytical model | Digital architectures |
| Shen et al. 2024 | <10-15% spike rate | Bit budget framework | Digital |
| General consensus | >90% sparsity | Conservative estimate | Digital |
| On neuromorphic HW | More relaxed | Native AC support | Neuromorphic (Loihi, etc.) |

### 5.2 Your Thesis Data in Context

From your NeuroBench results:
- **SNN Activation Sparsity: 74.16%** (meaning ~25.84% spike rate)
- **ANN Activation Sparsity: 59%**

At 25.84% spike rate, your SNN is **well above** the ~6-8% threshold needed for software energy superiority. This means:
- On conventional digital hardware: Your SNN is NOT more energy-efficient than a quantized ANN
- On neuromorphic hardware: The per-operation advantage (5.1x per op) still holds, but total energy depends on total operations

**Your energy calculation is honest:** SNN 976 nJ (1.08M ACs x 0.9 pJ) vs ANN 463 nJ (101K MACs x 4.6 pJ). The SNN does more total operations despite each being cheaper. This is the correct and honest result.

