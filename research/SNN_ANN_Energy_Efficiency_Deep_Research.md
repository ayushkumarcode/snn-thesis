# SNN vs ANN energy efficiency -- the full picture

Looked into this properly: energy comparisons, NeuroBench benchmarks, neuromorphic hardware measurements, AC vs MAC costs, spike sparsity thresholds, edge deployments. Consulted 40+ sources.

The energy efficiency narrative for SNNs is way more nuanced than people make it sound. **SNNs are NOT automatically more energy-efficient than ANNs** on conventional digital hardware. It depends on three things: (1) spike sparsity rates, which need to exceed ~92-93% for moderate time windows; (2) the hardware platform -- you need neuromorphic hardware to realize the theoretical gains; and (3) whether memory access and data movement costs are included. When you properly account for those, the bar for SNN energy superiority rises a lot.

The key threshold from Dampfhoffer et al. (2023) and Yan et al. (2024): SNNs need spike rates below roughly 6-8% (sparsity above 92-94%) at time window T=6 to beat quantized ANNs. Most real-world SNN implementations on vision tasks report spike rates of 20-40%, well above this. So the energy claim is questionable for many practical deployments on digital hardware, though neuromorphic hardware changes things significantly.

For my project specifically: my measured 74.16% activation sparsity (NeuroBench) translates to ~25.84% spike rate -- well above the 6-8% threshold for software-level energy superiority. But on neuromorphic hardware with native AC operations, the per-operation cost advantage (0.9 pJ/AC vs 4.6 pJ/MAC) gives a genuine 5.1x per-operation benefit. The honest framing: "SNNs achieve energy efficiency advantages on neuromorphic hardware through sparse AC operations, but require specialized hardware to realize these gains."

---

## 1. Energy comparisons: SNN vs ANN

### 1.1 The core papers

#### Dampfhoffer et al. (2023) -- "Are SNNs Really More Energy-Efficient Than ANNs?"
- IEEE TECI, Vol. 7, pp. 731+, 2023
- Key finding: SNNs with the IF model can compete with efficient ANNs when there is very high spike sparsity, between 0.15 and 1.38 spikes per synapse per inference.
- The main advantage of SNNs on digital hardware comes from **exploiting spike sparsity, NOT from replacing MAC with AC operations**.
- Many studies don't consider memory accesses, which are a huge fraction of energy consumption.
- For T=6 timesteps, SNNs need 92-93% sparsity to match optimized quantized ANNs.

#### Yan, Bai & Wong (2024) -- "Reconsidering the Energy Efficiency of SNNs"
- arXiv:2409.08290
- Establishes a fair baseline by mapping rate-encoded SNNs with T timesteps to functionally equivalent QNNs with ceil(log2(T+1)) bits.
- Sparsity thresholds by time window:

| Time Window (T) | Required Sparsity (Classical) | Required Sparsity (Spatial-Dataflow) |
|:---:|:---:|:---:|
| 6 | >92% | >93% |
| >16 | >97% | >97%+ |

- Experimental validation using VGG16 on CIFAR-10 with their sparsity regularization:
  - Achieved 94.19% sparsity at T=6
  - Energy: 0.85x (classical) and 0.78x (spatial-dataflow) relative to ANNs
  - Accuracy: 92.76%
- Energy model parameters used:

| Operation | Energy Cost |
|:---|:---:|
| 8-bit ADD | 0.03 pJ |
| 8-bit MUL | 0.2 pJ |
| SRAM (per bit) | 20 pJ |
| DRAM (per bit) | 2 nJ |
| NOC per hop | 10 pJ/bit |

- Without accounting for memory access and data movement overhead, SNNs appear efficient even at 0% sparsity. Realistic hardware analysis completely changes the picture.

#### Shen et al. (2024) -- "Are Conventional SNNs Really Efficient?" (CVPR 2024)
- Introduces the "Bit Budget" concept -- total computational work = bit-width x operations performed.
- SNNs only become more efficient than quantized ANNs when maintaining spike rates below ~10-15%.
- Reported actual spike rates: typical 20-40% on ImageNet-scale tasks -- above the crossover point.
- Conventional SNN implementations sacrifice efficiency gains by tolerating high spiking activity to maintain accuracy.

#### Hardware-Aware vs Hardware-Agnostic (2025) -- arXiv:2508.19654
- SNNs have ~50-60% efficiency advantage over CNNs with hardware-agnostic methodology, but hardware-aware results show SNNs do NOT surpass CNNs on classical computing architectures.
- SNNs require neuromorphic hardware for competitive energy efficiency.

#### Li et al. (2023) -- "Are SNNs Truly Energy-efficient? A Hardware Perspective" (arXiv:2309.03388)
- Hardware bottlenecks: repeated computations over timesteps, neuronal module overhead, crossbar non-idealities.
- Actual energy improvements differ significantly from estimated values.

### 1.2 Additional energy papers

**Spike-Thrift (Kundu et al., WACV 2021):** Attention-guided compression, up to 33.4x compression with no significant accuracy drop. Compressed SNNs: up to 12.2x better compute energy-efficiency vs ANNs.

**All In One Timestep (Castagnetti et al., 2025):** Multi-level spiking neurons reduce energy 2-3x vs binary SNNs. Reduces network activity by >20%. Achieves inference in 1 timestep (10x compression factor).

### 1.3 Summary of energy threshold literature

| Paper | Year | Venue | Spike Rate Threshold | Notes |
|:---|:---:|:---|:---:|:---|
| Dampfhoffer et al. | 2023 | IEEE TECI | 0.15-1.38 spikes/synapse/inference | Hardware-aware, includes memory |
| Yan et al. | 2024 | arXiv | <7-8% at T=6 | Includes data movement |
| Shen et al. | 2024 | CVPR | <10-15% | Bit budget framework |
| Li et al. | 2023 | arXiv | Varies by hardware | Hardware bottleneck analysis |
| HW-aware vs agnostic | 2025 | arXiv | N/A | 50-60% gap between methods |

**Consensus:** SNNs need spike rates below ~6-10% (sparsity >90-94%) to beat optimized quantized ANNs on digital hardware. On neuromorphic hardware, the threshold is more relaxed due to native AC support.

---

## 2. NeuroBench benchmark

### 2.1 Overview
- Yik et al., "The NeuroBench Framework for Benchmarking Neuromorphic Computing," Nature Communications 16:1589, February 2025
- 60+ institutions across industry and academia
- Current version: NeuroBench 2.2.0
- Website: neurobench.ai

### 2.2 Metric definitions

**Correctness:** Accuracy, mAP, MSE, R-squared, sMAPE (task-dependent)

**Complexity:**
- **Footprint:** Memory required for model representation (weights, parameters, buffers)
- **Connection Sparsity:** Ratio of zero weights to total (0=fully connected, 1=fully sparse)
- **Activation Sparsity:** Average proportion of zero activations across all neurons, timesteps, and samples
- **Synaptic Operations:**
  - Dense: all operations regardless of zeros
  - Eff_MACs: multiply-accumulates excluding zero activations/connections (non-binary)
  - Eff_ACs: accumulates with binary activations (spike-based)
- **Model Execution Rate:** Forward pass frequency in Hz

Key distinction: operations with non-binary activation are MACs; those with binary activation (spikes) are ACs.

### 2.3 Algorithm track benchmarks

1. **Keyword Few-Shot Class-Incremental Learning (FSCIL):** Audio keyword classification with continual learning (MSWC dataset)
2. **Event Camera Object Detection:** Prophesee 1MP dataset, COCO mAP
3. **Non-human Primate Motor Prediction:** Fingertip velocity prediction from cortical recordings
4. **Chaotic Function Prediction:** Mackey-Glass time series, sMAPE

### 2.4 Baseline results (from the Nature Comms paper)

#### Keyword FSCIL

| Model | Accuracy (Base/Avg) | Footprint | Activation Sparsity | Dense SynOps | Eff_MACs | Eff_ACs |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| M5 ANN | 97.09%/89.27% | 6.03e6 | 0.783 | 2.59e7 | 7.85e6 | 0 |
| SNN | 93.48%/75.27% | 1.36e7 | 0.916 | 3.39e6 | 0 | 3.65e5 |

SNN computes each sample over 200 passes, using an order of magnitude fewer effective AC operations than the ANN's MACs.

#### NHP Motor Prediction

| Model | R-squared | Footprint | Activation Sparsity | Eff_MACs | Eff_ACs |
|:---|:---:|:---:|:---:|:---:|:---:|
| ANN | 0.593 | 20,824 | 0.683 | 3,836 | 0 |
| SNN | 0.593 | 19,648 | 0.997 | 0 | 276 |

SNN achieves identical R-squared (0.593) with 0.997 activation sparsity, translating to only 276 ACs vs 3,836 MACs. That's a 13.9x operation count reduction BEFORE accounting for AC vs MAC energy difference. Pretty cool.

### 2.5 Energy estimation using NeuroBench metrics

NeuroBench gives you the building blocks:
- Energy_SNN = Eff_ACs x E_AC (where E_AC ~ 0.9 pJ at 45nm)
- Energy_ANN = Eff_MACs x E_MAC (where E_MAC ~ 4.6 pJ at 45nm)

This is the simplified hardware-agnostic approach. The hardware-aware approach adds memory access, data movement, and control overhead.

---

## 3. Neuromorphic hardware: measured energy numbers

### 3.1 Hardware comparison

| Platform | Process | Neurons | Synapses | Power | Energy/Syn. Op | Year |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Intel Loihi 1** | 14nm | 131K | 130M | ~0.5W active | ~23.6 pJ/SynOp | 2018 |
| **Intel Loihi 2** | Intel 4 | 1M | 120M | ~1W | Improved over Loihi 1 | 2021 |
| **IBM TrueNorth** | 28nm | 1M | 256M | 65-275 mW | ~26 pJ/SynEvent | 2014 |
| **IBM NorthPole** | 12nm | N/A (DNN) | N/A | N/A | N/A | 2023 |
| **SpiNNaker 1** | 130nm | 18 ARM cores/chip | Software-defined | ~1W/chip | ~5.8 uJ/syn event | 2014 |
| **SpiNNaker 2** | 22nm FDSOI | 153 ARM cores/chip | Hardware-defined | 10x vs S1 | Improved | 2021+ |
| **BrainScaleS-2** | 65nm | 512 | 131K | ~1W | Analog (low pJ range) | 2022 |
| **DarwinWafer** | N/A | 0.15B (wafer) | 6.4B (wafer) | ~100W | **4.9 pJ/SOP** | 2025 |
| **Innatera T1** | 28nm | SNN-based | Analog SNN | <10 mW total | <200 fJ/spike event | 2024 |

### 3.2 Detailed platform data

**Intel Loihi 1 (Davies et al., IEEE Micro 2018):**
- 128 neuromorphic cores, 14nm. ~131K neurons, up to 130M synapses.
- ~23.6 pJ per synaptic operation (measured). Idle power: 30 mW.
- Keyword spotting benchmark: 5.3x to 109.1x improvement in energy cost per inference vs conventional hardware. Advantage improves for larger networks.

**Intel Loihi 2 (2021):**
- Intel 4 process, 1M neurons, 120M synapses. ~1W. 10x faster spike processing.
- State Space Model benchmark: 1000x less energy than Jetson Orin Nano, 75x lower latency.
- Caveat: Jetson performs better in offline batched processing mode.

**IBM TrueNorth (Merolla et al., Science 2014):**
- 4,096 cores, 1M neurons, 256M synapses, 28nm. 65 mW typical. ~26 pJ/synaptic event. 46 GSOPS/W.

**IBM NorthPole (2023):**
- Not spiking -- optimized DNN inference chip. 5x more energy efficient than H100. ~4,000x faster than TrueNorth. All memory on-chip.

**SpiNNaker 1 (Furber et al., 2014):**
- 130nm, 18 ARM968 cores per chip. ~1W per chip. ~5.8 uJ per synaptic event. Full machine: 1M ARM cores, ~100 kW.

**SpiNNaker 2 (Mayr et al.):**
- 22nm FDSOI, 153 ARM cores. 10x neural sim capacity per watt vs SpiNNaker 1. Operates down to 0.5V.
- Sandia deployment (March 2025): 1,152 chips, 175M neurons. Claims 18x more efficient than GPUs.
- MNIST: 96.6% accuracy at ~23 uJ per image classification.

**BrainScaleS-2 (Heidelberg, 2022):**
- Mixed-signal. 1000x acceleration vs biological real time.
- Spikey (BrainScaleS precursor): 0.2 mJ per inference -- most efficient in the Ostrau benchmarking study.

**DarwinWafer (2025):**
- Wafer-scale: 64 chiplets. 0.15B neurons, 6.4B synapses. 4.9 pJ/SOP at 333 MHz. ~100W for entire wafer.

### 3.3 Single-neuron energy comparison (Ostrau et al. 2022)

Energy for simulating 1 second of a single neuron:

| Platform | Total Energy (J) |
|:---|:---:|
| Spikey (BrainScaleS precursor) | 1.49e-6 |
| SpiNNaker | 3.33e-4 |
| GPU (RTX 2070) | 3.18e-5 |
| Human Brain | 2.49e-10 |

The biological brain is ~4 orders of magnitude more efficient than current neuromorphic hardware. Humbling.

---

## 4. AC vs MAC energy cost

### 4.1 Horowitz ISSCC 2014 -- the canonical reference

This is THE foundational reference for energy per arithmetic operation. All values for 45nm CMOS at 0.9V.

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
