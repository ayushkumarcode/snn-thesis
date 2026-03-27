# SNN Edge Deployment Research: Deploying Spiking Neural Networks on Real Hardware

**Research Date:** 2026-02-25
**Scope:** Comprehensive investigation into deploying SNNs on edge devices, FPGAs, microcontrollers, and accessible hardware for an undergraduate thesis project.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Can SNNs Be Deployed on Real Hardware?](#2-can-snns-be-deployed-on-real-hardware)
3. [Frameworks Supporting Edge Deployment](#3-frameworks-supporting-edge-deployment)
4. [Power Consumption: SNN vs ANN on Edge Devices](#4-power-consumption-snn-vs-ann-on-edge-devices)
5. [Student Projects That Have Done This](#5-student-projects-that-have-done-this)
6. [FPGA-Based SNN Deployment for Undergraduates](#6-fpga-based-snn-deployment-for-undergraduates)
7. [Raspberry Pi and Accessible Hardware](#7-raspberry-pi-and-accessible-hardware)
8. [Recent Papers and Projects (2023-2026)](#8-recent-papers-and-projects-2023-2026)
9. [Concrete Hardware Cost Breakdown](#9-concrete-hardware-cost-breakdown)
10. [Feasibility Assessment and Recommended Project Paths](#10-feasibility-assessment-and-recommended-project-paths)
11. [Sources](#11-sources)

---

## 1. Executive Summary

Deploying spiking neural networks on real edge hardware is not only feasible but is an active and growing area of research with multiple proven pathways accessible to undergraduate students. The research reveals three tiers of deployment difficulty, all of which are achievable within a thesis project timeline:

**Tier 1 -- Software SNN on CPU (Easiest, 2-4 weeks to deploy):** Train an SNN using snnTorch or SpikingJelly in Python, then convert the trained model to optimized C code and run inference on a microcontroller (STM32, Arduino Portenta H7) or Raspberry Pi. This path has been demonstrated with concrete benchmarks: a C runtime achieves 21x speedup over Python snnTorch, and compiled models fit within 250 KB of SRAM.

**Tier 2 -- FPGA Accelerated SNN (Moderate, 6-10 weeks):** Train an SNN in snnTorch, apply quantization-aware training, and deploy to an FPGA using HLS (High-Level Synthesis). The open-neuromorphic/fpga-snntorch workshop from ISFPGA 2024 provides a complete pipeline for deploying on the AMD Kria KV260 ($199). The Spiker+ framework can auto-generate VHDL from a Python description, consuming only 180 mW per inference on small FPGAs.

**Tier 3 -- Neuromorphic Hardware (Advanced but possible):** Deploy via NIR (Neuromorphic Intermediate Representation) to SpiNNaker2, SynSense Xylo/Speck, or Intel Loihi 2. The NIR standard now connects 7 simulators to 4+ hardware platforms, and SynSense claims interns can deploy their first application within 1-2 months.

The strongest undergraduate thesis angle would combine Tier 1 and Tier 2: train an SNN for DVS gesture recognition using snnTorch, deploy it on both a microcontroller (C runtime) and an FPGA (HLS pipeline), then measure and compare power, latency, and accuracy across platforms. This produces a highly demonstrable project with concrete, measurable results.

---

## 2. Can SNNs Be Deployed on Real Hardware?

### 2.1 Yes -- Verified Across Multiple Platforms

SNNs have been successfully deployed on the following hardware categories, with published results:

**Microcontrollers (ARM Cortex-M):**
- STM32F407VG6 (Cortex-M4, 168 MHz, 192 KB RAM): eLSNN implementation achieving 54% lower execution time vs naive implementation
- Arduino Portenta H7 (Cortex-M7, 480 MHz, 1024 KB SRAM): N-MNIST inference with ~250 KB total memory (50 KB weights + 200 KB neuron states/buffers)
- ESP32 (dual-core, 240 MHz, 520 KB RAM): Evaluated for SNN sensor processing

**FPGAs:**
- PYNQ-Z2 (Xilinx Zynq XC7Z020): Multiple student projects, ~28 LUTs per neuron
- AMD Kria KV260 (Zynq UltraScale+): Official snnTorch-to-FPGA workshop pipeline
- Basys3/Cmod (Xilinx 7-Series): Low-cost SNN deployment demonstrated
- Xilinx Artix-7: MNIST recognition in 0.52 ms/image

**Neuromorphic Chips:**
- Intel Loihi 2: DVS gesture recognition at 89.64% accuracy on 37 cores
- SpiNNaker2: DVS gesture recognition at 94.13% accuracy, 459 mJ per gesture
- SynSense Speck: 320,000-neuron processor at milliwatt power
- SynSense Xylo: Audio processing at microwatt energy budget
- BrainChip Akida AKD1500: Sub-1W edge AI co-processor

### 2.2 Key Constraint: Spike Sparsity

The practical advantage of SNNs on edge hardware depends heavily on spike sparsity. Research shows:
- Below 0.44 spikes/synapse (VGG16) or 0.42 (AlexNet), SNNs are more energy-efficient than ANNs
- At 0.1 spike sparsity, SNNs are 3.6x more energy-efficient
- Above 0.5 spikes/synapse, SNNs cannot compete with ANNs on digital hardware

This means the choice of task, encoding scheme, and network architecture directly determines whether the SNN edge deployment offers genuine advantages.

---

## 3. Frameworks Supporting Edge Deployment

### 3.1 Training Frameworks (GPU, then export)

| Framework | Hardware Targets | Export Path | Key Feature |
|-----------|-----------------|-------------|-------------|
| snnTorch | FPGA (HLS), Loihi 2, SpiNNaker2 | NIR export, HLS C++ | Best tutorials, ISFPGA 2024 FPGA workshop |
| SpikingJelly | CPU/GPU simulation | Python/C conversion | Best DVS128 built-in support |
| Norse | CPU/GPU, via NIR | NIR export | PyTorch-native, good for small networks |
| Lava / Lava-DL | Intel Loihi 1/2 | Native Loihi compiler | Official Intel framework, cloud access only |
| Rockpool / Sinabs | SynSense Xylo/Speck | Native SynSense compiler | Direct hardware deployment |
| Nengo | SpiNNaker, Loihi | NengoDL, NengoFPGA | Functional brain modeling approach |

### 3.2 Deployment/Compilation Frameworks

**NIR (Neuromorphic Intermediate Representation):**
- Published in Nature Communications (2024)
- Connects 7 simulators (snnTorch, Norse, Lava, Nengo, Rockpool, Sinabs, Spyx) to 4 hardware platforms (Loihi 2, Speck, SpiNNaker2, Xylo)
- Defines a common set of computational primitives (LIF neurons, convolutions)
- Train once, deploy anywhere -- the "ONNX of neuromorphic computing"
- GitHub: https://github.com/neuromorphs/NIR

**Spiker+ (FPGA Auto-Generation):**
- Python-to-VHDL automatic generation
- Supports 6 neuron models (IF, I-order LIF, II-order LIF, hard/subtractive reset)
- 2 network architectures (feedforward fully-connected, fully-connected recurrent)
- Requires only 7,612 logic cells and 18 BRAMs
- 180 mW power consumption per inference
- Includes video tutorials for the complete workflow
- GitHub: https://github.com/smilies-polito/Spiker

**TENNLab Embedder:**
- Translates SNNs to portable, dependency-free C code libraries
- Targets microcontrollers and embedded von Neumann processors
- Low size, weight, and power (SWaP) focus
- From University of Tennessee Knoxville

**ModNEF (Open-Source FPGA Emulator):**
- Modular neuromorphic digital hardware architecture for FPGAs
- LIF neuron models with different emulation strategies
- Users control power consumption, memory, precision tradeoffs
- Evaluated on Zynq XC7Z020 with MNIST and N-MNIST
- Published in ACM Transactions on Architecture and Code Optimization

**S2NN-HLS:**
- Spiking neural network for Zynq devices via Vivado HLS
- Izhikevich neuron model (biologically realistic)
- DDR memory energy reduction up to 77%, PL energy reduction up to 76%
- Less than 2% energy of software-only implementation
- GitHub: https://github.com/eejlny/S2NN-HLS

### 3.3 The snnTorch-to-FPGA Pipeline (Most Recommended for Thesis)

The open-neuromorphic/fpga-snntorch repository provides the most complete, documented pipeline:

1. **Train** SNN using snnTorch (Python, GPU via Google Colab)
2. **Quantize** weights and states (quantization-aware training)
3. **Export** to HLS C++ using AMD Vitis HLS compiler
4. **Synthesize** hardware design (dataflow architecture for deep SNNs)
5. **Deploy** on AMD Kria KV260 using PYNQ Python interface
6. **Test** using provided bitstream, PYNQ scripts, and hardware handoff files

Workshop presented at ISFPGA 2024 by Jason Eshraghian (UC Santa Cruz) and Fabrizio Ottati (NXP Semiconductors). Repository: https://github.com/open-neuromorphic/fpga-snntorch

---

## 4. Power Consumption: SNN vs ANN on Edge Devices

### 4.1 Concrete Measurements

| Platform | Network Type | Task | Power/Energy | Accuracy |
|----------|-------------|------|--------------|----------|
| FPGA (Spiker+) | SNN (LIF) | MNIST | 180 mW per inference | Competitive |
| FPGA (SYNtzulu) | SNN | Time-series | 14.2 mW peak, 0.3 mW idle | -- |
| FPGA (Hybrid HNN) | SNN+ANN | Classification | 1,192 mW | 87% |
| FPGA (Pure ANN) | CNN | Classification | 1,248 mW | 88% |
| SENECA neuromorphic | SNN | Vision | 927 uJ (62.5% of ANN time) | -- |
| SENECA neuromorphic | ANN | Vision | 1,232 uJ | -- |
| Analog SNN chip | SNN (STDP) | MNIST | 530 uW at 10 MHz | -- |
| TrueNorth | SNN/CNN | DVS gesture | <200 mW | 96.5% |
| SpiNNaker2 | SNN (Q-SNN) | DVS gesture | 459 mJ per gesture | 94.13% |
| General estimate | SNN (STDP) | Various | ~5 mJ per inference | -- |
| General estimate | ANN | Various | ~200 mJ per inference | -- |
| FPGA Artix XC7A200T | CNN | ImageNet-class | 1,775 mW at 100 MHz | -- |

### 4.2 The Nuanced Reality

The headline claim "SNNs are 10-100x more efficient" requires significant caveats:

**When SNNs win:**
- On neuromorphic hardware (Loihi, SpiNNaker, Speck) where the hardware is designed for event-driven computation
- When spike sparsity is low (<0.44 spikes/synapse)
- For always-on, event-driven sensing tasks (e.g., DVS camera input)
- On FPGAs with spike-aware optimizations exploiting sparsity

**When SNNs lose:**
- On standard digital FPGAs without sparsity exploitation, SNNs are "clearly less energy efficient than their equivalent CNNs in the general case" (Efficiency analysis study, 2022)
- When membrane potentials must be stored in memory (unlike CNNs where neurons are computed sequentially)
- When spike sparsity exceeds 0.5 spikes/synapse

**Key insight for the thesis:** The most honest and valuable contribution would be to measure both SNN and ANN on the same edge hardware and present the real tradeoffs rather than claiming unconditional SNN superiority. This makes for a stronger thesis than cherry-picking favorable comparisons.

---

## 5. Student Projects That Have Done This

### 5.1 Purdue Polytechnic Capstone: FPGA SNN Lane-Following Robot

- **Level:** Undergraduate capstone (Senior project)
- **Task:** SNN controller for autonomous lane-following vehicle with obstacle avoidance
- **Hardware:** FPGA (unspecified Xilinx board)
- **Architecture:** 4 input neurons (12-bit to 8-bit scaling via CORDIC) -> 16 synapses -> 4 hidden neurons -> 8 synapses -> 2 output neurons
- **Outcome:** Functional lane-following with SNN replacing binary logic controller
