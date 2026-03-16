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
- **URL:** https://polytechnic.purdue.edu/capstone-project/fpga-implementation-of-spiking-neural-network-based-controller-for-lane-following

### 5.2 Washington University CSE462M (Spring 2025): SNNs on FPGAs

- **Level:** Undergraduate course project
- **Hardware:** PYNQ-Z2 (Xilinx Zynq XC7Z020)
- **Tools:** Xilinx Vivado, custom HDL
- **Key findings:** Initial single neuron consumed 13% of LUTs using 32-bit fixed-point; optimized to Q2.6 (8-bit) precision, dramatically reducing resource usage to ~28 LUTs/neuron
- **Challenge documented:** Precision vs. resource tradeoff is the critical design decision

### 5.3 UCSD CSE237D: PYNQ SNN Accelerator

- **Level:** Graduate course project (but scope achievable by strong undergrads)
- **Hardware:** PYNQ-Z1 (Xilinx Zynq)
- **Task:** SNN inference accelerator on FPGA
- **Researchers:** Srinithya Nagiri, Tanaya Kolankari

### 5.4 ANN-vs-SNN Comparison Project (GitHub)

- **Repository:** https://github.com/NicolaCST/ANN-vs-SNN
- **Focus:** Benchmarking performance and power consumption between ANNs and SNNs
- **Contains:** Jupyter notebook (VCS_Project.ipynb) + documentation
- **Relevance:** Could be extended with actual hardware measurements for a thesis

### 5.5 SNN Arduino Library for Robotics

- **Repository:** https://github.com/RishabhMalviya/SNN_Arduino
- **Focus:** LIF neuron implementation for Arduino-based robots
- **Approach:** Liquid state machine paradigm with LIF neurons
- **Application:** Neural robot control via L293D motor drivers
- **Level:** Accessible to undergraduates with Arduino experience

---

## 6. FPGA-Based SNN Deployment for Undergraduates

### 6.1 Feasibility Assessment: YES, with the right approach

FPGA-based SNN deployment is achievable for an undergraduate thesis. The key factors:

**What makes it feasible:**
- Spiker+ auto-generates VHDL from Python -- no need to hand-write HDL
- The ISFPGA 2024 workshop (fpga-snntorch) provides a complete bitstream and deployment scripts
- PYNQ framework allows Python-based interaction with the FPGA (no bare-metal programming)
- HLS (High-Level Synthesis) allows writing in C++ rather than Verilog/VHDL
- Multiple student projects at the undergraduate level have been completed successfully

**What makes it challenging:**
- Learning Xilinx Vivado toolchain (steep but manageable with tutorials)
- Understanding fixed-point arithmetic and quantization tradeoffs
- Debugging hardware designs is harder than debugging software
- Synthesis times can be long (30 minutes to hours per iteration)

**Recommended approach for an undergraduate:**

Option A -- Use Spiker+ (Fastest to results):
1. Design SNN architecture in Python using Spiker+ framework
2. Auto-generate VHDL accelerator
3. Deploy on PYNQ-Z2 or similar small FPGA
4. Measure power, latency, accuracy
5. Compare against software-only implementation

Option B -- Use snnTorch + HLS (Most educational):
1. Train SNN in snnTorch with quantization-aware training
2. Follow ISFPGA 2024 workshop pipeline
3. Deploy on AMD Kria KV260 using Vitis HLS
4. Use PYNQ Python interface for testing

Option C -- Use ModNEF (Most flexible):
1. Use the open-source modular FPGA SNN emulator
2. Configure LIF neurons with desired precision
3. Deploy on Zynq XC7Z020 (PYNQ-Z2)
4. Evaluate MNIST/N-MNIST classification

### 6.2 Required Skills and Learning Timeline

| Skill | Time to Learn | Resources |
|-------|---------------|-----------|
| snnTorch basics | 1-2 weeks | Official tutorials (7 tutorials available) |
| FPGA/Vivado basics | 2-3 weeks | PYNQ getting started guide, Xilinx tutorials |
| Vitis HLS | 1-2 weeks | AMD documentation, HLS examples |
| Fixed-point quantization | 1 week | snnTorch quantization tutorial |
| PYNQ Python overlay | 1 week | PYNQ documentation |
| End-to-end integration | 2-3 weeks | fpga-snntorch repository |

Total estimated learning curve: 8-12 weeks, which fits within a thesis timeline.

---

## 7. Raspberry Pi and Accessible Hardware

### 7.1 Raspberry Pi as SNN Platform

The Raspberry Pi is a viable SNN deployment target, but with important caveats:

**Software SNN inference (Python/C):**
- Run snnTorch or SpikingJelly models directly in Python on Raspberry Pi 4/5
- Python inference will be slow (~2.4 seconds per sample for N-MNIST on desktop, much slower on Pi)
- Converting to optimized C achieves 21x speedup, making real-time inference plausible
- PyTorch on Raspberry Pi 5 can achieve ~40 FPS for standard neural networks (relevant baseline)

**Hardware acceleration on Pi:**
- Raspberry Pi AI Kit ($70) features Hailo-8L with 13 TOPS, but this is designed for conventional ANNs, not SNNs
- No native SNN support on the Hailo-8L accelerator
- The Pi's ARM CPU can run SNN inference in C, but without the event-driven efficiency advantages

**Recommended Raspberry Pi approach:**
1. Train SNN using snnTorch on GPU (Google Colab)
2. Export model weights
3. Implement lightweight C SNN runtime on Raspberry Pi
4. Compare inference speed/power vs. equivalent ANN (TensorFlow Lite Micro or similar)
5. Measure actual power consumption using a USB power meter

### 7.2 Other Accessible Hardware Options

**Arduino Portenta H7 (~$80):**
- Cortex-M7 at 480 MHz, 1024 KB SRAM
- Proven for SNN inference: N-MNIST with ~250 KB memory footprint
- C runtime achieves usable inference times
- Good for demonstrating "SNN on a microcontroller" angle

**STM32 Discovery Boards (~$15-30):**
- STM32F407 (Cortex-M4, 168 MHz, 192 KB RAM)
- Demonstrated for eLSNN with 54% lower execution time vs naive implementation
- Very affordable, well-documented ecosystem (STM32CubeAI)

**ESP32 (~$5-10):**
- Dual-core 240 MHz, 520 KB RAM
- WiFi/Bluetooth for IoT demonstration
- Tight memory constraints but feasible for small SNNs

**BrainChip Akida Development Kit (price on application):**
- Actual neuromorphic SoC with SNN support
- Available for Raspberry Pi form factor
- Edge Impulse integration for training and deployment
- Sub-1W operation
- Limitation: Pricing not publicly available, may be expensive

**SynSense Xylo/Speck Development Kits (price on application):**
- True neuromorphic hardware
- Speck: 320,000 neurons, milliwatt power, vision processing
- Xylo: Microwatt energy budget, audio processing
- SynSense claims interns can deploy first application in 1-2 months
- Rockpool Python library for deployment
- Limitation: Academic pricing not publicly listed

### 7.3 Hybrid Approach: Raspberry Pi + FPGA

An interesting thesis angle is using a Raspberry Pi as the host controller connected to an FPGA accelerator:
- Pi handles data loading, preprocessing, results display
- FPGA handles SNN inference acceleration
- PYNQ-Z2 has a Raspberry Pi GPIO header for integration
- This mirrors real edge deployment architectures

---

## 8. Recent Papers and Projects (2023-2026)

### 8.1 Key Papers

| Year | Title | Key Contribution |
|------|-------|-----------------|
| 2025 | "Spiking neural networks on FPGA: A survey" (Neural Networks, Vol 186) | Comprehensive survey of FPGA SNN methodologies |
| 2025 | "Efficient Deployment of SNNs on SpiNNaker2 for DVS Gesture Recognition Using NIR" | Complete snnTorch->NIR->SpiNNaker2 pipeline, 94.13% accuracy |
| 2025 | "Compression and Inference of SNNs on Resource-Constrained Hardware" | 21x speedup C runtime vs Python, Arduino Portenta deployment |
| 2025 | "ModNEF: Open Source Modular Neuromorphic Emulator for FPGA" (ACM TACO) | Open-source FPGA SNN framework evaluated on MNIST/N-MNIST |
| 2025 | "A Robust, Open-Source Framework for SNNs on Low-End FPGAs" | Low-cost Artix-7 FPGA, 0.52 ms/image MNIST |
| 2024 | "Spiker+: Framework for SNN FPGA Accelerators at the Edge" | Auto-generate VHDL, 180 mW, 7612 logic cells |
| 2024 | "Neuromorphic Intermediate Representation" (Nature Communications) | NIR standard connecting 7 simulators to 4 hardware platforms |
| 2024 | "Energy-Aware FPGA Implementation of SNN with LIF Neurons" | Focus on energy measurement methodology for FPGA SNNs |
| 2024 | "Enabling Efficient On-Edge SNN Acceleration with Flexible FPGA Architectures" | Ultra-low area and power SNN on FPGA |
| 2024 | "SpikeExplorer: Hardware-Oriented Design Space Exploration for SNNs on FPGA" | Automated architecture search for FPGA-optimized SNNs |
| 2024 | "Energy efficient and low-latency SNNs on embedded microcontrollers" (Neural Computing and Applications) | eLSNN on STM32F4, spiking activity tuning |
| 2024 | ISFPGA 2024 Workshop: "Deploying SNNs to FPGAs via HLS" | Complete snnTorch-to-KV260 pipeline with code |
| 2023 | "An FPGA implementation of Bayesian inference with SNNs" (Frontiers) | Bayesian SNN on FPGA with uncertainty estimation |

### 8.2 Key GitHub Repositories

| Repository | Stars | Focus |
|------------|-------|-------|
| open-neuromorphic/fpga-snntorch | -- | ISFPGA 2024 workshop: snnTorch to FPGA deployment |
| smilies-polito/Spiker | -- | Python-to-VHDL SNN accelerator generation |
| neuromorphs/NIR | -- | Neuromorphic Intermediate Representation |
| eejlny/S2NN-HLS | -- | SNN on Zynq via Vivado HLS (Izhikevich model) |
| OpenHEC/SNN-simulator-on-PYNQcluster | -- | SNN simulation on PYNQ clusters |
| im-afan/snn-fpga | -- | SNN on low-end Basys3/Cmod FPGAs |
| RishabhMalviya/SNN_Arduino | -- | LIF neurons on Arduino for robotics |
| NicolaCST/ANN-vs-SNN | -- | Performance/power comparison benchmark |
| jeshraghian/snntorch | 3k+ | Primary SNN training framework |
| fangwei123456/spikingjelly | 3k+ | SpikingJelly SNN framework |

---

## 9. Concrete Hardware Cost Breakdown

### 9.1 Budget Tier: Under $50

| Hardware | Approximate Cost | SNN Deployment Path |
|----------|-----------------|-------------------|
| STM32F4 Discovery Board | $15-25 | C runtime for small SNNs, eLSNN |
| ESP32 DevKit | $5-10 | WiFi-enabled SNN sensor node |
| Arduino Nano 33 BLE | $20-25 | TinyML SNN with BLE connectivity |
| Raspberry Pi 4 (4GB) | $35 (if available) | Python/C SNN inference |

### 9.2 Mid-Range Tier: $50-200

| Hardware | Approximate Cost | SNN Deployment Path |
|----------|-----------------|-------------------|
| Raspberry Pi 5 (8GB) | $80 | Best CPU-based SNN inference |
| Arduino Portenta H7 | $80 | Proven C SNN runtime, N-MNIST |
| PYNQ-Z2 (Xilinx Zynq) | $120-180 | FPGA SNN with Python interface |
| Basys3 (Xilinx Artix-7) | $150-180 | Low-cost FPGA SNN |

### 9.3 Higher Tier: $200+

| Hardware | Approximate Cost | SNN Deployment Path |
|----------|-----------------|-------------------|
| AMD Kria KV260 | $199 | Official snnTorch-to-FPGA pipeline |
| BrainChip Akida Dev Kit | Contact vendor | True neuromorphic SoC |
| SynSense Xylo/Speck Kit | Contact vendor | Ultra-low-power neuromorphic |

### 9.4 Free/Cloud Options

| Platform | Cost | Access |
|----------|------|--------|
| Google Colab | Free | SNN training with GPU |
| Intel INRC (Loihi 2) | Free for researchers | Cloud-hosted Loihi systems, requires membership |
| SpiNNaker (Manchester) | Free for academics | Cloud access via sPyNNaker |

---

## 10. Feasibility Assessment and Recommended Project Paths

### 10.1 Overall Feasibility Verdict: HIGHLY FEASIBLE

A "deploy SNN on real hardware" thesis project is not only feasible but represents a strong, timely, and demonstrable thesis topic. The field has matured significantly in 2024-2025 with frameworks like Spiker+, NIR, and the fpga-snntorch workshop specifically designed to make this accessible.

### 10.2 Recommended Project Path A: "SNN Edge Deployment Benchmark" (Strongest Thesis)

**Concept:** Train an SNN for DVS128 gesture recognition, deploy on multiple platforms, measure and compare.

**Pipeline:**
1. Train SNN using snnTorch on DVS128 Gesture dataset (GPU/Colab)
2. Deploy as optimized C runtime on STM32 or Raspberry Pi
3. Deploy on FPGA via HLS (PYNQ-Z2 or Kria KV260)
4. Train equivalent ANN (CNN) for the same task
5. Deploy ANN on same platforms using TensorFlow Lite Micro / TFLite
6. Measure and compare: accuracy, latency, power consumption, memory footprint
7. Analyze when and why SNNs offer advantages (or not)

**Hardware cost:** $200-400 total (PYNQ-Z2 + STM32 Discovery + USB power meter)
**Timeline:** 12-16 weeks
**Novelty:** Honest, measured comparison on real hardware -- addresses the open question "Are SNNs really more efficient?"
**Demonstrability:** Live demo showing inference on multiple platforms with power measurements

### 10.3 Recommended Project Path B: "FPGA SNN Accelerator for DVS Gestures" (More Technical)

**Concept:** Build a complete pipeline from DVS data to FPGA inference.

**Pipeline:**
1. Train SNN on DVS128 using snnTorch with quantization-aware training
2. Use Spiker+ or HLS to generate FPGA accelerator
3. Deploy on PYNQ-Z2
4. Measure resource utilization (LUTs, BRAMs, DSPs), power, latency
5. Explore design space: precision (8-bit vs 16-bit), neuron count, architecture
6. Compare against published results

**Hardware cost:** $150-200 (PYNQ-Z2 board)
**Timeline:** 14-18 weeks
**Novelty:** Complete characterization of SNN FPGA accelerator design space
**Demonstrability:** PYNQ Jupyter notebook interface showing live FPGA inference

### 10.4 Recommended Project Path C: "SNN on Microcontroller for Edge AI" (Most Accessible)

**Concept:** Demonstrate practical SNN inference on ultra-low-cost hardware.

**Pipeline:**
1. Train SNN using snnTorch on MNIST or N-MNIST
2. Convert to optimized C runtime (following the compression paper methodology)
3. Deploy on STM32F4 Discovery or Arduino Portenta H7
4. Measure inference time, memory usage, power consumption
5. Compare against TensorFlow Lite Micro CNN on same hardware
6. Demonstrate keyword spotting or gesture classification application

**Hardware cost:** $15-80 (STM32 Discovery or Portenta H7)
**Timeline:** 8-12 weeks
**Novelty:** Practical SNN-vs-ANN comparison on commodity microcontroller
**Demonstrability:** Standalone sensor + microcontroller demo

### 10.5 Critical Success Factors

1. **Start with software simulation first.** Get the SNN trained and validated before touching hardware.
2. **Use snnTorch as the training framework.** It has the best FPGA deployment pipeline and tutorials.
3. **Choose quantization-aware training from the start.** 8-bit integer models deploy much more easily than 32-bit float.
4. **Measure power properly.** Use a USB power meter (e.g., Monsoon Power Monitor, or even a simple USB ammeter) for credible results.
5. **Do not oversell SNN efficiency.** The research shows the comparison is nuanced. An honest analysis makes a stronger thesis than overclaiming.
6. **Use N-MNIST or DVS128 Gesture as the benchmark.** These are standard neuromorphic datasets with well-known baselines.

### 10.6 Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| FPGA toolchain issues (Vivado) | Medium | Use Spiker+ auto-generation or pre-built bitstreams |
| SNN accuracy lower than ANN | Expected | This is a finding, not a failure -- document the tradeoff |
| Hardware procurement delays | Medium | Start with software simulation; have microcontroller as backup |
| Learning curve too steep | Low-Medium | Follow existing tutorials (ISFPGA 2024 workshop); start with Tier 1 |
| Power measurement difficulty | Medium | Use USB power meters; document methodology carefully |

---

## 11. Sources

### Frameworks and Tools
- [snnTorch Documentation](https://snntorch.readthedocs.io/)
- [snnTorch GitHub](https://github.com/jeshraghian/snntorch)
- [snnTorch NIR Export](https://snntorch.readthedocs.io/en/latest/snntorch.export_nir.html)
- [Lava Software Framework](https://lava-nc.org/)
- [Lava GitHub](https://github.com/lava-nc/lava)
- [NIR (Neuromorphic Intermediate Representation)](https://neuroir.org/)
- [NIR GitHub](https://github.com/neuromorphs/NIR)
- [Spiker+ GitHub](https://github.com/smilies-polito/Spiker)
- [S2NN-HLS GitHub](https://github.com/eejlny/S2NN-HLS)
- [SNN Arduino Library](https://github.com/RishabhMalviya/SNN_Arduino)
- [ModNEF Paper (ACM TACO)](https://dl.acm.org/doi/10.1145/3730581)
- [Open Neuromorphic](https://open-neuromorphic.org/)

### Workshop and Tutorial Resources
- [ISFPGA 2024 Workshop: fpga-snntorch](https://github.com/open-neuromorphic/fpga-snntorch)
- [Open Neuromorphic Workshops](https://open-neuromorphic.org/workshops/)
- [Neuromorphic Software Guide](https://open-neuromorphic.org/neuromorphic-computing/software/)
- [Neuromorphic Hardware Guide](https://open-neuromorphic.org/neuromorphic-computing/hardware/)

### Key Research Papers
- [NIR: A Unified Instruction Set for Brain-Inspired Computing (Nature Communications, 2024)](https://www.nature.com/articles/s41467-024-52259-9)
- [Spiker+: SNN FPGA Accelerators for Inference at the Edge (arXiv, 2024)](https://arxiv.org/html/2401.01141v1)
- [Efficient Deployment of SNNs on SpiNNaker2 via NIR (arXiv, 2025)](https://arxiv.org/html/2504.06748v1)
- [Compression and Inference of SNNs on Resource-Constrained Hardware (arXiv, 2025)](https://arxiv.org/html/2511.12136)
- [Energy-Aware FPGA Implementation of SNN with LIF Neurons (arXiv, 2024)](https://arxiv.org/html/2411.01628v1)
- [SNN FPGA Survey: Methodologies and Recent Advancements (Neural Networks, 2025)](https://www.sciencedirect.com/science/article/abs/pii/S0893608025001352)
- [Efficiency Analysis: SNN vs ANN on FPGAs (JSA, 2022)](https://www.sciencedirect.com/science/article/abs/pii/S1383762122002508)
- [Are SNNs Really More Energy-Efficient? Hardware-Aware Study (IEEE, 2022)](https://ieeexplore.ieee.org/iel7/7433297/7777658/09927729.pdf)
- [Energy Efficient SNNs on Embedded Microcontrollers (Neural Computing, 2024)](https://link.springer.com/article/10.1007/s00521-024-10191-5)
- [Edge Intelligence with Spiking Neural Networks (arXiv, 2025)](https://arxiv.org/html/2507.14069v1)
- [SpikeExplorer: HW-Oriented Design Space Exploration (Electronics, 2024)](https://www.mdpi.com/2079-9292/13/9/1744)
- [Enabling Efficient On-Edge SNN Acceleration on FPGA (Electronics, 2024)](https://www.mdpi.com/2079-9292/13/6/1074)
- [A Robust Open-Source Framework for SNNs on Low-End FPGAs (arXiv, 2025)](https://arxiv.org/html/2507.07284v1)
- [SNN-Based Near-Sensor Computing for Structural Health (Future Internet, 2021)](https://www.mdpi.com/1999-5903/13/8/219)
- [TENNLab: Generating SNN Code Libraries for Embedded Systems (ICONS, 2025)](https://neuromorphic.eecs.utk.edu/publications/2025-07-29-generating-spiking-neural-network-code-libraries-for-embedded-systems/)

### Student Projects
- [Purdue Capstone: FPGA SNN Lane-Following Robot](https://polytechnic.purdue.edu/capstone-project/fpga-implementation-of-spiking-neural-network-based-controller-for-lane-following)
- [Washington University CSE462M: SNNs on FPGAs (Spring 2025)](https://ese.washu.edu/documents/Spiking-Neural-Networks-on-FPGAs.pdf)
- [UCSD CSE237D: PYNQ SNN Accelerator](https://kastner.ucsd.edu/ryan/wp-content/uploads/sites/5/2014/03/admin/Pynq-SNN.pdf)
- [ANN-vs-SNN Comparison Project](https://github.com/NicolaCST/ANN-vs-SNN)

### Hardware Platforms
- [Intel Loihi 2](https://open-neuromorphic.org/neuromorphic-computing/hardware/loihi-2-intel/)
- [BrainChip Akida](https://brainchip.com/ip/)
- [SynSense Xylo](https://www.synsense.ai/products/xylo/)
- [SynSense Speck](https://www.synsense.ai/products/speck-2/)
- [AMD Kria KV260](https://www.amd.com/en/products/system-on-modules/kria/k26/kv260-vision-starter-kit.html)
- [PYNQ-Z2](https://www.seeedstudio.com/PYNQ-Z2-board-based-on-Xilinx-Zynq-C7Z020-SoC-p-2835.html)
- [Raspberry Pi AI Kit](https://www.raspberrypi.com/products/ai-kit/)
- [SNN Library Benchmarks](https://open-neuromorphic.org/blog/spiking-neural-network-framework-benchmarking/)
