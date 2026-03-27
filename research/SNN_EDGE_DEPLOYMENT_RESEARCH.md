# deploying SNNs on real edge hardware

i looked into whether it's actually possible to deploy SNNs on edge devices, FPGAs, microcontrollers, etc. for an undergrad thesis. the short answer: yes, and there are multiple proven pathways that are genuinely accessible. the field matured a lot in 2024-2025 with frameworks specifically designed to make this doable.

there are basically three tiers of difficulty:

**tier 1 -- software SNN on CPU (easiest, 2-4 weeks to deploy):** train SNN in snnTorch or SpikingJelly, convert to optimized C, run inference on a microcontroller (STM32, Arduino Portenta H7) or Raspberry Pi. a C runtime gets 21x speedup over Python snnTorch, and compiled models fit in ~250 KB SRAM.

**tier 2 -- FPGA accelerated SNN (moderate, 6-10 weeks):** train in snnTorch, do quantization-aware training, deploy to FPGA via HLS. the open-neuromorphic/fpga-snntorch workshop from ISFPGA 2024 has a complete pipeline for AMD Kria KV260 ($199). Spiker+ can auto-generate VHDL from Python, using only 180 mW per inference.

**tier 3 -- neuromorphic hardware (advanced but possible):** deploy via NIR to SpiNNaker2, SynSense Xylo/Speck, or Intel Loihi 2. NIR now connects 7 simulators to 4+ hardware platforms. SynSense claims interns can deploy their first app in 1-2 months.

the strongest thesis angle would be combining tier 1 and 2: train SNN for gesture recognition, deploy on both microcontroller and FPGA, measure and compare power, latency, and accuracy across platforms. concrete, demonstrable results.

---

## 1. can SNNs actually run on real hardware?

### yes -- verified across multiple platforms

**microcontrollers (ARM Cortex-M):**
- STM32F407VG6 (Cortex-M4, 168 MHz, 192 KB RAM): eLSNN implementation, 54% lower execution time vs naive
- Arduino Portenta H7 (Cortex-M7, 480 MHz, 1024 KB SRAM): N-MNIST inference, ~250 KB total memory (50 KB weights + 200 KB buffers)
- ESP32 (dual-core, 240 MHz, 520 KB RAM): evaluated for SNN sensor processing

**FPGAs:**
- PYNQ-Z2 (Xilinx Zynq XC7Z020): multiple student projects, ~28 LUTs per neuron
- AMD Kria KV260 (Zynq UltraScale+): official snnTorch-to-FPGA workshop pipeline
- Basys3/Cmod (Xilinx 7-Series): low-cost deployment demonstrated
- Xilinx Artix-7: MNIST recognition in 0.52 ms/image

**neuromorphic chips:**
- Intel Loihi 2: DVS gesture at 89.64%, 37 cores
- SpiNNaker2: DVS gesture at 94.13%, 459 mJ per gesture
- SynSense Speck: 320K neurons at milliwatt power
- SynSense Xylo: audio processing at microwatt budget
- BrainChip Akida AKD1500: sub-1W edge AI co-processor

### the spike sparsity constraint

the practical advantage depends heavily on spike sparsity:
- below 0.44 spikes/synapse (VGG16) or 0.42 (AlexNet), SNNs are more energy-efficient than ANNs
- at 0.1 spike sparsity, SNNs are 3.6x more efficient
- above 0.5 spikes/synapse, SNNs can't compete with ANNs on digital hardware

so the task, encoding scheme, and architecture directly determine whether edge deployment actually offers advantages.

---

## 2. frameworks that support edge deployment

### training frameworks (GPU, then export)

| Framework | Hardware targets | Export path | Key feature |
|-----------|----------------|-------------|-------------|
| snnTorch | FPGA (HLS), Loihi 2, SpiNNaker2 | NIR export, HLS C++ | Best tutorials, ISFPGA 2024 FPGA workshop |
| SpikingJelly | CPU/GPU simulation | Python/C conversion | Best DVS128 support |
| Norse | CPU/GPU, via NIR | NIR export | PyTorch-native, good for small networks |
| Lava / Lava-DL | Intel Loihi 1/2 | Native Loihi compiler | Official Intel framework, cloud only |
| Rockpool / Sinabs | SynSense Xylo/Speck | Native SynSense compiler | Direct hardware deployment |
| Nengo | SpiNNaker, Loihi | NengoDL, NengoFPGA | Functional brain modeling approach |

### deployment/compilation frameworks

**NIR (Neuromorphic Intermediate Representation):**
- published in Nature Communications (2024)
- connects 7 simulators (snnTorch, Norse, Lava, Nengo, Rockpool, Sinabs, Spyx) to 4 hardware platforms (Loihi 2, Speck, SpiNNaker2, Xylo)
- the "ONNX of neuromorphic computing" -- train once, deploy anywhere
- https://github.com/neuromorphs/NIR

**Spiker+ (FPGA auto-generation):**
- Python-to-VHDL automatic generation
- 6 neuron models, 2 network architectures
- only needs 7,612 logic cells and 18 BRAMs
- 180 mW power per inference
- video tutorials included
- https://github.com/smilies-polito/Spiker

**TENNLab Embedder:** translates SNNs to portable, dependency-free C code libraries. targets microcontrollers and embedded processors. from UT Knoxville.

**ModNEF:** modular neuromorphic FPGA architecture. users control power/memory/precision tradeoffs. evaluated on Zynq XC7Z020 with MNIST and N-MNIST. published in ACM TACO.

**S2NN-HLS:** SNN for Zynq via Vivado HLS. Izhikevich neuron model. DDR energy reduction up to 77%, PL energy reduction up to 76%. https://github.com/eejlny/S2NN-HLS

### the snnTorch-to-FPGA pipeline (probably the best path for a thesis)

the open-neuromorphic/fpga-snntorch repo has the most complete documented pipeline:

1. train SNN using snnTorch (Python, GPU via Colab)
2. quantize weights/states (quantization-aware training)
3. export to HLS C++ via AMD Vitis HLS
4. synthesize hardware design (dataflow architecture)
5. deploy on AMD Kria KV260 using PYNQ Python interface
6. test using provided bitstream and scripts

workshop presented at ISFPGA 2024 by Jason Eshraghian and Fabrizio Ottati. repo: https://github.com/open-neuromorphic/fpga-snntorch

---

## 3. power consumption: SNN vs ANN on edge hardware

### concrete measurements

| Platform | Type | Task | Power/Energy | Accuracy |
|----------|------|------|-------------|----------|
| FPGA (Spiker+) | SNN (LIF) | MNIST | 180 mW | Competitive |
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

### the reality is nuanced

the headline claim "SNNs are 10-100x more efficient" has caveats:

**when SNNs win:**
- on neuromorphic hardware designed for event-driven computation
- when spike sparsity is low (<0.44 spikes/synapse)
- for always-on, event-driven sensing (DVS input)
- on FPGAs with spike-aware sparsity exploitation

**when SNNs lose:**
- on standard digital FPGAs without sparsity exploitation, SNNs are "clearly less energy efficient than their equivalent CNNs in the general case" (from a 2022 efficiency analysis)
- when membrane potentials need to be stored in memory
- when spike sparsity exceeds 0.5 spikes/synapse

honestly, the most valuable thesis contribution would be measuring both SNN and ANN on the same edge hardware and presenting the real tradeoffs. that's a stronger thesis than cherry-picking favorable comparisons.

---

## 4. student projects that have done this

### Purdue Polytechnic capstone: FPGA SNN lane-following robot
- undergraduate capstone (senior project)
- SNN controller for autonomous lane-following with obstacle avoidance
- FPGA (Xilinx board): 4 input neurons -> 16 synapses -> 4 hidden -> 8 synapses -> 2 output neurons
- functional lane-following with SNN replacing binary logic controller

### Washington University CSE462M (Spring 2025): SNNs on FPGAs
- undergrad course project
- PYNQ-Z2 (Xilinx Zynq XC7Z020)
- key finding: initial single neuron used 13% of LUTs at 32-bit fixed-point; optimized to Q2.6 (8-bit), dramatically reducing to ~28 LUTs/neuron
- precision vs resource tradeoff is the critical design decision

### UCSD CSE237D: PYNQ SNN accelerator
- graduate course project (achievable scope for strong undergrads)
- PYNQ-Z1 (Xilinx Zynq)
- SNN inference accelerator on FPGA

### other relevant projects
- [ANN-vs-SNN comparison](https://github.com/NicolaCST/ANN-vs-SNN): benchmarking performance and power, could be extended with hardware measurements
- [SNN Arduino Library](https://github.com/RishabhMalviya/SNN_Arduino): LIF neurons for Arduino-based robots, liquid state machine paradigm

---

## 5. FPGA deployment for undergrads

### is it feasible? yes, with the right approach

**what makes it feasible:**
- Spiker+ auto-generates VHDL from Python -- no hand-written HDL
- ISFPGA 2024 workshop provides complete bitstream and scripts
- PYNQ lets you interact with the FPGA from Python (no bare-metal)
- HLS lets you write C++ instead of Verilog/VHDL
- multiple undergrad projects have done this successfully

**what makes it challenging:**
- Xilinx Vivado toolchain has a steep learning curve
- understanding fixed-point arithmetic and quantization tradeoffs
- hardware debugging is harder than software
- synthesis times can be 30 min to hours per iteration

### recommended approaches

**option A -- Spiker+ (fastest to results):** design in Python, auto-generate VHDL, deploy on PYNQ-Z2, measure power/latency/accuracy, compare vs software.

**option B -- snnTorch + HLS (most educational):** train with quantization-aware training, follow ISFPGA 2024 workshop, deploy on Kria KV260 via Vitis HLS, test with PYNQ.

**option C -- ModNEF (most flexible):** open-source modular FPGA emulator, configure LIF neurons with desired precision, deploy on Zynq.

### learning timeline

| Skill | Time | Resources |
|-------|------|-----------|
| snnTorch basics | 1-2 weeks | Official tutorials |
| FPGA/Vivado basics | 2-3 weeks | PYNQ getting started, Xilinx tutorials |
| Vitis HLS | 1-2 weeks | AMD docs, examples |
| Fixed-point quantization | 1 week | snnTorch quantization tutorial |
| PYNQ Python overlay | 1 week | PYNQ docs |
| End-to-end integration | 2-3 weeks | fpga-snntorch repo |

total: 8-12 weeks, fits within a thesis timeline.

---

## 6. Raspberry Pi and other accessible hardware

### Raspberry Pi as SNN platform

it works, but with caveats:

- Python inference is slow (~2.4s per sample for N-MNIST on desktop, worse on Pi)
- converting to optimized C gets 21x speedup, making real-time plausible
- the Pi AI Kit ($70, Hailo-8L, 13 TOPS) is for conventional ANNs, not SNNs
- no event-driven efficiency advantages on the Pi's ARM CPU

recommended approach: train on GPU (Colab), export weights, implement lightweight C runtime on Pi, compare vs ANN (TF Lite Micro), measure power with USB power meter.

### other accessible options

| Hardware | Cost | SNN deployment path |
|----------|------|-------------------|
| **Arduino Portenta H7** | ~$80 | Cortex-M7, proven for N-MNIST SNN inference, ~250 KB memory |
| **STM32 Discovery** | ~$15-30 | Cortex-M4, demonstrated for eLSNN with 54% speedup |
| **ESP32** | ~$5-10 | WiFi/BT for IoT demo, tight memory but feasible for small SNNs |
| **BrainChip Akida Dev Kit** | Contact vendor | Actual neuromorphic SoC, sub-1W, Edge Impulse integration |
| **SynSense Xylo/Speck Kits** | Contact vendor | True neuromorphic, microwatt/milliwatt operation, Rockpool library |

### interesting thesis angle: Raspberry Pi + FPGA hybrid

Pi as host controller (data loading, preprocessing, display) connected to FPGA (SNN inference acceleration). PYNQ-Z2 has a Pi GPIO header. mirrors real edge deployment architectures.

---

## 7. recent papers and projects (2023-2026)

### key papers

| Year | Title | What it contributes |
|------|-------|-------------------|
| 2025 | "Spiking neural networks on FPGA: A survey" (Neural Networks) | survey of FPGA SNN methodologies |
| 2025 | "Efficient Deployment of SNNs on SpiNNaker2 for DVS Gesture via NIR" | complete snnTorch->NIR->SpiNNaker2 pipeline, 94.13% |
| 2025 | "Compression and Inference of SNNs on Resource-Constrained Hardware" | 21x speedup C runtime, Arduino Portenta deployment |
| 2025 | "ModNEF: Open Source Modular Neuromorphic Emulator for FPGA" (ACM TACO) | open-source FPGA SNN framework, MNIST/N-MNIST |
| 2025 | "A Robust Open-Source Framework for SNNs on Low-End FPGAs" | Artix-7, 0.52 ms/image MNIST |
| 2024 | "Spiker+: Framework for SNN FPGA Accelerators at the Edge" | auto-generate VHDL, 180 mW, 7612 logic cells |
| 2024 | "NIR" (Nature Communications) | connects 7 simulators to 4 hardware platforms |
| 2024 | "Energy-Aware FPGA Implementation of SNN with LIF Neurons" | energy measurement methodology |
| 2024 | "SpikeExplorer: HW-Oriented Design Space Exploration for SNNs on FPGA" | automated architecture search |
| 2024 | "Energy efficient SNNs on embedded microcontrollers" (Neural Computing) | eLSNN on STM32F4 |
| 2024 | ISFPGA 2024 Workshop | complete snnTorch-to-KV260 pipeline with code |

### key GitHub repos

| Repo | Focus |
|------|-------|
| open-neuromorphic/fpga-snntorch | ISFPGA 2024: snnTorch to FPGA |
| smilies-polito/Spiker | Python-to-VHDL SNN accelerator |
| neuromorphs/NIR | Neuromorphic Intermediate Representation |
| eejlny/S2NN-HLS | SNN on Zynq via Vivado HLS |
| im-afan/snn-fpga | SNN on low-end Basys3/Cmod FPGAs |
| RishabhMalviya/SNN_Arduino | LIF neurons on Arduino |

---

## 8. hardware costs

### under $50

| Hardware | Cost | Path |
|----------|------|------|
| STM32F4 Discovery | $15-25 | C runtime for small SNNs |
| ESP32 DevKit | $5-10 | WiFi SNN sensor node |
| Arduino Nano 33 BLE | $20-25 | TinyML SNN with BLE |
| Raspberry Pi 4 (4GB) | $35 (if available) | Python/C SNN inference |

### $50-200

| Hardware | Cost | Path |
|----------|------|------|
| Raspberry Pi 5 (8GB) | $80 | best CPU-based SNN inference |
| Arduino Portenta H7 | $80 | proven C SNN runtime |
| PYNQ-Z2 | $120-180 | FPGA SNN with Python |
| Basys3 | $150-180 | low-cost FPGA SNN |

### $200+

| Hardware | Cost | Path |
|----------|------|------|
| AMD Kria KV260 | $199 | official snnTorch-to-FPGA pipeline |
| BrainChip Akida | Contact vendor | true neuromorphic SoC |
| SynSense Xylo/Speck | Contact vendor | ultra-low-power neuromorphic |

### free/cloud

| Platform | Cost | Access |
|----------|------|--------|
| Google Colab | Free | SNN training with GPU |
| Intel INRC (Loihi 2) | Free for researchers | requires membership |
| SpiNNaker (Manchester) | Free for academics | cloud via sPyNNaker |

---

## 9. recommended project paths

### overall: highly feasible

a "deploy SNN on real hardware" thesis is not only feasible but represents a strong, timely, demonstrable topic. frameworks like Spiker+, NIR, and fpga-snntorch were specifically designed to make this accessible.

### path A: "SNN edge deployment benchmark" (strongest thesis)

train SNN for DVS128 gesture recognition, deploy on multiple platforms, measure and compare.

1. train SNN in snnTorch on DVS128 (GPU/Colab)
2. deploy as optimized C on STM32 or Raspberry Pi
3. deploy on FPGA via HLS (PYNQ-Z2 or Kria KV260)
4. train equivalent CNN for same task
5. deploy CNN on same platforms via TF Lite Micro
6. measure accuracy, latency, power, memory for all
7. analyze when SNNs offer advantages (or not)

cost: $200-400 (PYNQ-Z2 + STM32 + USB power meter). timeline: 12-16 weeks. live demo with power measurements is very compelling.

### path B: "FPGA SNN accelerator for DVS gestures" (more technical)

1. train with quantization-aware training
2. generate FPGA accelerator via Spiker+ or HLS
3. deploy on PYNQ-Z2
4. measure resources (LUTs, BRAMs, DSPs), power, latency
5. explore design space: precision, neuron count, architecture

cost: $150-200. timeline: 14-18 weeks.

### path C: "SNN on microcontroller for edge AI" (most accessible)

1. train on MNIST or N-MNIST
2. convert to optimized C runtime
3. deploy on STM32F4 or Arduino Portenta H7
4. measure inference time, memory, power
5. compare against TF Lite Micro CNN on same hardware
6. demo keyword spotting or gesture classification

cost: $15-80. timeline: 8-12 weeks.

### critical success factors

1. start with software simulation. get the SNN trained before touching hardware.
2. use snnTorch -- best FPGA deployment pipeline.
3. choose quantization-aware training from the start. 8-bit deploys way easier than 32-bit.
4. measure power properly (USB power meter).
5. don't oversell SNN efficiency. the comparison is nuanced. an honest analysis is a stronger thesis.
6. use N-MNIST or DVS128 Gesture as benchmark -- standard with known baselines.

### risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| FPGA toolchain issues | Medium | Use Spiker+ auto-gen or pre-built bitstreams |
| SNN accuracy < ANN | Expected | That's a finding, not a failure |
| Hardware procurement delays | Medium | Start with software; have microcontroller backup |
| Learning curve too steep | Low-Medium | Follow ISFPGA 2024 workshop tutorials |
| Power measurement difficulty | Medium | USB power meters; document methodology |

---

## sources

### frameworks and tools
- [snnTorch](https://snntorch.readthedocs.io/) / [GitHub](https://github.com/jeshraghian/snntorch)
- [snnTorch NIR Export](https://snntorch.readthedocs.io/en/latest/snntorch.export_nir.html)
- [Lava](https://lava-nc.org/) / [GitHub](https://github.com/lava-nc/lava)
- [NIR](https://neuroir.org/) / [GitHub](https://github.com/neuromorphs/NIR)
- [Spiker+ GitHub](https://github.com/smilies-polito/Spiker)
- [S2NN-HLS GitHub](https://github.com/eejlny/S2NN-HLS)
- [SNN Arduino](https://github.com/RishabhMalviya/SNN_Arduino)
- [ModNEF (ACM TACO)](https://dl.acm.org/doi/10.1145/3730581)
- [Open Neuromorphic](https://open-neuromorphic.org/)

### workshop/tutorial resources
- [ISFPGA 2024: fpga-snntorch](https://github.com/open-neuromorphic/fpga-snntorch)
- [Open Neuromorphic Workshops](https://open-neuromorphic.org/workshops/)

### key papers
- [NIR (Nature Communications, 2024)](https://www.nature.com/articles/s41467-024-52259-9)
- [Spiker+ (arXiv, 2024)](https://arxiv.org/html/2401.01141v1)
- [SNNs on SpiNNaker2 via NIR (arXiv, 2025)](https://arxiv.org/html/2504.06748v1)
- [SNN Compression for Constrained Hardware (arXiv, 2025)](https://arxiv.org/html/2511.12136)
- [Energy-Aware FPGA SNN (arXiv, 2024)](https://arxiv.org/html/2411.01628v1)
- [SNN FPGA Survey (Neural Networks, 2025)](https://www.sciencedirect.com/science/article/abs/pii/S0893608025001352)
- [SNN vs ANN Efficiency on FPGAs (JSA, 2022)](https://www.sciencedirect.com/science/article/abs/pii/S1383762122002508)
- [SNNs on Embedded Microcontrollers (Neural Computing, 2024)](https://link.springer.com/article/10.1007/s00521-024-10191-5)
- [SpikeExplorer (Electronics, 2024)](https://www.mdpi.com/2079-9292/13/9/1744)
- [Open-Source SNNs on Low-End FPGAs (arXiv, 2025)](https://arxiv.org/html/2507.07284v1)
- [TENNLab Embedder (ICONS, 2025)](https://neuromorphic.eecs.utk.edu/publications/2025-07-29-generating-spiking-neural-network-code-libraries-for-embedded-systems/)

### student projects
