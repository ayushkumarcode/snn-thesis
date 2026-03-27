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
