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
