# State-of-the-Art: Neuromorphic Hardware for Audio Processing & Relevant Theses (2022--2026)

**Research Report for COMP30040 Thesis: SNN-based ESC-50 Sound Classification on SpiNNaker**

**Date:** 5 March 2026
**Context:** UoM undergraduate thesis deploying SNNs on SpiNNaker for ESC-50 environmental sound classification

---

## Executive Summary

This report surveys the state of neuromorphic hardware for audio processing and relevant graduate/undergraduate theses from 2022--2026. The field has seen substantial progress, with Intel Loihi 2 emerging as the dominant platform for audio benchmarks (achieving near-zero hardware-software accuracy gaps on speech tasks), SynSense's Xylo Audio setting records for ultra-low-power keyword spotting (291 uW, 95% accuracy), and SpiNNaker2 demonstrating on-chip learning for speech commands (91.12% on Google Speech Commands). BrainChip's Akida has entered the commercial audio market, and BrainScaleS-2 remains primarily a neuroscience research platform with limited audio classification work.

Our SpiNNaker deployment (33.1% +/- 6.9% on ESC-50, 12.8 pp hardware gap) faces a significantly harder task than the keyword spotting and digit recognition benchmarks used by other platforms. No other published work has deployed SNNs on neuromorphic hardware for the full 50-class ESC-50 dataset, making this thesis the first of its kind.

Key finding for contextualizing our results: the hardware-software accuracy gap varies enormously across the literature, from near-zero (Loihi 2 on SHD/SSC after careful quantization-aware training) to 7--13 pp (DYNAP-SE, our SpiNNaker work), depending on the complexity of the task, the quantization approach, and the degree of hardware-software co-design.

---

## Part 1: Neuromorphic Hardware for Audio Processing

### 1.1 Intel Loihi and Loihi 2

Loihi 2 is the most thoroughly benchmarked neuromorphic platform for audio tasks as of 2025.

#### Key Papers and Results

| Paper | Year | Venue | Task | Dataset | Hardware Acc. | Software Acc. | Gap | Energy |
|-------|------|-------|------|---------|--------------|---------------|-----|--------|
| Stewart et al., "Speech2Spikes" | 2023 | NICE | Keyword spotting | Google Speech Commands (35-way) | 88.5% (Loihi 1) | 88.5% (snnTorch) | ~0 pp | 109x less than GPU, 23x less than CPU |
| Shrestha et al., "Efficient Video & Audio Processing with Loihi 2" | 2024 | ICASSP | Audio classification, denoising | SHD, SSC | ~90% SHD (recurrent) | ~91% | <1 pp | 250x less than Jetson Orin Nano |
| Knight et al., "Complete Pipeline for SNNs with Synaptic Delays on Loihi 2" | 2025 | arXiv (2510.13757) | Keyword recognition | SHD, SSC | SHD: 90.9% (recurrent w/ delays); SSC: 69.8% (FF), 67.8% (recurrent) | SHD: 88.8%; SSC: 71.4% (FF), 65.3% (recurrent) | 0--2 pp | SHD: 0.36--0.46 mJ; 18x faster than Jetson |
| Yan et al., "Eventprop training for neuromorphic applications" | 2025 | arXiv (2503.04341) | Keyword recognition | SHD, SSC | SHD: ~99% (1024 hidden); SSC: ~97% (1024 hidden) | Same (negligible gap after 8-bit quant) | ~0 pp | 0.50 mJ (SHD, 1024 hidden); 200x less than Jetson |
| Shrestha et al., "Efficient Neuromorphic Signal Processing with Loihi 2" | 2022 | JSPS | Spectral transforms | Audio (STFT) | N/A | N/A | N/A | 47x less bandwidth (RF neurons) |
| Intel N-DNS Challenge | 2023--2024 | MLSys/ICASSP | Audio denoising | DNS Challenge | Comparable to NsNet2 | NsNet2 baseline | Near-zero | 42x lower latency, 149x lower energy vs edge GPU |
| S4D on Loihi 2 | 2024 | arXiv (2409.15022) | Sequence processing (incl. audio) | sMNIST, psMNIST, sCIFAR | <3 pp drop | Full precision | 1--3 pp | 1000x more energy efficient, 75x lower latency |

**Key takeaways for Loihi 2:**
- Near-zero hardware-software accuracy gap when using quantization-aware training and 8-bit integer weights
- Dominant platform for audio benchmarks in 2024--2025
- Energy advantage of 100--250x over edge GPUs (Jetson Orin Nano)
- Primarily benchmarked on keyword spotting (Google Speech Commands) and digit recognition (SHD), NOT environmental sound classification
- No published ESC-50 or ESC-10 deployment on Loihi

