# State-of-the-Art: Neuromorphic Hardware for Audio Processing & Relevant Theses (2022--2026)

**Research Report for COMP30040 Thesis: SNN-based ESC-50 Sound Classification on SpiNNaker**

**Date:** 5 March 2026
**Context:** UoM undergraduate thesis deploying SNNs on SpiNNaker for ESC-50 environmental sound classification

---

## Executive Summary

This report surveys the state of neuromorphic hardware for audio processing and relevant graduate/undergraduate theses from 2022--2026. The field has seen substantial progress, with Intel Loihi 2 emerging as the dominant platform for audio benchmarks (achieving near-zero hardware-software accuracy gaps on speech tasks), SynSense's Xylo Audio setting records for ultra-low-power keyword spotting (291 uW, 95% accuracy), and SpiNNaker2 demonstrating on-chip learning for speech commands (91.12% on Google Speech Commands). BrainChip's Akida has entered the commercial audio market, and BrainScaleS-2 remains primarily a neuroscience research platform with limited audio classification work.

Our SpiNNaker deployment (33.1% +/- 6.9% on ESC-50, 12.8 pp hardware gap) faces a significantly harder task than the keyword spotting and digit recognition benchmarks used by other platforms. No other published work has deployed SNNs on neuromorphic hardware for the full 50-class ESC-50 dataset, making this thesis the first of its kind.

Key finding for contextualizing our results: the hardware-software accuracy gap varies enormously across the literature, from near-zero (Loihi 2 on SHD/SSC after careful quantization-aware training) to 7--13 pp (DYNAP-SE, our SpiNNaker work), depending on the complexity of the task, the quantization approach, and the degree of hardware-software co-design.
