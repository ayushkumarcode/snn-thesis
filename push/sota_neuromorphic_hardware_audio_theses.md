# Neuromorphic Hardware for Audio Processing & Relevant Theses (2022-2026)

looking into how other neuromorphic hardware platforms handle audio tasks, and also what relevant theses exist. this is important for contextualizing our SpiNNaker deployment (33.1% +/- 6.9% on ESC-50, 12.8pp hardware gap).

the big picture: Intel Loihi 2 dominates the audio benchmarks right now (near-zero hardware-software accuracy gaps on speech tasks), SynSense's Xylo Audio is doing ultra-low-power keyword spotting (291 uW, 95% accuracy), and SpiNNaker2 has done on-chip learning for speech commands (91.12% on Google Speech Commands). but nobody has deployed SNNs on neuromorphic hardware for the full 50-class ESC-50 dataset before -- that's just us.

the hardware-software accuracy gap varies a lot depending on the task and platform -- from near-zero (Loihi 2 on SHD/SSC with quantization-aware training) to 7-13pp (DYNAP-SE, our SpiNNaker work). so our 12.8pp gap isn't great but it's not unusual either, especially given how much harder ESC-50 is compared to keyword spotting.

---

## Part 1: Neuromorphic Hardware for Audio Processing

### 1.1 Intel Loihi and Loihi 2

Loihi 2 is the most thoroughly benchmarked neuromorphic platform for audio as of 2025.

| Paper | Year | Venue | Task | Dataset | Hardware Acc. | Software Acc. | Gap | Energy |
|-------|------|-------|------|---------|--------------|---------------|-----|--------|
| Stewart et al., "Speech2Spikes" | 2023 | NICE | Keyword spotting | Google Speech Commands (35-way) | 88.5% (Loihi 1) | 88.5% (snnTorch) | ~0 pp | 109x less than GPU, 23x less than CPU |
| Shrestha et al., "Efficient Video & Audio Processing with Loihi 2" | 2024 | ICASSP | Audio classification, denoising | SHD, SSC | ~90% SHD (recurrent) | ~91% | <1 pp | 250x less than Jetson Orin Nano |
| Knight et al., "Complete Pipeline for SNNs with Synaptic Delays on Loihi 2" | 2025 | arXiv (2510.13757) | Keyword recognition | SHD, SSC | SHD: 90.9% (recurrent w/ delays); SSC: 69.8% (FF), 67.8% (recurrent) | SHD: 88.8%; SSC: 71.4% (FF), 65.3% (recurrent) | 0-2 pp | SHD: 0.36-0.46 mJ; 18x faster than Jetson |
| Yan et al., "Eventprop training for neuromorphic applications" | 2025 | arXiv (2503.04341) | Keyword recognition | SHD, SSC | SHD: ~99% (1024 hidden); SSC: ~97% (1024 hidden) | Same (negligible gap after 8-bit quant) | ~0 pp | 0.50 mJ (SHD, 1024 hidden); 200x less than Jetson |
| Shrestha et al., "Efficient Neuromorphic Signal Processing with Loihi 2" | 2022 | JSPS | Spectral transforms | Audio (STFT) | N/A | N/A | N/A | 47x less bandwidth (RF neurons) |
| Intel N-DNS Challenge | 2023-2024 | MLSys/ICASSP | Audio denoising | DNS Challenge | Comparable to NsNet2 | NsNet2 baseline | Near-zero | 42x lower latency, 149x lower energy vs edge GPU |
| S4D on Loihi 2 | 2024 | arXiv (2409.15022) | Sequence processing (incl. audio) | sMNIST, psMNIST, sCIFAR | <3 pp drop | Full precision | 1-3 pp | 1000x more energy efficient, 75x lower latency |

key takeaways for Loihi 2:
- near-zero hardware-software gap when using quantization-aware training and 8-bit integer weights
