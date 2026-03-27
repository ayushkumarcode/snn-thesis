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

#### Resonant and Fire Neurons for Audio

Loihi 2 supports programmable neuron models including Resonate-and-Fire (RF) neurons that can approximate spectrograms from audio inputs. The RF neurons intrinsically resonate to the strongest spectral components, producing modulated sparse spike outputs that encode the short-time Fourier spectrum 47x more efficiently than conventional STFT. This is a unique capability not available on SpiNNaker 1.

---

### 1.2 SpiNNaker and SpiNNaker2

#### SpiNNaker 1 (Our Platform)

| Paper | Year | Venue | Task | Accuracy | Notes |
|-------|------|-------|------|----------|-------|
| Dominguez-Morales et al. | 2016 | ICANN (LNCS 9886) | Pure tone classification (8 classes) | 99.8% (clean), 95% (SNR=3dB) | 130--1397 Hz tones only. Not environmental sounds. |
| Wall (thesis) | ~2016 | UoM eScholar | Auditory periphery model | N/A (biological model) | Cochlear model on SpiNNaker, not classification |
| **Our work** | **2026** | **COMP30040 / ICONS** | **ESC-50 (50-class)** | **33.1% +/- 6.9% (5-fold)** | **First ever ESC-50 deployment on SpiNNaker. FC2-only hybrid approach. Gap: 12.8 pp vs software SNN (46.0%).** |

SpiNNaker 1 has extremely limited published work on audio classification. Dominguez-Morales et al. (2016) is the only prior SpiNNaker audio classification paper, and it used simple pure tones (8 frequency classes), not real-world environmental sounds. Our ESC-50 deployment is a substantial advance in complexity.

#### SpiNNaker2

| Paper | Year | Venue | Task | Dataset | Accuracy | Notes |
|-------|------|-------|------|---------|----------|-------|
| Rostami et al., "E-prop on SpiNNaker 2" | 2022 | Frontiers Neurosci. | Speech classification | Google Speech Commands (12-class) | 91.12% | On-chip e-prop learning. 12x more energy efficient than V100 GPU. 682KB memory. Gap vs TF baseline: 0.08 pp. |
| Mayr et al., "Language Modeling on SpiNNaker2" | 2023 | arXiv (2312.09084) | Language modeling | N/A | N/A | First LM on neuromorphic hardware (EGRU). Not audio classification. |
| Vogginger et al., "Event-based backpropagation on SpiNNaker2" | 2024 | NeurIPS | On-chip training | Yin-Yang | Proof of concept | EventProp on SpiNNaker2. Not audio-specific. |

**SpiNNaker2 specifications:**
- 153 ARM cores, 19MB on-chip SRAM, 2GB DRAM
- 22nm FDSOI with adaptive body biasing
- Near-threshold operation down to 0.5V
- 10x improvement in neural simulation capacity per watt over SpiNNaker1
- Average power draw below 0.34W for inference

**Key takeaway:** SpiNNaker2 achieved 91.12% on 12-class Google Speech Commands with essentially zero hardware gap (0.08 pp), but this is a simpler task than 50-class ESC-50. Our 12.8 pp gap on ESC-50 should be interpreted in context of the much greater task difficulty and our use of the older SpiNNaker 1 platform with its binary input constraints.

---

### 1.3 BrainScaleS-2

BrainScaleS-2 is an accelerated mixed-signal neuromorphic platform from Heidelberg University.

**Hardware specifications:**
- 512 adaptive integrate-and-fire neurons
- 131K plastic synapses
- 1000x speedup (analog acceleration of neural dynamics)
- Mixed-signal: analog compute core with digital periphery

**Audio classification work:** Extremely limited. No published environmental sound classification results were found. The Spiking Heidelberg Digits (SHD) and Spiking Speech Commands (SSC) datasets were *created* by the Heidelberg group (Zenke Lab / Electronic Visions Group), but BrainScaleS-2 deployment results for these audio benchmarks are not prominently published. The platform is primarily used for neuroscience research rather than applied audio classification.

**Related audio work:** Haghighatshoar et al. (2023) investigated auditory sound source localization using neuromorphic architectures, but this is localization rather than classification.

---

### 1.4 BrainChip Akida

Akida is the first commercially available neuromorphic processor.
