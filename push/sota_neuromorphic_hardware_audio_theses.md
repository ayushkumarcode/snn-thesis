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
- dominant platform for audio benchmarks in 2024-2025
- energy advantage of 100-250x over edge GPUs (Jetson Orin Nano)
- primarily benchmarked on keyword spotting and digit recognition, NOT environmental sound classification
- no published ESC-50 or ESC-10 deployment on Loihi

Loihi 2 also supports Resonate-and-Fire neurons that can approximate spectrograms from audio inputs -- the RF neurons intrinsically resonate to the strongest spectral components, encoding STFT 47x more efficiently than conventional approach. this is a unique capability not available on SpiNNaker 1.

---

### 1.2 SpiNNaker and SpiNNaker2

#### SpiNNaker 1 (our platform)

| Paper | Year | Venue | Task | Accuracy | Notes |
|-------|------|-------|------|----------|-------|
| Dominguez-Morales et al. | 2016 | ICANN (LNCS 9886) | Pure tone classification (8 classes) | 99.8% (clean), 95% (SNR=3dB) | 130-1397 Hz tones only. Not environmental sounds. |
| Wall (thesis) | ~2016 | UoM eScholar | Auditory periphery model | N/A (biological model) | Cochlear model on SpiNNaker, not classification |
| **our work** | **2026** | **COMP30040 / ICONS** | **ESC-50 (50-class)** | **33.1% +/- 6.9% (5-fold)** | **First ever ESC-50 deployment on SpiNNaker. FC2-only hybrid. Gap: 12.8pp vs software SNN (46.0%).** |

SpiNNaker 1 has barely any published work on audio classification. Dominguez-Morales et al. (2016) is literally the only prior SpiNNaker audio classification paper and it used simple pure tones (8 frequency classes). our ESC-50 deployment is a big step up in complexity.

#### SpiNNaker2

| Paper | Year | Venue | Task | Dataset | Accuracy | Notes |
|-------|------|-------|------|---------|----------|-------|
| Rostami et al., "E-prop on SpiNNaker 2" | 2022 | Frontiers Neurosci. | Speech classification | Google Speech Commands (12-class) | 91.12% | On-chip e-prop learning. 12x more energy efficient than V100 GPU. 682KB memory. Gap vs TF baseline: 0.08 pp. |
| Mayr et al., "Language Modeling on SpiNNaker2" | 2023 | arXiv (2312.09084) | Language modeling | N/A | N/A | First LM on neuromorphic hardware (EGRU). Not audio classification. |
| Vogginger et al., "Event-based backpropagation on SpiNNaker2" | 2024 | NeurIPS | On-chip training | Yin-Yang | Proof of concept | EventProp on SpiNNaker2. Not audio-specific. |

SpiNNaker2 specs: 153 ARM cores, 19MB on-chip SRAM, 2GB DRAM, 22nm FDSOI with adaptive body biasing, near-threshold operation down to 0.5V, 10x improvement in neural simulation capacity per watt over SpiNNaker1, average power below 0.34W for inference.

important context: SpiNNaker2 got 91.12% on 12-class Google Speech Commands with essentially zero hardware gap (0.08pp), but that's a much simpler task than 50-class ESC-50. our 12.8pp gap should be interpreted accordingly -- way harder task + older platform + binary input constraints.

---

### 1.3 BrainScaleS-2

accelerated mixed-signal neuromorphic platform from Heidelberg. 512 adaptive integrate-and-fire neurons, 131K plastic synapses, 1000x speedup (analog acceleration).

audio classification work: basically none. no published environmental sound classification results. the SHD and SSC datasets were actually *created* by the Heidelberg group, but BrainScaleS-2 deployment results for these audio benchmarks aren't prominently published. the platform is primarily used for neuroscience research.

---

### 1.4 BrainChip Akida

first commercially available neuromorphic processor. event-based digital architecture. Akida 1.0 (2021), Akida 2.0/Pico (2024). <2mW (Akida), <1mW (Akida Pico).

audio applications: keyword spotting with DS-CNN (32 different keywords), TENNs (Temporal Event-based Neural Networks) that eliminate pre-processing steps -- no mel spectrogram needed, operates directly on raw audio. always-on voice activity detection at <1mW.

no published ESC-50 or ESC-10 results though. primarily keyword spotting and acoustic anomaly detection.

---

### 1.5 SynSense Xylo Audio

most extensively benchmarked neuromorphic chip for audio inference. all-digital, 28nm CMOS, up to 1000 LIF neurons, 16 input channels, 8 output. ultra-low power: 219 uW idle, 93 uW dynamic inference.
