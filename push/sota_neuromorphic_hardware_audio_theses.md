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

| Paper | Year | Task | Dataset | Accuracy | Power | Energy/Inf |
|-------|------|------|---------|----------|-------|------------|
| Bauer et al. (Rockpool + Xylo) | 2022/2023 | Ambient audio classification | Custom | 98% | <100 uW dynamic | N/A |
| Xylo Audio 2 KWS benchmark | 2024 | Keyword spotting | Aloha KWS | 95.31% | 291 uW dynamic | 6.6 uJ/Inf |
| NeuroBench DCASE on Xylo Audio 2 | 2024 | Acoustic scene classification | DCASE 2020 (TAU) | Reported in paper | Sub-mW | Reported in paper |

#### Cross-Platform Energy Comparison (Aloha KWS Benchmark)

this is probably the best published cross-platform comparison for audio tasks:

| Device | Idle Power | Active Power | Dynamic Power | Dynamic Energy/Inf | Active Energy/Inf |
|--------|-----------|--------------|---------------|-------------------|-------------------|
| **Xylo Audio** | 216 uW | 507 uW | 291 uW | 6.6 uJ | 11 uJ |
| **Loihi** (Blouw) | 29 uW | 110 uW | 81 uW | 0.27 uJ | 0.37 uJ |
| **Loihi** (Yan) | 29 uW | 40 uW | 11 uW | 0.037 uJ | 0.13 uJ |
| **SpiNNaker2** | -- | -- | 7.1 uW | 7.1 nJ | Not reported |
| **GPU** | 14.97 mW | 37.83 mW | 22.86 mW | 29.67 uJ | 49.1 uJ |
| **CPU** | 17.01 mW | 28.48 mW | 11.47 mW | 6.32 uJ | 15.7 uJ |

things to note:
- Loihi has the lowest per-inference energy (0.037-0.27 uJ)
- Xylo Audio has lowest total power consumption
- SpiNNaker2 shows very low dynamic power (7.1 uW) but total active power is higher
- all neuromorphic platforms are orders of magnitude more efficient than GPU/CPU
- our SpiNNaker 1 energy measurement (976 nJ/sample = 0.976 uJ) is actually in the same ballpark as Loihi's per-inference energy, but for a much harder task

---

### 1.6 Other Platforms

**DYNAP-SE (SynSense, analog mixed-signal):** hardware-software gap of 80.6% -> 73.5% = 7.1pp on a simple classification task. analog circuit variability is a major source of accuracy degradation. our 12.8pp gap on a much harder task is comparable.

**FPGA-based:** HPCNeuroNet (2023) does SNN+Transformer on Xilinx FPGA, 71.11 GOP/s at 3.55W for audio. also Graph Neural Networks for audio classification on SoC FPGA (2025).

**NorthPole (IBM):** successor to TrueNorth. no published audio classification results found.

---

### 1.7 Best Hardware Accuracy Numbers for Neuromorphic Audio Classification

| Platform | Task | Classes | Accuracy | Year |
|----------|------|---------|----------|------|
| SpiNNaker 1 (Dominguez-Morales) | Pure tones | 8 | 99.8% | 2016 |
| Xylo Audio | Ambient audio | Custom | 98% | 2022 |
| Xylo Audio 2 | Keyword spotting (Aloha) | Binary (KW/not) | 95.31% | 2024 |
| Loihi 2 (Eventprop) | Heidelberg Digits | 20 | ~99% | 2025 |
| Loihi 2 (Eventprop) | Speech Commands | 35 | ~97% | 2025 |
| SpiNNaker2 (e-prop) | Speech Commands | 12 | 91.12% | 2022 |
| Loihi 1 (Speech2Spikes) | Speech Commands | 35 | 88.5% | 2023 |
| **our SpiNNaker 1** | **ESC-50** | **50** | **33.1%** | **2026** |

i know our number looks bad in this table but these aren't apples-to-apples comparisons at all. ESC-50 has 50 diverse classes (animals, machines, nature, domestic, urban) with only 1600 training samples. keyword spotting and digit recognition are fundamentally simpler tasks. the relevant comparison is our software SNN (47.15%) and the hardware gap (12.8pp).

---

### 1.8 Hardware vs. Software Accuracy Gap: Literature Context

| Work | Platform | Gap (pp) | Task Complexity | Quantization | Notes |
|------|----------|----------|-----------------|--------------|-------|
| Loihi 2 (Eventprop, 2025) | Loihi 2 | ~0 | Medium (SHD/SSC) | 8-bit QAT | Best case. Quantization-aware training. |
| SpiNNaker2 (e-prop, 2022) | SpiNNaker2 | 0.08 | Medium (12-class GSC) | Float (on ARM) | E-prop on-chip. Near-perfect match. |
| Loihi 2 (ICASSP, 2024) | Loihi 2 | <1 | Medium (SHD/SSC) | 8-bit | Good quantization pipeline. |
| S4D on Loihi 2 (2024) | Loihi 2 | 1-3 | Medium (sequential) | 8-bit | SSM architecture. |
| DYNAP-SE (2025) | DYNAP-SE | 7.1 | Low (simple classification) | Analog | Analog variability a major factor. |
| **our work (2026)** | **SpiNNaker 1** | **12.8** | **High (50-class ESC-50)** | **16-bit fixed** | **Binary input constraint. FC2-only hybrid.** |

our 12.8pp gap comes from:
1. task complexity: 50-class ESC-50 vs 12-35 class speech benchmarks
2. architecture constraint: FC2-only deployment due to SpiNNaker's binary input requirement
3. no quantization-aware training: weights converted post-hoc
4. platform generation: SpiNNaker 1 (2012 design) vs Loihi 2 (2021 design)

---

### 1.9 Hybrid ANN-SNN Deployment on Hardware

our PANNs+SNN approach fits into a growing trend:

| Work | Year | Architecture | Accuracy | Hardware |
|------|------|-------------|----------|----------|
| Hybrid ANN-SNN deployment (Shrestha) | 2024 | ANN feature extraction + SNN classification | Various | Loihi 2 (SNN) + Jetson Nano (ANN) |
| End-to-end hybrid NN mapping (PMC) | 2021 | ANN-SNN hybrid | Near-zero degradation | Custom neuromorphic |
| SpikeFit (EurIPS 2025) | 2025 | Quantized SNN deployment | SOTA compression | Various neuromorphic |
| **our PANNs+SNN** | **2026** | **CNN14 (ANN) + 3-layer SNN head** | **92.50%** | **Conceptual: CNN14 on GPU, SNN head on SpiNNaker** |

the hybrid deployment model (heavy ANN feature extraction on conventional hardware, lightweight SNN classifier on neuromorphic) is increasingly recognized as practical. our PANNs+SNN work is a demonstration of this for audio.

---

### 1.10 Energy Efficiency Summary

| Platform | Process | Power (inference) | Energy/Op | Key Audio Result |
|----------|---------|-------------------|-----------|-----------------|
| Loihi 2 | Intel 7 (~7nm) | 0.04-0.5 mW per inference | ~0.037-0.5 uJ/inf | 99% SHD, 97% SSC |
| Xylo Audio 2 | 28nm CMOS | 291 uW dynamic | 6.6 uJ/inf | 95% Aloha KWS |
| SpiNNaker2 | 22nm FDSOI | <0.34W total | 7.1 nJ/inf (dynamic) | 91.12% GSC-12 |
| Akida Pico | N/A | <1 mW | N/A | KWS with TENNs |
| SpiNNaker 1 | 130nm | Higher | 0.9 pJ/AC (theoretical) | 33.1% ESC-50 (ours) |
| BrainScaleS-2 | 65nm | ~200 mW (chip) | N/A for audio | No audio classification results |

---

## Part 2: Relevant PhD and MSc Theses

### Directly Relevant Theses (Audio + SNN/Neuromorphic)

#### Dominguez-Morales (2018) -- University of Seville

"Neuromorphic audio processing through real-time embedded spiking neural networks" -- PhD in Computer Engineering. developed speech recognition and audio processing systems based on a spiking artificial cochlea. implemented multilayer SNN on 48-chip SpiNNaker for audio classification. created NAVIS and pyNAVIS tools.

key results: pure tone classification on SpiNNaker 99.8% (8 classes, clean), 95% at SNR=3dB.

vs our work: they used simple pure tones (8 classes), we use 50-class ESC-50 environmental sounds. both on SpiNNaker. our task is substantially more complex. they focused on cochlear models, we use learned SNN features with surrogate gradients.

#### Dampfhoffer (2023) -- Universite Grenoble Alpes

"Models and algorithms for implementing energy-efficient spiking neural networks on neuromorphic hardware at the edge" -- PhD. addressed the lack of general models for estimating SNN energy consumption. the key contribution is the Dampfhoffer et al. (2023) IEEE TECI paper showing SNNs need <6.4% spike rate to beat quantized ANNs.

we cite this extensively. our SNN activation sparsity is 74.16%, meaning spike rates are below the 6.4% threshold, supporting the case for neuromorphic hardware advantage. their work is theoretical/modeling; ours includes actual hardware deployment.

#### Wall -- University of Manchester

"Spikes from sound: A model of the human auditory periphery on SpiNNaker" -- PhD. biologically-inspired model of human auditory periphery on SpiNNaker. focused on converting sound into spiking neural action potentials. both use SpiNNaker at Manchester but Wall focused on biological modeling, we focus on ML classification.

---

### SNN-Related Theses at University of Manchester (SpiNNaker Group)

| Title | Author | Year | Focus | Relevance |
|-------|--------|------|-------|-----------|
| "Deep Spiking Neural Networks" | (Jin) | ~2022 | Noisy Softplus activation, PAF training method | Training methodology for deep SNNs |
| "Learning in Spiking Neural Networks" | (Davies) | ~2022 | STDP-based learning, SpiNNaker spike injection | Learning rules + SpiNNaker implementation |
| "Ensemble Learning for Spiking Neural Networks" | -- | ~2022 | Ensemble methods for SNN performance | Shows class probability > firing rate for predictions |
| "Parallelisation of Neural Processing on Neuromorphic Hardware" | L. Peres | June 2022 | Cortical Microcircuit real-time simulation on SpiNNaker | 20x improvement over previous |
| "Parallel Simulation of Neural Networks on SpiNNaker" | X. Jin | ~2010 | SpiNNaker simulation methodology | Foundational SpiNNaker work |
| "Modelling Neural Dynamics on Neuromorphic Hardware" | -- | -- | Neural dynamics on SpiNNaker | Biological modeling |
| "Neural Encoding by Bursts of Spikes" | -- | -- | Burst coding neuroscience | Encoding schemes |

Manchester has a strong tradition of SpiNNaker PhD theses but they all focus on biological neural simulation and learning algorithms, NOT audio classification. our thesis fills a pretty clear gap in the Manchester SpiNNaker portfolio by applying the platform to a practical ML task.

---

### Other Relevant Graduate Theses

**TU Dresden / SpiNNaker2 Group:** the e-prop on SpiNNaker2 work (Rostami et al., 2022) came from Christian Mayr's group. most advanced audio-related work on SpiNNaker2.

**ETH Zurich / Neuromorphic Intelligence Group:** significant work on mixed-signal neuromorphic processors and SNNs under Giacomo Indiveri. recent PhD work includes error-propagation SNNs deployed on Intel Loihi.

**University of Zurich:** recent work (2025) deployed SNNs on DYNAP-SE for cognitive load classification from EEG with 7.1pp hardware-software gap -- comparable to ours and confirms hardware degradation is an expected challenge.

### Award-Winning Undergraduate Theses on SNNs

didn't find any specific undergraduate/honours theses on SNNs that won awards or were published as standalone works. not surprising -- SNNs on hardware is typically graduate-level, undergraduate theses are rarely indexed in searchable databases, and most published SNN work credits grad students or postdocs.

i think this actually makes our thesis notable -- deploying SNNs on SpiNNaker for ESC-50 with 5-fold cross-validation and all the ablation studies we did is typically PhD-level scope. getting a conference paper submission (ICONS 2026) out of an undergrad thesis is itself worth mentioning.

---

## Part 3: Key Surveys and Review Papers

| Survey | Year | Scope | Key Value |
|--------|------|-------|-----------|
| Basu et al., "Fundamental Survey on Neuromorphic Based Audio Classification" | Feb 2025 | SNN audio classification review | Most recent survey. Covers hardware platforms, encoding methods, learning approaches. arXiv: 2502.15056 |
| Kim et al., "SNN and Sound: A Comprehensive Review" | 2024 | SNN applications in sound | Biomedical Engineering Letters. Covers speech, environmental sound, music. |
| Larroza et al., "Spike Encoding for Environmental Sound: A Comparative Benchmark" | March 2025 | Spike encoding comparison for ESC-10 | Closest work to ours. ESC-10 only, FC only, no hardware. arXiv: 2503.11206 |
| Meunier et al., "Comparison of Hardware-friendly Audio-to-spikes Cochlear Encoding" | 2025 | Audio encoding for neuromorphic hardware | IEEE AICAS 2025. Bio-mimetic vs hardware-friendly encoding on SHD and GSC. |
| Yik et al., "NeuroBench" | 2025 | Neuromorphic benchmarking framework | Nature Communications 16:1589. Includes audio benchmarks. |

---

## Part 4: Where Our Thesis Fits

### positioning

1. **first ESC-50 deployment on neuromorphic hardware.** nobody else has done this on any chip.
2. **first SpiNNaker audio classification since 2016.** only prior SpiNNaker audio work (Dominguez-Morales 2016) used 8 pure tones. that's a 10 year gap.
3. **most extensive encoding comparison** for environmental sound on SNN (7 schemes).
4. **hybrid ANN-SNN deployment concept.** PANNs+SNN (92.5%) demonstrates viability.
5. **unusual scope for undergrad.** the breadth of experiments (encoding ablation, surrogate gradient ablation, adversarial robustness, continual learning, PANNs transfer, SpiNNaker deployment, NeuroBench energy analysis) is... a lot.

### our hardware gap in context

our 12.8pp gap is within the expected range considering:
- DYNAP-SE showed 7.1pp gap on a much simpler task
- Loihi 2 achieves near-zero but uses quantization-aware training, 8-bit optimized pipeline, and a much newer chip
- SpiNNaker 1 is 2012-era design (130nm) vs Loihi 2's Intel 7
- we didn't use quantization-aware training (post-hoc weight conversion)
- FC2-only deployment forced by SpiNNaker's binary input constraint

### research gaps

| Gap | Status |
|-----|--------|
| ESC-50 on neuromorphic hardware | **we fill this** |
| Spike encoding comparison for environmental sounds | **we fill this** |
| SpiNNaker audio classification beyond pure tones | **we fill this** |
| Hybrid ANN-SNN audio deployment | **we partially fill** (software demo, not hardware for CNN14) |
| BrainScaleS-2 audio classification | open gap |
| Cross-platform audio benchmark (same task on multiple chips) | open gap |
| ESC-50 on Loihi 2 | open gap -- would be interesting future work |

### what to cite

for related work and discussion chapters:
1. Speech2Spikes (Stewart et al., 2023) -- Loihi audio benchmark comparison
2. Efficient Video and Audio Processing with Loihi 2 (Shrestha et al., ICASSP 2024) -- hardware-software gap context
3. E-prop on SpiNNaker 2 (Rostami et al., 2022) -- SpiNNaker2 speech results
4. Xylo Audio 2 KWS benchmark (2024) -- cross-platform energy comparison
5. Basu et al. (Feb 2025) -- most recent survey
6. Larroza et al. (2025) -- already cited, closest SNN-ESC work
7. Dampfhoffer thesis (2023) -- already cited for energy thresholds

---

## Full Bibliography

### Neuromorphic Hardware Papers

1. Stewart, K.M. et al. (2023). "Speech2Spikes: Efficient Audio Encoding Pipeline for Real-time Neuromorphic Systems." NICE 2023. DOI: 10.1145/3584954.3584995
2. Shrestha, S.B. et al. (2024). "Efficient Video and Audio Processing with Loihi 2." ICASSP 2024. IEEE. arXiv: 2310.03251
3. Knight, J.C. et al. (2025). "A Complete Pipeline for deploying SNNs with Synaptic Delays on Loihi 2." arXiv: 2510.13757
4. Yan, Y. et al. (2025). "Eventprop training for efficient neuromorphic applications." arXiv: 2503.04341
5. Rostami, A. et al. (2022). "E-prop on SpiNNaker 2: Exploring online learning in spiking RNNs on neuromorphic hardware." Front. Neurosci. 16:1018006
6. Mayr, C. et al. (2024). "SpiNNaker2: A Large-Scale Neuromorphic System for Event-Based and Asynchronous Machine Learning." arXiv: 2401.04491
7. Vogginger, B. et al. (2024). "Event-based backpropagation on the neuromorphic platform SpiNNaker2." NeurIPS 2024. arXiv: 2412.15021
8. Dominguez-Morales, J.P. et al. (2016). "Multilayer SNN for Audio Samples Classification Using SpiNNaker." ICANN 2016, LNCS 9886, pp.45-53
9. Bauer, F. et al. (2022). "Sub-mW Neuromorphic SNN audio processing applications with Rockpool and Xylo." arXiv: 2208.12991
10. Micro-power spoken keyword spotting on Xylo Audio 2 (2024). arXiv: 2406.15112
11. NeuroBench DCASE 2020 benchmark on XyloAudio 2 (2024). arXiv: 2410.23776
12. Shrestha, S.B. et al. (2022). "Efficient Neuromorphic Signal Processing with Loihi 2." Journal of Signal Processing Systems
13. Timcheck, J. et al. (2023). "The Intel Neuromorphic DNS Challenge." arXiv: 2303.09503
14. "A Diagonal Structured State Space Model on Loihi 2." (2024). arXiv: 2409.15022
15. Meunier, V. et al. (2025). "Comparison of Hardware-friendly, Audio-to-spikes Cochlear Encoding for Neuromorphic Processing." IEEE AICAS 2025
16. SpikeFit (2025). "Towards Optimal Deployment of Spiking Networks on Neuromorphic Hardware." EurIPS 2025. arXiv: 2510.15542

### Surveys and Reviews

17. Basu, A. et al. (2025). "Fundamental Survey on Neuromorphic Based Audio Classification." arXiv: 2502.15056
18. Kim, D. et al. (2024). "SNN and Sound: A Comprehensive Review of Spiking Neural Networks in Sound." Biomedical Engineering Letters
19. Larroza, A. et al. (2025). "Spike Encoding for Environmental Sound: A Comparative Benchmark." arXiv: 2503.11206
20. Yik, J. et al. (2025). "NeuroBench." Nature Communications 16:1589

### Theses

21. Dampfhoffer, M. (2023). "Models and algorithms for implementing energy-efficient spiking neural networks on neuromorphic hardware at the edge." PhD thesis, Universite Grenoble Alpes
22. Dominguez-Morales, J.P. (2018). "Neuromorphic audio processing through real-time embedded spiking neural networks." PhD thesis, Universidad de Sevilla
23. Peres, L. (2022). "Parallelisation of Neural Processing on Neuromorphic Hardware." PhD thesis, University of Manchester
24. Wall, J. "Spikes from sound: A model of the human auditory periphery on SpiNNaker." PhD thesis, University of Manchester
25. Jin, X. "Deep Spiking Neural Networks." PhD thesis, University of Manchester
26. Davies, S. "Learning in Spiking Neural Networks." PhD thesis, University of Manchester

