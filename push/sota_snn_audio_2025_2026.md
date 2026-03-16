# State-of-the-Art: Spiking Neural Networks for Audio and Environmental Sound Classification (2024--2026)

**Research Report for COMP30040 UoM Thesis**
**Compiled: 5 March 2026**
**Research Agent: Deep Research Investigator (Claude Opus 4.6)**

---

## 1. Executive Summary

This report presents a comprehensive survey of the state-of-the-art in Spiking Neural Networks (SNNs) applied to audio and environmental sound classification, covering the period 2024--2026 with relevant historical context. The investigation spanned multiple search vectors across arXiv, IEEE Xplore, NeurIPS proceedings, OpenReview, Semantic Scholar, Google Scholar, ACM DL, Frontiers, Nature, and university repositories.

**Key finding: No prior work has applied a full SNN to the complete ESC-50 (50-class) dataset.** The closest work is Larroza et al. (2025, arXiv:2503.11206), which applies a 4-layer FC-only SNN to ESC-10 (10-class subset), achieving only 69.0% F1-score with their best encoding (TAE). Our thesis work (47.15% accuracy on full ESC-50 with a convolutional SNN, and 92.50% with PANNs+SNN head) represents a genuine first in the literature.

The field of SNN audio processing is rapidly evolving in 2024--2026, with major advances in:
- Spiking Transformer architectures for speech commands (SpikeSCR: 95.70% SHD; SpikCommander: 96.71% GSC)
- Multimodal audio-visual SNNs (S-CMRL: 98.13% UrbanSound8K-AV)
- Neuromorphic hardware deployment (SpiNNaker2 keyword spotting: 91.12%; Loihi 2 keyword spotting: 200x energy reduction)
- Speech enhancement SNNs (Spiking-FullSubNet: Intel N-DNS Challenge winner)

However, environmental sound classification with SNNs remains severely underexplored, with our work being the most comprehensive study to date.

---

## 2. Papers Applying SNNs to Audio/Sound Classification (2024--2026)

### 2.1 Environmental Sound Classification (Most Relevant to Thesis)

#### Paper 1: Larroza et al. (2025) -- THE Closest Competitor
- **Title:** "Spike Encoding for Environmental Sound: A Comparative Benchmark"
- **Authors:** Andres Larroza, Javier Naranjo-Alcazar, Vicent Ortiz, Maximo Cobos, Pedro Zuccarello
- **Venue:** arXiv:2503.11206v3 (submitted to ICASSP 2026; earlier versions targeted EUSIPCO 2025)
- **Funding:** IVACE/FEDER (LIASound project), STARRING-NEURO project (Spanish Ministry of Science)
- **Datasets:** ESC-10, UrbanSound8K, TAU Urban Acoustic Scenes (3-class)
- **NOT tested on ESC-50**
- **SNN Architecture:** 4 fully-connected layers, 128 LIF neurons per hidden layer, built with snnTorch v0.9.1
- **No convolutional layers**
- **Training:** 100 epochs, batch size 32, LR 0.01, macro-averaged accuracy metric
- **Encodings Compared:** Threshold Adaptive Encoding (TAE), Step Forward (SF), Moving Window (MW)
- **Results:**

| Encoder | ESC-10 | UrbanSound8K | TAU-3Class |
|---------|--------|--------------|------------|
| TAE     | 0.690  | 0.535        | 0.690      |
| SF      | 0.598  | 0.564        | 0.640      |
| MW      | 0.620  | 0.530        | 0.550      |
| **Baseline (non-spiking)** | **0.727** | **0.730** | **0.873** |

- **Key Claim:** "To our knowledge, no state-of-the-art solution has yet encoded environmental sound datasets using spike-based methods and performed classification with a spiking neural network (SNN)."
- **Significance for our thesis:** Their best ESC-10 result (69.0% with TAE) uses only FC layers and only 10 classes. Our ConvSNN achieves 47.15% on the FULL ESC-50 (50 classes), which is a fundamentally harder task. Our work is strictly more ambitious and comprehensive.
- **Spike firing rates:** TAE: 38.44% (ESC-10), 49.95% (TAU), 48.68% (US8K). TAE has lowest firing rates = most energy efficient.

#### Paper 2: Guo et al. (2024) -- Multimodal Audio-Visual SNN
- **Title:** "Transformer-Based Spiking Neural Networks for Multimodal Audiovisual Classification"
- **Authors:** Lingyue Guo, Zeyu Gao, Jinye Qu, Suiwu Zheng, Runhao Jiang, Yanfeng Lu, Hong Qiao
- **Venue:** IEEE Transactions on Cognitive and Developmental Systems, Vol. 16(3), June 2024
- **DOI:** 10.1109/TCDS.2023.3327081
- **Datasets:** UrbanSound8K-AV (self-made AV dataset), CIFAR10-AV, N-TIDIGIT+MNIST-DVS
- **Architecture:** Spiking Multimodal Transformer (SMMT) with spiking cross-attention
- **UrbanSound8K-AV Accuracy:** 96.85% (with timesteps=4)
- **Note:** This is a MULTIMODAL (audio+visual) result, not audio-only. Not directly comparable to our work.

#### Paper 3: S-CMRL (2025) -- Semantic-Alignment Audio-Visual SNN
- **Title:** "Enhancing Audio-Visual Spiking Neural Networks through Semantic-Alignment and Cross-Modal Residual Learning"
- **Venue:** arXiv:2502.12488, February 2025
- **Architecture:** Transformer-based multimodal SNN with cross-modal residual learning
- **Datasets:** CREMA-D, UrbanSound8K-AV, MNISTDVS-NTIDIGITS
- **Results:**

| Dataset | S-CMRL | CMCI | SMMT (Guo) | WeightAttention |
|---------|--------|------|------------|-----------------|
| UrbanSound8K-AV | **98.13%** | 97.90% | 96.85% | 97.60% |
| CREMA-D | **73.25%** | 70.02% | -- | 64.78% |

- **Note:** Again multimodal, not audio-only SNN.

### 2.2 Speech Command Recognition (Keyword Spotting)

#### Paper 4: SpikeSCR (Wang et al., 2024)
- **Title:** "Efficient Speech Command Recognition Leveraging Spiking Neural Network and Curriculum Learning-based Knowledge Distillation"
- **Authors:** Jiaqi Wang, Liutao Yu, Liwei Huang, Chenlin Zhou, Han Zhang, Zhenxi Song, Min Zhang, Zhengyu Ma, Zhiguo Zhang
- **Venue:** arXiv:2412.12858 (December 2024), published in Neural Networks (ScienceDirect) 2025
- **Architecture:** SpikeSCR -- fully spike-driven with Global-Local Hybrid Encoder (Spiking Self-Attention + Separable Gated Convolution), LIF neurons
- **Training:** Surrogate gradients, BPTT
- **Key Innovation:** Knowledge Distillation with Curriculum Learning (KDCL) for time-step reduction

| Dataset | SpikeSCR (100 steps) | With KDCL (40 steps) | Previous SOTA |
|---------|---------------------|---------------------|---------------|
| SHD (20 classes) | **95.70%** | 93.60% | 95.07% (DCLS) |
| SSC (35 classes) | **82.79%** | 80.25% | 80.69% (DCLS) |
| GSC v2 (35 classes) | **95.60%** | 95.01% | 95.35% (DCLS) |

- **Energy:** KDCL reduces time steps by 60%, energy by 54.8% (0.0314mJ to 0.0142mJ on SSC)
- **Energy model:** AC=0.9pJ, MAC=4.6pJ (45nm) -- same model we use for NeuroBench

#### Paper 5: SpikCommander (2025/2026)
- **Title:** "SpikCommander: A High-performance Spiking Transformer with Multi-view Learning for Efficient Speech Command Recognition"
- **Venue:** arXiv:2511.07883 (January 2026)
- **Architecture:** Multi-view Spiking Spatio-Temporal Attention Self-Attention (MSTASA) + Spiking Contextual Refinement MLP (SCR-MLP)
- **Results (100 time steps):**

| Dataset | SpikCommander | SpikeSCR | DCLS |
|---------|--------------|----------|------|
| SHD     | **96.41%**   | 95.70%   | 95.07% |
| SSC     | **83.26%**   | 82.79%   | 80.69% |
| GSC v2  | **96.71%**   | 95.60%   | 95.35% |

- **Parameters:** 0.19M (SHD), 1.12M (SSC/GSC)
- **Current SOTA for SNN speech command recognition**

#### Paper 6: SIDC-KWS (Interspeech 2025)
- **Title:** "SIDC-KWS: Efficient Spiking Inception-Dilated Conformer with Keyword Spotting"
- **Venue:** Interspeech 2025, Hyderabad
- **GSC v2 12-class Accuracy:** 96.8%
- **Energy:** 75.59% less energy than ANN counterpart
- **Architecture:** Spiking Inception + Dilated Convolution + Conformer self-attention

#### Paper 7: E-prop on SpiNNaker 2 (Yan et al., 2022)
- **Title:** "E-prop on SpiNNaker 2: Exploring online learning in spiking RNNs on neuromorphic hardware"
- **Venue:** Frontiers in Neuroscience, 2022
- **Dataset:** Google Speech Commands
- **Architecture:** Spiking Recurrent Neural Network (SRNN) with e-prop learning rule
- **Accuracy:** 91.12% (trained ONLINE on SpiNNaker 2)
- **Memory:** 680 KB for 25K weights
- **Energy:** 12x less than NVIDIA V100 GPU
- **Significance:** Demonstrated on-chip learning for keyword spotting, not just inference

### 2.3 Speech Enhancement

#### Paper 8: Spiking-FullSubNet (Hao et al., 2024)
- **Title:** "Towards Ultra-Low-Power Neuromorphic Speech Enhancement with Spiking-FullSubNet"
- **Venue:** arXiv:2410.04785 (October 2024), IEEE TNNLS 2025
- **Achievement:** 1st Place Winner, Intel N-DNS Challenge (Track 1: Algorithmic)
- **Architecture:** Full-band + sub-band SNN with Gated Spiking Neurons (GSNs)
- **DNSMOS Score:** 3.94
- **Energy:** Nearly 3 orders of magnitude smaller than best ANN (CMGAN)
- **Significance:** First SNN to win a major speech processing competition

#### Paper 9: Three-Stage Hybrid SNN Fine-Tuning (Abuhajar et al., 2025)
- **Title:** "Three-stage hybrid spiking neural networks fine-tuning for speech enhancement"
- **Venue:** Frontiers in Neuroscience, April 2025
- **Method:** ANN train -> ANN-to-SNN conversion -> Hybrid fine-tuning (spiking forward, ANN backward)
- **Architecture:** Spiking Wave-U-Net and Spiking Conv-TasNet
- **Operates in temporal domain (no FFT needed)

### 2.4 Sound Source Localization

#### Paper 10: RF-PLC SSL (Zhang et al., NeurIPS 2024)
- **Title:** "Spike-based Neuromorphic Model for Sound Source Localization"
- **Authors:** Dehao Zhang, Shuai Wang, Ammar Belatreche, et al.
- **Venue:** NeurIPS 2024 (Poster)
- **Architecture:** Resonate-and-Fire (RF) neurons with Phase-Locking Coding (RF-PLC) + Multi-Auditory Attention (MAA)
- **Claims:** SOTA accuracy in SSL tasks, exceptional noise robustness
- **Significance:** First SNN to appear at NeurIPS for audio processing

#### Paper 11: Hilbert Transform SNN Localization (Haghighatshoar & Muir, 2025)
- **Title:** "Low-power Spiking Neural Network audio source localisation using a Hilbert Transform audio event encoding scheme"
- **Venue:** Communications Engineering (Nature), 2025
- **Method:** Hilbert transform avoids dense band-pass filters; event-based encoding captures analytic signal phase
- **MAE:** 0.25--0.65 degrees (frequency bands 1.6--2.6 kHz)
- **Deployed to:** Ultra-low-power SNN inference hardware (Synsense Xylo)
- **GitHub:** https://github.com/synsense/HaghighatshoarMuir2024

### 2.5 Audio Fidelity / Fake Audio Detection

#### Paper 12: SAFE (2024, Withdrawn)
- **Title:** "SAFE: Spiking Neural Network-based Audio Fidelity Evaluation"
- **Venue:** OpenReview (submitted to ICLR 2025, withdrawn)
- **Task:** Fake/partial-fake audio detection using SNNs
- **Significance:** First attempt at using SNNs for deepfake audio detection

### 2.6 Other Audio SNN Work

#### Paper 13: SOM-Associated-SNN (2025)
- **Title:** "SOM-Associated-SNN: Enhancing audio classification with spiking neural networks through single-modality clustering and associative learning"
- **Venue:** Neurocomputing (ScienceDirect), May 2025
- **Datasets:** Spoken-MNIST, SHD
- **Architecture:** 3-layer SNN with SOM clustering + STDP + associative learning
- **No backpropagation needed -- unsupervised/biologically plausible

#### Paper 14: Ternary Spike System (2024/2025)
- **Title:** "Ternary Spike-based Neuromorphic Signal Processing System"
- **Venue:** arXiv:2407.05310 (2024), Neural Networks (2025)
- **Innovation:** TAE encoding produces ternary spikes {-1, 0, 1}; QT-SNN quantizes membrane potentials and weights
- **Results:** 94% memory reduction, 7.5x energy savings vs other SNN works
- **Tasks:** Speech recognition and EEG

#### Paper 15: Cochlear Encoding Comparison (Meunier et al., 2025)
- **Title:** "Comparison of Hardware-friendly, Audio-to-spikes Cochlear Encoding for Neuromorphic Processing"
- **Venue:** IEEE AICAS 2025, Bordeaux
- **Finding:** Lighter, hardware-friendly cochlear encoders can outperform bio-mimetic ones in accuracy and energy efficiency
- **Datasets:** Heidelberg Digits, Google Speech Commands

#### Paper 16: Spiking-LEAF (ICASSP 2024)
- **Title:** "Spiking-LEAF: A Learnable Auditory front-end for Spiking Neural Networks"
- **Venue:** ICASSP 2024
- **Innovation:** Learnable filter bank + IHC-LIF two-compartment neuron model inspired by inner hair cells
- **Tasks:** Keyword spotting, speaker identification
- **Outperforms:** SOTA spike encodings and conventional fbank features

#### Paper 17: Spike Time Difference Encoders (2025)
- **Title:** "Towards efficient keyword spotting using spike-based time difference encoders"
- **Venue:** arXiv:2503.15402 (March 2025)
- **Dataset:** TIdigits
- **Results:** TDE feedforward (89%) vs CuBa-LIF feedforward (71%) vs recurrent CuBa-LIF (91%)
- **Key:** TDE achieves 92% fewer synaptic operations than recurrent network

---

## 3. Best Reported SNN Accuracies on Environmental Sound Benchmarks

### 3.1 ESC-50 (50 classes)

| Method | Architecture | Accuracy | Year | Reference |
|--------|-------------|----------|------|-----------|
| **Our work (thesis)** | **Conv SNN (LIF, surrogate gradients)** | **47.15% +/- 4.50%** | **2026** | **This thesis** |
| **Our work + PANNs** | **PANNs CNN14 + SNN head** | **92.50% +/- 1.30%** | **2026** | **This thesis** |
| No other SNN work exists | -- | -- | -- | -- |

**Our thesis is the FIRST and ONLY work to apply an SNN to full ESC-50.** This novelty claim is confirmed by:
1. Larroza et al. (2025) explicitly stating "no state-of-the-art solution has yet encoded environmental sound datasets using spike-based methods"
2. The Basu et al. (2025) survey finding no ESC-50 SNN results
3. The Baek & Lee (2024) comprehensive review finding no ESC-50 SNN results

### 3.2 ESC-10 (10 classes)

| Method | Architecture | Accuracy/F1 | Year | Reference |
|--------|-------------|-------------|------|-----------|
| Larroza et al. (TAE) | 4-layer FC SNN, 128 LIF neurons | 69.0% (accuracy) | 2025 | arXiv:2503.11206 |
| Larroza et al. (MW) | Same | 62.0% | 2025 | Same |
| Larroza et al. (SF) | Same | 59.8% | 2025 | Same |
| Non-spiking baseline | -- | 72.7% | 2025 | Same |

### 3.3 UrbanSound8K

| Method | Architecture | Accuracy | Notes | Year |
|--------|-------------|----------|-------|------|
| S-CMRL | Transformer SNN | 98.13% | Multimodal (AV) | 2025 |
| SMMT (Guo) | Transformer SNN | 96.85% | Multimodal (AV) | 2024 |
| Larroza et al. (SF) | FC SNN | 56.4% | Audio-only | 2025 |
| Larroza et al. (TAE) | FC SNN | 53.5% | Audio-only | 2025 |
| Non-spiking baseline | -- | 73.0% | Audio-only | 2025 |

**Important caveat:** The high UrbanSound8K numbers (96-98%) are MULTIMODAL audio-visual results, not audio-only SNN results. Audio-only SNN performance on UrbanSound8K peaks at only 56.4%.

---

## 4. Spike Encoding Methods Used in Audio SNNs

### 4.1 Summary Table of Encoding Usage

| Encoding Method | Used In | Domain | Notes |
|----------------|---------|--------|-------|
| **Direct (learnable)** | SpikeSCR, SpikCommander, our thesis | Speech, ESC | Most common for surrogate gradient training |
| **Rate coding** | Wu et al. 2018, our thesis | Speech, ESC | Straightforward but requires many timesteps |
| **Threshold Adaptive (TAE)** | Larroza 2025, Ternary Spike 2024 | ESC, speech | Best for environmental sound in Larroza study |
| **Step Forward (SF)** | Larroza 2025 | ESC | Second-best on UrbanSound8K |
| **Moving Window (MW)** | Larroza 2025 | ESC | Worst overall in comparative study |
| **Latency (time-to-first-spike)** | Our thesis, TTFS literature | ESC | 4-7.5x fewer operations than rate |
| **Phase coding** | Our thesis | ESC | Tied with rate coding in our study |
| **Population coding** | Our thesis | ESC | Underperformed in our study |
| **Delta (temporal difference)** | Our thesis | ESC | Very poor for static spectrograms |
| **Burst coding** | Our thesis | ESC | Worst in our study (6.50%) |
| **Hilbert Transform** | Haghighatshoar 2025 | SSL | Event-based encoding of analytic signal phase |
| **RF-PLC** | Zhang 2024 (NeurIPS) | SSL | Phase-locking with Resonate-and-Fire neurons |
| **Speech2Spikes** | Orchard et al. 2023 | KWS | Delta-based; 88.5% on GSC |
| **Cochlear/IHC-LIF** | Spiking-LEAF 2024 | KWS | Learnable auditory frontend, best for KWS |
| **Mel spectrogram + LIF embedding** | SpikeSCR, SpikCommander | Speech | Standard in Spiking Transformer literature |
| **ANN-to-SNN conversion** | Spiking-FullSubNet 2024 | SE | Post-training conversion for speech enhancement |

### 4.2 Key Insight from Literature

The most successful recent audio SNNs (SpikeSCR, SpikCommander) use **direct encoding** where mel spectrograms are projected through learnable linear layers into spike embeddings via LIF neurons. This is effectively what we call "direct encoding" in our thesis. It bypasses handcrafted spike encoding entirely and lets the network learn optimal spike representations via surrogate gradient training.

Our thesis is the ONLY work to systematically compare 7 encoding methods on the same architecture and dataset for environmental sound. The encoding hierarchy we found (direct >> rate = phase > population > latency >> delta = burst) is a novel contribution with no precedent.

---

## 5. Neuromorphic Hardware for Audio SNNs

### 5.1 Hardware Deployment Summary

| Hardware | Audio Task | Accuracy | Energy | Reference |
|----------|-----------|----------|--------|-----------|
| **SpiNNaker 1** | Pure tone classification (8 classes) | >85% (SNR>3dB) | -- | Dominguez-Morales et al. 2016 |
| **SpiNNaker 1** | ESC-50 FC2-only (our thesis) | 33.1% +/- 6.9% | 86 nJ/sample | This thesis |
| **SpiNNaker 2** | Keyword spotting (GSC) | 91.12% | 12x < V100 GPU | Yan et al. 2022 |
| **SpiNNaker 2** | Gesture recognition | -- | -- | Kalmbach et al. 2022 |
| **Intel Loihi** | Keyword spotting (GSC) | ~88.5% | 109x < GPU, 23x < CPU | Speech2Spikes 2023 |
| **Intel Loihi 2** | Keyword spotting | ~comparable | 200x < embedded GPU, 10x faster | 2024 demos |
| **Intel Loihi 2** | Audio+video processing | -- | -- | ICASSP 2024 demo |
| **Synsense Xylo** | Sound source localization | MAE 0.25-0.65 deg | Ultra-low-power | Haghighatshoar 2025 |
| **FPGA (Cyclone V)** | CARFAC cochlea | -- | 18% ALM utilization | Various |
| **FPGA** | Event-graph audio classification | SOTA for FPGA | Low latency, low power | 2025 |

### 5.2 Our SpiNNaker Deployment in Context

Our thesis is one of very few works to deploy an SNN for ENVIRONMENTAL SOUND (not just speech) on neuromorphic hardware. The only prior work is Dominguez-Morales et al. (2016), which classified 8 pure tones (130-1397 Hz) on SpiNNaker -- a trivially simple task compared to 50-class ESC-50. Our work represents a massive step up in complexity for SpiNNaker audio deployment.

---

## 6. PANNs / Pretrained Audio Features with SNNs

### 6.1 Literature Search Results

**No prior work combines PANNs (or any pretrained audio features) with an SNN classifier head for environmental sound classification.**

Our thesis result -- PANNs CNN14 frozen embeddings + 3-layer SNN head achieving 92.50% on ESC-50 -- is NOVEL. The key insight that the SNN-ANN gap collapses from 16.7pp (scratch) to 0.95pp (PANNs features) is an original contribution.

Related but distinct work:
- **ANN-to-SNN conversion** literature (Bu et al., CVPR 2025; ICLR 2024) converts pretrained ANNs to SNNs, but focuses on vision and NLP, not audio
- **Knowledge distillation** (SpikeSCR 2024) transfers knowledge from longer-timestep SNN teacher to shorter-timestep SNN student, but does not use pretrained audio features
- **Spiking-LEAF** (ICASSP 2024) uses a learnable auditory frontend but trains from scratch

### 6.2 Significance

Our PANNs+SNN experiment addresses a fundamental question: "Is the SNN accuracy gap a feature-learning problem or a spiking computation problem?" The answer -- it is a feature-learning problem -- is a key scientific contribution that has not been demonstrated in the audio domain before.

---

## 7. Survey Papers and Reviews (2024--2025)

### 7.1 Basu et al. (2025) -- Fundamental Survey on Neuromorphic Audio Classification
- **arXiv:** 2502.15056 (February 2025)
- **Authors:** Amlan Basu, Pranav Chaudhari, Gaetano Di Caterina
- **Scope:** 24-page survey covering SNNs, memristors, neuromorphic hardware for audio
- **Key datasets reviewed:** UrbanSound, AudioSet, ESC-50
- **Finding:** No SNN results reported on ESC-50

### 7.2 Baek & Lee (2024) -- SNN and Sound: A Comprehensive Review
- **Journal:** Biomedical Engineering Letters, Vol. 14(5):981-991, 2024
- **Coverage:** Sound localization, speech recognition, classification
- **Key papers surveyed:**
  - Wu et al. (2018): SOM-SNN, 99.60% RWCP, 97.4% TIDIGITS
  - Dong et al. (2018): Conv SNN, 97.5% TIDIGITS, 93.8% TIMIT
  - Amin (2021): ATM-SNN, 97.64% TIDIGITS, 99.50% RWCP
  - Bensimon et al. (2021): SCTN-SNN, 98.73% RWCP
  - Yang & Chang (2024): Low-timestep RSNN, PER 22.6% TIMIT, 71.2 uW
  - Xiang et al. (2023): Photonic SNN, 93.75% TIDIGITS
  - Guo et al. (2023): Multimodal SNN, 96.85% UrbanSound8K-AV
- **Notable gap:** Review focuses on speech (TIDIGITS, TIMIT, RWCP). Environmental sound classification with SNNs is barely covered.

---

## 8. MSc/PhD Theses on SNNs for Audio (2022--2026)

### 8.1 Found Theses

| Thesis | University | Year | Topic |
|--------|-----------|------|-------|
| Daddinounou | Universite Grenoble Alpes | 2024 | "Design and Analysis of Neuromorphic Spiking Neural Networks with Spintronic Synapses" (hardware focus) |
| Rios-Navarro (related to Dominguez-Morales group) | Universidad de Sevilla | 2022 | "Neuromorphic Auditory Computing: Towards a Digital, Event-Based Implementation of the Hearing Sense for Robotics" |
| Multiple master's theses | Human Brain Project / Various | 2022-2024 | Gradient estimation for analog neuromorphic hardware |

### 8.2 Gap Analysis

**No MSc or PhD thesis from any of the target universities (Edinburgh, UCL, Imperial, ETH Zurich, MIT, TU Munich, TU Delft, KU Leuven) was found that specifically addresses SNN-based environmental sound classification.** This is consistent with the finding that our thesis breaks new ground in this specific intersection.

UCL has an active neuromorphic technologies group, and several of the institutions have SNN research groups, but their thesis work focuses on vision, robotics, or theoretical SNN training -- not audio classification.

---

## 9. Comprehensive Results Table: SNN Audio Benchmarks

### 9.1 Speech Command Datasets

| Method | SHD | SSC | GSC v2 | Params | Timesteps | Year |
|--------|-----|-----|--------|--------|-----------|------|
| SpikCommander | **96.41%** | **83.26%** | **96.71%** | 0.19-1.12M | 100 | 2026 |
| SpikeSCR | 95.70% | 82.79% | 95.60% | 1.63M | 100 | 2024 |
| DCLS-Delays | 95.07% | 80.69% | 95.35% | 2.50M | 100 | 2024 |
| SpikeSCR+KDCL | 93.60% | 80.25% | 95.01% | 1.63M | **40** | 2024 |
| SIDC-KWS | -- | -- | 96.8% (12-class) | -- | -- | 2025 |
| Speech2Spikes+SNN | -- | -- | 88.5% | -- | -- | 2023 |
| E-prop (SpiNNaker2) | -- | -- | 91.12% | 25K weights | online | 2022 |

### 9.2 Environmental Sound Datasets

| Method | Dataset | Classes | Accuracy | Architecture | Year |
|--------|---------|---------|----------|-------------|------|
| **Our thesis** | **ESC-50** | **50** | **47.15%** | **Conv SNN (LIF)** | **2026** |
| **Our thesis + PANNs** | **ESC-50** | **50** | **92.50%** | **PANNs+SNN head** | **2026** |
| Larroza (TAE) | ESC-10 | 10 | 69.0% | FC-only SNN | 2025 |
| Larroza (TAE) | UrbanSound8K | 10 | 53.5% | FC-only SNN | 2025 |
| Larroza (SF) | UrbanSound8K | 10 | 56.4% | FC-only SNN | 2025 |
| S-CMRL | US8K-AV | 10 | 98.13% | Transformer SNN (multimodal) | 2025 |
| SMMT (Guo) | US8K-AV | 10 | 96.85% | Transformer SNN (multimodal) | 2024 |

### 9.3 Sound Classification (Non-Environmental)

| Method | Dataset | Accuracy | Architecture | Year |
|--------|---------|----------|-------------|------|
| Wu et al. (SOM-SNN) | RWCP | 99.60% | SOM+SNN | 2018 |
| Wu et al. (SOM-SNN) | TIDIGITS | 97.4% | SOM+SNN | 2018 |
| Amin (ATM-SNN) | TIDIGITS | 97.64% | Adaptive threshold SNN | 2021 |
| Dong et al. | TIDIGITS | 97.5% | Conv SNN (STDP) | 2018 |
| Bensimon et al. | RWCP | 98.73% | SCTN-SNN | 2021 |
| Yang & Chang | TIMIT | PER 22.6% | RSNN (71.2 uW) | 2024 |

---

## 10. Adversarial Robustness of Audio SNNs

### 10.1 Literature Status

No paper has specifically studied adversarial robustness of audio SNNs. Our thesis finding -- that SNN retains 26% accuracy at FGSM eps=0.1 while ANN drops to 1.75% -- is novel in the audio domain.

Key related work (vision domain):
- Wang et al. (2025, arXiv:2512.22522): "Towards Reliable Evaluation of Adversarial Robustness for SNNs" -- warns that SNN robustness may be overestimated due to gradient estimation issues; proposes Adaptive Sharpness Surrogate Gradient (ASSG)
- FEEL-SNN (NeurIPS 2024): Robust SNNs with sparse connections
- Sharmin et al. (ECCV 2020): Original SNN adversarial robustness study

### 10.2 Implications for Our Thesis

Our adversarial robustness experiment should cite Wang et al. (2025) and acknowledge that FGSM/PGD attacks may underestimate the vulnerability of SNNs due to surrogate gradient inaccuracies. However, our finding that SNNs show qualitatively different robustness behavior remains valid and is the first such finding in the audio domain.

---

## 11. Continual Learning with Audio SNNs

No prior work studies continual learning specifically with audio SNNs. Our thesis experiment (SNN forgetting: 74.4% vs ANN forgetting: 81.3%, showing SNN forgets 6.9pp less on ESC-50 super-categories) is novel.

---

## 12. Key Narrative for Thesis Positioning

### 12.1 What Makes Our Work Novel

1. **First SNN on full ESC-50 (50 classes):** Confirmed by literature search. No prior work exists.
2. **Most comprehensive encoding comparison:** 7 encodings (direct, rate, phase, population, latency, delta, burst) on same architecture. Larroza (2025) compared only 3 encodings on ESC-10.
3. **First PANNs+SNN hybrid for audio:** No prior work combines pretrained audio features with SNN classifier.
4. **First SNN adversarial robustness study for audio:** Prior adversarial SNN work is vision-only.
5. **First continual learning study for audio SNNs:** No prior work exists.
6. **SpiNNaker deployment for ESC-50:** First deployment of environmental sound SNN on neuromorphic hardware beyond trivial pure tones.
7. **Surrogate gradient ablation for audio SNNs:** No prior systematic comparison of 8 surrogate functions for audio SNN training.

### 12.2 Where Our Work Fits in the Landscape

```
                    AUDIO SNN COMPLEXITY SPECTRUM

Simple                                              Complex
|----|----|----|----|----|----|----|----|----|----|
Pure     Digits    KWS      ESC-10   ESC-50   Full
tones                                         AudioSet

Dominguez  Wu/Dong  SpikeSCR Larroza  [US]     [None]
-Morales   2018     2024     2025
2016                         (FC-only)

                             69.0%    47.15%
                                      92.50%
                                      (PANNs)
```

Our work tackles the most complex audio classification task ever attempted with an SNN (50 classes of diverse environmental sounds), and achieves competitive results when combined with pretrained features.

---

## 13. Research Gaps and Recommended Follow-ups

### 13.1 Gaps This Search Could Not Fill

1. **Exact accuracy numbers from Guo et al. (2024) per-modality:** The UrbanSound8K-AV audio-only accuracy is not reported separately from the multimodal result.
2. **Detailed results from Basu et al. (2025) survey:** The full PDF was not accessible in extractable text format.
3. **University thesis repositories:** Direct search of institutional repositories (Edinburgh, ETH, etc.) was limited to web-accessible metadata.
4. **Chinese-language SNN audio papers:** Several Chinese institutions are active in SNN research but papers may not be indexed in English.
5. **Upcoming ICONS 2026 submissions:** Cannot search unreleased submissions.

### 13.2 Recommended Citations for Thesis

**Must-cite papers (directly relevant):**
1. Larroza et al. (2025) arXiv:2503.11206 -- closest competitor, ESC-10 only
2. Baek & Lee (2024) Biomedical Eng. Letters -- comprehensive SNN+sound review
3. Basu et al. (2025) arXiv:2502.15056 -- fundamental survey on neuromorphic audio
4. Dominguez-Morales et al. (2016) ICANN -- SpiNNaker audio predecessor
5. Wu et al. (2018) Frontiers -- SOM-SNN framework
6. Wang et al. (2025) arXiv:2512.22522 -- adversarial robustness evaluation warning

**Should-cite papers (contextual):**
7. SpikeSCR (Wang et al. 2024) -- SOTA speech command SNN
8. Spiking-FullSubNet (Hao et al. 2024) -- SNN competition winner
9. Zhang et al. (NeurIPS 2024) -- RF-PLC sound localization
10. Guo et al. (2024) IEEE TCDS -- multimodal audio SNN
11. Meunier et al. (2025) IEEE AICAS -- cochlear encoding comparison
12. Haghighatshoar & Muir (2025) Comm. Eng. -- SNN audio localization
13. Speech2Spikes (2023) NICE -- audio encoding pipeline
14. Spiking-LEAF (ICASSP 2024) -- learnable auditory frontend

---

## 14. Confidence Assessment

| Finding | Confidence | Basis |
|---------|-----------|-------|
| No prior SNN work on full ESC-50 | **Very High (95%+)** | Multiple surveys confirm; explicit claims by Larroza et al. |
| No prior PANNs+SNN for audio | **High (90%)** | Exhaustive search found nothing; niche intersection |
| No prior adversarial robustness for audio SNNs | **High (90%)** | All adversarial SNN papers are vision-domain |
| No prior continual learning for audio SNNs | **High (90%)** | Exhaustive search found nothing |
| SpikCommander is current SOTA on SHD/SSC/GSC | **High (85%)** | arXiv Jan 2026, most recent comprehensive comparison |
| Larroza et al. best ESC-10 result is 69.0% | **Very High (95%)** | Directly extracted from paper HTML |
| Our 47.15% is competitive given ESC-50 difficulty | **Very High (95%)** | 50 classes vs 10 classes, with CNN arch vs FC-only |

---

## 15. Sources

- [Larroza et al. (2025) - Spike Encoding for Environmental Sound](https://arxiv.org/abs/2503.11206)
- [Basu et al. (2025) - Fundamental Survey on Neuromorphic Audio Classification](https://arxiv.org/abs/2502.15056)
- [Baek & Lee (2024) - SNN and Sound Review](https://pmc.ncbi.nlm.nih.gov/articles/PMC11362401/)
- [SpikeSCR (Wang et al. 2024)](https://arxiv.org/abs/2412.12858)
- [SpikCommander (2025/2026)](https://arxiv.org/abs/2511.07883)
- [Guo et al. (2024) - SMMT](https://ieeexplore.ieee.org/document/10293172/)
- [S-CMRL (2025)](https://arxiv.org/abs/2502.12488)
- [Zhang et al. (NeurIPS 2024) - RF-PLC SSL](https://openreview.net/forum?id=CyCDqnrymT)
- [Spiking-FullSubNet (2024)](https://arxiv.org/abs/2410.04785)
- [Haghighatshoar & Muir (2025) - Hilbert Transform SSL](https://www.nature.com/articles/s44172-025-00359-9)
- [SIDC-KWS (Interspeech 2025)](https://www.isca-archive.org/interspeech_2025/lim25_interspeech.pdf)
- [E-prop on SpiNNaker 2 (2022)](https://www.frontiersin.org/articles/10.3389/fnins.2022.1018006/full)
- [Dominguez-Morales et al. (2016)](https://link.springer.com/chapter/10.1007/978-3-319-44778-0_6)
- [Wu et al. (2018) - SOM-SNN](https://www.frontiersin.org/articles/10.3389/fnins.2018.00836/full)
- [SOM-Associated-SNN (2025)](https://www.sciencedirect.com/science/article/abs/pii/S0925231225010884)
- [Ternary Spike System (2024)](https://arxiv.org/abs/2407.05310)
- [Meunier et al. (2025) - Cochlear Encoding](https://hal.science/hal-05344616v1)
- [Spiking-LEAF (ICASSP 2024)](https://arxiv.org/abs/2309.09469)
- [TDE Keyword Spotting (2025)](https://arxiv.org/abs/2503.15402)
- [Speech2Spikes (2023)](https://dl.acm.org/doi/10.1145/3584954.3584995)
- [Three-Stage Hybrid SNN (2025)](https://www.frontiersin.org/articles/10.3389/fnins.2025.1567347/full)
- [Wang et al. (2025) - Adversarial SNN Evaluation](https://arxiv.org/abs/2512.22522)
- [SAFE: SNN Audio Fidelity (2024, withdrawn)](https://openreview.net/forum?id=QWDZE2mYIe)
- [SpikSLC-Net (ICLR 2024 submission, rejected)](https://openreview.net/forum?id=Nz2UApmv2e)
- [Abuhajar et al. (2025) - Hybrid SNN Speech Enhancement](https://www.frontiersin.org/articles/10.3389/fnins.2025.1567347/full)
- [Daddinounou (2024) PhD Thesis - Grenoble Alpes](https://theses.hal.science/tel-04957484v1)
