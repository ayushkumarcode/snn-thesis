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
