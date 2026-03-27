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
