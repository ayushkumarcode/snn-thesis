# State-of-the-Art in Transfer Learning for Spiking Neural Networks (2024-2026)

**Research Report for COMP30040 Thesis**
**Date: 5 March 2026**
**Context: PANNs+SNN head achieves 92.50% on ESC-50; scratch SNN: 47.15%; gap collapse from 16.7pp to 0.95pp with equal features**

---

## Executive Summary

This report surveys the state-of-the-art in transfer learning for Spiking Neural Networks (SNNs), covering ANN-to-SNN conversion, knowledge distillation, and hybrid ANN-SNN architectures with a focus on 2024-2026 publications. The field has undergone a dramatic acceleration: ANN-to-SNN conversion has matured to the point where single-timestep lossless conversion is now possible for vision models (ICML 2025, CVPR 2025), knowledge distillation from ANN teachers to SNN students has produced at least 8 distinct methods in 2024-2025 alone, and hybrid ANN-SNN architectures with pretrained ANN feature extractors and SNN classifier heads have emerged as a practical deployment paradigm for neuromorphic hardware.

Critically, **the thesis finding that the SNN-ANN accuracy gap collapses from 16.7pp to 0.95pp when both receive equal-quality features is consistent with -- but extends -- the broader literature**. The closest parallel is Spiking Vocos (2025), which achieves ANN-comparable audio quality at 14.7% energy using self-architectural distillation. The SAFE paper (ICLR 2025 submission) uses a CNN feature extractor + SNN classifier for audio fidelity and finds comparable performance with fewer parameters. However, **no prior work has demonstrated this gap-collapse phenomenon specifically for environmental sound classification on ESC-50**, making the thesis finding genuinely novel.

The report identifies zero prior works combining PANNs (or any AudioSet-pretrained model) with an SNN classifier head for ESC-50. This confirms the thesis approach occupies a clear gap in the literature.

---

## 1. ANN-to-SNN Conversion: Recent Advances (2024-2026)

### 1.1 Training-Free Conversion Methods

The dominant trend in 2024-2025 is training-free conversion of pretrained ANNs to SNNs, eliminating the need for SNN-specific training entirely.

| Paper | Venue | Key Result | Timesteps | Notes |
|-------|-------|------------|-----------|-------|
| Bu et al., "Inference-Scale Complexity" | CVPR 2025 | Near-lossless on ImageNet, segmentation, detection | Variable | Channel-wise threshold balancing; leverages open-source pretrained ANNs directly |
| STA (Spatio-Temporal Approximation) | ICLR 2024 | Converts CLIP ViT-B/32 to SNN; retains zero-shot capability | Low | First training-free conversion of a pretrained Transformer; inherits CLIP transferability |
| He et al., "Differential Coding" | ICML 2025 | SOTA accuracy with reduced spike count and energy | Variable | Transmits rate changes rather than absolute rates; threshold iteration optimization |
| "One Timestep Is Enough" (PMSM) | arXiv 2025 | 81.6% ImageNet with T=1 (ViT-S) | **1** | Polarity Multi-Spike Mapping; 4-level spiking neurons |
| "All In One Timestep" | arXiv 2025 | 75.12% ImageNet with T=4 | 4 | Exponentially fewer timesteps than prior work |
| Training-Free Spiking Transformers | arXiv 2025 | Near-lossless on CV, NLU, NLG | Low | Universal Group Operators + Spatial Rectification Self-Attention |
| PASCAL | TMLR 2025 | Mathematically equivalent to quantized ANN | Minimal | Proves inhibitory (negative) spikes essential; per-layer optimal quantization |
| Wang et al., "Negative Spikes" | IJCAI 2025 | Outperforms two-stage algorithm by 1.29% at T=4 | 4 | Leaky ReLU-based neuron model; joint layer calibration |
| LAS | arXiv 2025 | Loss-less conversion of LLMs (OPT-66B) | Low | Outlier-Aware Threshold neurons; fully spike-driven LLMs |

**Key Insight for Thesis:** The thesis uses 25 timesteps for the SNN, which is generous by modern conversion standards. Recent work achieves near-lossless conversion with T=1-4. However, conversion methods target pretrained ANNs, not from-scratch SNN training. The thesis's direct training approach (surrogate gradients, 25 timesteps) follows a fundamentally different paradigm.

### 1.2 Conversion Specifically for Audio

Audio-specific ANN-to-SNN conversion remains extremely sparse:

**Abuhajar et al. (2025) -- "Three-Stage Hybrid SNN Fine-Tuning for Speech Enhancement"**
- Venue: Frontiers in Neuroscience (April 2025)
- Method: (1) Train ANN (Wave-U-Net or ConvTasNet), (2) Convert to SNN, (3) Hybrid fine-tuning (spiking forward pass, ANN backward pass)
- Application: Speech enhancement (not classification)
- Key result: Hybrid fine-tuning recovers most of the ANN's speech quality
- **Relevance to thesis:** This is the closest methodological parallel to the thesis's PANNs+SNN approach -- ANN features transferred to SNN domain -- but for speech enhancement rather than classification.

**DPSNN (Sun & Bohte, 2024) -- "Spiking Neural Network for Low-Latency Streaming Speech Enhancement"**
- Encoder-separator-decoder architecture
- Spiking neurons in separator; non-spiking encoder/decoder
- Time-domain masking approach
- **Relevance:** Demonstrates that hybrid ANN-SNN is practical for audio but uses a fundamentally different architecture (generative, not classification).

**No papers were found performing ANN-to-SNN conversion specifically for environmental sound classification (ESC-50, UrbanSound8K, etc.).**

### 1.3 Timestep-Accuracy Trade-off

The literature establishes a clear relationship between conversion timesteps and accuracy:

| Timesteps | Typical Accuracy Loss | Method Class |
|-----------|----------------------|--------------|
| T >= 256 | ~0% (lossless) | Rate coding conversion |
| T = 16-64 | 0.5-2% | Threshold balancing + calibration |
| T = 4-8 | 1-3% | Quantization-aware conversion |
| T = 1 | 0-5% | Multi-level neurons (PMSM) |

The thesis's T=25 falls in a comfortable zone where modern conversion methods achieve near-lossless accuracy. However, the thesis trains directly with surrogate gradients rather than converting, making this comparison illustrative rather than directly applicable.

---

## 2. Knowledge Distillation: ANN Teacher to SNN Student

### 2.1 Comprehensive Taxonomy of Methods (2023-2025)

The field has produced a rich taxonomy of ANN-to-SNN knowledge distillation approaches:

| Method | Venue | Year | Approach | Key Innovation |
|--------|-------|------|----------|----------------|
| Xu et al. (BKD) | CVPR 2023 | 2023 | ANN-SNN joint training with KD | Blurred KD: random blurred SNN features restore ANN features |
| BKDSNN | ECCV 2024 | 2024 | Feature-level BKD | Outperforms prior SOTA by 4.51% on ImageNet (CNN topology) |
| SAKD (Qiu et al.) | Neural Networks 178 | 2024 | Self-architectural KD | Bilevel: (1) transfer ANN weights to SNN, (2) mimic ANN behavior |
| Efficient Logit-based KD | ICML 2025 | 2025 | Temporal-wise logit distillation | Full-range timestep deployment without retraining |
| SAMD + NLD (Liu et al.) | arXiv 2025 | 2025 | Saliency-scaled activation map + noise-smoothed logits | Addresses continuous-vs-sparse distribution mismatch |
| HTA-KL | arXiv 2025 | 2025 | Head-tail aware KL divergence | Balances high- and low-probability regions in distillation |
| Enhanced Self-Distillation | NeurIPS 2025 | 2025 | Rate-based self-distillation | Projects SNN firing rates onto lightweight ANN branches |
| Cross KD (CKD) | arXiv 2025 | 2025 | Bidirectional ANN-SNN transfer | Semantic similarity + sliding replacement for cross-modality |
| Temporal Separation + Entropy | arXiv 2025 | 2025 | Temporal entropy regularization | Separates knowledge along temporal dimension |
| BSD | arXiv 2025 | 2025 | Bidirectional spike-based distillation | Biologically plausible; stimulus-to-concept encoding |

### 2.2 Key Distillation Findings

**Distribution Mismatch Problem:**
The fundamental challenge identified across multiple 2025 papers is that ANN outputs are continuous while SNN outputs are sparse and discrete. Straightforward alignment of intermediate features and logits neglects this architectural difference (Liu et al., 2025). Solutions include:
- Gaussian noise smoothing of SNN logits (NLD)
- Saliency-scaled activation maps (SAMD)
- Blurred feature restoration (BKD, BKDSNN)

**Self-Architectural Distillation:**
SAKD (Qiu et al., 2024) and Spiking Vocos (2025) both use the *same architecture* for teacher ANN and student SNN, which avoids the capacity gap problem. This is directly relevant to the thesis: the PANNs+SNN head and PANNs+ANN head have the same 3-layer architecture, differing only in LIF vs ReLU activation. The 0.95pp gap validates that self-architectural transfer is highly effective.

**Typical Accuracy Recovery:**
- BKDSNN (ECCV 2024): SNN reaches within 0.93-4.51pp of ANN on ImageNet depending on architecture
- SAKD (2024): SNN achieves comparable performance to ANN teacher using same architecture
- Efficient Logit KD (ICML 2025): Near-ANN performance across full range of timesteps

### 2.3 Distillation for Audio
