# Transfer Learning for SNNs -- What's Out There (2024-2026)

so i went deep into the literature on transfer learning for SNNs because our PANNs+SNN head gets 92.50% on ESC-50 while scratch SNN only gets 47.15%. the gap collapse from 16.7pp to 0.95pp with equal features felt like it should be a known thing, and i wanted to see if anyone else has shown this.

turns out the field has exploded recently. ANN-to-SNN conversion can now do single-timestep lossless conversion for vision models (ICML 2025, CVPR 2025), there are at least 8 distinct knowledge distillation methods from 2024-2025 alone, and hybrid ANN-SNN architectures are becoming a real deployment paradigm.

the key finding from all this reading: **our gap collapse result (16.7pp to 0.95pp) is consistent with the broader literature but nobody has shown it for environmental sound classification before.** closest thing is Spiking Vocos (2025) which gets ANN-comparable audio quality at 14.7% energy using self-architectural distillation. SAFE (ICLR 2025 submission) uses CNN features + SNN classifier for audio fidelity. but nobody has done this for ESC-50.

**zero prior works combine PANNs (or any AudioSet-pretrained model) with an SNN classifier head for ESC-50.** that's our gap in the literature.

---

## 1. ANN-to-SNN Conversion: Recent Advances

### Training-Free Conversion Methods

the dominant trend in 2024-2025 is training-free conversion -- just take a pretrained ANN and convert it directly to an SNN.

| Paper | Venue | Key Result | Timesteps | Notes |
|-------|-------|------------|-----------|-------|
| Bu et al., "Inference-Scale Complexity" | CVPR 2025 | Near-lossless on ImageNet, segmentation, detection | Variable | Channel-wise threshold balancing; leverages open-source pretrained ANNs directly |
| STA (Spatio-Temporal Approximation) | ICLR 2024 | Converts CLIP ViT-B/32 to SNN; retains zero-shot capability | Low | First training-free conversion of a pretrained Transformer |
| He et al., "Differential Coding" | ICML 2025 | SOTA accuracy with reduced spike count and energy | Variable | Transmits rate changes rather than absolute rates |
| "One Timestep Is Enough" (PMSM) | arXiv 2025 | 81.6% ImageNet with T=1 (ViT-S) | **1** | Polarity Multi-Spike Mapping; 4-level spiking neurons |
| "All In One Timestep" | arXiv 2025 | 75.12% ImageNet with T=4 | 4 | Exponentially fewer timesteps than prior work |
| Training-Free Spiking Transformers | arXiv 2025 | Near-lossless on CV, NLU, NLG | Low | Universal Group Operators + Spatial Rectification Self-Attention |
| PASCAL | TMLR 2025 | Mathematically equivalent to quantized ANN | Minimal | Proves inhibitory (negative) spikes essential |
| Wang et al., "Negative Spikes" | IJCAI 2025 | Outperforms two-stage algorithm by 1.29% at T=4 | 4 | Leaky ReLU-based neuron model |
| LAS | arXiv 2025 | Loss-less conversion of LLMs (OPT-66B) | Low | Outlier-Aware Threshold neurons; fully spike-driven LLMs |

for our thesis: we use 25 timesteps which is generous by modern conversion standards. recent work gets near-lossless with T=1-4. but conversion methods target pretrained ANNs, not from-scratch SNN training like we do. different paradigm entirely.

### Conversion for Audio Specifically

audio-specific ANN-to-SNN conversion is basically nonexistent:

**Abuhajar et al. (2025) -- "Three-Stage Hybrid SNN Fine-Tuning for Speech Enhancement"**
- Frontiers in Neuroscience (April 2025)
- (1) Train ANN (Wave-U-Net or ConvTasNet), (2) Convert to SNN, (3) Hybrid fine-tuning
- for speech enhancement, not classification
- this is probably the closest methodological parallel to our PANNs+SNN approach

**DPSNN (Sun & Bohte, 2024) -- "Spiking Neural Network for Low-Latency Streaming Speech Enhancement"**
- encoder-separator-decoder architecture, spiking neurons only in separator
- demonstrates hybrid ANN-SNN works for audio but totally different architecture

**nobody has done ANN-to-SNN conversion for environmental sound classification.**

### Timestep-Accuracy Trade-off

| Timesteps | Typical Accuracy Loss | Method Class |
|-----------|----------------------|--------------|
| T >= 256 | ~0% (lossless) | Rate coding conversion |
| T = 16-64 | 0.5-2% | Threshold balancing + calibration |
| T = 4-8 | 1-3% | Quantization-aware conversion |
| T = 1 | 0-5% | Multi-level neurons (PMSM) |

our T=25 sits in a comfortable zone. but again, we train directly with surrogate gradients rather than converting, so this comparison is more illustrative than anything.

---

## 2. Knowledge Distillation: ANN Teacher to SNN Student

### Methods Taxonomy (2023-2025)

there's been a ton of work here:

| Method | Venue | Year | Approach | Key Innovation |
|--------|-------|------|----------|----------------|
| Xu et al. (BKD) | CVPR 2023 | 2023 | ANN-SNN joint training with KD | Blurred KD: random blurred SNN features restore ANN features |
| BKDSNN | ECCV 2024 | 2024 | Feature-level BKD | Outperforms prior SOTA by 4.51% on ImageNet (CNN topology) |
| SAKD (Qiu et al.) | Neural Networks 178 | 2024 | Self-architectural KD | Bilevel: (1) transfer ANN weights to SNN, (2) mimic ANN behavior |
| Efficient Logit-based KD | ICML 2025 | 2025 | Temporal-wise logit distillation | Full-range timestep deployment without retraining |
| SAMD + NLD (Liu et al.) | arXiv 2025 | 2025 | Saliency-scaled activation map + noise-smoothed logits | Addresses continuous-vs-sparse distribution mismatch |
| HTA-KL | arXiv 2025 | 2025 | Head-tail aware KL divergence | Balances high- and low-probability regions |
| Enhanced Self-Distillation | NeurIPS 2025 | 2025 | Rate-based self-distillation | Projects SNN firing rates onto lightweight ANN branches |
| Cross KD (CKD) | arXiv 2025 | 2025 | Bidirectional ANN-SNN transfer | Semantic similarity + sliding replacement |
| Temporal Separation + Entropy | arXiv 2025 | 2025 | Temporal entropy regularization | Separates knowledge along temporal dimension |
| BSD | arXiv 2025 | 2025 | Bidirectional spike-based distillation | Biologically plausible; stimulus-to-concept encoding |

### Key Distillation Findings

the fundamental challenge everyone keeps running into: ANN outputs are continuous while SNN outputs are sparse and discrete. straightforward alignment doesn't work because of this architectural mismatch (Liu et al., 2025). solutions include gaussian noise smoothing of SNN logits, saliency-scaled activation maps, blurred feature restoration.

self-architectural distillation (SAKD, Spiking Vocos) uses the *same architecture* for teacher ANN and student SNN, which avoids the capacity gap problem. this is directly relevant to us -- our PANNs+SNN head and PANNs+ANN head have the same 3-layer architecture, just LIF vs ReLU. the 0.95pp gap validates that self-architectural transfer works really well.

typical accuracy recovery:
- BKDSNN (ECCV 2024): SNN within 0.93-4.51pp of ANN on ImageNet
- SAKD (2024): comparable performance using same architecture
- Efficient Logit KD (ICML 2025): near-ANN across full range of timesteps

### Distillation for Audio

**Spiking Vocos (Chen et al., 2025)** -- this is probably the single most relevant paper for our work:
- neural vocoder (audio generation)
- self-architectural distillation from ANN Vocos to Spiking Vocos
- UTMOS=3.74, PESQ=3.45, consuming only **14.7% of ANN energy**
- spiking ConvNeXt module + amplitude shortcut path
- shows that: (1) self-architectural distillation works for audio, (2) SNN can match ANN quality with right transfer approach, (3) energy savings are dramatic (85.3% reduction)

**SAFE: SNN-based Audio Fidelity Evaluation (ICLR 2025 submission)**
- fake audio detection
- CNN feature extraction (up to maxpool) + 3 spiking layers (128, 10, 2 neurons)
- comparable accuracy to ANN with fewer parameters
- **directly parallels our approach**: CNN extracts features, SNN classifies

**SpikeVoice (ACL 2024)**
- first SNN-based TTS system
- ANN-comparable quality at 10.5% energy consumption

---

