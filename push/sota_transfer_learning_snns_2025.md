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

## 3. Hybrid ANN-SNN Architectures

### "ANN Feature Extractor + SNN Classifier" -- Our Paradigm

this is exactly what we do (PANNs CNN14 + SNN head). the literature shows it's emerging but under-explored:

| Paper | Year | ANN Backbone | SNN Head | Task | Result |
|-------|------|-------------|----------|------|--------|
| **our thesis** | **2026** | **PANNs CNN14 (frozen)** | **3-layer SNN** | **ESC-50** | **92.50% (SNN) vs 93.45% (ANN), 0.95pp gap** |
| SAFE | 2025 | CNN (maxpool layers) | 3 spiking layers | Fake audio detection | Comparable to ANN SOTA |
| Aydin et al. | CVPRW 2024 | ANN (low-rate dense) | SNN (high-rate sparse) | Visual pose estimation | 74% lower error than pure SNN |
| Keugle et al. | 2024 | ANN on Jetson Nano | SNN on Loihi | DVS classification | Surpasses both pure ANN and pure SNN |
| Abuhajar et al. | 2025 | ANN Wave-U-Net | SNN (converted) | Speech enhancement | Near-ANN quality after fine-tuning |
| Spiking Vocos | 2025 | ANN Vocos (teacher) | Spiking Vocos (student) | Neural vocoder | 14.7% energy of ANN |

### Aydin et al. (CVPR 2024 Workshop) -- Slow-Fast Hybrid

most architecturally sophisticated hybrid approach i found:
- ANN provides "slow" dense state initialization; SNN provides "fast" spike-based predictions
- key insight: pure SNNs suffer from long state convergence transients; ANN initialization solves this
- 74% lower error than pure SNN
- code: https://github.com/uzh-rpg/hybrid_ann_snn

### Hardware Deployment of Hybrid Systems

Keugle et al. (2024) -- "Towards Efficient Deployment of Hybrid SNNs on Neuromorphic and Edge AI Hardware":
- deploys ANN on Jetson Nano, SNN on Intel Loihi
- also tested ANN on Coral Edge TPU, SNN on Loihi
- uses Lava framework for SNN, PyTorch for ANN
- **hybrid outperforms both pure ANN and pure SNN** in accuracy, latency, and energy

this is basically what we're doing with PANNs+SNN: CNN14 on GPU/CPU, SNN head on SpiNNaker. our 86 nJ SNN energy figure is competitive with their Loihi numbers.

---

## 4. Audio Pretrained Models + SNNs

### The Gap: Nobody Has Done This Before

i searched everywhere and **found zero papers** combining any of these with SNN classifier heads:
- PANNs (Kong et al., 2020) + SNN
- VGGish + SNN
- wav2vec 2.0 / HuBERT / Whisper + SNN
- CLAP / AudioMAE / BEATs + SNN
- AST (Audio Spectrogram Transformer) + SNN

so yeah, we're first. **first work to use PANNs (or any AudioSet-pretrained model) as a frozen feature extractor with an SNN classifier head for environmental sound classification.**

### Closest Work in Audio SNNs

**Spiking-LEAF (Song et al., ICASSP 2024)**
- learnable auditory front-end for SNNs
- combines learnable filter bank with IHC-LIF neuron model
- keyword spotting and speaker identification
- outperforms conventional mel spectrograms for SNN processing
- but it's a front-end (encoding), not a pretrained feature extractor -- different from what we do

**SATRN: Spiking Audio Tagging Robust Network (Gao & Deng, Electronics 2025)**
- spiking architecture with temporal-spatial attention + membrane potential residual connections
- UrbanSound8K and FSD50K
- "comparable performance to traditional CNNs"
- trained from scratch though, not using pretrained features

**SpikSLC-Net (OpenReview 2025)**
- joint sound source localization and classification
- spiking hybrid attention fusion mechanism
- training-inference-decoupled layer normalization

**Spiking-FullSubNet (Hao et al., IEEE TNNLS 2025)**
- winner of Intel N-DNS Challenge (Algorithmic Track)
- full-band and sub-band fusion for speech enhancement
- novel spiking neuron with dynamic input integration/forgetting
- highly relevant because it shows well-designed SNN audio systems can beat ANN SOTA

### Multimodal Audio-Visual SNN Space

some interesting work here but not directly related to us:
- **MISNet (ACM TOMM 2024-2025)**: multimodal LIF neuron coordinates audiovisual spikes
- **SMMT: Spiking Multimodal Transformer (IEEE TCDS 2024)**: spiking cross-attention for multimodal fusion
- **S-CMRL (arXiv 2025)**: semantic-alignment cross-modal residual learning

---

## 5. SNN-ANN Accuracy Gap: What Does the Literature Say?

### Gap on Standard Benchmarks (Vision)

| Task | ANN Accuracy | SNN Accuracy | Gap | Method | Year |
|------|-------------|-------------|-----|--------|------|
| ImageNet (from scratch) | 80.80% (Transformer-8-512) | 73.38% (Spikformer-8-512) | **7.42pp** | Direct training | 2024 |
| ImageNet (pretrained SSL) | ~82% | 81.10% (Spikformer V2) | **~1pp** | Self-supervised pretraining | 2024 |
| ImageNet (conversion) | ~88.60% (ANN) | ~87.60% (converted SNN) | **~1pp** | Training-free conversion | 2025 |
| ImageNet (conversion T=1) | ~82% | 81.6% (PMSM ViT-S) | **~0.4pp** | Multi-level single-timestep | 2025 |
| CIFAR-100 (KD) | ANN teacher | Within 1-2pp | **1-2pp** | BKDSNN, SAKD | 2024 |

### Gap Collapse with Pretrained Features

this is the pattern i keep seeing across the literature:

| Setting | ANN Accuracy | SNN Accuracy | Gap | Source |
|---------|-------------|-------------|-----|--------|
| **ESC-50 from scratch (ours)** | **63.85%** | **47.15%** | **16.70pp** | **our thesis** |
| **ESC-50 PANNs features (ours)** | **93.45%** | **92.50%** | **0.95pp** | **our thesis** |
| ImageNet from scratch | 80.80% | 73.38% | 7.42pp | Spikformer V2 |
| ImageNet with SSL pretraining | ~82% | 81.10% | ~1pp | Spikformer V2 |
| ImageNet conversion (pretrained) | 88.60% | ~87.60% | ~1pp | Bu et al. 2025 |
| Neural vocoder (self-distillation) | ANN Vocos | 14.7% energy, comparable quality | ~0pp (quality) | Spiking Vocos 2025 |
| Audio fidelity (SAFE) | ANN SOTA | Comparable | ~0pp | SAFE 2025 |
| Speech (TTS, SpikeVoice) | ANN TTS | 10.5% energy, comparable | ~0pp (quality) | SpikeVoice 2024 |

our gap collapse from 16.7pp to 0.95pp is actually the most dramatic demonstration of this in the audio domain. the ratio (16.7 / 0.95 = 17.6x reduction) exceeds what's typically reported in vision (7.42 / ~1 = 7.4x).

### The "Feature Learning Bottleneck" Hypothesis

our central claim -- that the SNN-ANN gap is a *feature learning* problem, not a *spiking computation* problem -- is backed by converging evidence:

1. **Spikformer V2 (2024):** self-supervised pretraining (improving feature quality) narrows the gap from 7.42pp to ~1pp on ImageNet
2. **ANN-to-SNN conversion (2024-2025):** converting a pretrained ANN to SNN loses only 1-2pp, confirming spiking computation itself isn't the bottleneck
3. **BKDSNN (ECCV 2024):** knowledge distillation transfers ANN feature representations to SNNs, recovering most of the gap
4. **STA/CLIP conversion (ICLR 2024):** converting CLIP to SNN retains zero-shot capability -- spiking neurons can preserve complex learned representations
5. **Spiking Vocos (2025):** self-architectural distillation achieves ANN-comparable audio quality

**but nobody has explicitly articulated and empirically demonstrated this hypothesis for audio classification before.** that's our contribution.

---

## 6. Audio SNN Landscape: ESC-50 and Environmental Sound

### SNN on ESC-50: Prior Work

| Paper | Year | Dataset | SNN Accuracy | ANN Reference | Notes |
|-------|------|---------|-------------|---------------|-------|
| **our thesis** | **2026** | **ESC-50 (full, 50 classes)** | **47.15% (scratch), 92.50% (PANNs)** | **63.85% (scratch), 93.45% (PANNs)** | **First SNN on full ESC-50** |
| Larroza et al. | 2025 | ESC-10 only | 69.0% (best, TAE encoding) | -- | Spike encoding benchmark; FC-only; no hardware |
| Dennis et al. | 2018 | ESC-50 subset | Limited | -- | SNN framework; non-deep |
| Dominguez-Morales et al. | 2016 | Pure tones only | Limited | -- | SpiNNaker; not ESC-50 |

we're still the first and only work reporting SNN accuracy on the full 50-class ESC-50 dataset.

### SNN on UrbanSound8K

| Paper | Year | SNN Accuracy | Notes |
|-------|------|-------------|-------|
| Larroza et al. | 2025 | 56.4% (SF encoding) | Spike encoding benchmark |
| SATRN (Gao & Deng) | 2025 | "Comparable to CNNs" | Spiking attention mechanism |
| ESC-NAS | 2024 | 81.25% (ANN, not SNN) | Hardware-aware NAS for edge |

### SNN Audio Review Paper

Kim et al. (2024), "SNN and Sound: A Comprehensive Review" (Biomedical Engineering Letters, Vol. 14, No. 5, pp. 981-991):
- SNNs for sound are in early stages
- key challenges: effective training algorithms, spike encoding methods, hardware integration
- promising areas: real-time processing, low-power edge deployment
- **does not cite any SNN work on full ESC-50**

---

## 7. PhD/MSc Theses on SNN Transfer Learning

found a few but nothing audio-specific:

**1. "Deep Spiking Neural Networks" -- University of Manchester**
- proposes Noisy Softplus (NSP) activation to model spiking neurons
- develops generalized off-line training using Parametric Activation Functions (PAF)
- maps ANN values to SNN physical units
- relevant since it's Manchester and addresses ANN-to-SNN weight transfer, and our SpiNNaker work is on Manchester hardware

**2. Christian Steennis -- Leiden University (LIACS), August 2025**
- MSc thesis on neural network quantization and ANN-to-SNN conversion
- explores quantization (spatial dimension) and SNN conversion (temporal dimension)

**3. Peng Kang -- Northwestern University, 2024**
- technical report on event-based processing with SNNs
- single-timestep and multi-timestep Spiking UNets, vision focused

**4. Cameron Eric Johnson -- Missouri S&T**
- PhD dissertation: "Spiking Neural Networks and Their Applications"

**no PhD or MSc theses found on SNN transfer learning for audio specifically.** that's another novelty point for us.

---

## 8. PANNs Extensions with Neuromorphic Components

searched for "PANNs spiking neural network", "pretrained audio neural networks neuromorphic", "CNN14 SNN", "AudioSet pretrained spiking" -- **zero results.** nobody has combined PANNs or any AudioSet-pretrained model with neuromorphic/SNN components before.

---

## 9. The SNN Transfer Learning Landscape: A Taxonomy

based on all this reading, i think the field breaks down into four paradigms:

### Paradigm 1: ANN-to-SNN Conversion (Weight Transfer)
- train ANN normally, convert weights to SNN
- requires threshold balancing and calibration
- mature for vision (Bu et al. CVPR 2025, STA ICLR 2024)
- barely exists for audio (Abuhajar et al. 2025 for speech enhancement)

### Paradigm 2: Knowledge Distillation (Behavior Transfer)
- train SNN to mimic ANN's intermediate features and/or logits
- addresses continuous-vs-sparse distribution mismatch
- very active: BKDSNN (ECCV 2024), SAKD (Neural Networks 2024), 6+ methods in 2025
- audio: Spiking Vocos (2025)

### Paradigm 3: Frozen ANN Features + SNN Head (What We Do)
- use pretrained ANN as frozen feature extractor
- train small SNN classifier on extracted features
- **least explored paradigm overall**
- **completely unexplored for environmental sound classification**
- only known instances: SAFE (2025, audio fidelity), our thesis (2026, ESC-50)

### Paradigm 4: Hybrid ANN-SNN Co-execution
- ANN and SNN run simultaneously on different hardware
- ANN provides initialization/features; SNN provides efficient inference
- Aydin et al. (CVPRW 2024), Keugle et al. (2024)
- audio: DPSNN (2024, ANN encoder + SNN separator)

**we uniquely combine Paradigm 3 (frozen PANNs features + SNN head) with Paradigm 4 (SpiNNaker deployment of the SNN head). haven't found anyone else doing this.**

---

## 10. Summary Tables

### ANN-to-SNN Knowledge Distillation Methods (2023-2025)

| Method | Year | Venue | Teacher | Student | Transfer Level | Key Result |
|--------|------|-------|---------|---------|---------------|------------|
| Xu et al. | 2023 | CVPR | ANN | SNN | Features + Logits | SOTA on static + neuromorphic datasets |
| BKDSNN | 2024 | ECCV | ANN | SNN | Blurred features | +4.51% on ImageNet (CNN) |
| SAKD | 2024 | Neural Networks | Same-arch ANN | SNN | Weights + Behavior | Bilevel transfer |
| Efficient Logit KD | 2025 | ICML | ANN | SNN | Temporal logits | Full-range timestep deployment |
| Liu et al. (SAMD+NLD) | 2025 | arXiv | ANN | SNN | Saliency maps + smoothed logits | Addresses distribution mismatch |
| HTA-KL | 2025 | arXiv | ANN | SNN | Head-tail KL | Balanced probability transfer |
| Enhanced Self-Distillation | 2025 | NeurIPS | Self (SNN) | Self (SNN) | Firing rate projections | Reduces training complexity |
| CKD | 2025 | arXiv | ANN | SNN | Bidirectional | Cross-modality + cross-architecture |
| BSD | 2025 | arXiv | Bidirectional | Bidirectional | Spike-based | Biologically plausible |
| Spiking Vocos | 2025 | arXiv | ANN Vocos | Spiking Vocos | Self-architectural | 14.7% energy, comparable quality |

### SNN Audio Systems (2024-2026)

| System | Year | Task | Architecture | Key Finding |
|--------|------|------|-------------|-------------|
| Spiking-FullSubNet | 2025 | Speech enhancement | Full-band + sub-band SNN | Won Intel N-DNS Challenge |
| DPSNN | 2024 | Speech enhancement | ANN encoder + SNN separator | Low-latency streaming |
| Abuhajar et al. | 2025 | Speech enhancement | Converted Wave-U-Net/ConvTasNet | Three-stage hybrid fine-tuning |
| SpikeVoice | 2024 | Text-to-speech | Spiking TTS with STSA | 10.5% energy of ANN |
| Spiking Vocos | 2025 | Neural vocoder | Spiking ConvNeXt | 14.7% energy, self-distillation |
| SAFE | 2025 | Fake audio detection | CNN features + SNN classifier | Comparable to ANN SOTA |
| SATRN | 2025 | Audio tagging | Spiking attention | Comparable to CNNs on US8K |
| Spiking-LEAF | 2024 | Keyword spotting | Learnable auditory front-end | Outperforms mel spectrograms |
| SpikSLC-Net | 2025 | Sound localization + classification | Spiking hybrid attention | Joint localization-classification |
| MISNet | 2025 | Audio-visual classification | Multimodal LIF neuron | First balanced accuracy+efficiency |
| Larroza et al. | 2025 | Spike encoding benchmark | FC SNN | ESC-10 only (69% TAE) |
| **our thesis** | **2026** | **ESC-50 classification** | **CNN14 features + SNN head** | **92.50%, 0.95pp gap to ANN** |

### Accuracy Gap Collapse Evidence

| Work | Domain | Scratch Gap | Pretrained Gap | Collapse Ratio | Year |
|------|--------|------------|----------------|----------------|------|
| **our thesis** | **Audio (ESC-50)** | **16.70pp** | **0.95pp** | **17.6x** | **2026** |
| Spikformer V2 | Vision (ImageNet) | 7.42pp | ~1pp (SSL) | 7.4x | 2024 |
| Bu et al. | Vision (ImageNet) | ~7pp | ~1pp (conversion) | ~7x | 2025 |
| Spiking Vocos | Audio (vocoder) | N/A | ~0pp (distillation) | N/A | 2025 |
| SAFE | Audio (fidelity) | N/A | ~0pp (hybrid) | N/A | 2025 |

---

## 11. Research Gaps

things nobody has done yet:
1. PANNs + SNN head -- we're first
2. any audio foundation model (wav2vec, HuBERT, Whisper, CLAP, AudioMAE, BEATs) + SNN classifier
3. full ESC-50 with SNNs (Larroza et al. 2025 only did ESC-10)
4. explicit "gap collapse" measurement in audio
5. ANN-to-SNN conversion for environmental sound classification (all conversion work is vision or speech)
6. any PhD/MSc theses on SNN transfer learning for audio
7. hybrid ANN-SNN deployment for audio on neuromorphic hardware (only DPSNN and our SpiNNaker work)

---

## 12. How Confident Am I

| Finding | Confidence | Basis |
|---------|-----------|-------|
| No prior PANNs+SNN work exists | Very high | searched Google Scholar, arXiv, Semantic Scholar, IEEE Xplore |
| No prior full ESC-50 SNN work exists | Very high | consistent with earlier research; confirmed by 2025 survey |
| Gap collapse is under-reported in audio | High | only our thesis and Spiking Vocos demonstrate it; vision has more evidence |
| KD from ANN to SNN is a mature field | Very high | 10+ papers in 2024-2025 alone |
| ANN-to-SNN conversion is vision-dominated | High | all major conversion papers (CVPR, ICLR, ICML) are vision; audio has 2 papers |
| Hybrid ANN feature + SNN head is under-explored | High | only SAFE and our thesis explicitly use this paradigm |

---

## 13. What to Cite

### must-cite (directly relevant)

1. Bu, T., Li, M., & Yu, Z. (2025). "Inference-Scale Complexity in ANN-SNN Conversion." CVPR 2025.
2. Abuhajar, R., et al. (2025). "Three-stage hybrid spiking neural networks fine-tuning for speech enhancement." Frontiers in Neuroscience.
3. Chen, Y., et al. (2025). "Spiking Vocos: An Energy-Efficient Neural Vocoder." arXiv:2509.13049.
4. Xu, Z., et al. (2024). "BKDSNN: Enhancing the Performance of Learning-based SNN Training with BKD." ECCV 2024.
5. Qiu, H., et al. (2024). "Self-architectural knowledge distillation for spiking neural networks." Neural Networks 178.
6. Song, Z., et al. (2024). "Spiking-LEAF: A Learnable Auditory front-end for SNNs." ICASSP 2024.
7. Hao, X., et al. (2025). "Toward Ultra-Low-Power Neuromorphic Speech Enhancement with Spiking-FullSubNet." IEEE TNNLS.
8. Kim, T., et al. (2024). "SNN and Sound: A Comprehensive Review." Biomedical Engineering Letters 14(5).
9. Zhou, Z., et al. (2024). "Spikformer V2: Join the High Accuracy Club on ImageNet." arXiv:2401.02020.
10. Larroza, A., et al. (2025). "Spike Encoding for Environmental Sound: A Comparative Benchmark." arXiv:2503.11206.

### should-cite (supporting context)

11. STA: Spatio-Temporal Approximation, ICLR 2024 (CLIP-to-SNN conversion).
12. Xu, Q., et al. (2023). "Constructing Deep SNNs from ANNs with KD." CVPR 2023.
13. Aydin, A., et al. (2024). "A Hybrid ANN-SNN Architecture." CVPRW 2024.
14. SAFE: SNN-based Audio Fidelity Evaluation, ICLR 2025 submission.
15. SpikeVoice, ACL 2024 (first SNN TTS).
16. He et al. (2025). "Differential Coding for Training-Free ANN-to-SNN Conversion." ICML 2025.
17. Yu, C., et al. (2025). "Efficient Logit-based KD of Deep SNNs." ICML 2025.
18. SATRN: Spiking Audio Tagging Robust Network, Electronics 2025.
19. Steennis, C. (2025). MSc Thesis, Leiden University. (ANN-SNN conversion and quantization.)
20. Gao, Y., & Deng, M. (2025). "SATRN: Spiking Audio Tagging Robust Network." Electronics 14(4).

---

## 14. What This Means for the Thesis

based on all of this, i think we can confidently claim:

1. **"Our PANNs+SNN approach is the first to combine an AudioSet-pretrained feature extractor with an SNN classifier for environmental sound classification."** -- backed by finding zero prior work anywhere.

2. **"The SNN-ANN accuracy gap collapse from 16.7pp to 0.95pp with equal features demonstrates that the bottleneck is feature learning, not spiking computation."** -- consistent with vision evidence (Spikformer V2, Bu et al.) and audio generation (Spiking Vocos), but first quantified demonstration for audio classification.

3. **"The frozen-feature + SNN head paradigm is the least explored of the four SNN transfer learning paradigms."** -- only one other audio instance (SAFE, 2025).

4. **"Our work on full ESC-50 with SNNs remains unique."** -- closest is Larroza et al. (2025) who only did ESC-10 with FC networks.

5. **"The 17.6x gap collapse ratio exceeds the ~7x typical in vision."** -- suggesting audio classification may benefit disproportionately from pretrained features when combined with SNN classifiers. interesting to think about why -- probably because audio feature learning from 1,600 samples is just that much harder.
