# SNNs for Audio and Environmental Sound Classification -- What Exists (2024-2026)

went through a ton of papers on SNNs for audio, covering 2024-2026 with some older context. searched arXiv, IEEE Xplore, NeurIPS proceedings, OpenReview, Semantic Scholar, Google Scholar, ACM DL, Frontiers, Nature, and university repositories.

**the big takeaway: nobody has applied a full SNN to the complete ESC-50 (50-class) dataset before us.** closest is Larroza et al. (2025, arXiv:2503.11206) who used a 4-layer FC-only SNN on ESC-10 (10-class subset) and got 69.0% F1-score with their best encoding (TAE). our thesis (47.15% on full ESC-50 with convolutional SNN, 92.50% with PANNs+SNN head) is genuinely first.

the field is moving fast though -- spiking Transformers for speech commands are getting impressive (SpikeSCR: 95.70% SHD; SpikCommander: 96.71% GSC), multimodal audio-visual SNNs are hitting 98%+ on UrbanSound8K-AV, and neuromorphic hardware deployment is getting real (SpiNNaker2 91.12%, Loihi 2 with 200x energy reduction). but environmental sound classification specifically? barely explored.

---

## 2. Papers Applying SNNs to Audio/Sound Classification

### Environmental Sound Classification (most relevant to us)

#### Larroza et al. (2025) -- the closest competitor
- "Spike Encoding for Environmental Sound: A Comparative Benchmark"
- arXiv:2503.11206v3 (submitted to ICASSP 2026)
- Datasets: ESC-10, UrbanSound8K, TAU Urban Acoustic Scenes (3-class). **NOT tested on ESC-50**
- Architecture: 4 fully-connected layers, 128 LIF neurons per hidden layer, snnTorch v0.9.1. **no convolutional layers**
- Training: 100 epochs, batch 32, LR 0.01
- Encodings compared: Threshold Adaptive Encoding (TAE), Step Forward (SF), Moving Window (MW)

| Encoder | ESC-10 | UrbanSound8K | TAU-3Class |
|---------|--------|--------------|------------|
| TAE     | 0.690  | 0.535        | 0.690      |
| SF      | 0.598  | 0.564        | 0.640      |
| MW      | 0.620  | 0.530        | 0.550      |
| **Baseline (non-spiking)** | **0.727** | **0.730** | **0.873** |

they explicitly state: "To our knowledge, no state-of-the-art solution has yet encoded environmental sound datasets using spike-based methods and performed classification with a spiking neural network (SNN)."

their best ESC-10 result (69.0% with TAE) uses only FC layers and only 10 classes. our ConvSNN gets 47.15% on the full ESC-50 (50 classes) -- fundamentally harder task. spike firing rates: TAE has 38.44% (ESC-10), lowest = most energy efficient.

#### Guo et al. (2024) -- Multimodal Audio-Visual SNN
- "Transformer-Based Spiking Neural Networks for Multimodal Audiovisual Classification"
- IEEE TCDS, Vol. 16(3), June 2024
- UrbanSound8K-AV Accuracy: 96.85% (timesteps=4)
- but this is MULTIMODAL (audio+visual), not audio-only. not directly comparable.

#### S-CMRL (2025) -- Semantic-Alignment Audio-Visual SNN
- arXiv:2502.12488, February 2025
- Transformer-based multimodal SNN with cross-modal residual learning

| Dataset | S-CMRL | CMCI | SMMT (Guo) | WeightAttention |
|---------|--------|------|------------|-----------------|
| UrbanSound8K-AV | **98.13%** | 97.90% | 96.85% | 97.60% |
| CREMA-D | **73.25%** | 70.02% | -- | 64.78% |

again multimodal, not audio-only.

### Speech Command Recognition (Keyword Spotting)

#### SpikeSCR (Wang et al., 2024)
- "Efficient Speech Command Recognition Leveraging SNN and Curriculum Learning-based Knowledge Distillation"
- arXiv:2412.12858, published in Neural Networks 2025
- fully spike-driven, Global-Local Hybrid Encoder (Spiking Self-Attention + Separable Gated Convolution)
