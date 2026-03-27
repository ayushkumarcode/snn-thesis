# Multimodal SNNs: Can You Combine Different Data Types in One Spiking Network?

so i wanted to figure out if combining different data types (vision, audio, event camera, IMU) in a single SNN is actually a thing, and whether it's doable for an undergrad thesis.

turns out it's a real and rapidly growing area since 2023. there are multiple working implementations for audio-visual classification, event camera + RGB fusion, and sensor fusion for robotics. what really caught my eye: a paper from August 2024 does exactly the simplified version i was thinking about (MNIST digits + audio digits fusion) and gets 98.43% accuracy. so yeah, this is doable at the undergrad level if you keep the scope tight. enough existing work to build on, enough open problems to contribute to.

---

## 1. Has Anyone Done Vision + Audio in an SNN?

**yes -- at least 6 major papers from 2023-2025.**

### Key Papers

| Paper/System | Year | Datasets | Accuracy | Key Innovation |
|---|---|---|---|---|
| **SMMT** (Spiking Multi-Modal Transformer) | 2023 | CREMA-D, UrbanSound8K-AV | ~66% (CREMA-D) | Spiking Cross-Attention (SCA) for audio-visual fusion |
| **MISNet** (Multimodal Interaction Spiking Network) | 2024 | 5 audio-visual datasets | Competitive | MLIF neuron that synchronizes audiovisual spikes in a single neuron |
| **Bjorndahl et al.** | Aug 2024 | N-MNIST + SHD | 98.43% | Early/middle/late fusion comparison for digit recognition |
| **S-CMRL** | Feb 2025 | CREMA-D, UrbanSound8K-AV, MNISTDVS-NTIDIGITS | 73.25% (CREMA-D), 99.28% (MNISTDVS-NTIDIGITS) | Semantic-alignment + cross-modal residual learning |
| **TAAF** | May 2025 | CREMA-D, AVE, EAD | 77.55% (CREMA-D) | Temporal attention-guided adaptive fusion for modality imbalance |
| **Oikonomou et al.** | Nov 2024 | Various | Survey | Bio-inspired multimodal perception for robotics |

### Fusion Strategies

three dominant approaches:
1. **Early fusion**: combining raw spike trains before feature extraction
2. **Late fusion (concatenation)**: separate unimodal branches merging at the decision layer -- simplest to implement
