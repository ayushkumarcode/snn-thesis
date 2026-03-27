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
3. **Cross-modal attention**: spiking attention that lets modalities guide each other

the Bjorndahl paper is really relevant here -- they found that **late fusion (concatenation)** gets nearly identical results to more complex fusion strategies, and fusion depth barely matters. so the simplest approach works. nice.

### Sources
- [SMMT - IEEE Xplore](https://ieeexplore.ieee.org/iel7/7274989/10552653/10293172.pdf)
- [MISNet - ACM TOMM](https://dl.acm.org/doi/10.1145/3721981)
- [Bjorndahl et al. - arXiv 2409.00552](https://arxiv.org/abs/2409.00552)
- [S-CMRL - arXiv 2502.12488](https://arxiv.org/html/2502.12488)
- [S-CMRL Code - GitHub](https://github.com/Brain-Cog-Lab/S-CMRL)
- [TAAF - arXiv 2505.14535](https://arxiv.org/abs/2505.14535)
- [Oikonomou et al. - arXiv 2411.14147](https://arxiv.org/abs/2411.14147)

---

## 2. Event Camera + Conventional Camera Fusion

**one of the most active areas in neuromorphic computing right now (2024-2025).**

### Key Papers

| Paper/System | Year | Task | Key Contribution |
|---|---|---|---|
| **SFDNet** | 2025 | Object Detection | Fully spiking RGB-event fusion detector with LIMF neuron; SOTA on PKU-DAVIS-SOD |
| **SSTFormer** | 2023-2025 | Frame-Event Recognition | Hybrid SNN-ANN with Memory Support Transformer for RGB + spiking event encoding |
| **DSF-Net** | 2025 | High-Speed Detection | Dynamic sparse fusion of event-RGB via spike-triggered attention |
| **SpikeFET** | 2025 | Object Tracking | First fully spiking frame-event tracker |
| **SNNPTrack** | 2025 | RGBE Tracking | SNN-based prompt learning for RGB-Event tracking (ICASSP 2025) |
| **RGB-Event Collision Prediction** | 2025 | Collision Prediction | Self-attention fusion for UAV collision prediction (IJCNN 2025) |

### Standard Pattern

- **RGB branch**: processes conventional camera frames (sometimes ANN layers)
- **Event branch**: processes DVS event streams with spiking neurons (LIF/PLIF)
- **Fusion module**: combines features via attention, concatenation, or cross-modal mechanisms

SSTFormer is worth noting -- it has a **publicly available codebase** (https://github.com/Event-AHU/SSTFormer) and a custom PokerEvent dataset with 114 classes and 27,102 frame-event pairs.

### Sources
- [SFDNet - MDPI Electronics](https://www.mdpi.com/2079-9292/14/6/1105)
- [SSTFormer - arXiv](https://arxiv.org/abs/2308.04369)
- [SSTFormer Code - GitHub](https://github.com/Event-AHU/SSTFormer)
- [DSF-Net - ACM MM 2025](https://dl.acm.org/doi/10.1145/3746027.3755846)
- [SpikeFET - arXiv](https://arxiv.org/pdf/2505.20834)
- [RGB-Event Collision - IJCNN 2025](https://arxiv.org/html/2505.04258v2)

---

## 3. Sensor Fusion with SNNs (IMU + Camera etc)

**emerging area, especially for robotics and navigation.**

### Key Research

| Paper/System | Year | Sensors | Platform | Key Finding |
|---|---|---|---|---|
| **Lopez-Osorio et al.** | 2024 | Neuromorphic vision + feedback sensors | SpiNNaker | Real-time robot adaptation using SNN sensor fusion |
| **Spiking Neural-Invariant Kalman Fusion** | Jan 2026 | IMU | CPU/GPU | SNN extracts motion features from noisy IMU, fused with Kalman filter for pose estimation |
| **Loihi-2 Sensor Fusion** | Aug 2024 | Camera, LIDAR, GPS, IMU | Intel Loihi-2 | 100x more efficient than CPU, 30x more than GPU |
| **Marine-Inspired Navigation** | 2025 | Multiple underwater sensors | Neuromorphic | Multi-modal fusion for autonomous subaquatic navigation |

the Loihi-2 result is pretty wild: **100x energy efficiency over CPU and ~30x over GPU** for sensor fusion tasks across AIODrive, Oxford Radar RobotCar, nuScenes, etc. that's the practical motivation for neuromorphic sensor fusion right there.

### Sources
- [Lopez-Osorio et al. - Wiley](https://advanced.onlinelibrary.wiley.com/doi/10.1002/aisy.202300646)
- [Spiking Kalman - arXiv](https://arxiv.org/abs/2601.08248)
- [Loihi-2 Sensor Fusion - arXiv](https://arxiv.org/abs/2408.16096)
- [Marine Navigation - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12608416/)
- [Neuromorphic Solutions Book - Springer](https://link.springer.com/book/10.1007/978-3-031-63565-6)

---

## 4. Is This Still an Open Problem?

**yes -- multiple survey papers explicitly call it out.**

### Known Gaps

1. **Limited exploration**: "Imaging applications remain strongly biased toward classification and regression tasks, with limited exploration of segmentation or multimodal integration" (SNN Imaging Survey, 2025)

2. **Dataset scarcity**: lack of large-scale multimodal neuromorphic datasets

3. **Modality imbalance**: when vision is way stronger than audio, the network just ignores audio. TAAF (2025) addresses this but it's still open

4. **Temporal alignment**: synchronizing spike trains from modalities at different temporal scales -- not solved in general

5. **Scalability beyond classification**: extending to detection, segmentation, generation is mostly unexplored

6. **Bio-plausibility**: most current approaches use surrogate gradients (not bio-plausible). combining multimodal fusion with STDP is barely explored

7. **Missing modality robustness**: what if one modality is absent or corrupted? barely studied for SNNs

### Research Questions That Could Work for an Undergrad

- how does fusion strategy (early/mid/late) affect robustness to noise in one modality?
- can STDP-based learning do competitive multimodal fusion?
- what's the minimum network size for effective multimodal SNN fusion?
- how does temporal resolution mismatch between modalities affect things?
- can knowledge distillation from ANN multimodal models help SNN ones?

### Sources
- [SNN Imaging Survey - MDPI Sensors](https://www.mdpi.com/1424-8220/25/21/6747)
- [SNN Multimodal Neuroimaging Review - Frontiers](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1623497/full)
- [SNN Ubiquitous Computing Survey - arXiv](https://arxiv.org/html/2506.01737v1)

---

## 5. How Complex Would a Simplified Version Be?

### Minimum Viable Multimodal SNN (Late Fusion, Digit Classification)

based on Bjorndahl et al. (2024):

```
Visual Branch (N-MNIST):
  - Input: 34x34x2 spike trains (300 timesteps)
  - 2-3 spiking conv layers with LIF neurons
  - Output: 128-dim feature vector

Auditory Branch (SHD):
  - Input: 700 channels spike trains
  - 2-3 spiking FC layers with LIF neurons
  - Output: 128-dim feature vector

Fusion:
  - Concatenation: 256-dim combined vector
  - Classification head: 1-2 FC layers -> 10 classes

ESTIMATED PARAMETERS: ~100K-500K (tiny by modern standards)
TRAINING TIME: Hours on a single GPU (even laptop)
CODE COMPLEXITY: ~300-500 lines Python (snnTorch/SpikingJelly)
```

### Comparison to Other Project Types

| Project Type | Approx Params | Training Time | Code Lines | Difficulty |
|---|---|---|---|---|
| Single-modality SNN (MNIST) | 50K-200K | <1 hour | 150-250 | Beginner |
| **Multimodal SNN (digit fusion)** | **100K-500K** | **2-6 hours** | **300-500** | **Intermediate** |
| Audio-visual SNN (CREMA-D) | 1M-5M | 1-2 days | 800-1500 | Advanced |
| RGB-Event object detection | 5M-20M | Days-weeks | 2000+ | PhD-level |

### Why It's Manageable

1. **Both datasets are small**: N-MNIST is 28x28, SHD is 700 channels
2. **Both are pre-spiked**: no custom encoding needed, data is already spikes
3. **Late concatenation is trivial**: just `torch.cat([visual_features, audio_features], dim=1)`
4. **Well-established training**: surrogate gradients + cross-entropy loss
5. **Existing code**: S-CMRL on GitHub, snnTorch tutorials, multiple papers

---

## 6. Bibliography (2023-2026)

### Core Multimodal SNN Papers

1. **Gu et al. (2023)** - "Transformer-Based Spiking Neural Networks for Multimodal Audiovisual Classification" - IEEE TCDS. [Link](https://ieeexplore.ieee.org/iel7/7274989/10552653/10293172.pdf)

2. **Wang et al. (2023-2025)** - "SSTFormer: Bridging Spiking Neural Network and Memory Support Transformer for Frame-Event based Recognition" - IEEE TCDS. [Link](https://arxiv.org/abs/2308.04369)

3. **Bjorndahl et al. (Aug 2024)** - "Digit Recognition using Multimodal Spiking Neural Networks" - Submitted to IEEE ICASSP 2025. [Link](https://arxiv.org/abs/2409.00552) -- **most relevant for undergrad project**

4. **MISNet (2024)** - "Towards Energy-efficient Audio-visual Classification via Multimodal Interactive Spiking Neural Network" - ACM TOMM. [Link](https://dl.acm.org/doi/10.1145/3721981)

5. **Oikonomou et al. (Nov 2024)** - Survey on bio-inspired multimodal perception for robotics. [Link](https://arxiv.org/abs/2411.14147)

6. **S-CMRL (Feb 2025)** - "Enhancing Audio-Visual Spiking Neural Networks through Semantic-Alignment and Cross-Modal Residual Learning." [Link](https://arxiv.org/abs/2502.12488)

7. **TAAF (May 2025)** - "Spiking Neural Networks with Temporal Attention-Guided Adaptive Fusion for imbalanced Multi-modal Learning" - ACM MM 2025. [Link](https://arxiv.org/abs/2505.14535)

### Event Camera + RGB

8. **SFDNet (2025)** - Spiking RGB-event fusion for object detection. [Link](https://www.mdpi.com/2079-9292/14/6/1105)

9. **DSF-Net (2025)** - Dynamic sparse event-RGB fusion. [Link](https://dl.acm.org/doi/10.1145/3746027.3755846)

10. **SpikeFET (2025)** - Fully spiking frame-event tracker. [Link](https://arxiv.org/pdf/2505.20834)

### Sensor Fusion

11. **Lopez-Osorio et al. (2024)** - SNN sensor fusion on SpiNNaker. [Link](https://advanced.onlinelibrary.wiley.com/doi/10.1002/aisy.202300646)

12. **Loihi-2 Sensor Fusion (2024)** - 100x efficiency over CPU. [Link](https://arxiv.org/abs/2408.16096)

13. **Spiking Kalman (2026)** - SNN + Kalman for IMU localization. [Link](https://arxiv.org/abs/2601.08248)

### Surveys

14. [SNN Multimodal Neuroimaging Review (2025)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1623497/full)
15. [SNN Frameworks Benchmark (2025)](https://www.mdpi.com/2673-4117/6/11/304)
16. [SNN and Sound Review (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11362401/)

---

## 7. Could This Work as an Undergrad Project?

