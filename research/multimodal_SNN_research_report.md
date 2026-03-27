# Multimodal Spiking Neural Networks: Comprehensive Research Report

**Date**: 2026-02-25
**Research Focus**: Combining different data types (vision, audio, event camera, IMU) in a single spiking neural network
**Purpose**: Assess feasibility for an undergraduate thesis project

---

## Executive Summary

Multimodal SNNs -- combining different sensory data types within a single spiking neural network -- represent an **active and rapidly growing research area** that has seen significant acceleration since 2023. The field is no longer purely theoretical: multiple working implementations exist for audio-visual classification, event camera + RGB fusion, and sensor fusion for robotics. Critically, a paper published in August 2024 demonstrates *exactly* the simplified version proposed (MNIST digits + audio digits fusion), achieving 98.43% accuracy. This means a multimodal SNN thesis project is **achievable at the undergraduate level**, provided the scope is carefully bounded. The area has enough existing work to build upon but enough open problems to contribute meaningfully.

**Verdict: This is feasible as an undergraduate project. It sits at the boundary between "well-explored for PhDs" and "emerging for undergrads" -- an ideal sweet spot for a thesis that can demonstrate both competence and novelty.**

---

## 1. Has Anyone Combined Vision + Audio in an SNN?

**Yes -- this is now an established sub-field with at least 6 major papers from 2023-2025.**

### Key Papers and Systems

| Paper/System | Year | Datasets | Accuracy | Key Innovation |
|---|---|---|---|---|
| **SMMT** (Spiking Multi-Modal Transformer) | 2023 | CREMA-D, UrbanSound8K-AV | ~66% (CREMA-D) | Spiking Cross-Attention (SCA) mechanism for audio-visual fusion |
| **MISNet** (Multimodal Interaction Spiking Network) | 2024 | 5 audio-visual datasets | Competitive | MLIF neuron that synchronizes audiovisual spikes in a single neuron |
| **Bjorndahl et al.** | Aug 2024 | N-MNIST + SHD | 98.43% | Early/middle/late fusion comparison for digit recognition |
| **S-CMRL** | Feb 2025 | CREMA-D, UrbanSound8K-AV, MNISTDVS-NTIDIGITS | 73.25% (CREMA-D), 99.28% (MNISTDVS-NTIDIGITS) | Semantic-alignment + cross-modal residual learning |
| **TAAF** | May 2025 | CREMA-D, AVE, EAD | 77.55% (CREMA-D) | Temporal attention-guided adaptive fusion for modality imbalance |
| **Oikonomou et al.** | Nov 2024 | Various | Survey | Bio-inspired multimodal perception for robotics |

### Architecture Patterns

Three dominant fusion strategies have emerged:
1. **Early fusion**: Combining raw spike trains before feature extraction
2. **Late fusion (concatenation)**: Separate unimodal branches that merge at the decision layer -- simplest to implement
3. **Cross-modal attention**: Spiking attention mechanisms that allow modalities to guide each other

The Bjorndahl et al. (2024) paper is particularly relevant: they found that **late fusion (concatenation)** achieves nearly identical results to more complex fusion strategies, and the fusion depth has minimal impact on accuracy. This means the simplest approach works well.

### Sources
- [SMMT - IEEE Xplore](https://ieeexplore.ieee.org/iel7/7274989/10552653/10293172.pdf)
- [MISNet - ACM TOMM](https://dl.acm.org/doi/10.1145/3721981)
- [Bjorndahl et al. - arXiv 2409.00552](https://arxiv.org/abs/2409.00552)
- [S-CMRL - arXiv 2502.12488](https://arxiv.org/html/2502.12488)
- [S-CMRL Code - GitHub](https://github.com/Brain-Cog-Lab/S-CMRL)
- [TAAF - arXiv 2505.14535](https://arxiv.org/abs/2505.14535)
- [Oikonomou et al. - arXiv 2411.14147](https://arxiv.org/abs/2411.14147)

---

## 2. Combining Event Camera Data + Conventional Camera Data

**Yes -- this is one of the most active areas in neuromorphic computing (2024-2025).**

### Key Papers and Systems

| Paper/System | Year | Task | Key Contribution |
|---|---|---|---|
| **SFDNet** | 2025 | Object Detection | Fully spiking RGB-event fusion detector with LIMF neuron; state-of-the-art on PKU-DAVIS-SOD |
| **SSTFormer** | 2023-2025 | Frame-Event Recognition | Hybrid SNN-ANN with Memory Support Transformer for RGB + spiking event encoding |
| **DSF-Net** | 2025 | High-Speed Detection | Dynamic sparse fusion of event-RGB via spike-triggered attention |
| **SpikeFET** | 2025 | Object Tracking | First fully spiking frame-event tracker |
| **SNNPTrack** | 2025 | RGBE Tracking | SNN-based prompt learning for RGB-Event tracking (ICASSP 2025) |
| **RGB-Event Collision Prediction** | 2025 | Collision Prediction | Self-attention fusion for UAV collision prediction (IJCNN 2025) |

### Technical Approach

The standard pattern is:
- **RGB branch**: Processes conventional camera frames (sometimes using ANN layers)
- **Event branch**: Processes DVS event streams using spiking neurons (LIF/PLIF)
- **Fusion module**: Combines features via attention, concatenation, or cross-modal mechanisms

SSTFormer is notable because it has a **publicly available codebase** on GitHub (https://github.com/Event-AHU/SSTFormer) and a custom PokerEvent dataset with 114 classes and 27,102 frame-event pairs.

### Sources
- [SFDNet - MDPI Electronics](https://www.mdpi.com/2079-9292/14/6/1105)
- [SSTFormer - arXiv](https://arxiv.org/abs/2308.04369)
- [SSTFormer Code - GitHub](https://github.com/Event-AHU/SSTFormer)
- [DSF-Net - ACM MM 2025](https://dl.acm.org/doi/10.1145/3746027.3755846)
- [SpikeFET - arXiv](https://arxiv.org/pdf/2505.20834)
- [RGB-Event Collision - IJCNN 2025](https://arxiv.org/html/2505.04258v2)
