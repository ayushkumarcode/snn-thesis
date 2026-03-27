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
