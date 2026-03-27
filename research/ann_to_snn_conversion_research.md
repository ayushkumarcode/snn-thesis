# ANN-to-SNN Conversion: Comprehensive Research Report for Undergraduate Thesis Direction

**Research Date:** 2026-02-25
**Scope:** Evaluating ANN-to-SNN conversion as a practical and contributory undergraduate thesis direction

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [State of the Art (2024-2026)](#2-state-of-the-art-2024-2026)
3. [Available Tools and Frameworks](#3-available-tools-and-frameworks)
4. [Accuracy Loss During Conversion](#4-accuracy-loss-during-conversion)
5. [Which Architectures Convert Best](#5-which-architectures-convert-best)
6. [Timestep Requirements](#6-timestep-requirements)
7. [Undergraduate Contribution Opportunities](#7-undergraduate-contribution-opportunities)
8. [Recent Papers with Reproducible Code](#8-recent-papers-with-reproducible-code)
9. [Time to Get a Working Pipeline](#9-time-to-get-a-working-pipeline)
10. [Thesis Framing Recommendations](#10-thesis-framing-recommendations)
11. [Consolidated Accuracy Tables](#11-consolidated-accuracy-tables)
12. [Research Gaps and Open Problems](#12-research-gaps-and-open-problems)
13. [Risk Assessment](#13-risk-assessment)
14. [Sources](#14-sources)

---

## 1. Executive Summary

ANN-to-SNN conversion is one of the two dominant methods for building deep spiking neural networks (the other being direct training with surrogate gradients). The core idea is straightforward: take a pre-trained artificial neural network, replace ReLU activations with integrate-and-fire spiking neurons, normalize thresholds, and run inference where spike rates encode activation values. This is the **most cost-effective** method for obtaining high-accuracy SNNs because it leverages the mature ANN training ecosystem.

**Key finding for thesis viability:** This is an excellent undergraduate thesis direction. The conversion pipeline is well-supported by existing tools (SpikingJelly, snn_toolbox, snntorch, and standalone paper implementations), the core experiments are reproducible within weeks, and there are clear contribution opportunities in under-explored domains and architecture comparisons. The field is actively producing top-venue publications (ICML 2024/2025, CVPR 2025, NeurIPS 2023, ECCV 2024) with open-source code, making it both current and accessible.

**The strongest thesis framing would be:** "Evaluating the Practicality of ANN-to-SNN Conversion for [Specific Domain/Architecture]" -- where the specific domain is chosen to be something not yet comprehensively studied (medical imaging, audio classification, lightweight architectures like MobileNet/EfficientNet, or a head-to-head tool comparison).

---

## 2. State of the Art (2024-2026)

### 2.1 The Evolution of ANN-to-SNN Conversion

ANN-to-SNN conversion has evolved through three major phases:

**Phase 1 (2015-2019): Basic Rate Coding**
- Replace ReLU with IF neurons, normalize weights/thresholds
- Required 500-2500+ timesteps for competitive accuracy
- Limited to VGG-like architectures on CIFAR-10/MNIST
- Key papers: Diehl et al. 2015, Sengupta et al. 2019

**Phase 2 (2020-2023): Optimized Conversion with Reduced Latency**
- Introduction of threshold balancing, weight normalization, calibration
- Reduction to 32-256 timesteps while maintaining accuracy
- Extension to ResNets, deeper architectures, ImageNet-scale
- Key papers: SNN Calibration (ICML 2021), QCFS (ICLR 2022), unified framework (ICML 2023)

**Phase 3 (2024-2026): Ultra-Low Latency and Beyond-CNN Architectures**
- Conversion with 1-8 timesteps achieving near-ANN accuracy
- First successful Transformer-to-SNN conversions
