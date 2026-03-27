# DVS128 Gesture Recognition with SNNs: Current State of the Field

**Research Date:** 2026-02-25
**Purpose:** Comprehensive assessment for undergraduate thesis project planning

---

## Executive Summary

DVS128 Gesture recognition using Spiking Neural Networks is approaching a **saturated benchmark**. The dataset, originally introduced by IBM in 2017 with a baseline of ~94%, has seen accuracy climb to 99.6% (TENNs-PLEIADES) and even 100% (STREAM) as of 2024-2025. The dataset contains only 1,464 samples (1,176 train / 288 test) across 11 gesture classes, making it small by modern standards. This saturation means that simply achieving high accuracy is no longer a meaningful contribution.

However, several genuine research opportunities remain that are well-suited to an undergraduate thesis: (1) systematic comparison studies across architectures/frameworks that nobody has done cleanly, (2) efficiency-focused investigations (accuracy vs. parameters vs. timesteps vs. energy), (3) event representation ablation studies, and (4) neuron model comparisons (LIF vs. PLIF vs. others). The key differentiator for a good undergrad project is NOT achieving SOTA accuracy -- it is rigorous experimental methodology, reproducibility, and genuine analysis.

**SpikingJelly** is the recommended framework. It has built-in DVS128 support, faster training via CUDA kernels, and a working end-to-end example achieving ~96% accuracy out of the box. snnTorch's DVS loader (spikevision) is **deprecated and broken**, requiring the Tonic library as a workaround, adding pipeline complexity.

---

## 1. State-of-the-Art Accuracy on DVS128 Gesture

### Current Leaderboard (Papers with Code, as of Feb 2025)

| Rank | Method | Accuracy (%) | Year | Parameters | Timesteps | Type |
|------|--------|-------------|------|------------|-----------|------|
| 1 | TENNs-PLEIADES | 99.59 (100 w/ filter) | 2024 | 192K | Variable | Temporal Neural Network |
| 2 | SG-SNN | 99.3 | 2025 | Not reported | Multiple | Self-Organizing Glial SNN |
| 3 | Spikeformer | 98.96 | 2024 | Large | 16 | Spiking Transformer |
| 4 | MSVIT | 98.80 | 2025 | Not reported | 16 | Multi-Scale Vision Transformer |
| 5 | SpikePoint | 98.74 | 2024 | Small (~1/21 of ANN equiv.) | 16 | Point-based SNN |
| 6 | Spike-HAR++ | 98.26 | 2024 | Lightweight | - | Spiking Transformer |
| 7 | STREAM | 100.0 | 2024 | Not reported | - | Temporal Kernel Network |
| 8 | EgoEvGesture | 96.97 | 2025 | - | - | Egocentric SNN |
| 9 | SpikingJelly baseline | 96.18 | 2021 | ~1.5M (est.) | 16 | 5-layer CSNN + LIF |
| 10 | snnTorch example | ~90.6 | 2024 | Small | 300 | 3-layer CSNN + LIF |

**Key observation:** The gap between the SpikingJelly tutorial baseline (96.18%) and SOTA (99.6%) is only ~3.4 percentage points. On a test set of only 288 samples, this difference amounts to roughly 10 correctly classified samples.

### Accuracy Progression Over Time

- 2017 (IBM original): ~94.59% (TrueNorth EEDN)
- 2020-2021: 95-97% (PLIF, TA-SNN, DECOLLE)
- 2022-2023: 97-98% (SEW-ResNet, Spikformer, attention mechanisms)
- 2024-2025: 98.7-99.6% (SpikePoint, SG-SNN, TENNs-PLEIADES)

**Source:** [Papers with Code DVS128 Gesture Benchmark](https://paperswithcode.com/sota/gesture-recognition-on-dvs128-gesture), [CatalyzeX DVS128 Gesture Dataset](https://www.catalyzex.com/s/Dvs128%20Gesture%20Dataset)

---

