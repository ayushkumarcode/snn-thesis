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

