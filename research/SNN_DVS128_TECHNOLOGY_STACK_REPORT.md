# Comprehensive Technology Stack Report: Spiking Neural Network DVS128 Gesture Recognition

**Research Date:** 2026-02-23
**Scope:** Full-stack investigation for building an SNN-based DVS128 Gesture Recognition system

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [SpikingJelly Framework](#2-spikingjelly-framework)
3. [snnTorch Framework](#3-snntorch-framework)
4. [DVS128 Data Pipeline](#4-dvs128-data-pipeline)
5. [Tonic Library](#5-tonic-library)
6. [Training Infrastructure](#6-training-infrastructure)
7. [Evaluation and Visualization](#7-evaluation-and-visualization)
8. [End-to-End Examples](#8-end-to-end-examples)
9. [ANN Comparison and Frame Integration](#9-ann-comparison-and-frame-integration)
10. [State-of-the-Art Benchmarks](#10-state-of-the-art-benchmarks)
11. [Recommended Stack Configuration](#11-recommended-stack-configuration)
12. [Sources](#12-sources)

---

## 1. Executive Summary

Building a Spiking Neural Network for DVS128 Gesture Recognition requires integrating several key components: a neuromorphic dataset loader, an SNN framework built on PyTorch, appropriate preprocessing transforms, and visualization/evaluation tools. The two leading SNN frameworks are **SpikingJelly** (from the Peking University group, published in Science Advances) and **snnTorch** (from UC Santa Cruz, by Jason Eshraghian). Both are built on PyTorch and support surrogate gradient-based backpropagation for training deep SNNs.

SpikingJelly provides a more complete built-in pipeline for DVS128 Gesture classification, including its own dataset loader, a pre-built DVSGestureNet model, and full training scripts. snnTorch offers a more modular, tutorial-driven approach and integrates with the **Tonic** library for neuromorphic dataset loading. For a thesis project, SpikingJelly offers the fastest path to a working baseline, while snnTorch offers better educational resources and more flexible visualization tools.

The DVS128 Gesture dataset consists of 1176 training and 288 test samples across 11 gesture classes, recorded using a Dynamic Vision Sensor (DVS128) camera at 128x128 resolution. Raw data comes in AEDAT 3.1 format as polarity events (x, y, timestamp, polarity). Events must be converted to frame tensors for batch training, which both frameworks handle through various binning strategies.

Training on a single GPU (e.g., RTX 2080 Ti with 12GB VRAM) takes approximately 18-28 seconds per epoch, with 64-256 epochs typically needed to reach peak accuracy of approximately 96-97%. Apple M-series MacBooks can be used via PyTorch MPS backend, though the CuPy/Triton acceleration backends for SpikingJelly will not be available. Energy efficiency can be estimated without neuromorphic hardware using the **syops** library, which computes synaptic operations (SynOps) and estimates energy based on 45nm technology costs.

---

## 2. SpikingJelly Framework

### 2.1 Version Requirements

| Dependency | Version |
|---|---|
| Python | >= 3.6 (recommended: 3.9 - 3.11) |
| PyTorch | >= 2.2.0 (tested on 2.7.1) |
| torchvision | Required (version matching PyTorch) |
| torchaudio | Required (version matching PyTorch) |
| numpy | Required (no pinned version) |
| scipy | Required (no pinned version) |
| tensorboard | Required |
| einops | Required |
| tqdm | Required |
| h5py | Required |
| matplotlib | Required |

**Source:** https://github.com/fangwei123456/spikingjelly/blob/master/setup.py

### 2.2 Installation

```bash
# Install PyTorch first (visit pytorch.org for your platform)
pip install torch torchvision torchaudio

# Install SpikingJelly stable release
