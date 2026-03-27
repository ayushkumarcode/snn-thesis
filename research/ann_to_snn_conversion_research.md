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
- First conversion of non-ReLU architectures (ConvNeXt, MLP-Mixer, ResMLP)
- Training-free conversion methods eliminating retraining requirements
- Extension to object detection, semantic segmentation, video classification

### 2.2 Landmark Papers (2024-2026)

| Paper | Venue | Key Contribution |
|-------|-------|-----------------|
| Sign Gradient Descent-based Neuronal Dynamics | ICML 2024 | First to convert ConvNeXt, MLP-Mixer, ResMLP (beyond ReLU) |
| Optimal ANN-SNN Conversion with Group Neurons | ICASSP 2024 | ResNet-34 on ImageNet: 73.61% at T=2 |
| SpikeYOLO | ECCV 2024 (Best Paper Candidate) | Integer-valued training + spike-driven inference for detection |
| Inference-Scale Complexity in ANN-SNN Conversion | CVPR 2025 | Training-free conversion at inference-scale; classification, segmentation, detection, video |
| Differential Coding for Training-Free ANN-to-SNN | ICML 2025 | Novel differential coding reduces spike counts and energy |
| Towards High-performance Spiking Transformers | ICLR 2025 | Spiking Transformer: 88.60% top-1, only 1% loss at T=4 |
| One-Timestep is Enough | 2025 | Scale-and-Fire neurons achieve high accuracy at T=1 |

### 2.3 Current Research Frontiers

1. **Ultra-low latency (T=1-4)**: The holy grail -- achieving ANN-level accuracy with minimal timesteps
2. **Transformer conversion**: Moving beyond CNNs to convert attention-based architectures
3. **Beyond-ReLU activation conversion**: Converting architectures using GELU, SiLU, Swish
4. **Training-free conversion**: No retraining needed -- just convert and run
5. **Domain expansion**: Object detection, segmentation, video, NLP tasks
6. **Energy-accuracy co-optimization**: Jointly optimizing for accuracy and neuromorphic energy efficiency
7. **Adaptive inference**: Dynamically adjusting timesteps per input for efficiency

---

## 3. Available Tools and Frameworks

### 3.1 Tool Comparison Matrix

| Feature | snn_toolbox | SpikingJelly (ann2snn) | snnTorch | Custom Paper Code |
|---------|------------|----------------------|---------|-------------------|
| **Input framework** | Keras, PyTorch, Caffe, Lasagne | PyTorch only | PyTorch only | Varies (usually PyTorch) |
| **Conversion method** | Weight normalization + threshold balancing | MaxNorm / RobustNorm / Scaling | Basic IF neuron replacement | Method-specific (QCFS, calibration, etc.) |
| **Backend simulators** | INIsim (built-in), Brian2, pyNN, SpiNNaker, Loihi | SpikingJelly's own simulator | snnTorch simulator | Usually custom |
| **Hardware deployment** | SpiNNaker, Loihi support | Limited | No | Varies |
| **Documentation quality** | Good (ReadTheDocs) | Good (English + Chinese) | Excellent (tutorials, Colab) | Varies (often minimal) |
| **Learning curve** | Moderate | Low (PyTorch-native) | Low (PyTorch-native) | High (read the paper) |
| **Maintenance (2024-25)** | Low activity; ~387 stars, some stale dependencies | Active; published in Science Advances; high activity | Active; good tutorial ecosystem | Depends on authors |
| **Best for** | Multi-backend deployment, SpiNNaker/Loihi | Fast prototyping, research | Learning, education, visualization | State-of-the-art results |
| **GitHub** | NeuromorphicProcessorProject/snn_toolbox | fangwei123456/spikingjelly | jeshraghian/snntorch | Various |

### 3.2 snn_toolbox (Detailed)

- **Repository:** https://github.com/NeuromorphicProcessorProject/snn_toolbox
- **Stars:** ~387 | **Forks:** ~103 | **Open issues:** 6
- **Pipeline:** Load ANN -> Parse layers -> Normalize parameters -> Convert to SNN -> Simulate
- **Supported layers:** Conv2D, Dense, BatchNorm (absorbed into preceding layer), Pooling, Flatten
- **Known limitations:**
  - Conv1D has normalization issues (GitHub issue #129)
  - Keras version compatibility problems (issue #87)
  - Lower maintenance activity in 2024-2025
  - ResNet accuracy degradation during conversion (issue #66)
- **Configuration:** Extensive config file system documented at ReadTheDocs
- **Best use case:** If you need SpiNNaker or Loihi deployment, or multi-backend comparison

### 3.3 SpikingJelly ann2snn (Detailed)

- **Repository:** https://github.com/fangwei123456/spikingjelly
- **Published:** Science Advances (2023) -- high credibility
- **Conversion modes:**
  - `MaxNorm`: Uses max activation values for threshold setting
  - `RobustNorm`: Uses 99.9% activation quantile (more robust to outliers)
  - Scaling mode: User-specified scaling parameters
- **Process:** ReLU -> IF neurons; AvgPool -> spatial downsampling
- **Pre-built examples:** `resnet18_cifar10.py`, `cnn_mnist.py`
- **Speed advantage:** Up to 11x speedup over other frameworks when T=32
- **Full-stack:** Supports neuromorphic datasets, ANN2SNN, surrogate gradients, and bio-plausible rules
- **Best use case:** Research prototyping, fast iteration, if already using PyTorch

### 3.4 snnTorch (Detailed)

- **Repository:** https://github.com/jeshraghian/snntorch
- **Tutorials:** Comprehensive series (Tutorial 1 through 7+), Colab notebooks
- **Conversion approach:** Simpler, more educational -- demonstrates basic ANN-to-SNN concepts
- **Example:** `Saad-data/ANN-to-SNN-Conversion-with-snnTorch` demonstrates a simple FC + ReLU -> SNN conversion
- **Best use case:** Learning the fundamentals, educational projects, clear visualization of spike dynamics

### 3.5 Standalone Research Code (Most Recommended for State-of-the-Art)

For an undergraduate thesis aiming at current results, the standalone paper implementations are often the best starting point:

