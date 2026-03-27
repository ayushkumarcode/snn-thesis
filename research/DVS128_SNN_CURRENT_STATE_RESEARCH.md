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

## 2. Most Common Architectures

### Architecture Taxonomy

| Architecture Type | Description | Representative Methods | Typical Accuracy |
|------------------|-------------|----------------------|-----------------|
| **Convolutional SNN (CSNN)** | Conv layers + spiking neurons (LIF/PLIF) + pooling. The workhorse. | SpikingJelly baseline, DECOLLE | 93-97% |
| **Spiking Transformer** | Self-attention adapted for spike-based computation (SSA). | Spikformer, Spikeformer, MSVIT, Spike-HAR++ | 97-99% |
| **Recurrent SNN (RSNN/SCRNN)** | Recurrent connections for temporal modeling. | SCRNN, ALIF-based models | 92-96% |
| **Point-based SNN** | Process events as 3D point clouds, avoiding frame conversion. | SpikePoint | 98.74% |
| **Temporal Kernel Networks** | Structured temporal convolutions (not strictly SNN). | TENNs-PLEIADES, STREAM | 99.6-100% |
| **Self-Organizing SNN** | Topographic maps + glial cell mechanisms. | SG-SNN | 99.3% |
| **Hybrid (ANN-SNN)** | ANN-to-SNN conversion or knowledge distillation. | HSD, BKDSNN | 95-98% |
| **Lightweight/Pruned SNN** | Focus on parameter efficiency. | NSPDI-SNN (<7K params), LightSNN | 94-97% |

### Most Common Architecture for New Projects: CSNN

The **Convolutional SNN** with LIF or PLIF neurons is by far the most common starting point. The standard architecture pattern used in SpikingJelly's tutorial and most papers is:

```
{Conv2d-BatchNorm-LIF-MaxPool}*N -> Flatten -> FC-LIF -> FC-LIF -> Output
```

Typical configuration: 5 convolutional blocks with 128 channels, 3x3 kernels, followed by 2 fully connected layers (512, then 11 classes).

### Neuron Models

| Neuron | Description | DVS128 Gesture Impact |
|--------|-------------|----------------------|
| **LIF** | Leaky Integrate-and-Fire. Fixed decay constant. Standard choice. | Baseline ~96% |
| **PLIF** | Parametric LIF. Learnable decay per layer. ICCV 2021. | ~1-2% improvement over LIF |
| **IF** | Integrate-and-Fire. No leak. Simpler but less expressive. | Lower accuracy |
| **ALIF** | Adaptive LIF. Learnable threshold adaptation. | Improved temporal modeling |

**Key finding:** PLIF (learnable membrane time constant) consistently outperforms fixed-parameter LIF neurons, and is less sensitive to hyperparameter initialization. This is supported by the SpikingJelly authors' own ICCV 2021 paper.

**Source:** [PLIF Paper (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/papers/Fang_Incorporating_Learnable_Membrane_Time_Constant_To_Enhance_Learning_of_Spiking_ICCV_2021_paper.pdf), [GitHub: Parametric-LIF](https://github.com/fangwei123456/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron)

---

## 3. Framework Comparison: SpikingJelly vs. snnTorch

### Head-to-Head Comparison

| Feature | SpikingJelly | snnTorch |
|---------|-------------|---------|
| **DVS128 Loader** | Built-in, works well. Auto-downloads AEDAT, converts to npz/frames. | DEPRECATED (spikevision broken). Must use Tonic library. |
| **DVS128 Tutorial** | Complete end-to-end classification tutorial with code | Partial (Tutorial 7 uses NMNIST, not DVS128 directly) |
| **Pre-built DVS Model** | Yes: classify_dvsg example with full training script | No pre-built DVS128 model |
| **Training Speed** | ~18s/epoch (CUDA backend) to ~28s/epoch (PyTorch backend) on RTX 2080 Ti | Slower. No custom CUDA kernels. torch.compile does not help snnTorch significantly. |
| **CUDA Acceleration** | CuPy backend: 0.26s fwd+bwd for 16K neuron benchmark. Up to 11x speedup at T=32. | No custom CUDA. Relies on standard PyTorch. |
| **Neuron Models** | LIF, PLIF, IF, QIF, EIF, Izhikevich | Leaky, Synaptic, Alpha, Recurrent LIF, LSTM-based |
| **Documentation Quality** | Good but partly in Chinese. English docs available. | Excellent. Very tutorial-driven, beginner-friendly. |
| **Community/Stars** | ~3.3K GitHub stars | ~2.5K GitHub stars |
| **Academic Paper** | Science Advances (2023) | NeurIPS Workshop |
| **Multi-step Processing** | Native support. SeqToANNContainer for parallel timestep processing. | Sequential processing only (loop over timesteps). |
| **Surrogate Gradients** | ATan, Sigmoid, many others | ATan, Fast Sigmoid, Straight-Through, Triangular |
| **Active Development** | Active (last commit recent) | Active (v0.9.4 latest) |
| **PyPI Package** | `pip install spikingjelly` | `pip install snntorch` |

### Framework Benchmark (Open Neuromorphic, 2024)

For a 16K neuron network forward+backward pass:

| Framework | Time (seconds) | Notes |
|-----------|---------------|-------|
| SpikingJelly (CuPy) | 0.26 | Fastest by significant margin |
| Lava DL (SLAYER) | ~0.4-0.5 | Custom CUDA |
| Sinabs EXODUS | ~0.4-0.5 | Custom CUDA |
| Norse (torch.compile) | ~0.5-0.7 | JAX-competitive with compilation |
| snnTorch | ~1.0+ | No significant torch.compile speedup |
| Spyx (JAX) | ~0.3-0.4 | JAX-based, different ecosystem |

### Verdict for Thesis Project

**SpikingJelly is the clear winner for DVS128 Gesture recognition.** It has:
- A working DVS128 data pipeline out of the box
- A complete classification example achieving 96.18% accuracy
- Significantly faster training through CUDA kernels
- The PLIF neuron model (learnable parameters) built-in

snnTorch is better for learning fundamentals and has better documentation, but its DVS128 support is broken and requires workarounds through Tonic. If you choose snnTorch, budget extra time for data pipeline setup.

**Source:** [SNN Library Benchmarks - Open Neuromorphic](https://open-neuromorphic.org/blog/spiking-neural-network-framework-benchmarking/), [SpikingJelly GitHub](https://github.com/fangwei123456/spikingjelly), [snnTorch GitHub Issue #285](https://github.com/jeshraghian/snntorch/issues/285)
