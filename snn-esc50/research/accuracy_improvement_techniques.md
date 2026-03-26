# SNN Accuracy Improvement Techniques: Exhaustive Research Report

**Date:** 2026-03-25
**Context:** ESC-50 environmental sound classification, Conv SNN (622K params), 47.15% SNN vs 63.85% ANN
**Objective:** Techniques to close or significantly narrow the 16.7pp SNN-ANN accuracy gap

---

## Executive Summary

This report catalogues every promising technique found across ~40 recent papers (2022-2026) that could improve our SNN's accuracy on ESC-50. The techniques fall into three tiers of expected impact:

**Tier 1 -- High Impact, Moderate Effort (likely +5-15pp):**
1. ANN-to-SNN Conversion (preserve ~60-63% of ANN's 63.85%)
2. Knowledge Distillation from ANN teacher (typically +3-11pp)
3. Hybrid Training: ANN init + SNN fine-tune (typically +5-10pp)
4. TET Loss Function (reported +10pp on DVS-CIFAR10, consistently +2-5pp elsewhere)
5. Learnable Neuron Parameters (PLIF/GLIF) (+1-3pp)

**Tier 2 -- Moderate Impact, Moderate Effort (likely +2-5pp):**
6. Temporal Batch Normalization (TEBN/tdBN/BNTT)
7. Ternary Spikes (negative spikes)
8. Residual/Skip Connections (SEW-ResNet style)
9. Channel/Temporal Attention (TCJA-SNN)
10. Advanced Surrogate Gradients (AdaLi, learnable)

**Tier 3 -- Lower/Uncertain Impact, Higher Effort:**
11. NAS for SNNs
12. Self-supervised pre-training
13. Spiking-LEAF auditory front-end
14. EventMix/neuromorphic augmentation
15. Quantization-aware training

---

## 1. ANN-to-SNN Conversion

### 1a. PASCAL: Precise ANN-SNN Conversion via Spike Accumulation

- **Paper:** "PASCAL: Precise and Efficient ANN-SNN Conversion using Spike Accumulation and Adaptive Layerwise Activation" (2025, arXiv:2505.01730)
- **Key idea:** Augments IF neuron with spike accumulation/inhibition so integrate-fire process EXACTLY reproduces ANN activation statistics. Uses QCFS (Quantization-Clip-Floor-Shift) activation and configures per-layer optimal timesteps. Supports negative (inhibitory) spikes.
- **Accuracy:** Near-lossless. ResNet-34 achieves ~74% on ImageNet matching source ANN, with 64x fewer timesteps than prior conversion methods.
- **Applicability to us:** VERY HIGH. We have a trained ANN at 63.85%. PASCAL could convert it to an SNN preserving most of that accuracy. Our architecture (Conv+BN+MaxPool+FC) is standard and compatible. Key question: whether our 25 timesteps are sufficient or if we need PASCAL's adaptive layerwise timesteps.
- **Implementation complexity:** MEDIUM. Need to: (1) retrain ANN with QCFS activation (replace ReLU), (2) apply conversion algorithm, (3) optionally fine-tune. Code available at https://github.com/BrainSeek-Lab/PASCAL
- **Expected result:** 55-62% accuracy (preserving 85-97% of ANN accuracy)

### 1b. Data-Driven Threshold and Potential Initialization

- **Paper:** Bojkovic et al., "Data Driven Threshold and Potential Initialization for Spiking Neural Networks" (AISTATS 2024, PMLR 238:4771-4779)
- **Key idea:** After ANN-to-SNN conversion, optimally initialize SNN thresholds and membrane potentials using ANN activation distributions. Theoretical framework minimizes conversion error.
- **Accuracy:** SOTA on CIFAR10/100/ImageNet across architectures. Conversion loss typically 1-3%.
- **Applicability:** HIGH. Direct drop-in after any conversion method. Code: https://github.com/srinuvaasu/data_driven_init
- **Implementation complexity:** LOW-MEDIUM. Post-conversion calibration step.
- **Expected result:** Additional +1-3pp on top of any conversion baseline

### 1c. One-Timestep Conversion via Scale-and-Fire Neurons
