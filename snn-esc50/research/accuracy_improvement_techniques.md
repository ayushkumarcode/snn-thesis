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

- **Paper:** "One-Timestep is Enough: Achieving High-Performance ANN-to-SNN Conversion via Scale-and-Fire Neurons" (2025, arXiv:2510.23383)
- **Key idea:** Scale-and-Fire Neuron (SFN) enables single-timestep inference by adopting parameters estimated from ANN activation distribution, refined via Bayesian optimization. Eliminates temporal integration errors.
- **Accuracy:** Best accuracy among binary/ternary/multi-level SNNs with 1 timestep.
- **Applicability:** MEDIUM. Ultra-low latency scenario. May not preserve full accuracy on complex tasks.
- **Implementation complexity:** MEDIUM.

### 1d. Negative Spikes for Conversion

- **Paper:** "A Fast and Accurate ANN-SNN Conversion Algorithm with Negative Spikes" (IJCAI 2025)
- **Key idea:** Allow negative spikes during conversion to better approximate ReLU/LeakyReLU outputs. Outperforms two-stage methods by 1.29% at T=4.
- **Accuracy:** Better than two-stage at T=4 than two-stage at T=8.
- **Applicability:** HIGH. Compatible with our architecture.
- **Implementation complexity:** MEDIUM.

---

## 2. Knowledge Distillation (ANN Teacher -> SNN Student)

### 2a. SAKD: Self-Architectural Knowledge Distillation

- **Paper:** Xu et al., "Self-architectural knowledge distillation for spiking neural networks" (Neural Networks 178, 2024; originally OpenReview 2022)
- **Key idea:** Use same-architecture ANN as teacher. Directly transfer pre-trained ANN weights to SNN, then distill both intermediate features and logits. Bilevel teacher-student training.
- **Accuracy:** ResNet-18 on CIFAR-100: 80.48% with 4 timesteps, SURPASSING the ANN (79.90%). +3.49% over non-distilled SNN. On ImageNet: SEW-ResNet152 achieves 77.30% (SNN SOTA).
- **Applicability:** VERY HIGH. Our ANN and SNN share identical architecture (Conv+BN+MaxPool+FC). SAKD is designed exactly for this case. The ANN teacher is already trained at 63.85%.
- **Implementation complexity:** MEDIUM. Need to: (1) load ANN teacher weights, (2) modify training loop to compute feature-level + logit-level KD loss, (3) balance KD loss with task loss.
- **Expected result:** 50-58% (could potentially match or exceed ANN with perfect distillation)

### 2b. Constructing Deep SNNs from ANNs with KD (CVPR 2023)

- **Paper:** Xu, Li et al., "Constructing Deep Spiking Neural Networks from Artificial Neural Networks with Knowledge Distillation" (CVPR 2023, pp. 7886-7895)
- **Key idea:** ANN-SNN joint training with KD. SNN student learns rich features from ANN teacher through the KD method, avoiding training from scratch with non-differentiable spikes. Strong noise immunity.
- **Accuracy:** Improved noise robustness (relevant for ESC-50 audio).
- **Applicability:** HIGH. Same architecture teacher-student is ideal.
- **Implementation complexity:** MEDIUM.

### 2c. MD-SNN: Membrane Potential-Aware Distillation

- **Paper:** "MD-SNN: Membrane Potential-aware Distillation on Quantized Spiking Neural Network" (2024, arXiv:2512.04443)
- **Key idea:** Distills using MEMBRANE POTENTIALS (not just spikes/logits) between teacher and student. Single T=4 teacher guides multiple students via temporal membrane alignment.
- **Accuracy:** +0.48-1.06% over quantized baselines. 30% training FLOPs reduction.
- **Applicability:** MEDIUM-HIGH. Primarily for quantized SNNs but membrane distillation idea is broadly useful.
- **Implementation complexity:** MEDIUM.

### 2d. Enhanced Self-Distillation for SNNs

- **Paper:** "Enhanced Self-Distillation Framework for Efficient Spiking Neural Network Training" (2024, arXiv:2510.06254)
- **Key idea:** SNN distills knowledge from its own later timesteps to earlier timesteps.
- **Applicability:** MEDIUM. No external teacher needed.
- **Implementation complexity:** LOW-MEDIUM.

---

## 3. Hybrid Training (ANN Init + SNN Fine-tune)

### 3a. Rathi et al. Hybrid Conversion + STDB

- **Paper:** Rathi et al., "Enabling Deep Spiking Neural Networks with Hybrid Conversion and Spike Timing Dependent Backpropagation" (ICLR 2020)
- **Key idea:** Fast ANN-to-SNN conversion for weight initialization, then fine-tune with spike-timing dependent backpropagation (STDB). Achieves similar accuracy to full conversion with MUCH fewer timesteps, and faster convergence than training from scratch.
- **Accuracy:** Conversion loss 2-3% after fine-tuning, with far fewer timesteps.
- **Applicability:** VERY HIGH. This is the simplest high-impact change: (1) convert our 63.85% ANN, (2) fine-tune as SNN for a few epochs. Code: https://github.com/nitin-rathi/hybrid-snn-conversion
- **Implementation complexity:** MEDIUM. Need to map ReLU->LIF, set thresholds, then fine-tune.
- **Expected result:** 55-62% accuracy

### 3b. Three-Stage Hybrid Fine-Tuning for Audio

- **Paper:** "Three-stage hybrid spiking neural networks fine-tuning for speech enhancement" (Frontiers in Neuroscience, April 2025)
- **Key idea:** (1) Train ANN, (2) convert to SNN, (3) hybrid fine-tune where forward pass uses spikes but backward pass uses ANN signals. Applied to Wave-U-Net and ConvTasNet for AUDIO processing.
- **Accuracy:** Significant improvement over unconverted SNNs on noisy VCTK and TIMIT.
- **Applicability:** VERY HIGH. This is specifically for audio SNNs. Same three-stage approach directly applicable to our pipeline.
- **Implementation complexity:** MEDIUM.
- **Expected result:** Best of both worlds -- should preserve most of ANN's 63.85%

---

## 4. Loss Function Improvements

### 4a. TET: Temporal Efficient Training

- **Paper:** Deng et al., "Temporal Efficient Training of Spiking Neural Network via Gradient Re-weighting" (ICLR 2022)
- **Key idea:** Instead of accumulating spikes then applying loss once, optimize every timestep's pre-synaptic inputs independently. Compensates for momentum loss in surrogate gradient descent, converges to flatter minima with better generalizability. Also improves temporal scalability.
- **Accuracy:** +10pp on DVS-CIFAR10 (83% vs 73% prior SOTA). Consistent improvements on CIFAR-10/100/ImageNet.
- **Applicability:** VERY HIGH. We already use per-timestep CE loss (standard snnTorch approach) -- but TET adds gradient re-weighting and momentum compensation. Trivial to implement. Code: https://github.com/Gus-Lab/temporal_efficient_training
- **Implementation complexity:** LOW. Modify loss function only. ~20 lines of code change.
- **Expected result:** +2-5pp (47% -> 49-52%)

**CRITICAL NOTE:** Our current training already does per-timestep CE loss (line 71 of train.py: `loss += criterion(mem_out[step], targets)`). TET's key addition is gradient re-weighting to avoid bad local minima, plus a regularization term. This is probably our single easiest win.

### 4b. Temporal Regularization Training

- **Paper:** "Temporal Regularization Training: Unleashing the Potential of Spiking Neural Networks" (2025, arXiv:2506.19256)
- **Key idea:** Regularize temporal consistency of SNN outputs across timesteps.
- **Applicability:** MEDIUM. Complementary to TET.
- **Implementation complexity:** LOW.

---

## 5. Learnable Neuron Parameters

### 5a. PLIF: Parametric LIF with Learnable Membrane Time Constant

- **Paper:** Fang et al., "Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks" (ICCV 2021)
- **Key idea:** Make membrane time constant (tau, which controls beta in snnTorch) a learnable per-layer parameter. Different layers can learn different decay rates suited to their function.
- **Accuracy:** CIFAR-10: 93.50% (vs lower with fixed beta). Robust to initialization.
- **Applicability:** VERY HIGH. snnTorch already supports this! Just set `learn_beta=True` in `snn.Leaky()`. Zero architecture change needed.
- **Implementation complexity:** TRIVIALLY LOW. Change 4 lines in snn_model.py:
  ```python
  self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)
  ```
- **Expected result:** +1-2pp

### 5b. Learnable Threshold
