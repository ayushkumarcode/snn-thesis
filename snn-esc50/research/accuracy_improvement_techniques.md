# accuracy improvement techniques -- research notes

date: 2026-03-25
context: ESC-50 conv SNN (622K params), 47.15% SNN vs 63.85% ANN -- trying to close that 16.7pp gap

went through ~40 recent papers (2022-2026). techniques fall into three rough tiers:

tier 1 (likely +5-15pp): ANN-to-SNN conversion, knowledge distillation, hybrid ANN init + SNN fine-tune, TET loss, learnable neuron params (PLIF/GLIF)

tier 2 (likely +2-5pp): temporal batch normalization, ternary spikes, residual/skip connections, channel/temporal attention, advanced surrogate gradients

tier 3 (uncertain, higher effort): NAS, self-supervised pretraining, Spiking-LEAF auditory front-end, EventMix augmentation, QAT
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

- **Paper:** Various (Diet-SNN, LTR-LIF, etc.)
- **Key idea:** Make firing threshold a learnable parameter. snnTorch supports `learn_threshold=True`.
- **Accuracy:** +1.58% improvement reported.
- **Applicability:** VERY HIGH. Same as above -- single flag change.
- **Implementation complexity:** TRIVIALLY LOW.
  ```python
  self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_threshold=True, threshold=1.0)
  ```
- **Expected result:** +0.5-2pp

### 5c. GLIF: Gated Leaky Integrate-and-Fire

- **Paper:** Yao et al., "GLIF: A Unified Gated Leaky Integrate-and-Fire Neuron for Spiking Neural Networks" (NeurIPS 2022)
- **Key idea:** Fuse multiple bio-features (membrane leak, input integration, firing threshold, reset) with learnable gating factors. Enlarges representation space beyond standard LIF.
- **Accuracy:** ResNet-19 on CIFAR-100: 77.35% (+2.92% over Dspike). CIFAR-10: 95.03%.
- **Applicability:** MEDIUM. Would require replacing snnTorch's Leaky with custom GLIF neuron. Code: https://github.com/Ikarosy/Gated-LIF
- **Implementation complexity:** MEDIUM. Custom neuron model, but well-documented code available.
- **Expected result:** +2-4pp

### 5d. Per-Neuron Learnable Parameters

- **Paper:** snnTorch documentation
- **Key idea:** Instead of per-layer beta, initialize beta as a VECTOR (one per neuron) and set learn_beta=True. Each neuron learns its own optimal decay rate.
- **Applicability:** VERY HIGH. Supported natively in snnTorch.
- **Implementation complexity:** LOW.
  ```python
  import torch
  beta_init = torch.ones(256) * 0.95  # for FC1 layer with 256 neurons
  self.lif3 = snn.Leaky(beta=beta_init, spike_grad=spike_grad, learn_beta=True)
  ```
- **Expected result:** +0.5-1pp on top of per-layer learnable

---

## 6. Temporal Batch Normalization

### 6a. TEBN: Temporal Effective Batch Normalization

- **Paper:** "Temporal Effective Batch Normalization in Spiking Neural Networks" (NeurIPS 2022)
- **Key idea:** Assigns different learnable weights to each timestep to rescale presynaptic inputs. Smooths temporal distributions, stabilizes gradient norm.
- **Accuracy:** CIFAR-100: 74.37% (vs 66.6% BNTT, +7.77pp). DVS-CIFAR10: 75.1% (vs 60.5% NeuNorm, +14.6pp).
- **Applicability:** HIGH. Replace standard BN with TEBN. Compatible with our Conv-BN-MaxPool-LIF pattern.
- **Implementation complexity:** MEDIUM. Need to implement TEBN layer (not in snnTorch by default) or use SpikingJelly which has it.
- **Expected result:** +2-5pp

### 6b. tdBN: Threshold-Dependent Batch Normalization

- **Paper:** Zheng et al. (2021)
- **Key idea:** Extends BN to temporal dimension, incorporates firing threshold as hyperparameter. Can be folded into weights for zero inference overhead.
- **Accuracy:** Enables training deep SNNs from scratch with good accuracy.
- **Applicability:** HIGH. Our architecture already uses BN.
- **Implementation complexity:** MEDIUM.

### 6c. BNTT: Batch Normalization Through Time

- **Paper:** Panda et al. (Frontiers in Neuroscience, 2021). Code: https://github.com/Intelligent-Computing-Lab-Panda/BNTT-Batch-Normalization-Through-Time
- **Key idea:** Separate BN parameters for each timestep. Learns time-varying input distribution.
- **Accuracy:** Enables training deep SNNs with 25-30 timesteps (matches our setup).
- **Applicability:** HIGH. Direct replacement for our BN layers.
- **Implementation complexity:** MEDIUM.

---

## 7. Ternary Spikes (Negative Spikes)

- **Paper:** Guo et al., "Ternary Spike: Learning Ternary Spikes for Spiking Neural Networks" (AAAI 2024)
- **Key idea:** Replace binary {0,1} spikes with ternary {-1, 0, +1}. Dramatically increases information capacity while preserving event-driven and multiplication-free advantages. Learnable ternary threshold.
- **Accuracy:** CIFAR-10: 95.60% (ResNet19), +0.09pp over prior binary. CIFAR-100: ~+7% over binary with 2 timesteps. Key: ternary representation far exceeds binary in information capacity.
- **Applicability:** MEDIUM-HIGH. Would need to modify neuron model or use custom implementation. Not natively in snnTorch. Code: https://github.com/yfguo91/Ternary-Spike
- **Implementation complexity:** MEDIUM. Custom neuron firing function.
- **Expected result:** +2-5pp (especially beneficial for our information-bottleneck problem)

---

## 8. Residual/Skip Connections

### 8a. SEW ResNet: Spike-Element-Wise Residual Learning

- **Paper:** Fang et al., "Deep Residual Learning in Spiking Neural Networks" (NeurIPS 2021). Code: https://github.com/fangwei123456/Spike-Element-Wise-ResNet
- **Key idea:** Element-wise addition of spike tensors (not membrane potentials) for skip connections. Enables identity mapping, overcomes vanishing/exploding gradients. First time >100 layer SNN training possible.
- **Accuracy:** DVS Gesture: 97.92% (vs 90.97% for standard spiking ResNet, +6.95pp).
- **Applicability:** MEDIUM. Our current architecture has NO skip connections. Adding them would increase model complexity slightly but could significantly help gradient flow through 4 LIF layers.
- **Implementation complexity:** MEDIUM. Add skip connections between conv blocks.
- **Expected result:** +2-4pp

---

## 9. Attention Mechanisms

### 9a. TCJA-SNN: Temporal-Channel Joint Attention

- **Paper:** "TCJA-SNN: Temporal-Channel Joint Attention for Spiking Neural Networks" (IEEE TNNLS, 2024)
- **Key idea:** Squeeze-and-excitation adapted for SNNs. Compress spike stream, apply 1D convolution attention independently on temporal and channel dimensions.
- **Accuracy:** Up to +15.7% on neuromorphic datasets. Consistent improvements across benchmarks.
- **Applicability:** MEDIUM-HIGH. Add temporal-channel attention after conv layers. Small parameter overhead.
- **Implementation complexity:** MEDIUM.
- **Expected result:** +1-3pp

### 9b. SECA: Spiking Efficient Channel Attention

- **Paper:** "Sa-SNN: Spiking Attention Neural Network for Image Classification" (PeerJ, 2024)
- **Key idea:** Local cross-channel interaction via 1D convolution without dimensionality reduction.
- **Applicability:** MEDIUM. Lightweight addition to our conv layers.
- **Implementation complexity:** LOW-MEDIUM.

---

## 10. Advanced Surrogate Gradients

### 10a. Gygax & Zenke: Theoretical Underpinnings

- **Paper:** Gygax & Zenke, "Elucidating the Theoretical Underpinnings of Surrogate Gradient Learning" (Neural Computation 37(5):886-925, 2025)
- **Key idea:** Proves that surrogate gradient = derivative of escape noise function for stochastic neurons. Spike Rate Escape (SRE) is theoretically grounded because it matches the stochastic firing model.
- **Applicability:** Confirms our choice of SRE is theoretically optimal. Not an accuracy improvement per se, but validates our approach.
- **Implementation complexity:** N/A (theoretical).

### 10b. AdaLi: Adaptive Lightweight Surrogate Gradients

- **Paper:** "Adaptive and lightweight surrogate gradients: enhancing training efficiency of spiking neural networks" (Frontiers in Neuroscience, 2026)
- **Key idea:** Lightweight surrogate that reduces computational complexity. Adaptive mechanism adjusts gradient based on training epoch. Resolves gradient vanishing/explosion.
- **Accuracy:** Outperforms baselines in both efficiency and accuracy.
- **Applicability:** MEDIUM. Would need custom surrogate gradient implementation.
- **Implementation complexity:** MEDIUM.

### 10c. Learnable Surrogate Gradients

- **Paper:** "Learnable Surrogate Gradient for Direct Training Spiking Neural Networks" (IJCAI 2023)
- **Key idea:** Make surrogate gradient function parameters learnable alongside network weights.
- **Applicability:** MEDIUM. snnTorch may not directly support this -- would need custom implementation.
- **Implementation complexity:** MEDIUM.

---

## 11. Second-Order / Synaptic Neuron Models

### 11a. Synaptic Neuron with Learnable Alpha

- **Paper:** snnTorch built-in (Synaptic neuron model)
- **Key idea:** 2nd-order LIF with separate synaptic current and membrane potential decay rates (alpha and beta). More biologically realistic. snnTorch supports `learn_alpha=True`.
- **Accuracy:** May not consistently outperform Leaky on simple tasks; alpha tends toward 0 on simple datasets but may help with complex temporal patterns in audio.
- **Applicability:** HIGH for audio. Audio spectrograms have temporal structure that could benefit from synaptic dynamics.
- **Implementation complexity:** LOW. Replace `snn.Leaky` with `snn.Synaptic`.
  ```python
  self.lif1 = snn.Synaptic(alpha=0.9, beta=0.95, spike_grad=spike_grad,
                            learn_alpha=True, learn_beta=True)
  ```
- **Expected result:** +0-2pp (uncertain, worth testing)

---

## 12. Spiking Transformers / Attention Architectures

### 12a. Spikformer V2

- **Paper:** "Spikformer V2: Join the High Accuracy Club on ImageNet with an SNN Ticket" (2024, arXiv:2401.02020)
- **Key idea:** Spiking Self-Attention (SSA) with sparse spike-form Q, K, V. No softmax. Self-supervised pretraining.
- **Accuracy:** ImageNet: 81.10% (1 timestep after SSL). CIFAR-100: ~80%.
- **Applicability:** LOW for our case. These are vision transformers requiring ViT architecture. Our 622K param CNN is too small for transformer overhead.
- **Implementation complexity:** HIGH. Complete architecture redesign.

### 12b. QKFormer

- **Paper:** "QKFormer: Hierarchical Spiking Transformer using Q-K Attention" (2024, arXiv:2403.16552)
- **Key idea:** Linear complexity Q-K attention with binary values. 85.65% ImageNet.
- **Applicability:** LOW. Same as Spikformer -- requires transformer architecture.
- **Implementation complexity:** HIGH.

---

## 13. Self-Supervised / Contrastive Pre-training for SNNs

### 13a. SpikeCLR

- **Paper:** "SpikeCLR: Contrastive Self-Supervised Learning for Few-Shot Event-Based Vision" (2026, arXiv:2603.16338)
- **Key idea:** Contrastive self-supervised framework for SNNs. Learn visual representations from unlabeled event data. Outperforms supervised in low-data regimes.
- **Applicability:** MEDIUM. ESC-50 has only 2000 samples -- could benefit from self-supervised pre-training on AudioSet or other large audio corpora.
- **Implementation complexity:** HIGH. Need to adapt from vision to audio, implement contrastive pipeline.

### 13b. NeuroMoCo

- **Paper:** "Self-Supervised Contrastive Learning In Spiking Neural Networks" (MVIP 2024)
- **Key idea:** Momentum contrastive learning for SNNs with MixInfoNCE loss tailored to temporal characteristics.
- **Applicability:** MEDIUM. Same potential benefit for small-data regime.
- **Implementation complexity:** HIGH.

---

## 14. Spiking Audio Front-End

### 14a. Spiking-LEAF

- **Paper:** "Spiking-LEAF: A Learnable Auditory front-end for Spiking Neural Networks" (2024, arXiv:2309.09469)
- **Key idea:** Learnable Gabor filter bank + PCEN + IHC-LIF two-compartment neuron model. Bio-inspired cochlear processing. Captures multi-scale temporal dynamics.
- **Accuracy:** Outperforms SOTA auditory front-ends on keyword spotting and speaker ID in accuracy, noise robustness, and encoding efficiency.
- **Applicability:** MEDIUM-HIGH. Could replace our mel-spectrogram front-end with a learned spiking front-end. More biologically plausible. May improve encoding quality.
- **Implementation complexity:** HIGH. Complete front-end redesign. No public code found.

---

## 15. Neuromorphic Data Augmentation

### 15a. EventMix

- **Paper:** "EventMix: An Efficient Augmentation Strategy for Event-Based Data" (Information Sciences, 2023)
- **Key idea:** CutMix extended to 3D (spatiotemporal) for event/spike data. Gaussian Mixture Model generates random 3D masks for arbitrary-shape mixing.
- **Accuracy:** Improved classification on neuromorphic datasets for both ANNs and SNNs.
- **Applicability:** MEDIUM. Could adapt for spike train augmentation.
- **Implementation complexity:** MEDIUM.

### 15b. NDA: Neuromorphic Data Augmentation

- **Paper:** "Neuromorphic Data Augmentation for Training Spiking Neural Networks" (ECCV 2022, arXiv:2203.06145)
- **Key idea:** Geometric augmentations designed for event-based data. Stabilizes SNN training, reduces generalization gap.
- **Applicability:** LOW-MEDIUM. Our input is mel-spectrograms, not raw events.
- **Implementation complexity:** MEDIUM.

**NOTE:** Our standard SpecAugment augmentation FAILED (Decision #43: accuracy dropped from 47.15% to 40.75%). SNN-specific augmentation in spike domain might work better since it preserves temporal spike structure.

---

## 16. Regularization and Sparsity

### 16a. Spike Budgeting / L1 Regularization

- **Paper:** "High-performance deep spiking neural networks with 0.3 spikes per neuron" (Nature Communications, 2024)
- **Key idea:** L1 regularization on spike rates pushes sparsity below 0.3 spikes/neuron while maintaining accuracy. Our SNN has 25.8% spike rate (73.6% sparsity).
- **Accuracy:** CIFAR-10 maintained with <0.3 spikes/neuron.
- **Applicability:** MEDIUM. May improve generalization via regularization effect.
- **Implementation complexity:** LOW. Add L1 penalty on spike outputs.

### 16b. Dropout for SNNs

- **Paper:** Various
- **Key idea:** Standard dropout (0.2-0.5) after spiking layers. Our ANN uses Dropout(0.3) but our SNN does NOT use any dropout.
- **Applicability:** VERY HIGH. Our SNN has NO dropout -- adding it could help with overfitting on small ESC-50 dataset.
- **Implementation complexity:** TRIVIALLY LOW. Add `nn.Dropout(0.3)` before fc2.
- **Expected result:** +1-3pp (addressing overfitting on 1600 training samples)

---

## 17. Multiscale Temporal Dynamics

### 17a. TS-LIF: Temporal Segment LIF

- **Paper:** "TS-LIF: A Temporal Segment Spiking Neuron Network for Time Series Forecasting" (2025, arXiv:2503.05108)
- **Key idea:** Dual-compartment neuron where dendritic and somatic compartments capture different frequency components. Provides functional heterogeneity for multi-scale temporal processing.
- **Applicability:** MEDIUM. Audio has multi-scale temporal structure that could benefit.
- **Implementation complexity:** HIGH. Custom neuron model.

### 17b. Temporal Dendritic Heterogeneity

- **Paper:** "Temporal dendritic heterogeneity incorporated with spiking neural networks for learning multi-timescale dynamics" (Nature Communications, 2024)
- **Key idea:** Multi-compartment neuron with heterogeneous timing factors on different dendritic branches. Automatically learns multi-timescale dynamics.
- **Applicability:** MEDIUM. Audio classification benefits from multi-timescale processing.
- **Implementation complexity:** HIGH.

---

## 18. Integer-Valued Training

- **Paper:** "Integer-Valued Training and Spike-Driven Inference Spiking Neural Network" (ECCV 2024, Best Paper Candidate)
- **Key idea:** I-LIF neuron activates integer values during training while maintaining spike-driven inference. Reduces quantization errors. SpikeYOLO architecture.
- **Accuracy:** COCO: 66.2% mAP@50 (+15pp over prior SNN SOTA). Gen1: 67.2% (beats ANN equivalent).
- **Applicability:** MEDIUM. Primarily for object detection but integer-valued training concept is general.
- **Implementation complexity:** MEDIUM-HIGH.

---

## 19. Curriculum Learning for SNNs

- **Paper:** "Curriculum Design Helps Spiking Neural Networks to Classify Time Series" (2024, arXiv:2401.10257)
- **Key idea:** Change neuron activation state during training, progressively training from easy to hard. Improves accuracy and convergence.
- **Applicability:** MEDIUM. Could train on ESC-10 first (easy), then fine-tune on ESC-50 (hard).
- **Implementation complexity:** LOW.

---

## PRIORITY-RANKED IMPLEMENTATION PLAN

Based on impact vs effort, here is the recommended implementation order:

### Immediate Wins (< 1 hour each, combinable):

| # | Technique | Change | Expected Gain |
|---|-----------|--------|---------------|
| 1 | **Learnable beta** | `learn_beta=True` in all 4 LIF neurons | +1-2pp |
| 2 | **Learnable threshold** | `learn_threshold=True` in all 4 LIF neurons | +0.5-2pp |
| 3 | **Dropout in SNN** | Add `nn.Dropout(0.3)` before fc2 | +1-3pp |
| 4 | **Use SRE surrogate** | Already done (spike_rate_escape is best) | +0pp (already optimal) |

### Short-Term Wins (1-4 hours each):

| # | Technique | Change | Expected Gain |
|---|-----------|--------|---------------|
| 5 | **TET loss** | Modify loss with gradient re-weighting | +2-5pp |
| 6 | **Per-neuron learnable beta** | Beta as vector, not scalar | +0.5-1pp |
| 7 | **Synaptic neuron** | Replace Leaky with Synaptic (2nd order) | +0-2pp |
| 8 | **Spike L1 regularization** | Add L1 penalty on spike outputs | +0.5-1pp |

### Medium-Term (1-2 days each):

| # | Technique | Change | Expected Gain |
|---|-----------|--------|---------------|
| 9 | **Knowledge distillation** | ANN teacher -> SNN student (SAKD) | +3-11pp |
| 10 | **Hybrid training** | Convert ANN weights, fine-tune as SNN | +5-15pp |
| 11 | **ANN-to-SNN conversion** | PASCAL or threshold calibration | +8-15pp |
| 12 | **TEBN / BNTT** | Replace BN with temporal BN | +2-5pp |
| 13 | **Skip connections** | SEW-style residual in conv blocks | +2-4pp |
| 14 | **Ternary spikes** | {-1, 0, +1} instead of {0, 1} | +2-5pp |
| 15 | **Channel attention** | TCJA/SECA after conv layers | +1-3pp |

### Longer-Term (3+ days):

| # | Technique | Change | Expected Gain |
|---|-----------|--------|---------------|
| 16 | **GLIF neuron** | Custom gated LIF implementation | +2-4pp |
| 17 | **Self-supervised pre-training** | Contrastive learning on audio | +3-5pp |
| 18 | **Spiking-LEAF front-end** | Replace mel-spectrogram | Uncertain |
| 19 | **NAS** | Automated architecture search | +2-5pp |

---

## MOST PROMISING SINGLE TECHNIQUE

**Hybrid Training (ANN init + SNN fine-tune)** is the single most impactful technique. Rationale:

1. We ALREADY have a trained ANN at 63.85%
2. Our ANN and SNN share IDENTICAL architecture (same Conv layers, same FC layers)
3. Weight transfer is straightforward (just load ANN weights, add LIF neurons)
4. Fine-tuning with surrogate gradients starting from good weights should converge much faster
5. Expected accuracy: 55-62% (vs current 47.15%), representing a 8-15pp improvement

The implementation would be:
1. Load trained ANN weights into SNN (Conv, BN, FC weights are identical)
2. Initialize LIF membrane potentials to 0
3. Set thresholds based on ANN activation statistics (data-driven initialization)
4. Fine-tune with spike_rate_escape surrogate gradient for 10-20 epochs
5. Use TET loss for optimal convergence

This combination of ANN initialization + data-driven thresholds + TET loss + learnable neuron parameters could potentially reach 58-63% accuracy, nearly closing the gap entirely.

---

## COMBINATION STRATEGY

The most effective approach combines multiple complementary techniques:

**Phase 1: Quick wins on current architecture (30 minutes)**
- Add `learn_beta=True`, `learn_threshold=True` to all LIF neurons
- Add `nn.Dropout(0.3)` before fc2 in SNN
- Expected: 47.15% -> ~50-52%

**Phase 2: TET loss (2 hours)**
- Implement TET gradient re-weighting
- Expected: 50-52% -> ~53-55%

**Phase 3: ANN-to-SNN hybrid (1 day)**
- Load ANN weights into SNN
- Data-driven threshold initialization
- Fine-tune with all Phase 1+2 improvements
- Expected: ~58-63%

**Phase 4: Knowledge distillation (1 day)**
- SAKD with ANN teacher
- Expected: potentially match or exceed ANN (63-65%)

---

## Research Gaps and Uncertainties

1. **No SNN benchmarks on full ESC-50** -- we are literally the first (confirmed by Larroza et al. 2025 which only did ESC-10)
2. **Audio-specific SNN augmentation** is underexplored -- EventMix/NDA are for vision
3. **Small dataset regime** (2000 samples) is particularly challenging for SNNs -- most SNN papers benchmark on CIFAR/ImageNet with 50K+ samples
4. **Membrane potential vs spike count readout** -- our current approach (sum of membrane potentials) is already the recommended approach; spike count readout would likely be worse
5. **Interaction effects** between techniques are unknown -- combining learnable params + TET + KD might have diminishing returns or synergies

---

## Key References

1. Gygax & Zenke (2025). Neural Computation 37(5):886-925. Surrogate gradient theory.
2. Deng et al. (2022). ICLR 2022. TET loss.
3. Fang et al. (2021). ICCV 2021. PLIF learnable time constant.
4. Fang et al. (2021). NeurIPS 2021. SEW ResNet.
5. Yao et al. (2022). NeurIPS 2022. GLIF gated neuron.
6. Xu et al. (2024). Neural Networks 178. SAKD knowledge distillation.
7. Xu et al. (2023). CVPR 2023. ANN-SNN KD construction.
8. Rathi et al. (2020). ICLR 2020. Hybrid SNN conversion.
9. Bojkovic et al. (2024). AISTATS 2024. Data-driven threshold init.
10. PASCAL (2025). arXiv:2505.01730. Near-lossless conversion.
11. Guo et al. (2024). AAAI 2024. Ternary spikes.
12. Three-stage hybrid (2025). Frontiers in Neuroscience. Audio SNN fine-tuning.
13. TET (2022). ICLR 2022. Temporal efficient training.
14. TEBN (2022). NeurIPS 2022. Temporal effective batch normalization.
15. TCJA-SNN (2024). IEEE TNNLS. Temporal-channel joint attention.
16. Spiking-LEAF (2024). arXiv:2309.09469. Learnable audio front-end.
17. SpikeCLR (2026). arXiv:2603.16338. Contrastive self-supervised SNN.
18. SpikeYOLO (2024). ECCV 2024. Integer-valued training.
19. AdaLi (2026). Frontiers in Neuroscience. Adaptive surrogate gradients.
20. MD-SNN (2024). arXiv:2512.04443. Membrane potential distillation.
