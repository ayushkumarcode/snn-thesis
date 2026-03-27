# accuracy improvement techniques -- research notes

date: 2026-03-25
context: ESC-50 conv SNN (622K params), 47.15% SNN vs 63.85% ANN -- trying to close that 16.7pp gap

went through ~40 recent papers (2022-2026). techniques fall into three rough tiers:

tier 1 (likely +5-15pp): ANN-to-SNN conversion, knowledge distillation, hybrid ANN init + SNN fine-tune, TET loss, learnable neuron params (PLIF/GLIF)

tier 2 (likely +2-5pp): temporal batch normalization, ternary spikes, residual/skip connections, channel/temporal attention, advanced surrogate gradients

tier 3 (uncertain, higher effort): NAS, self-supervised pretraining, Spiking-LEAF auditory front-end, EventMix augmentation, QAT

---

## 1. ANN-to-SNN conversion

### 1a. PASCAL (2025, arXiv:2505.01730)

"Precise ANN-SNN Conversion via Spike Accumulation and Adaptive Layerwise Activation." augments IF neuron with spike accumulation/inhibition so the integrate-fire process EXACTLY reproduces ANN activation statistics. uses QCFS activation and per-layer optimal timesteps. supports negative spikes.

near-lossless: ResNet-34 gets ~74% on ImageNet matching source ANN, with 64x fewer timesteps than prior conversion. directly applicable since we have a trained ANN at 63.85%. standard architecture (Conv+BN+MaxPool+FC) is compatible. question is whether T=25 is enough or if we need adaptive layerwise timesteps.

implementation: retrain ANN with QCFS activation (replace ReLU), apply conversion, optionally fine-tune. code at https://github.com/BrainSeek-Lab/PASCAL

expected: 55-62% accuracy (preserving 85-97% of ANN)

### 1b. data-driven threshold and potential initialization

Bojkovic et al., AISTATS 2024. after conversion, optimally initialize thresholds and membrane potentials from ANN activation distributions. conversion loss typically 1-3%. direct drop-in after any conversion method. code: https://github.com/srinuvaasu/data_driven_init

expected: additional +1-3pp on top of any conversion

### 1c. one-timestep conversion (scale-and-fire)

arXiv:2510.23383. single-timestep inference via Scale-and-Fire Neuron. best accuracy among binary/ternary/multi-level SNNs at T=1. medium applicability for us though.

### 1d. negative spikes for conversion (IJCAI 2025)

allows negative spikes during conversion to better approximate ReLU/LeakyReLU. outperforms two-stage by 1.29% at T=4.

---

## 2. knowledge distillation (ANN teacher -> SNN student)

### 2a. SAKD (Neural Networks 2024)

Xu et al. "Self-architectural knowledge distillation for spiking neural networks." same-architecture ANN as teacher, directly transfer weights then distill intermediate features + logits.

ResNet-18 on CIFAR-100: 80.48% with 4 timesteps, SURPASSING the ANN (79.90%). +3.49% over non-distilled. on ImageNet: SEW-ResNet152 gets 77.30% (SNN SOTA).

this is really interesting for us because our ANN and SNN share identical architecture. the ANN teacher is already trained at 63.85%. implementation: load ANN teacher weights, modify training loop for feature-level + logit-level KD loss.

expected: 50-58% (could potentially match ANN with perfect distillation)

### 2b. ANN-SNN KD construction (CVPR 2023)

Xu, Li et al. joint ANN-SNN training with KD. SNN learns from ANN teacher through KD, avoiding training from scratch. also showed improved noise robustness (relevant for ESC-50 audio).

### 2c. MD-SNN (arXiv:2512.04443)

distills using MEMBRANE POTENTIALS not just spikes/logits. +0.48-1.06% over quantized baselines. 30% training FLOPs reduction. primarily for quantized SNNs but the membrane distillation idea is broadly useful.

### 2d. enhanced self-distillation (arXiv:2510.06254)

SNN distills from its own later timesteps to earlier ones. no external teacher needed.

---

## 3. hybrid training (ANN init + SNN fine-tune)

### 3a. Rathi et al. (ICLR 2020)

fast ANN-to-SNN conversion for weight init, then fine-tune with spike-timing dependent backprop. similar accuracy to full conversion with MUCH fewer timesteps and faster convergence than from scratch.

this is probably the simplest high-impact change: (1) convert our 63.85% ANN, (2) fine-tune as SNN for a few epochs. code: https://github.com/nitin-rathi/hybrid-snn-conversion

expected: 55-62%

### 3b. three-stage hybrid for audio (Frontiers in Neuroscience, April 2025)

this one is specifically for audio SNNs. three stages: (1) train ANN, (2) convert to SNN, (3) hybrid fine-tune where forward uses spikes but backward uses ANN signals. applied to Wave-U-Net and ConvTasNet for audio processing.

directly applicable to our pipeline. expected: preserve most of ANN's 63.85%.

---

## 4. loss function improvements

### 4a. TET: temporal efficient training (ICLR 2022)

Deng et al. instead of accumulating spikes then one loss, optimize every timestep's pre-synaptic inputs independently. compensates for momentum loss in surrogate gradient descent, converges to flatter minima.

+10pp on DVS-CIFAR10 (83% vs 73% prior SOTA). consistent improvements elsewhere. code: https://github.com/Gus-Lab/temporal_efficient_training

our current training already does per-timestep CE loss (line 71 of train.py: `loss += criterion(mem_out[step], targets)`). TET's addition is gradient re-weighting to avoid bad local minima plus a regularization term. this is probably our single easiest win. ~20 lines of code change.

expected: +2-5pp (47% -> 49-52%)

### 4b. temporal regularization training (arXiv:2506.19256)

regularizes temporal consistency across timesteps. complementary to TET.

---

## 5. learnable neuron parameters

### 5a. PLIF -- learnable membrane time constant (ICCV 2021)

Fang et al. make tau (controls beta) learnable per-layer. CIFAR-10: 93.50%. robust to initialization.

snnTorch already supports this! just set `learn_beta=True`:
```python
self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)
```

literally 4 line changes. expected: +1-2pp.

### 5b. learnable threshold

various papers (Diet-SNN, LTR-LIF). snnTorch supports `learn_threshold=True`:
```python
self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_threshold=True, threshold=1.0)
```

single flag change. expected: +0.5-2pp.

### 5c. GLIF -- gated LIF (NeurIPS 2022)

Yao et al. fuses multiple bio-features with learnable gating factors. ResNet-19 on CIFAR-100: 77.35% (+2.92% over Dspike). would need custom neuron replacing snnTorch's Leaky. code: https://github.com/Ikarosy/Gated-LIF

expected: +2-4pp but more work.

### 5d. per-neuron learnable parameters

instead of per-layer beta, initialize as a vector (one per neuron):
```python
beta_init = torch.ones(256) * 0.95
self.lif3 = snn.Leaky(beta=beta_init, spike_grad=spike_grad, learn_beta=True)
```

natively supported. expected: +0.5-1pp on top of per-layer.

---

## 6. temporal batch normalization

### 6a. TEBN (NeurIPS 2022)

"Temporal Effective Batch Normalization." different learnable weights per timestep for rescaling presynaptic inputs. smooths temporal distributions, stabilizes gradient norm. CIFAR-100: 74.37% (vs 66.6% BNTT = +7.77pp). DVS-CIFAR10: 75.1% (vs 60.5% NeuNorm = +14.6pp).

would replace our standard BN. compatible with Conv-BN-MaxPool-LIF pattern. not in snnTorch by default (SpikingJelly has it). expected: +2-5pp.

### 6b. tdBN (Zheng et al. 2021)

extends BN to temporal dimension, incorporates threshold. can be folded into weights for zero inference overhead.

### 6c. BNTT (Frontiers 2021)

Panda et al. separate BN params for each timestep. learns time-varying distributions. code: https://github.com/Intelligent-Computing-Lab-Panda/BNTT-Batch-Normalization-Through-Time

matches our 25 timestep setup. expected: improvement similar to TEBN.

---

## 7. ternary spikes

Guo et al. AAAI 2024. {-1, 0, +1} instead of {0, 1}. dramatically increases info capacity while keeping event-driven and multiply-free. CIFAR-10: 95.60%, CIFAR-100: +7% over binary at T=2.

not in snnTorch natively, needs custom implementation. code: https://github.com/yfguo91/Ternary-Spike. expected: +2-5pp (especially for our information-bottleneck issue).

---

## 8. residual/skip connections

### SEW ResNet (NeurIPS 2021)

Fang et al. element-wise addition of spike tensors for skip connections. enables identity mapping, first >100 layer SNN training. DVS Gesture: 97.92% (vs 90.97% standard = +6.95pp). code: https://github.com/fangwei123456/Spike-Element-Wise-ResNet

our current architecture has NO skip connections. adding them would help gradient flow through our 4 LIF layers. expected: +2-4pp.

---

## 9. attention mechanisms

### 9a. TCJA-SNN (IEEE TNNLS 2024)

temporal-channel joint attention. squeeze-and-excitation adapted for SNNs. up to +15.7% on neuromorphic datasets. add after conv layers, small param overhead. expected: +1-3pp.

### 9b. SECA (PeerJ 2024)

local cross-channel interaction via 1D conv. lightweight. expected: modest improvement.

---

## 10. advanced surrogate gradients

### 10a. Gygax & Zenke (Neural Computation 2025)

proves surrogate gradient = derivative of escape noise function. confirms our SRE choice is theoretically optimal. not an accuracy improvement per se but validates our approach.

### 10b. AdaLi (Frontiers in Neuroscience 2026)

adaptive lightweight surrogate. reduces complexity, resolves gradient vanishing/explosion.

### 10c. learnable surrogates (IJCAI 2023)

make surrogate function parameters learnable. would need custom implementation.

---

## 11. second-order neuron models

### synaptic neuron with learnable alpha

snnTorch built-in. 2nd-order LIF with separate synaptic and membrane decay (alpha and beta). more biologically realistic. audio spectrograms have temporal structure that could benefit from synaptic dynamics.

```python
self.lif1 = snn.Synaptic(alpha=0.9, beta=0.95, spike_grad=spike_grad,
                          learn_alpha=True, learn_beta=True)
```

low effort. expected: +0-2pp (uncertain but worth testing for audio).

---

## 12. spiking transformers

Spikformer V2 (2024) gets 81.10% ImageNet with 1 timestep. QKFormer gets 85.65%. but these require ViT architecture -- our 622K CNN is too small for transformer overhead. not applicable.

---

## 13. self-supervised pretraining

SpikeCLR (2026, arXiv:2603.16338) -- contrastive framework. ESC-50 has only 2000 samples so could benefit from pretraining on AudioSet. but high implementation complexity (adapt from vision to audio).

NeuroMoCo (MVIP 2024) -- momentum contrastive with MixInfoNCE loss. same potential benefit for small data.

---

## 14. spiking audio front-end

Spiking-LEAF (2024, arXiv:2309.09469) -- learnable Gabor filter bank + PCEN + IHC-LIF. outperforms SOTA auditory front-ends on KWS and speaker ID. could replace our mel-spectrogram front-end. but high complexity and no public code.

---

## 15. neuromorphic augmentation

EventMix (Information Sciences 2023) -- CutMix extended to 3D spatiotemporal. could adapt for spike trains.

NDA (ECCV 2022) -- geometric augmentations for event data. our input is mel spectrograms though, not raw events.

note: our standard SpecAugment augmentation FAILED (accuracy dropped from 47.15% to 40.75%). SNN-specific augmentation in spike domain might work better since it preserves temporal structure.

---

## 16. regularization and sparsity

"0.3 spikes per neuron" (Nature Comms 2024) -- L1 reg pushes sparsity below 0.3 spikes/neuron while maintaining accuracy. our SNN has 25.8% rate. may improve generalization.

dropout for SNNs -- our ANN uses Dropout(0.3) but our SNN has NO dropout. adding `nn.Dropout(0.3)` before fc2 could help with overfitting on 1600 training samples. expected: +1-3pp. trivially low effort.

---

## 17. multiscale temporal dynamics

TS-LIF (arXiv:2503.05108) -- dual-compartment capturing different frequency components. audio has multi-scale temporal structure.

temporal dendritic heterogeneity (Nature Comms 2024) -- multi-compartment with heterogeneous timing. automatically learns multi-timescale dynamics.

both high implementation complexity.

---

## 18. integer-valued training

I-LIF neuron (ECCV 2024, best paper candidate) -- activates integer values during training, maintains spike-driven inference. SpikeYOLO: 66.2% mAP on COCO (+15pp over prior SNN SOTA). primarily for detection but concept is general.

---

## 19. curriculum learning

"Curriculum Design Helps SNNs Classify Time Series" (2024, arXiv:2401.10257). progressively train easy to hard. could train on ESC-10 first, fine-tune on ESC-50. low complexity.

---

## implementation priority

### immediate wins (< 1 hour each, combinable):
1. learnable beta: `learn_beta=True` in all 4 LIF neurons -- +1-2pp
2. learnable threshold: `learn_threshold=True` -- +0.5-2pp
3. dropout in SNN: `nn.Dropout(0.3)` before fc2 -- +1-3pp
4. SRE surrogate: already done, we're optimal

### short-term (1-4 hours):
5. TET loss: gradient re-weighting -- +2-5pp
6. per-neuron learnable beta: vector not scalar -- +0.5-1pp
7. synaptic neuron: replace Leaky with Synaptic -- +0-2pp
8. spike L1 regularization -- +0.5-1pp

### medium-term (1-2 days):
9. knowledge distillation (SAKD): ANN teacher -> SNN student -- +3-11pp
10. hybrid training: convert ANN weights, fine-tune as SNN -- +5-15pp
11. ANN-to-SNN conversion (PASCAL) -- +8-15pp
12. TEBN / BNTT: replace BN -- +2-5pp
13. skip connections (SEW-style) -- +2-4pp
14. ternary spikes -- +2-5pp
15. channel attention (TCJA/SECA) -- +1-3pp

### longer-term (3+ days):
16. GLIF neuron -- +2-4pp
17. self-supervised pretraining -- +3-5pp
18. Spiking-LEAF front-end -- uncertain
19. NAS -- +2-5pp

---

## most promising single technique

hybrid training (ANN init + SNN fine-tune). rationale:

1. we ALREADY have a trained ANN at 63.85%
2. ANN and SNN share IDENTICAL architecture
3. weight transfer is straightforward (load weights, add LIF neurons)
4. fine-tuning from good weights should converge much faster
5. expected: 55-62% (8-15pp improvement)

the implementation:
1. load trained ANN weights into SNN
2. initialize LIF membranes to 0
3. set thresholds from ANN activation statistics (data-driven init)
4. fine-tune with SRE surrogate for 10-20 epochs
5. use TET loss for optimal convergence

combining ANN init + data-driven thresholds + TET loss + learnable params could potentially reach 58-63%, nearly closing the gap entirely.

---

## combination strategy

phase 1 (quick wins, 30 min):
- learn_beta=True, learn_threshold=True on all LIF neurons
- nn.Dropout(0.3) before fc2
- expected: 47.15% -> ~50-52%

phase 2 (TET loss, 2 hours):
- gradient re-weighting implementation
- expected: 50-52% -> ~53-55%

phase 3 (hybrid, 1 day):
- load ANN weights, data-driven threshold init, fine-tune with phase 1+2
- expected: ~58-63%

phase 4 (KD, 1 day):
- SAKD with ANN teacher
- potentially match or exceed ANN (63-65%)

---

## research gaps

1. no SNN benchmarks on full ESC-50 -- we're literally the first
2. audio-specific SNN augmentation is underexplored (EventMix/NDA are vision)
3. small dataset regime (2000 samples) is particularly hard for SNNs -- most papers use CIFAR/ImageNet with 50K+
4. membrane potential vs spike count readout -- our approach (sum of membranes) is already reccomended
5. interaction effects between techniques are unknown -- combining everything might have diminishing returns

---

## references

1. Gygax & Zenke (2025). Neural Computation 37(5):886-925.
2. Deng et al. (2022). ICLR 2022. TET.
3. Fang et al. (2021). ICCV 2021. PLIF.
4. Fang et al. (2021). NeurIPS 2021. SEW ResNet.
5. Yao et al. (2022). NeurIPS 2022. GLIF.
6. Xu et al. (2024). Neural Networks 178. SAKD.
7. Xu et al. (2023). CVPR 2023. ANN-SNN KD.
8. Rathi et al. (2020). ICLR 2020. Hybrid conversion.
9. Bojkovic et al. (2024). AISTATS 2024. Data-driven init.
10. PASCAL (2025). arXiv:2505.01730.
11. Guo et al. (2024). AAAI 2024. Ternary spikes.
12. Three-stage hybrid (2025). Frontiers in Neuroscience.
13. TET (2022). ICLR 2022.
14. TEBN (2022). NeurIPS 2022.
15. TCJA-SNN (2024). IEEE TNNLS.
16. Spiking-LEAF (2024). arXiv:2309.09469.
17. SpikeCLR (2026). arXiv:2603.16338.
18. SpikeYOLO (2024). ECCV 2024.
19. AdaLi (2026). Frontiers in Neuroscience.
20. MD-SNN (2024). arXiv:2512.04443.
