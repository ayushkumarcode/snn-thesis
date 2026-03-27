# deep research notes: SNN inference energy reduction

date: 27 march 2026
context: ESC-50 SNN (622K params, T=25, 968 nJ/sample, spike rate 26.4%, best accuracy 61.65% with dendritic+delays)

---

went through 50+ papers on this. the three most impactful findings:

1. SPARQ (March 2026) gets 330x energy reduction by combining early exit + INT8 quantization + SNN sparsity. on AlexNet/CIFAR-10: 888 mJ down to 2.68 mJ. the key insight is techniques are MULTIPLICATIVE -- combining 3 orthogonal strategies gives 330x, not 3+3+3.

2. multi-level neurons (Oct 2025) enable T=1 inference at competitive accuracy: CIFAR-100 at 73.75% with N=4 levels vs ~70% for binary T=5. energy drops 2-3x vs binary AND timesteps drop from 5 to 1 (another 5x). combined ~10-15x.

3. unstructured pruning (ICLR 2024) achieves 91x energy efficiency with only 0.63% remaining connections and 2.19% accuracy loss. our own pruning data shows 93.2% accuracy retained at 90% pruning -- we're extremely pruning-resilient.

projected combined strategy:
- spike rate reg (26.4% to 6%): 4.4x
- adaptive early exit (T=25 to avg T=7): 3.6x
- weight pruning (to 70% sparsity): 2.5x
- combined: ~40x reduction to ~24 nJ/sample

---

## 1. early exit / adaptive timestep

### 1.1 SEENN (NeurIPS 2023) -- the foundation paper

Li et al. from Yale/Panda Lab. code at https://github.com/Intelligent-Computing-Lab-Yale/SEENN

SEENN-II with ResNet-19 gets 96.1% accuracy with avg 1.08 timesteps on CIFAR-10 (baseline T=4). two variants: SEENN-I does confidence thresholding (simpler), SEENN-II uses RL with auxiliary network. 4x fewer timesteps at same accuracy.

how it works: SEENN-I computes max(softmax(accumulated_membrane_potential)) at each timestep, exits if above threshold. SEENN-II trains an RL agent jointly with the SNN to learn per-sample optimal stopping. compatible with both directly-trained and converted models.

SpiNNaker 1 compatibility: HIGH. processes timestep-by-timestep natively, confidence check runs on ARM core between timesteps.

### 1.2 SPARQ (arXiv:2603.14380, March 2026) -- the combined champion

this is the big one. 330x lower system energy (AlexNet CIFAR-10: 888 mJ to 2.68 mJ). 90x fewer synaptic ops. 5.15% higher accuracy than quantized SNNs.

breakdown of how the 330x is achieved (multiplicative):
1. SNN sparsity: ~5x (binary spikes, only active neurons compute)
2. INT8 quantization: ~4x (8-bit vs 32-bit, 0.03pJ per AC vs 0.9pJ)
3. early exit (RL-guided): ~16x (most samples exit at layer 1 or 2)
4. combined: 5 * 4 * 16 = ~320x

RL exit mechanism: state = (exit index, discretized max softmax confidence), actions = {exit_now, continue}, reward = +1 + alpha*savings if correct, -1 if wrong. Q-learning with eta=0.1, gamma=0.9, alpha=0.3.

energy model (45nm CMOS): E_spike = N_AC * 0.03 pJ (8-bit), E_LIF = N_neurons * T * 1.0 pJ, E_mem = 80 pJ/byte.

SpiNNaker 1 compatibility: MEDIUM. early exit works, INT8 possible with custom synapse models (native is 16-bit fixed), RL agent runs on ARM core. implementation is high complexity though.

### 1.3 CPT-SNN (Neurocomputing 2025)

95.44% CIFAR-10 with avg 2.72 timesteps. CPT-Block transforms previous timestep output into inhibitory current for current timestep, creating temporal connections that accelerate convergence. dynamic confidence with adaptive per-timestep thresholds.

SpiNNaker 1 compatibility: HIGH -- inhibitory current feedback is natively supported.

### 1.4 SpikeCP (IEEE JSTSP 2025) -- zero retraining

Cohen et al. wraps ANY pretrained SNN with conformal prediction. calibrate on held-out set, at inference generate predicted set at each checkpoint, stop when predicted set is small enough (1-2 labels typically). formal guarantee: Pr(true_label in predicted_set) >= p_target.

this is the LOWEST-EFFORT approach -- zero retraining, just calibration. SpiNNaker compatibility excellent (thresholding logic on ARM core).

### 1.5 ED-sKWS (arXiv:2406.12726) -- the ONLY early exit paper for audio SNNs

this is critical for us. 128-neuron model: 2.85 uJ vs 5.36 uJ baseline = 47% energy savings. 512-neuron: 23.68 uJ vs 45.41 uJ = 48% savings. avg 60.46 timesteps of 98 total = 39% timestep reduction. accuracy loss only 0.08%.

uses Cumulative Temporal (CT) loss which differs from TET by ACCUMULATING historical membrane potentials rather than treating timesteps independently. better for audio because of temporal continuity.

architecture is FC-SNN though, not convolutional. datasets: Google Speech Commands v2 (35-class), SC-100 (100-class). directly applicable to ESC-50 but our model is conv-based.

### 1.6 cutoff + RCS (Frontiers 2025)

two techniques: (1) Top-K cutoff stops when predictions stabilize, (2) RCS regularizer trains network to be accurate early. 1.76-2.76x fewer timesteps on CIFAR-10, 1.64-1.95x on event-based.

### confidence metrics comparison

| metric | description | used by | pros | cons |
|--------|-------------|---------|------|------|
| max(softmax) | max softmax prob | SEENN-I, SPARQ | simple | overconfident on OOD |
| softmax entropy | H = -sum(p*log(p)) | various | better calibrated | needs log |
| spike count stability | predictions stable for k steps | Top-K cutoff | no training | needs buffer |
| conformal set size | predicted set <= threshold | SpikeCP | formal guarantees | larger sets |
| RL policy | learned exit | SEENN-II, SPARQ | optimal | needs RL training |

recommendation: start with SpikeCP (zero retraining) or max(softmax) thresholding (simplest), move to RL-based if promising.

---

## 2. spike rate reduction / sparsity enforcement

### 2.1 the 6.4% threshold -- why this matters

Dampfhoffer et al. (IEEE TECI 2023): SNNs with T in [5,10] need spike rate below 6.4% to beat quantized ANNs. Yang et al. (arXiv:2409.08290, 2024): at T=6, neuron sparsity must exceed 93%. we're at 26.4% = 4.1x above break-even. this is THE most critical problem.

### 2.2 energy-aware spike budgeting (arXiv:2602.12236, Feb 2026)

tested on frame-based data: MNIST gets 47% spike reduction (15.31% to 8.07%) while IMPROVING accuracy by 2.31%. CIFAR-10 gets 17.6% reduction with +1.76% accuracy. the insight is spike budgeting acts as regularization.

how it works: proportional feedback controller adjusts lambda_rate in real-time. delta_lambda = eta * (r_spike - r_target). target range for frame data: 5-12%. also makes beta and V_thr learnable per-layer (O(L) params only).

training objective: L_total = L_task + lambda_rate * (r_spike - r_target)^2

directly applicable. our snnTorch loop already has access to per-layer spike rates.

### 2.3 layer-wise regularization

the key observation from our pareto data (fold 1):
- L1=0.0: 42.25% acc, rate=1.49% (output only)
- L1=1e-5: 46.00% acc, rate=2.15% (BEST accuracy!)
- L1=1e-4: 38.50%, rate=0.55%
- L1=1e-3: 45.25%, rate=0.62%

our OUTPUT spike rate is already very low (0.6-2.2%). the 26.4% overall rate comes from HIDDEN layers. so layer-specific reg targeting conv1/conv2 is the right approach:
```
loss = CE_loss + lambda_conv * (spk1.mean() + spk2.mean()) + lambda_fc * spk3.mean()
```

### 2.4 TRT (arXiv:2506.19256, 2025)

temporal regularization training -- time-decaying reg prioritizes early timesteps with stronger constraints. forces SNN to classify quickly = naturally reduces total spike count. +2.96% CIFAR-10, +4.70% CIFAR-100 over standard direct training. would synergize with our rhythm/dendritic models.

### 2.5 what accuracy can we maintain at <5% spike rate?

from the literature: CIFAR-10 at ~5% spike rate still gets ~92-94%. CIFAR-100 at ~5% gets ~70-73%. for our ESC-50 at ~6%: expect ~40-44% (matches our pareto curve where L1=1e-3 gives 45.25% at 0.62% output rate).

moderate spike reduction (26.4% to 6-10%) typically costs <5% accuracy. below 3% it degrades more rapidly.

---

## 3. temporal compression

### 3.1 multi-level SNNs (arXiv:2510.24637, Oct 2025)

Castagnetti, Pegatoquet, Miramond. "All in One Timestep." multi-level neuron generates N*T+1 quantization intervals.

key results (VGG16):
| config | CIFAR-10 | CIFAR-100 | energy |
|--------|----------|-----------|--------|
| binary [T=4, N=1] | 94.40% | 70.18% | 8.5M nJ |
| multi-level [T=1, N=4] | 94.34% | 73.75% | 4.41M nJ |
| ANN | ~95% | ~76% | 12.9M nJ |

T=1 with N=4 levels MATCHES T=4 binary accuracy while using 0.51x the energy. also introduces "barrier neurons" after residual connections to prevent spike avalanche effect (reduces activity 20-30%).

SpiNNaker 1 compatibility: LOW. only supports binary spikes natively. would need rapid bursts to encode multi-level, or SpiNNaker 2 which supports graded spikes.

### 3.2 scale-and-fire neurons (arXiv:2510.23383, Oct 2025)

proves mathematically that multi-timestep integrate-and-fire can be replaced by single-timestep multi-threshold neuron. 88.8% top-1 on ImageNet at T=1 with SFormer (1B params). SpiNNaker 1 compatibility: LOW.

### 3.3 one-hot multi-level LIF (IEEE Access 2025)

one-hot encoding across multiple binary "lanes" -- preserves event-driven operation while increasing info capacity. 2% better than VGG16 SNN, 20x lower energy than VGG16 ANN, 3x latency reduction with <1% accuracy loss.

SpiNNaker 1 compatibility: MEDIUM. multiple parallel spike lanes can be implemented as multiple populations.

### 3.4 what's the minimum T for 50-class problems?

from literature: T=1 binary gives ~7% on our model but ~70% on CIFAR-100 with optimized architectures. T=1 multi-level N=4 gives ~73.75% on CIFAR-100. T=5-6 is the optimal energy/accuracy tradeoff per Yang et al. 2024.

for our ESC-50: our ablation shows T=7 gives 36.5% (90% of T=25). estimated T=5 with TRT training: ~35-37%. estimated T=3 with multi-level N=4: ~30-33%.

---

## 4. quantization

### 4.1 QP-SNN (ICLR 2025)

weight rescaling (ReScaW) for 2-bit + singular value structured pruning (SVS). CIFAR-10: 98.74% model size reduction, 78.69% SOPs reduction, 77.45% power reduction, only 2.44% accuracy decrease. "quantize first, then prune" strategy.

### 4.2 ternary spike (AAAI 2024)

Guo et al. {-1, 0, +1} instead of {0, 1}. CIFAR-10: 95.60% (2 timesteps). outperforms TET and RecDis-SNN by ~7% at T=4 on CIFAR-100. negligible energy overhead. information capacity doubles per spike. SpiNNaker compatibility: HIGH since it supports exc/inh synapses natively.

### 4.3 SpQuant-SNN (Frontiers 2024)

membrane potential compressed to ternary {-1, 0, 1}. 13x memory reduction, >4.7x FLOPs reduction, <1.8% accuracy degradation. uses Stacked Gradient Surrogation for low-precision training.

### 4.4 AGMM-BSNN (AAAI 2025) -- binary weights

weights are {-1, +1}, each synapse is add/subtract. solves the weight sign flipping problem during training with adaptive gradient modulation. SpiNNaker compatibility: HIGH -- add/subtract on ARM is 1 cycle vs multiply 3-5 cycles. 32x less weight memory.

### 4.5 ReverB-SNN (ICML 2025) -- reversed bits

instead of binary spikes + full-precision weights, use real-valued spikes + binary weights. ImageNet ResNet34 T=4: 70.91% (+3.22% over previous SoTA). only 0.52% more energy than vanilla SNN. SpiNNaker compatibility: LOW (real-valued spikes not natively supported).

### 4.6 S2NN (arXiv:2509.24266) -- sub-bit quantization

weights encoded with LESS than 1 bit by sharing binary kernels across groups. outlier-aware quantization + membrane potential feature distillation. SpiNNaker: LOW.

### 4.7 MD-SNN (arXiv:2512.04443) -- membrane potential distillation

14.85x lower energy-delay-area product, 2.64x higher TOPS/W. single T=4 teacher guides students at T=1,2,3,4. 30% training FLOPs reduction.

### 4.8 how much does quantization actually save on neuromorphic hardware?

| operation | 32-bit float | 16-bit fixed | 8-bit int | binary |
|-----------|-------------|-------------|-----------|--------|
| MAC energy (45nm) | 4.6 pJ | ~2.3 pJ | ~1.1 pJ | ~0.1 pJ |
| AC energy (45nm) | 0.9 pJ | ~0.45 pJ | ~0.2 pJ | ~0.03 pJ |
| SpiNNaker 1 native | no | yes (s16.15) | custom | custom |

SpiNNaker 1 uses 16-bit fixed-point natively. going to 8-bit needs custom synapse models but roughly halves memory per weight, allowing 2x more synapses per core.

---

## 5. pruning for extreme sparsity

### 5.1 unstructured pruning (ICLR 2024) -- 91x

Shi et al. code: https://github.com/xyshi2000/Unstructured-Pruning

0.63% remaining connections: 91x energy efficiency, only 2.19% accuracy loss on CIFAR-10. 8.5M SOPs. first application of unstructured NEURON pruning to deep SNNs.

SpiNNaker compatibility: MEDIUM. weight pruning doesnt help SpiNNaker directly (still sends to same core), but neuron pruning = fewer cores.

### 5.2 lottery ticket hypothesis in SNNs (ECCV 2022, oral)

Kim et al. code: https://github.com/Intelligent-Computing-Lab-Panda/Exploring-Lottery-Ticket-Hypothesis-in-SNNs

winning tickets at 97% sparsity. 98.13% sparsity with <2% accuracy drop on SVHN/F-MNIST. Early-Time ticket finds sparse connectivity 38% faster.

our model's extreme pruning resilience (93.2% retained at 90%) strongly suggests winning tickets exist in our network.

### 5.3 NDSNN (DAC 2023) -- neurogenesis dynamics

99% sparsity with up to 20.52% accuracy IMPROVEMENT on Tiny-ImageNet. drop-and-grow acts as regularization, preventing overfitting. our augmented SNN results showed similar patterns.

### 5.4 dynamic spatio-temporal pruning (Frontiers 2025)

code: https://github.com/gzxdu/SNN_Pruning

temporal pruning removes redundant timesteps per-layer. 98.18% parameter reduction on DVS128 with 0.69% accuracy IMPROVEMENT. 50% temporal redundancy reduction.

### 5.5 spiking brain compression (NeurIPS 2025 workshop)

one-shot post-training compression -- no retraining! extends Optimal Brain Surgeon to SNNs using Van Rossum Distance. compression time 2-3 orders of magnitude faster than iterative methods.

this matters because post-training = can be applied to our EXISTING trained models immediately.

### 5.6 maximum pruning rates that maintain useful accuracy

from literature:
- 90%: typically <3% loss (our model: 2.75 pp)
- 95%: typically <5%
- 97%: 5-10% (lottery tickets can mitigate)
- 99%: variable, sometimes accuracy IMPROVES (NDSNN)

for our model: 90% already demonstrated at 37.75% (vs 40.50% baseline). with fine-tuning, expect 39-40%. at 95% with IMP + fine-tuning: expect ~36-38%.

---

## 6. combined / multiplicative strategies

### why techniques are multiplicative

from SPARQ and QP-SNN: orthogonal efficiency techniques multiply, not add.

for our model:
| technique | factor | mechanism | orthogonal? |
|-----------|--------|-----------|-------------|
| spike rate 26.4% to 6% | 4.4x | fewer ACs/timestep | yes |
| timesteps T=25 to T=7 | 3.6x | fewer timesteps | yes |
| pruning 70% removed | 2.5x | fewer synapses | mostly |
| quantization 16-bit to 8-bit | 1.5x | cheaper per op | yes |
| combined | ~60x | | |
| energy | 968/60 = ~16 nJ | | |

but some interaction (not perfectly multiplicative): spike rate reduction partly overlaps with pruning. typical interaction factor ~0.7-0.8x. realistic: ~40-50x rather than 60x.

### SPARQ's 330x breakdown

1. SNN sparsity: ~5x
2. INT8 quantization: ~4x
3. early exit (RL): ~16x
4. combined: 5 * 4 * 16 = 320x (plus minor = 330x)

### recommended combined strategy

phase 1 (immediate, ~1-2 days each):
1. spike rate reg to 6% (~4.4x)
2. SpikeCP early exit wrapper (~2.5-3x)
3. combined: ~11-13x = ~75-88 nJ

phase 2 (this week):
4. IMP to 80% + fine-tune (~2x)
5. combined: ~22-26x = ~37-44 nJ

phase 3 (if time):
6. INT8 QAT (~1.5x)
7. spiking brain compression (~1.3x)
8. all combined: ~43-51x = ~19-23 nJ

---

## 7. novel / unconventional approaches (oct 2025 - march 2026)

### 7.1 prosperity (arXiv:2503.03379) -- product sparsity

Wei et al. discovers combinatorial similarities in matrix multiplication -- if two rows share many zero positions, inner products share intermediate results. reuses computations across similar spike patterns. SpikeBERT density reduced to 1.23% (from 13.19%). 7.4x speedup, 193x energy efficiency over A100 GPU. needs custom hardware though.

### 7.2 phi (ISCA 2025) -- hierarchical sparsity

two-level: vector-wise patterns (k-means selected) + element-wise sparse correction. 3.45x speedup, 4.93x energy efficiency over SoTA SNN accelerators. also needs custom hardware.

### 7.3 NeuEdge (arXiv:2602.02439, Feb 2026)

312x energy savings vs conventional DNNs on autonomous drone. 91-96% accuracy across vision and audio. 847 GOp/s/W. 89% core utilization on Loihi 2. designed for Loihi but principles transfer.

### 7.4 SpiNNaker2 deployment (arXiv:2504.06748, Apr 2025)

94.13% on DVS128 Gesture on-chip. two quantization pipelines: PTQ with percentile scaling and QAT with adaptive threshold. 8-bit on-chip inference works. lessons apply to SpiNNaker 1.

### 7.5 multi-core neuromorphic training (Nature Comms 2026)

1.05 TFLOPS/W at FP16 at 28nm. 55-85% reduction in memory access vs A100. shows neuromorphic is becoming competitive for training, not just inference.

### 7.6 sigma-delta on Loihi 2 (arXiv:2505.06417)

17x fewer synaptic ops. only 0.056x the ops of equivalent ANN. encodes only CHANGES in activation. SpiNNaker 1 compatibility LOW (needs graded spikes) but concept applies to our input encoding.

### 7.7 SpikeFit (EurIPS 2025 workshop)

with only 4 unique weight values: outperforms all SoTA SNN compression. Fisher Spike Contribution pruning + Clusterization-Aware Training. SpiNNaker 1 compatibility: HIGH -- 4 discrete values is very SpiNNaker-friendly, could use 2-bit lookups.

### 7.8 LightSNN (arXiv:2503.21846)

98x speedup over SNASNet in architecture search. sparsity-aware Hamming distance fitness. could find more energy-efficient architecture than our hand-designed one.

### 7.9 spiking SqueezeNet (arXiv:2602.09717)

88.1% energy reduction vs CNN counterpart. 15.7x higher efficiency. only 1% below CNN SqueezeNet.

---

## 8. SpiNNaker 1 specific considerations

energy per synaptic operation varies a lot:
- theoretical best: ~20 nJ/event
- cortical simulation measured: ~0.63-5.9 uJ/event (system-level)
- SpiNNaker 2: ~10 pJ/event (630x improvement)

what maps well to SpiNNaker 1:
| technique | fit | why |
|-----------|-----|-----|
| early exit (confidence) | excellent | ARM cores check between timesteps |
| spike rate reg | excellent | fewer multicast packets |
| structured pruning | good | fewer populations = fewer cores |
| ternary spikes | good | exc/inh natively supported |
| binary weights | good | add/subtract on ARM |
| SpikeFit (4 values) | good | 2-bit weight lookup |
| SpikeCP (conformal) | excellent | zero retrain, ARM thresholding |
| multi-level neurons | poor | binary spikes only |
| sigma-delta | poor | no graded spikes |

SpiNNaker 1 optimization path:
1. reduce spike rate from 26.4% to <7% -> fewer multicast packets
2. early exit at avg T=7 -> fewer sim timesteps
3. structured pruning -> fewer cores
4. binary/4-value weights -> less SDRAM access

---

## 9. research gaps and open questions

1. no paper combines early exit + spike reg for AUDIO SNNs. ED-sKWS is the only audio early exit and uses FC-only.
2. no ESC-50 specific energy optimization exists. closest is Larroza et al. (ESC-10, FC-only).
3. multi-level neurons on SpiNNaker 1 are unexplored. could be simulated via burst encoding.
4. spike budgeting only tested on continual learning -- standard single-task is unexplored.
5. post-training compression (spiking brain compression) untested on audio.

---

## 10. experiment priority

### tier 1: this week (highest impact, lowest effort)

1a. SpikeCP early exit wrapper (0.5 days) -- apply to existing model, no retraining. calibrate on validation fold. expected avg T drops from 25 to ~8-12 = 2-3x. low risk.

1b. layer-wise spike reg to 6% (1 day) -- per-layer L1 penalty (strong on conv, weak on FC), learnable beta/V_thr. expected 4.4x. low-medium risk (may lose 3-5%).

1c. post-training pruning + fine-tune (1 day) -- IMP to 80%, fine-tune 10 epochs. expected 2-2.5x. low risk (model tolerates 90%).

### tier 2: if time allows

2a. CT loss from ED-sKWS (1-2 days) -- better early exit with cumulative temporal loss.
2b. ternary spikes (1-2 days) -- same accuracy at fewer timesteps.
2c. SpikeFit 4-value weights (2 days) -- better SpiNNaker mapping.

### tier 3: maximum impact

3a. validate multiplicativity of combined spike reg + early exit + pruning
3b. binary weight training (AGMM-BSNN)
3c. KD to 60K model

---

## confidence assesment

| finding | confidence | basis |
|---------|------------|-------|
| spike rate reg to 6% = 4x energy | HIGH | Dampfhoffer, Yang, our pareto data |
| early exit = 2-3x timestep reduction | HIGH | SEENN, CPT-SNN, SpikeCP, our temporal ablation |
| combined = 10-40x | MEDIUM-HIGH | SPARQ shows 330x but different tasks |
| 90% pruning maintains >90% accuracy | HIGH | our own data |
| multi-level T=1 maintains >50% on 50-class | MEDIUM | only tested on CIFAR-100, not audio |
| SpiNNaker 1 benefits from these | MEDIUM | some need custom models |

---

## sources

- SEENN NeurIPS 2023: https://arxiv.org/abs/2304.01230
- SPARQ arXiv 2026: https://arxiv.org/abs/2603.14380
- CPT-SNN Neurocomputing 2025: https://www.sciencedirect.com/science/article/abs/pii/S0925231225009257
- SpikeCP IEEE JSTSP 2025: https://arxiv.org/abs/2305.11322
- ED-sKWS arXiv 2024: https://arxiv.org/abs/2406.12726
- Cutoff+RCS Frontiers 2025: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1522788/full
- Spike Budgeting arXiv 2026: https://arxiv.org/abs/2602.12236
- Yang et al. arXiv 2024: https://arxiv.org/abs/2409.08290
- Dampfhoffer IEEE TECI 2023: https://cea.hal.science/cea-03852141/file/Are_SNNs_Really_More_Energy_Efficient_Than_ANNs__An_In_Depth_Hardware_Aware_Study_versionacceptee.pdf
- All in One Timestep arXiv 2025: https://arxiv.org/abs/2510.24637
- Scale-and-Fire arXiv 2025: https://arxiv.org/abs/2510.23383
- One-Hot M-LIF IEEE Access 2025: https://ieeexplore.ieee.org/document/10906567/
- QP-SNN ICLR 2025: https://arxiv.org/abs/2502.05905
- Ternary Spike AAAI 2024: https://arxiv.org/abs/2312.06372
- SpQuant-SNN Frontiers 2024: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1440000/full
- AGMM-BSNN AAAI 2025: https://arxiv.org/abs/2502.14344
- ReverB-SNN ICML 2025: https://arxiv.org/abs/2506.07720
- S2NN arXiv 2025: https://arxiv.org/abs/2509.24266
- MD-SNN arXiv 2025: https://arxiv.org/abs/2512.04443
- SpikeFit EurIPS 2025: https://arxiv.org/abs/2510.15542
- Unstructured Pruning ICLR 2024: https://openreview.net/forum?id=eoSeaK4QJo
- LTH in SNNs ECCV 2022: https://arxiv.org/abs/2207.01382
- NDSNN DAC 2023: https://dl.acm.org/doi/10.1109/DAC56929.2023.10247810
- Dynamic Spatio-Temporal Pruning Frontiers 2025: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1545583/full
- Spiking Brain Compression arXiv 2025: https://arxiv.org/abs/2506.03996
- Prosperity arXiv 2025: https://arxiv.org/abs/2503.03379
- Phi ISCA 2025: https://arxiv.org/abs/2505.10909
- NeuEdge arXiv 2026: https://arxiv.org/abs/2602.02439
- SpiNNaker2 DVS arXiv 2025: https://arxiv.org/abs/2504.06748
- SDNN on Loihi 2 arXiv 2025: https://arxiv.org/abs/2505.06417
- Spike-Thrift WACV 2021: https://openaccess.thecvf.com/content/WACV2021/papers/Kundu_Spike-Thrift_Towards_Energy-Efficient_Deep_Spiking_Neural_Networks_by_Limiting_Spiking_WACV_2021_paper.pdf
- TRT arXiv 2025: https://arxiv.org/abs/2506.19256
- Spiking SqueezeNet arXiv 2026: https://arxiv.org/abs/2602.09717
- LightSNN arXiv 2025: https://arxiv.org/abs/2503.21846
- SpiNNaker2 Q-Networks arXiv 2025: https://arxiv.org/abs/2507.23562
- Nature Comms Neuromorphic Training 2026: https://www.nature.com/articles/s41467-026-70586-x
- Neuromorphic Audio Survey arXiv 2025: https://arxiv.org/abs/2502.15056
- ESC Spike Encoding arXiv 2025: https://arxiv.org/abs/2503.11206
