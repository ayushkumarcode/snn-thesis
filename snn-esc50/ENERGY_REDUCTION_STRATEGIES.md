# energy reduction strategies -- notes toward 10-100x target

research date: 26 march 2026
context: our SNN is 622K params, T=25, 968 nJ/sample, 1.08M ACs, spike rate 26.4%, accuracy 47.15%

---

the big picture is that our spike rate of 26.4% is 4x above the break-even threshold of ~6.4% (Dampfhoffer 2023, Yang et al. 2024). until we get below ~7%, our SNN isn't actually more efficient than a quantized ANN on digital hardware. so that's the thing we need to fix first.

rough estimates for what combining techniques could get us:

| strategy | reduction | acc impact | how hard |
|----------|-----------|------------|----------|
| adaptive timestep (T=25 -> avg T=7) | 3.5x | -2-5% | easy |
| spike rate reg (26% -> 6%) | 4.3x | -3-5% | easy |
| weight quantization (32b -> 8b) | 2-4x memory | -1-2% | medium |
| structured pruning (50-70%) | 2-3x | -2-4% | medium |
| depthwise separable conv | 3-5x | -1-3% | medium |
| KD to tiny model (622K -> 60K) | 10x | -5-8% | medium |
| input masking (skip silence) | 1.5-2x | 0% | easy |
| always-on duty cycle argument | 10-100x | 0% | easy (system-level) |

realistic combined (1+2+4+7): 15-30x -> ~32-65 nJ/sample
aggressive combined (1+2+3+4+5+6): 50-200x -> ~5-19 nJ/sample
system-level with duty cycle: 100-1000x effective

---

## 1. adaptive timestep / early exit

not all samples need T=25. easy samples like a dog bark or siren can be classified at T=3-5, while hard samples (similar urban sounds) need more time. the idea is to let avg inference timesteps drop from 25 to like 5-10.

papers i found on this:

**SEENN (NeurIPS 2023)** - Li et al.
- SEENN-II with ResNet-19 gets 96.1% on CIFAR-10 with avg 1.08 timesteps -- that's wild
- two variants: SEENN-I does confidence thresholding, SEENN-II uses RL
- 4x fewer timesteps at same accuracy

**CPT-SNN (Neurocomputing 2025)**
- 95.44% CIFAR-10 with avg 2.72 timesteps
- feeds previous timestep outputs back as inhibitory currents
- dynamic confidence adjusts timesteps per sample

**SPARQ (arXiv 2603.14380, March 2025)** -- this one is really interesting
- 330x lower system energy vs baseline SNN (2.68 mJ vs 888 mJ on AlexNet)
- 90-96x fewer synaptic ops
- RL agent learns optimal exit points
- combines early exit + INT8 quantization + SNN
- per-class routing so easy classes exit at layer 1, hard ones go deeper
- probably the most directly relevant paper for what we're trying to do

**SpikeCP (IEEE JSTSP 2025)** - conformal prediction wrapper
- wraps ANY pretrained SNN with reliability guarantees
- zero retraining needed, just calibration
- statistical guarantee that predicted set includes true label with probability >= 1-alpha
- most practical to implement since you dont retrain anything

**Cutoff + Regularization (Frontiers in Neuroscience 2025)**
- Top-K cutoff: stop when top predictions stabilize
- RCS regularizer helps optimize spike timing for early cutoff
- 1.76-2.76x fewer timesteps on CIFAR-10
- 1.64-1.95x on event-based datasets

our temporal ablation already shows the potential:
- T=7: 36.5% (90% of full accuracy) -- 72% energy saving
- T=10: 38.25% (94%) -- 60% saving
- T=15: 40.25% (99%) -- 40% saving

so the plan would be confidence-based early exit:
```
for each sample:
  run timestep t = 1, 2, 3, ...
  compute confidence = max(softmax(spike_count_so_far))
  if confidence > threshold AND stable for k steps:
    STOP
  if t == T_max (25): return prediction
```

expected savings: avg T drops from 25 to ~7-10, giving 2.5-3.5x energy reduction

SpiNNaker compatibility is good here -- it naturally runs timestep-by-timestep, and you can implement confidence check on ARM core between steps. multicast packets only sent when neurons fire so early stop is basically free.

---

## 2. spike rate reduction (this is the critical one)

our spike rate is 26.4% (73.6% sparsity). the literature is pretty clear on this:
- Dampfhoffer et al. 2023 (IEEE TECI): need spike rate < 6.4% (sparsity > 93.6%) to beat quantized ANNs
- Yang et al. 2024 (arXiv:2409.08290): at T=6, need sparsity > 93%
- we're 4x above break-even

**Energy-Aware Spike Budgeting (arXiv 2602.12236, 2025)**
- adaptive spike scheduler enforces energy constraints during training
- reduces spike rates by up to 47% while IMPROVING accuracy on frame-based datasets
- learnable LIF parameters (beta, threshold) auto-tune for target spike budget
- directly applicable to us

**Spike-Thrift (WACV 2021)** - Kundu et al.
- attention-guided compression gives 33.4x compression, 12.2x energy efficiency vs ANNs

our own L1 regularization results tell an interesting story:
- lambda=0.0: 42.25% acc, 18.63 spikes/sample, rate=1.49%
- lambda=1e-5: 46.0% acc, 26.89 spikes, rate=2.15% (BEST)
- lambda=1e-4: 38.5% acc, 6.82 spikes, rate=0.55%
- lambda=1e-3: 45.25% acc, 7.80 spikes, rate=0.62%

wait -- our OUTPUT spike rate is already very low (0.6-2.2%), but NeuroBench measures ALL layer spikes (26.4%). the issue is in the HIDDEN layers (conv1, conv2, fc1).

so the targeted strategy is: apply layer-wise spike regularization specifically to conv1/conv2 while keeping output unconstrained.

from Yang et al. 2024:
1. L2 weight penalty lambda_1 = 0.05
2. activation penalty lambda_2 = 1e-6
3. keep T small (T=6 optimal for energy)
4. target >= 93% sparsity per layer

implementation:
```python
# layer-wise spike loss
loss = ce_loss + lambda_conv * (spk1.mean() + spk2.mean()) + lambda_fc * spk3.mean()
# target: reduce from 26.4% firing to <7% firing
```

if spike rate goes from 26.4% to 6%, thats 4.4x fewer synaptic operations. and on SpiNNaker, fewer spikes = fewer multicast packets = less router congestion (which is already a bottleneck for us).

---

## 3. weight and membrane potential quantization

**QP-SNN (ICLR 2025)** - joint quantization + structured pruning. weight rescaling for better bit utilization, pruning criterion uses singular value of spatiotemporal spike activities. state of the art.

**SpQuant-SNN (Frontiers in Neuroscience 2024)**
- compresses membrane potential to TERNARY
- 13x memory reduction, >4.7x FLOPs reduction, <1.8% accuracy degradation
- uses something called Stacked Gradient Surrogation for low-precision training

**Ternary Spike (AAAI 2024)** - Guo et al.
- spikes are {-1, 0, +1} instead of {0, 1}
- inhibitory spikes carry more information
- binary sparsity 16.42% vs ternary 18.27% (minimal overhead)
- actually improves accuracy from richer encoding

**Binary Weight SNNs (2025)**
- AGMM-BSNN: 96.13% CIFAR-10 with 2 timesteps
- weights are {-1, +1}, each synapse is just add/subtract
- on SpiNNaker ARM cores: addition is 1 cycle vs multiply ~3-5 cycles

**MD-SNN (2025)** - membrane potential-aware distillation for quantized SNNs

for our model, phase 1 would be QAT to INT8 weights. SpiNNaker already uses 16-bit fixed point (s16.15), so going to INT8 halves memory bandwidth. PyTorch QAT is well-supported. phase 2 would be exploring binary weights for FC layers -- FC1 (2304x256) and FC2 (256x50) dominate, and binary weights mean add/subtract instead of multiply.

expected savings: 2-4x from reduced memory bandwidth, additional 2x from simpler arithmetic.

SpiNNaker compatibility is medium -- 16-bit is native, 8-bit needs custom synapse models, binary is possible with custom weight handling. main benefit is reduced memory footprint per core.

---

## 4. structured pruning

**Dynamic Spatio-Temporal Pruning (Frontiers in Neuroscience 2025)**
- 0.63% remaining connections gives 91x energy efficiency gain
- only 8.5M SOPs, 2.19% accuracy loss on CIFAR-10
- 98.18% parameter reduction on DVS128

**Lottery Ticket Hypothesis in SNNs (ECCV 2022, oral)**
- winning tickets at 97% sparsity without huge performance hit
- 98.13% sparsity with <2% accuracy drop

our own pruning results are pretty encouraging:
- 0% pruning: 40.5%
- 30% pruning: 41.75% (actually IMPROVES)
- 50%: 39.75%
- 70%: 37.5%
- 90%: 37.75% (retains 93.2% of accuracy)

our SNN is incredibly pruning-resilient -- at 90% we only lose 2.75 pp. plan is to do iterative magnitude pruning to 70-90%, fine-tune after, and use structured pruning (channel pruning) for actual speedup on SpiNNaker.

at 70% pruning thats 2-3x fewer ops. at 90% could be 5-10x. good SpiNNaker compatibility since fewer synapses = fewer routing table entries = less memory per core.

---

## 5. efficient architecture (depthwise separable)

**LitE-SNN (2024)** - lightweight efficient SNN via spatial + temporal NAS compression. first to propose compressive timestep search.

**Spiking SqueezeNet (arXiv 2602.09717, Feb 2025)**
- 88.1% energy reduction vs CNN equivalent, 15.7x efficiency
- only 1% below CNN-SqueezeNet accuracy

current architecture ops per timestep:
```
Conv1: ~1.3M ACs (when all input=1)
Conv2: ~63M potential ACs
FC1: 589,824 ACs
FC2: 12,800 ACs
```

with depthwise separable Conv2:
```
Depthwise: 32*3*3*32*108 = 995,328
Pointwise: 64*32*32*108 = ~7M
Total: ~8M vs ~63M = 8x reduction in conv2 alone
```

would replace Conv2d(32,64,3) with DepthwiseConv2d(32,32,3) + Conv2d(32,64,1). expected 3-5x total since convolutions dominate.

SpiNNaker compatibility is low-medium though. depthwise separable is harder to map efficiently on SpiNNaker 1 since population-based routing works best with standard dense connectivity. better suited for SpiNNaker 2 or software sim.

---

## 6. knowledge distillation to tiny model

**Distilling Spikes (2020)** - first KD specifically for SNNs, transfers learning from big SNN to small SNN.

**LaSNN (Neurocomputing 2025)** - three stages: train ANN teacher, convert to SNN, layer-wise distillation.

**SAKD (Neural Networks 2024)** - bilevel teacher-student training with ANN weight transfer.

**Edge Audio KD (Discover Applied Sciences 2024)** - 85.4% ESC-50 with compressed model via KD. comparable to SOTA with much less compute.

teacher would be our 622K param SpikingCNN (47.15%). student would be a tiny version:
- Conv1: 1->8 channels (not 32)
- Conv2: 8->16 (not 64)
- FC1: 576->64 (not 2304->256)
- FC2: 64->50

that's ~10-15K params (40-60x smaller). training loss:
```python
loss = alpha * CE(student_out, labels) +
       (1-alpha) * KL(student_logits/T, teacher_logits/T) +
       beta * MSE(student_hidden_spikes, teacher_hidden_spikes)
```

expected 40-60x fewer operations per timestep. accuracy probably 35-42% based on KD literature suggesting 70-90% retention.

good SpiNNaker compatibility -- smaller model fits on fewer cores, simpler routing tables, less memory.

---

## 7. input-level sparsity (skip silent regions)

**BAE (Frontiers 2020)** - biologically plausible auditory encoding with psychoacoustic masking
- 50.48% of spikes discarded on TIDIGITS (speech)
- 39.38% reduction on RWCP (environmental sounds -- similar to ESC-50!)
- 97.4% accuracy maintained on TIDIGITS
- this is directly applicable to audio SNNs

**Sigma-Delta Neural Networks** - only process CHANGES in activation between timesteps
- 17x synaptic op reduction on Loihi 2
- 0.056x the synaptic ops of equivalent ANN

audio spectrograms have massive temporal redundancy -- between adjacent time frames, most mel bins barley change.

three approaches we could try:

approach A - spectrogram delta encoding:
```python
# instead of full spectrogram at each timestep
# delta: x[t] = spectrogram[t] - spectrogram[t-1]
# most values near zero = very sparse input
```

approach B - psychoacoustic masking:
```python
# for each mel bin at each time frame:
# if energy < masking_threshold(neighboring_bins):
#   dont encode (set to 0)
# reduces encoding spikes by ~40% for environmental sounds
```

approach C - activity-gated input (simplest):
```python
# if spectrogram_energy[time_window] < threshold:
#   skip this timestep entirely (zero input)
#   SNN membranes just decay (very cheap)
```

expected savings: 1.5-2x conservatively since our input is already spectrogram-based.

SpiNNaker loves this -- zero input = zero spikes = zero packets = zero computation. event-driven architecture means silence is literally free.

---

## 8. always-on monitoring energy model (system-level)

this is the killer argument for neuromorphic audio and i think people miss this when they just compare per-inference energy.

the comparison "968 nJ vs 454 nJ per inference" misses the REAL deployment scenario.

in real-world environmental sound monitoring:
- sound events are RARE (a dog barks for ~2s out of every 60s)
- a fire alarm goes off maybe once per year
- significant sound events cover ~5-20% of time
- most time is silence/background noise

ANN on a microcontroller:
- MUST wake up every frame (every 25ms)
- compute mel spectrogram + inference REGARDLESS of input
- ~454 nJ/inference * 40 inferences/sec * 86400 sec/day = 1.57 J/day

SNN on neuromorphic hardware (Xylo, SpiNNaker):
- event-driven: ZERO computation during silence
- only processes when sound energy exceeds threshold
- with 10% duty cycle: 0.157 J/day -- 10x less
- with 2% duty cycle (typical surveillance): 0.031 J/day -- 50x less

reference hardware numbers:

Xylo Audio 2 (SynSense): 251-298 uW dynamic, 216-217 uW idle, 6.6 uJ/inference, 95.31% on Google Speech Commands

SpiNNaker 2: ~10 pJ/synaptic event, ~7.1 uJ/inference for KWS, DVFS down to 0.5V for 10x over SpiNNaker 1

comparison table (keyword spotting):
| platform | dynamic energy/inference |
|----------|------------------------|
| Xylo Audio 2 | 6.6 uJ |
| SpiNNaker 2 | 7.1 uJ |
| Loihi | 37 uJ |
| MOVIDIUS | 1,500 uJ |
| Jetson Nano | 5,580 uJ |
| CPU | 6,320 uJ |
| GPU | 29,670 uJ |

thats 800-4,500x energy advantage of neuromorphic vs GPU for audio inference.

for a 24-hour monitoring scenario:
```
ANN (ARM Cortex-M4, ~100uW idle + 500uW active):
  active: 0.5mW * 0.1 * 86400s = 4.32 J
  idle: 0.1mW * 0.9 * 86400s = 7.78 J (must still sample)
  total: ~12.1 J/day

SNN on SpiNNaker 2:
  active: 10pJ/synop * 1.08M synops * 1/sec * 0.1 * 86400s = 0.93 J
  idle: ~0 (truly event-driven)
  total: ~0.93 J/day = 13x advantage

SNN on Xylo Audio 2 with wake-on-event VAD chip:
  VAD idle: ~1 uW * 86400s = 0.086 J
  inference: 6.6 uJ * 1/sec * 0.1 * 86400 = 0.057 J
  total: ~0.14 J/day = 86x advantage over ANN
```

---

## combined strategy recommendations (prioritized)

### tier 1: do now (high impact, easy)

1a. confidence-based early exit -- add confidence check in inference loop. avg T drops from 25 to ~8 (3x energy reduction). <2% accuracy impact. 1-2 days.

1b. spike rate regularization -- add layer-wise L1 spike loss. spike rate drops 26.4% to 6-8% (3-4x). -3-5% accuracy. 1 day (modify training loop).

1c. input silence gating -- skip timesteps where spectrogram energy < threshold. 30-50% fewer timesteps (1.5-2x). 0% accuracy impact. half a day.

combined tier 1: 3x * 3.5x * 1.5x = ~16x energy reduction
estimated: 968 nJ -> ~60 nJ (below ANN's 454 nJ!)

### tier 2: if time allows

2a. weight pruning 70-90% -- already tolerates 90% at 93% retained accuracy. additional 2-3x. 2-3 days.

2b. INT8 QAT -- additional 2x. 2 days.

2c. KD to 10-15K model -- additional 10x. 3-5 days.

### tier 3: architecture redesign

3a. depthwise separable convolutions -- 3-5x in conv layers. 3-4 days, retrain from scratch, risk of not working well with LIF.

3b. single-timestep SNN -- iterative timestep reduction T=25 -> T=1. 25x timestep reduction but significant accuracy loss. 1 week.

---

## paper framing (for ICONS 2026)

the key narrative points:

1. software simulation is misleading -- NeuroBench shows SNN at 968 nJ vs ANN 454 nJ. makes SNN look WORSE. but this ignores neuromorphic hardware advantage (AC at 0.9pJ vs MAC at 4.6pJ), event-driven processing (zero cost during silence), and duty cycle advantage.

2. the spike rate problem is solvable -- 26.4% is above 6.4% break-even, but with targeted regularization we can push below 7%. that alone makes the SNN energy-favorable.

3. adaptive timesteps are a unique SNN advantage -- ANNs must process every sample the same way. SNNs can naturally exit early. this is a fundamental architectural advantage.

4. the deployment scenario matters -- for a 24-hour environmental sound monitor: ANN ~12 J/day (always processing), SNN on neuromorphic hardware ~0.1-1 J/day (event-driven). 10-100x advantage in real deployment.

---

## critical thresholds from literature

| metric | threshold | source | ours now | target |
|--------|-----------|--------|----------|--------|
| spike rate | <6.4% | Dampfhoffer 2023 | 26.4% | <6% |
| sparsity | >93% | Yang et al. 2024 | 73.6% | >94% |
| timesteps | T<=6 optimal | Yang et al. 2024 | T=25 | T=5-7 |
| weight sparsity | >90% viable | our pruning data | 0% | 70-90% |
| model size | <50K viable | KD literature | 622K | 60-100K |

---

## references

early exit / adaptive timestep:
- SEENN: Li et al. "SEENN: Towards Temporal Spiking Early Exit Neural Networks." NeurIPS 2023.
- CPT-SNN: "CPT-SNN: A spiking neural network that can combine the previous timestep." Neurocomputing 2025.
- SPARQ: "SPARQ: Spiking Early-Exit Neural Networks for Energy-Efficient Edge AI." arXiv:2603.14380, March 2025.
- SpikeCP: "Knowing When to Stop: Delay-Adaptive SNN Classifiers with Reliability Guarantees." IEEE JSTSP 2025.
- Cutoff+RCS: "Optimizing event-driven spiking neural network with regularization and cutoff." Frontiers in Neuroscience 2025.

spike rate and energy thresholds:
- Dampfhoffer et al. "Are SNNs Really More Energy-Efficient Than ANNs? An In-Depth Hardware-Aware Study." IEEE TECI 2023.
- Yang et al. "Reconsidering the energy efficiency of spiking neural networks." arXiv:2409.08290, 2024.
- Horowitz. "Computing's energy problem." ISSCC 2014. (AC=0.9pJ, MAC=4.6pJ)
- Energy-Aware Spike Budgeting. arXiv:2602.12236, 2025.
- Spike-Thrift: Kundu et al. WACV 2021.

quantization:
- QP-SNN: "Quantized and Pruned Spiking Neural Networks." ICLR 2025.
- SpQuant-SNN: "Ultra-low precision membrane potential with sparse activations." Frontiers in Neuroscience 2024.
- Ternary Spike: Guo et al. AAAI 2024.
- AGMM-BSNN: arXiv:2502.14344, 2025.
- MD-SNN: arXiv:2512.04443, 2025.

pruning:
- Dynamic Spatio-Temporal Pruning. Frontiers in Neuroscience 2025.
- LTH in SNNs: Kim et al. ECCV 2022 (oral).
- ICLR 2024 unstructured pruning framework.

architecture:
- LitE-SNN: arXiv:2401.14652, 2024.
- Spiking SqueezeNet: arXiv:2602.09717, Feb 2025.
- SpikeBottleNet: arXiv:2410.08673, 2024.

knowledge distillation:
- Distilling Spikes: Kushawaha et al. 2020.
- LaSNN: Neurocomputing 2025.
- SAKD: Neural Networks 2024.
- Edge Audio KD: Discover Applied Sciences 2024.

sigma-delta / event-driven:
- SDNN on Loihi 2: arXiv:2505.06417, 2025.
- BAE (Auditory Masking): Frontiers in Neuroscience 2020.

hardware:
- Xylo Audio 2: arXiv:2406.15112, 2024.
- SpiNNaker 2: arXiv:2401.04491, 2024.
- SpiNNaker 1: ~1W per chip, 18 ARM968 cores, 16-bit weights, ~630 nJ/synop.
- NASP NeuroVoice: microwatt-level always-on VAD. 2025.
- Loihi vs SpiNNaker 2 KWS: Yan et al. Neuromorph. Comput. Eng. 2021.

single-timestep and multi-level:
- "One Timestep Is All You Need." arXiv:2110.05929.
- "All in One Timestep." arXiv:2510.24637, 2025.

cross-domain:
- Biological auditory coding: TINS 2019.
- Compressed Sensing SNN: IEEE 2024.

---

## appendix: operation breakdown of our model

per-timestep operation count:
```
Layer          | Params  | Ops (dense)    | Ops (at 26.4% spike rate)
Conv1 (1->32)  | 320     | 320 * H*W      | ~320 * 64*216 = 4.4M (input dense)
Conv2 (32->64) | 18,496  | 18496 * H/2*W/2| ~18496 * 32*108 * 0.264 = 16.9M
FC1 (2304->256)| 590,080 | 590,080        | 590,080 * 0.264 = 155.8K
FC2 (256->50)  | 12,850  | 12,850         | 12,850 * 0.264 = 3.4K

Total per step: ~21.3M ACs (at 26.4% spike rate)
Total T=25:     ~533M ACs -> but NeuroBench says 1.08M?
```

NeuroBench counts "effective ACs" which accounts for actual spike activity per sample. our measured 1.08M ACs / 25 timesteps = 43,200 ACs per timestep -- so actual per-sample activity is much lower than theoretical max.

energy at different spike rates (T=25):
| spike rate | ACs (est) | energy (AC*0.9pJ) | vs ANN (454 nJ) |
|------------|-----------|-----------------------|-----------------|
| 26.4% (current) | 1.08M | 968 nJ | 2.1x worse |
| 15% | ~614K | ~553 nJ | 1.2x worse |
| 10% | ~409K | ~368 nJ | 1.2x better |
| 6% | ~245K | ~221 nJ | 2.1x better |
| 3% | ~123K | ~110 nJ | 4.1x better |

energy at different timesteps (spike rate 6%):
| timesteps | ACs (est) | energy | vs ANN |
|-----------|-----------|--------|--------|
| T=25 | ~245K | ~221 nJ | 2.1x better |
| T=15 | ~147K | ~132 nJ | 3.4x better |
| T=10 | ~98K | ~88 nJ | 5.2x better |
| T=7 | ~69K | ~62 nJ | 7.3x better |
| T=5 | ~49K | ~44 nJ | 10.3x better |
| T=3 | ~29K | ~26 nJ | 17.5x better |
| T=1 | ~10K | ~9 nJ | 50x better |

the 10x target: spike rate 6% + T=10 = ~88 nJ (10.9x better than ANN)
the 100x target: spike rate 3% + T=5 + 50% pruning = ~11 nJ (41x better)
for 100x vs ANN: need spike rate 3% + T=3 + tiny model = ~3-5 nJ

---

## bottom line

10x is achievable:
1. reduce spike rate 26.4% to 6% (4.4x) -- spike regularization
2. reduce timesteps 25 to 7 (3.6x) -- adaptive early exit
3. combined: ~16x -> ~60 nJ (7.6x better than ANN)

100x also requires:
4. pruning to 70% sparsity (2-3x more)
5. model compression via KD (5-10x more)
6. always-on duty cycle (10-50x more)

the system-level argument (always-on monitoring with event-driven processing) is honestly the strongest case. a 24-hour environmental sound monitor using SNN on neuromorphic hardware gets 10-100x lower energy than an equivalent ANN system, even before any model-level optimizations.
