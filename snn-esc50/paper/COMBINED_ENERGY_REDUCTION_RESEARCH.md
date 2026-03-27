# combining snn energy reduction techniques -- reading notes

date: 27 march 2026
context: audio SNN classifier, ESC-50, 622K params, 968 nJ/sample, 26.4% spike rate, T=25. best accuracy so far: 61.65% (dendritic+delays), 61.10% (rhythm). ANN baseline: 63.85% at 454 nJ.

basically trying to figure out how to stack multiple optimization techniques for 10-100x energy reduction without tanking accuracy.

---

## the big picture from reading all these papers

the literature pretty strongly supports multiplicative gains when you combine techniques, but theres some important caveats about interaction effects and double-counting that i keep seeing. heres what i've gathered:

1. **SPARQ (march 2026)** is the gold standard right now: SNN + early exit + INT8 quantization = 330x system energy reduction and 96x fewer synaptic operations on AlexNet/CIFAR-10. this is the most relevant paper for what we're trying to do.

2. technique combinations are genuinely multiplicative when they hit orthogonal dimensions: early exit reduces timesteps (temporal), spike regularization reduces per-step ops (spatial-activity), pruning removes connections (structural), quantization reduces per-operation cost (precision). these four axes are mostly independent.

3. the order you apply them matters -- ICLR 2026 work proves the "Progressive Intensity Hypothesis" which says you should apply weaker perturbations first, stronger later. for SNNs: train base model -> spike reg -> quantize -> prune. joint training can beat sequential but its harder.

4. diminishing returns are real but slow. first 3-4 techniques combine near-multiplicatively, after 4 things start to saturate.

5. for our specific model: early exit (T=7 avg) + spike reg (to 6%) + 90% pruning + INT8 could realistically get us 50-200x reduction (to ~5-19 nJ/sample), which would make our SNN decisively cheaper than the ANN. an aggressive combo including rhythm + reduced-T training could maybe approach 500x.

6. **KD hurts everything** -- every combo with KD we tried performed worse. this matches the literature finding that "a high-performance teacher often does not produce a good student." dont bother with KD unless doing specialized SNN-aware distillation.

---

## papers combining multiple techniques with actual measured gains

### SPARQ -- the one to beat (arXiv 2603.14380, march 2026)

combo: SNN conversion + INT8 quantization-aware training + RL-guided early exit
architecture: AlexNet, LeNet, 5-layer MLP on CIFAR-10, MNIST

individual contribution breakdown (AlexNet/CIFAR-10):
- baseline ANN: 922.4 uJ system energy
- SNN (T=32): 6.55 uJ (140x reduction from event-driven processing)
- + quantization (INT8): reduces per-operation cost
- + early exit (RL policy): further reduces average operations
- QDSNN (combined): 2.68 mJ system vs 888 mJ baseline SNN = 330x total

operation reduction:
- synaptic ops: 0.27M ACs (QDSNN) vs 26.06M (SNN) = 96x reduction
- total ops including LIF: <=7.64M vs 37.59M = 5x reduction
- interesting thing: LIF overhead dominates after optimization -- synaptic ops drop 96x but total only drops 5x because LIF updates are a fixed cost

the RL exit policy is neat: state = current exit index + discretized max softmax confidence, action = exit now or continue, reward = +1+0.3*savings if correct, -1 if wrong. trained 5k-10k episodes with Q-learning (lr=0.1, gamma=0.9).

accuracy: 78% on CIFAR-10 (vs 77% baseline SNN, 74.3% QSNN) -- actually IMPROVED accuracy with combined techniques which is cool

highly applicable to our model. architecture is comparable (conv SNN, direct training). the RL exit mechanism is architecture-agnostic. question is whether 330x translates from image to audio.

source: https://arxiv.org/abs/2603.14380

### QP-SNN: quantized and pruned SNNs (ICLR 2025)

combo: uniform quantization (ReScaW) + structured pruning (SVS criterion)
architecture: ResNet-20, Spikingformer on CIFAR-10/100, DVS datasets

synergy evidence (table 4, CIFAR-100):
- baseline: 69.16%
- + ReScaW only: 73.40% (+4.24pp)
- + SVS only: 73.32% (+4.16pp)
- + both combined: 73.89% (+4.73pp)
- combined > either alone = genuinely synergistic

compression results (CIFAR-10, 2-bit, 9.61% connection ratio):
- model size reduction: 98.74%
- SOP reduction: 78.69%
- power reduction: 77.45%
- accuracy loss: only 2.44%

the key insight is that quantization and pruning are synergistic for SNNs because the SVS criterion correctly identifies which kernels to prune by analyzing spatiotemporal spike activity patterns. random or magnitude-based pruning + quantization gives worse results.

source: https://openreview.net/forum?id=MiPyle6Jef

### SpQuant-SNN: triple compression (Frontiers in Neuroscience 2024)

combo: ternary membrane potential quantization + weight quantization (2-8 bit) + spatial-channel pruning

individual vs combined:
- membrane quant alone: ~13x memory reduction
- weight quant alone: 2-4x
- channel pruning alone: variable
- combined: 13x memory + >4.7x FLOPs + <1.8% accuracy loss

the cool thing here is ternary membrane potential -- quantize to {-1.0, 0.0, 1.0} per timestep. this is unique to SNNs (ANNs dont have membrane potential) and stacks with weight quant and pruning.

reasonably applicable to us. ternary membrane is SNN-specific and orthogonal to everything else. could layer this on top.

source: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1440000/full

### spiking brain compression (SBC, NeurIPS 2025)

combo: one-shot post-training pruning + quantization (joint)

results:
- 97% sparsity with <=1% accuracy loss on neuromorphic tasks
- 4-bit symmetric quantization with 0.17-7.71% accuracy loss
- compression time: 2-3 orders of magnitude faster than iterative methods
- scales to SEW-ResNet 152 on ImageNet

big advantage: post-training (no retraining needed). so you can apply it AFTER all other training-time optimizations. thats really nice.

source: https://arxiv.org/abs/2506.03996

### ADMM SNN compression (IEEE TNNLS 2021)

combo: connection pruning + weight quantization + activity regularization -- the first "kitchen sink" paper for SNNs

formulates pruning + quantization as constrained optimization, solves via ADMM + STBP. add activity reg to reduce spike events. applied individually ("moderate") or jointly ("aggressive").

validated on MNIST, N-MNIST, CIFAR-10, CIFAR-100. this is the foundational paper proving all three SNN compression axes can be combined.

source: https://arxiv.org/abs/1911.00822

### NeuEdge (february 2026)

combo: hybrid temporal coding (4.7x spike reduction) + adaptive thresholding (67% energy reduction in low-activity) + hardware-aware mapping

ablation breakdown (CIFAR-10, power):
- baseline (rate coding): 380 mW
- + hybrid encoding: 312 mW (18% reduction)
- + hardware-aware mapping: 294 mW (23% cumulative)
- + adaptive threshold: 201 mW (47% cumulative)
- full NeuEdge: 187 mW (51% cumulative)
- vs GPU baseline: 312x energy improvement

BUT the 312x number conflates algorithm improvements with hardware differences (neuromorphic vs GPU). the algorithmic improvement alone is about 2x (51% reduction). the 312x comes from deploying on Loihi 2 instead of Jetson Nano GPU. gotta be careful with these numbers.

source: https://arxiv.org/abs/2602.02439

### energy-aware spike budgeting (arXiv 2602.12236, february 2025)

combo: experience replay + learnable LIF parameters + adaptive spike scheduler

results -- genuine Pareto improvement:
- accuracy: +2.31pp over baseline
- forgetting: -2.77pp over baseline
- spike reduction: 47%
- all three improved simultaneously which is a rare pareto improvement

the mechanism: on frame-based data, spike budgeting acts as implicit regularization making the network BETTER while using fewer spikes. one of those rare cases where combining techniques gives better accuracy AND energy. pretty cool.

source: https://arxiv.org/abs/2602.12236

### dynamic spatio-temporal pruning (Frontiers in Neuroscience 2025)

combo: LAMPS layer-adaptive spatial pruning + adaptive temporal pruning

results:
- 98% parameter reduction across 4 datasets
- 0.63% remaining connections at extreme pruning
- 91x energy efficiency gain at extreme pruning
- only 8.5M SOPs, 2.19% accuracy loss on CIFAR-10
- DVS datasets: accuracy IMPROVED with temporal pruning (50% and 20% time reduction)

spatial and temporal pruning are synergistic -- removing temporal redundancy helps spatial pruning find better sparse subnetworks. makes sense intuitively.

source: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1545583/full

---

## which combinations are synergistic vs conflicting

### early exit + spike rate regularization: synergistic (compound)

these operate on orthogonal dimensions:
- early exit reduces NUMBER OF TIMESTEPS (T=25 to T=7 average)
- spike reg reduces SPIKES PER TIMESTEP (26.4% to 6%)
- combined: (25/7) * (26.4/6) = 3.57 * 4.4 = 15.7x -- genuinely multiplicative

literature backs this up: the "regularization and cutoff" paper (Frontiers 2025) combines both and shows compound benefits. SPARQ combines early exit with quantization (same principle). CPT-SNN achieves T=2.72 average with confidence-based exit.

potential positive interaction: spike reg could make early exit EASIER because sparser representations converge faster to confident predictions.

should always combine these.

### pruning + quantization: synergistic (with caveats)

from ICLR 2026 (prune-then-quantize or quantize-then-prune?):
- Progressive Intensity Hypothesis: apply WEAKER perturbation first
- for general NNs: quantize-then-prune is better (quantization is weaker)
- for SNNs specifically (QP-SNN): synergistic regardless of order, but combined ReScaW+SVS beats either alone

caveat from CVPR 2025 (GETA):
- more heavily pruned models are MORE sensitive to quantization
- at extreme pruning (>90%), quantization hurts more
- joint co-optimization avoids this

practical: prune to 70-90%, then quantize to INT8. or use joint framework like QP-SNN. do NOT prune to 99% then quantize to INT4 -- thats where accuracy collapses.

synergistic up to ~90% pruning + INT8. conflicting at extreme settings (>95% pruning + <4-bit).

### low timesteps + spike regularization: partially redundant

analysis:
- reducing T from 25 to 5 = 5x fewer ops
- reducing spike rate from 26.4% to 6% = 4.4x fewer ops per step
- combined: 5 * 4.4 = 22x -- still multiplicative

but training at low T with spike reg may reduce accuracy more than expected because you get a double information bottleneck -- fewer timesteps AND fewer spikes per step. our temporal ablation shows T=5 gives 33.5% (vs T=25 at 47.15%). adding spike reg on top of T=5 would probably push below 30%.

multiplicative in energy, possibly super-linear in accuracy loss. need to validate experimentally.

### KD + pruning: context-dependent

from the literature: KD -> small student -> prune further works in ANN world. for SNNs: KD alone is problematic (distribution mismatch between ANN teacher logits and SNN student logits). our results: KD HURTS everything.

from arXiv 2511.06902 (2025): "Prior KD methods simply align raw hidden features and output logits between teacher ANN and student SNN, ignoring discrepancies in distributions" -- this explains our negative results. naive KD is harmful. specialized SNN-aware KD (SAMD, NLD, SAKD) might help but adds complexity.

skip KD unless implementing specialized SNN-aware distillation. not worth the hassle for us.

### early exit + hierarchical classification: strongly synergistic

hierarchical: classify super-category (5 classes) then fine class (10 classes). early exit: stop at super-category level if confidence is high. combined: many easy samples classified in ~2 timesteps at the coarse level, hard samples go to fine classifier at ~10-15 timesteps.

most ESC-50 super-categories are easy to distinguish (Animals vs Urban is trivial). maybe 60-70% of samples exit at stage 1, reducing average ops by 5-8x.

this would be pretty novel -- not yet demonstrated in SNN literature for audio.

### rhythm oscillatory + spike regularization: partially redundant

rhythm already reduces firing rates (neurons in 'OFF' state dont fire). adding spike reg on top may have diminishing returns since rhythmic neurons are already sparse. though rhythm controls WHEN neurons fire and regularization controls HOW MANY -- slightly different dimensions.

our evidence: rhythm + L1=1e-5 gave 60.20% (vs rhythm alone 61.10%). the L1 penalty slightly hurt accuracy (-0.9pp) but reduced spike count. the energy gain from L1 on top of rhythm is probably smaller than L1 on baseline.

weakly synergistic at best.

### rhythm oscillatory + pruning: probably synergistic but untested

rhythm doesnt change weights or structure. pruning removes connections. these are orthogonal: rhythm modulates temporal activity, pruning modulates structural connectivity. combined: fewer active neurons (rhythm) * fewer connections per active neuron (pruning).

no direct evidence but the theoretical orthogonality seems strong.

---

## combined training strategies

### simultaneous vs sequential

from ICLR 2026: the Progressive Intensity Hypothesis says apply weaker perturbations first, stronger later.

for SNNs, recommended order:
1. train base model to convergence
2. add spike regularization (weak -- just modifies loss)
3. apply quantization-aware training (moderate -- reduces precision)
4. apply structured pruning (strong -- removes connections)
5. fine-tune the compressed model
6. add early exit mechanism (post-training, no architecture change)

from CVPR 2025 (GETA): joint structured pruning + quantization during training outperforms sequential by a "large margin" but is harder to implement and less stable. "nonlinear superposition of numerical errors and structural information losses."

practical plan for our model -- two-phase approach:
- phase 1 (training-time): train with rhythm oscillatory + spike regularization + quantization-aware training simultaneously. these are all loss/training mods that dont structurally change the model.
- phase 2 (post-training): apply structured pruning + early exit. these are post-hoc optimizations.

### multi-objective training: accuracy + energy as joint loss

MONAS-ESNN (WACV 2025): multi-objective NAS using NSGA-II evolutionary algorithm. optimizes both accuracy and spike count simultaneously.

energy-aware spike budgeting (2025): direct spike budget in the loss. learnable LIF parameters auto-tune for target budget. achieves pareto improvement on frame-based data.

SynOp loss (Frontiers 2020): direct SynOp count in the loss: L = L_task + lambda * N_synops. but excessive SynOp loss causes neuron silencing which is bad.

recommended loss for our model:
```
L = L_CE + lambda_spike * (sum of layer spike rates) + lambda_budget * max(0, spike_count - target)
```
where target = 6% of total possible spikes.

### pareto-optimal training

LitE-SNN (IJCAI 2024): joint optimization of architecture, spatial compression, and temporal compression. CompConv block supports pruning + mixed-precision quant in a unified search.

SpikeFit (EurIPS 2025): clusterization-aware training learns optimal discrete weight values. combined with Fischer Spike Contribution structured pruning. only 4 unique synaptic weight values needed.

these frameworks are for searching novel architectures. since our architecture is fixed (thesis constraint), we'd apply the optimization techniques rather than the full NAS framework.

---

## specific combinations for our model

### tier 1 combo: early exit (T=7 avg) + spike reg (6%) + silence gating

expected reduction: 3.5x * 4.4x * 1.5x = ~23x (968 nJ -> ~42 nJ)

literature support:
- early exit (T=7): our temporal ablation shows T=7 = 90% of accuracy. SEENN achieves 1.08 avg timesteps on CIFAR-10. CPT-SNN achieves 2.72 avg. conservative T=7 is realistic.
- spike reg (6%): Yang et al. 2024 achieves 94.19% sparsity (5.81% firing) at 92.76% accuracy on CIFAR-10. we need to go from 26.4% to 6%. literature suggests <5% accuracy loss.
- silence gating: BAE (2020) discards 39.38% of spikes on environmental sounds. our ESC-50 has 5-second clips with lots of silence. conservative 1.5x estimate.

no single paper combines exactly these three. but the orthogonality is strong -- early exit reduces T, spike reg reduces per-step activity, silence gating reduces input. Energy ~ T * r * I * ops_per_spike, so reductions multiply cleanly.

expected accuracy impact: -3 to -8% relative to rhythm baseline (61.1% -> 56-59%). acceptable for the energy gains.

### tier 2 combo: tier 1 + 90% pruning + INT8 quantization

expected reduction: 23x * 5x * 2x = ~230x (968 nJ -> ~4 nJ)

realism:
- 90% pruning: our experiments show SNN retains 93.2% of accuracy at 90% pruning. with fine-tuning expect minimal loss. 5x energy reduction from 10x fewer connections.
- INT8: SpiNNaker 1 already uses 16-bit fixed-point. going to 8-bit halves memory bandwidth.

but 230x assumes perfect multiplicativity. in practice pruning interacts with spike reg (pruned neurons cant fire so some spike reduction is "free"), and quantization interacts with pruning (more pruned models more sensitive to quant noise). realistic combined: probably 100-150x.

literature evidence at this scale: dynamic spatio-temporal pruning (2025) gets 91x, SPARQ gets 330x, ACE-SNN gets 100-10,000x (with hardware co-optimization). so 100-200x is realistic, 230x is optimistic but within published ranges.

### nuclear option: T=1 multi-level + binary weights + 95% pruning

expected reduction: 25x * 2x * 20x = 1000x (968 nJ -> ~1 nJ)

accuracy concern: SEVERE. T=1 binary on our model gives 7.25% (from temporal ablation). T=1 with 4-level neurons: maybe 25-30%. binary weights: typically -2 to -5%. 95% pruning: our results show ~36-38% at this level before binary weights. combined: probably 15-25% -- barely above chance for 50 classes.

only viable for super-category level classification or with PANNs features.

### biological combo: fly brain + early exit + silence gating

random sparse projection (fixed, no training) + WTA + confidence exit

expected: fly circuit 10-50x cheaper + early exit 2-3x + silence gating 1.5x = 30-225x

accuracy: completely unknown but intriguing. 50 odors maps to 50 classes. with PANNs features: maybe 70-85%. with raw mel features: maybe 20-35%.

this would be very novel and worth a quick experiment with PANNs embeddings. if it works, the paper narrative ("we replaced a deep SNN with a fly brain circuit") would be extraordinary.

### practical combo: reduced-T training (T=5) + spike reg + pruning

same 622K model, trained from scratch at T=5 with L1 spike penalty, pruned post-training

expected reduction: T=5 5x + spike reg 4.4x + 70% structured pruning 2x = 44x (968 nJ -> ~22 nJ)

accuracy estimate:
- T=5 baseline: ~33.5% (from temporal ablation, w/o retraining)
- T=5 from-scratch: probably 38-42% (model learns to use fewer timesteps)
- + spike reg: -2 to -4% -> 34-40%
- + 70% pruning (with fine-tuning): -1 to -2% -> 33-39%
- expected: ~35-40% at ~22 nJ

vs ANN: 63.85% at 454 nJ. practical combo: 35-40% at 22 nJ. thats 20x more energy-efficient with 24-29pp accuracy gap.

most practical option. can be implemented with existing infrastructure. this is the experiment to run first.

---

## energy accounting for combined techniques

### correct energy model

from Yang et al. (2024) and Dampfhoffer et al. (2023):

```
E_total = T * [E_synops + E_LIF + E_mem]

where:
  E_synops = N_active_synapses * r * E_AC     (spike-driven computation)
  E_LIF = N_neurons * E_update                (membrane update, every timestep)
  E_mem = N_params * E_read + N_states * E_write   (memory access)

  E_AC = 0.9 pJ (accumulate on neuromorphic HW) or 0.03 pJ (8-bit ADD)
  E_MAC = 4.6 pJ (ANN multiply-accumulate)
  E_SRAM = 20 pJ/bit
  E_DRAM = 2 nJ/bit
```

### how each technique affects the formula

| technique | reduces T | reduces r | reduces N_synapses | reduces E_per_op | reduces E_mem |
|-----------|-----------|-----------|-------------------|------------------|---------------|
| early exit | yes (main) | no | no | no | no |
| spike reg | no | yes (main) | no | no | no |
| pruning | no | indirectly | yes (main) | no | yes |
| quantization | no | no | no | yes (main) | yes |
| silence gating | no | yes (input) | no | no | no |
| depthwise arch | no | no | yes (by design) | no | no |
| rhythm oscillatory | possibly | yes (off neurons) | no | no | no |
| hierarchical cascade | yes (exit at coarse) | no | yes (smaller per-stage) | no | no |

### double-counting pitfalls

**pitfall 1: spike reg + pruning overlap**
pruning removes connections which means pruned neurons might fire less. if you prune 90% of connections, some spike reduction comes for free. claiming 4.4x from spike reg AND 10x from pruning when they overlap is wrong. correct approach: measure spike rate AFTER pruning. if pruning alone drops spike rate to 15%, then adding spike reg to reach 6% is only 2.5x further, not 4.4x.

**pitfall 2: early exit + temporal pruning**
both reduce timesteps. if you train with temporal pruning (T 25->10) AND add early exit (T 10->5 average), you get 5x total, not 2.5x * 2x. same thing, no double-counting, but the framing matters.

**pitfall 3: LIF overhead is NOT reduced by pruning or spike reg**
every neuron updates its membrane potential every timestep regardless of spiking activity. this is a FIXED COST proportional to N_neurons * T. only reducing T (early exit) or N_neurons (neuron pruning, not just synapse pruning) reduces this. Yang et al. (2024) show LIF overhead can be 50-80% of total energy on neuromorphic hardware. so spike reg alone may only reduce 20-50% of total energy (the spike-driven part), not the full energy. this is a big deal.

**pitfall 4: memory access dominates on digital hardware**
on GPUs/CPUs, memory access is 10-100x more expensive than computation. unstructured pruning doesnt help if the sparse matrix still requires full memory reads. only STRUCTURED pruning (removing entire channels/neurons) gives real memory savings. quantization helps by reducing bits per parameter.

### how to present combined results

recommended approach (from NeuroBench and SPARQ):
1. report NeuroBench metrics (Effective ACs, Effective MACs) for fair comparison
2. report TOTAL energy including memory access
3. use ablation table showing each technique added incrementally
4. report accuracy alongside energy at each step

example table:

| config | accuracy | avg T | spike rate | SOPs | energy (nJ) | reduction |
|--------|----------|-------|------------|------|-------------|-----------|
| baseline (T=25) | 61.10% | 25 | 26.4% | 1.08M | 968 | 1x |
| + spike reg (6%) | ~58% | 25 | 6.0% | 245K | ~220 | ~4.4x |
| + early exit | ~57% | 7 | 6.0% | 68K | ~63 | ~15x |
| + 70% pruning | ~55% | 7 | 5.0% | 20K | ~22 | ~44x |
| + INT8 quant | ~54% | 7 | 5.0% | 20K | ~13 | ~74x |

(numbers are estimated -- actual values require experiments)

---

## interaction effects and unexpected results

### cases where combining gave BETTER accuracy

energy-aware spike budgeting (2025): spike budgeting on MNIST gave +2.31pp accuracy AND 47% spike reduction. mechanism: implicit regularization -- forcing sparsity prevents overfitting. the network was over-spiking and constraining it improved generalization.

QP-SNN (ICLR 2025): combined ReScaW + SVS: 73.89% vs baseline 69.16% -- improvement of 4.73pp. quant + pruning together was better than either alone. weight rescaling for quantization also improved pruning quality.

our own results: 30% pruning gave 41.75% vs unpruned 40.5% -- improvement of 1.25pp. mild pruning acts as regularizer for our small dataset. kind of makes sense.

### cases where combinations were worse than expected

KD + everything: every combo with KD performed worse. root cause: distribution mismatch between ANN teacher and SNN student logits. the teacher's continuous activations mislead the student's binary spike-based learning.

aggressive quant + aggressive pruning (from QP-SNN): at extreme settings (2-bit + 90%+ pruning), accuracy degrades super-linearly. the model just doesnt have enough representational capacity for both constraints. ICLR 2026 confirms: more heavily pruned models are more sensitive to quantization.

spike reg + very low T (theoretical concern): both reduce information throughput. at T=1 with 6% spike rate, each neuron fires ~0.06 times per sample. thats just not enough information for 50 classes.

### non-obvious interactions

rhythm oscillatory + early exit (positive): rhythm creates periodic bursts of high-confidence predictions. early exit can detect these burst peaks and exit during them. the oscillatory modulation CREATES natural exit opportunities. havent tested this but its a nice idea.

pruning + stochastic resonance (positive): our results show SNN is highly pruning-resilient (93.2% retained at 90% pruning). stochastic resonance detected at sigma=0.02 (+0.25pp). noise injection might help pruned models recover from precision loss. untested but there's biological precedent (brain maintains function despite neuron loss + noise).

quantization + ternary spikes (synergistic): standard SNN has binary spikes {0,1} with float weights. ternary spikes {-1,0,+1} give richer info per spike. combine with quantized weights: each synapse is just add/subtract from a small codebook. total computation: table lookup + add/subtract -- no multiplication at all.

---

## the "kitchen sink" question

### has anyone combined ALL techniques?

closest: SPARQ (SNN + quant + early exit) and SpikeFit (quant + pruning + deployment optimization). nobody has combined ALL of: early exit + spike reg + pruning + quant + architecture optimization + input sparsity + hierarchical classification.

ADMM SNN (2021) combines three (pruning + quant + activity reg) and notes they can be applied "in either a single way for moderate compression or a joint way for aggressive compression."

### diminishing returns curve

based on published results:

| # techniques | typical combined reduction | incremental benefit of next |
|-------------|--------------------------|----------------------------|
| 1 | 3-5x | -- |
| 2 | 10-20x | 3-5x |
| 3 | 30-100x | 3-5x |
| 4 | 50-200x | 1.5-2x |
| 5 | 80-330x | 1.2-1.5x |
| 6+ | 100-500x | <1.3x (diminishing) |

the sweet spot is 3-4 techniques. after that each additional technique gives diminshing returns because:
1. techniques start to overlap (spike reg + pruning both reduce activity)
2. accuracy degrades faster than energy improves
3. implementation complexity grows super-linearly

### theoretical minimum energy for 50-class classifier

Landauer limit: kB*T*ln(2) = 2.9e-21 J at room temperature per bit erasure.

50 classes needs log2(50) = 5.64 bits of information.

absolute thermodynamic minimum: 5.64 * 2.9e-21 J = 1.6e-20 J = 0.016 fJ per classification. we're using 968 nJ. so we're about 60 trillion times above the Landauer limit. plenty of room to optimize.

practical minimum with our architecture:
- after 95% pruning: ~31K params remain
- at 0.9 pJ per AC: 31K * 0.9 pJ = 28 nJ per timestep
- at T=1: 28 nJ total + memory overhead
- realistic min: ~5-10 nJ (limited by memory access)

with a fly-brain circuit:
- 50 output neurons, ~50K random connections (binary)
- at 0.9 pJ per AC: ~50K * 0.05 (5% WTA) * 0.9 pJ = 2.25 nJ
- single timestep: ~2-5 nJ total

---

## experimental plan

### priority 1: practical combo (run first)
reduced-T training (T=5) + spike regularization (L1 targeting 6%) + 70% structured pruning
- expected: ~22 nJ/sample at 35-40% accuracy
- implementation: modify training script for T=5, add layer-wise L1 loss, prune post-training
- probably 1-2 days training on CSF3

### priority 2: tier 1 combo with rhythm
rhythm oscillatory (T=25) + early exit (avg T=7) + spike regularization
- expected: ~42 nJ/sample at 56-59% accuracy
- implementation: train rhythm model with spike reg, add confidence-based early exit post-training
- probably 2-3 days (rhythm training + early exit calibration)

### priority 3: tier 2 with post-training compression
take best model from priority 2, apply SBC (Spiking Brain Compression)
- expected: ~10-15 nJ at 53-57% accuracy
- implementation: apply SBC (one-shot, no retraining) for 90% pruning + INT8
- 1 day (SBC is post-training)

### priority 4: fly brain experiment
random projection + WTA on PANNs embeddings
- expected: unknown accuracy (possibly 70-85%), ~2-5 nJ
- implementation: fixed random sparse matrix, WTA inhibition, linear readout
- half a day (simple architecture)

---

## references table

| paper | techniques combined | max reduction | accuracy | year | venue |
|-------|-------------------|---------------|----------|------|-------|
| SPARQ | SNN + quant + early exit | 330x energy | 78% (CIFAR-10) | 2026 | arXiv |
| QP-SNN | quant + structured pruning | 98.74% model, 78.69% SOPs | -2.44% | 2025 | ICLR |
| SpQuant-SNN | ternary MP + quant + pruning | 13x mem, 4.7x FLOPs | -1.8% | 2024 | Frontiers |
| SBC | post-training prune + quant | 97% sparsity | <=1% loss | 2025 | NeurIPS |
| ADMM SNN | prune + quant + activity reg | "aggressive" | validated | 2021 | TNNLS |
| NeuEdge | encoding + adaptive thresh + HW | 312x (vs GPU) | 91-96% | 2026 | arXiv |
| spike budgeting | replay + learnable LIF + scheduler | 47% spikes, +2.31% acc | pareto | 2025 | arXiv |
| dynamic ST pruning | LAMPS spatial + temporal | 91x energy | -2.19% | 2025 | Frontiers |
| Spike-Thrift | attention + compression | 33.4x compress, 12.2x energy | minimal | 2021 | WACV |
| SEENN | RL early exit | 5.5x fewer timesteps | 96.07% (CIFAR-10) | 2023 | NeurIPS |
| CPT-SNN | confidence + previous timestep | T=2.72 avg | 95.44% | 2025 | Neurocomp |
| SpikeFit | CAT quant + FSC pruning | M=4 weights | SOTA | 2025 | EurIPS |
| GETA | joint pruning + quantization | varied | competitive | 2025 | CVPR |
| LitE-SNN | NAS + spatial + temporal compress | competitive | competitive | 2024 | IJCAI |
| 0.3 spikes/neuron | TTFS + L1 reg + quant | extreme sparsity | matches ANN | 2024 | Nature Comms |
| PQ order (ICLR 2026) | compression order analysis | up to 49.9pp diff | varies | 2026 | ICLR |
| Rhythm-SNN | oscillatory modulation | 100x+ (audio denoising) | SOTA | 2025 | Nature Comms |
| QUEST | quant-aware + energy-aware + device | 93x vs ANN | 89.6% | 2025 | arXiv |
| MONAS-ESNN | multi-obj NAS for accuracy+spikes | pareto-optimal | SOTA | 2025 | WACV |

---

## confidence levels

| finding | confidence | basis |
|---------|------------|-------|
| 3-4 techniques combine near-multiplicatively | high | multiple papers, physics supports orthogonality |
| 100x combined reduction is achievable | high | SPARQ 330x, NeuEdge 312x (with HW), dynamic pruning 91x |
| accuracy loss is 3-8% for first 3 techniques | medium | dataset/arch dependent, limited audio SNN data |
| KD hurts in our pipeline | high | our experiments + literature mismatch explanation |
| 1000x at useful accuracy (>30%) | low | no published evidence for small models on complex tasks |
| fly brain gets >30% on ESC-50 | medium | theoretical match (50 classes ~ 50 odors) but untested |
| rhythm + early exit synergistic | medium | theoretical argument but untested |
| order matters (quantize before prune) | high | ICLR 2026 with theoretical guarantees |

---

## research gaps

1. no paper combines >3 SNN-specific techniques with measured multiplicative gains on a single model. max is SPARQ with 3. a 4+ technique paper would be novel.

2. no combined optimization paper for audio SNNs at all. everything is on vision (CIFAR, ImageNet, DVS). ESC-50 is completley uncharted.

3. rhythm oscillatory + compression is unexplored. the Rhythm-SNN Nature Comms paper focuses on accuracy and robustness, not compression.

4. no systematic study of interaction effects between >2 SNN techniques. two-way interactions are studied. three-way and higher are unknown.

5. no energy measurements of combined techniques on SpiNNaker. all reported results are simulation or Loihi/TrueNorth.

---

## bottom line

for the thesis, implement the practical combo: train rhythm model at T=5 with spike reg targeting 6%, post-training 70% structured pruning + fine-tune, confidence-based early exit, measure energy with NeuroBench.

expected: ~20-30 nJ/sample at 35-45% accuracy. that flips the narrative from "SNN is 2.1x WORSE" to "SNN is 15-23x BETTER" with a documented accuracy tradeoff.

for the ICONS paper, report as a pareto curve: show the accuracy-energy tradeoff across all combinations. the story is: "combined optimization provides a continuum from 61.10% accuracy at 968 nJ to 35% accuracy at 20 nJ."
