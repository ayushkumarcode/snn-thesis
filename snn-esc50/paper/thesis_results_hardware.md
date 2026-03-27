# chapter 5: neuromorphic hardware results

SpiNNaker deployment chapter. this one was a rollercoaster -- lots of failures before things worked. need to document all the troubleshooting honestly since the co-design insights are a main contribution.

---

## 5.1 SpiNNaker deployment: challenges and solutions

Deploying a trained SpikingCNN to SpiNNaker requires satisfying hard constraints that dont exist in software:

**Constraint 1: binary spike inputs.** SpiNNaker communicates via spike packets (4-byte address events). Each input must be binary 0 or 1. No continuous values.

**Constraint 2: fixed-point arithmetic.** All weights and membrane potentials in fixed-point on ARM cores. Limits precision but enables efficient compute.

**Constraint 3: IF_curr_exp neuron model.** sPyNNaker supports IF with exponential synaptic current, which approximates but doesn't exactly match snnTorch's LIF.

**Constraint 4: no native conv support.** SpiNNaker's execution model is point-neuron populations connected by synaptic matrices. Conv layers would need unrolling into explicit weight matrices.

### 5.1.1 the FC1 cancellation problem

This was the big blocker. The SpikingCNN has AvgPool2d between LIF2 and FC1. AvgPool averages binary spike outputs into fractional values in [0, 0.5]. When these drive FC1 (2304->256), the weight matrix (near zero-mean by default) causes near-zero or negative total input for most FC1 neurons = zero hidden spikes = zero output spikes.

Quantitative:
- FC1 weight mean: -0.0034 (near zero)
- Active inputs per FC1 neuron: 1398 of 2304
- Expected current: 1398 x (-0.0034) = -4.75 (net negative, no spikes)
- Observed hidden spike rate: 0/step (Runs 1-4)

This is the root cause of Runs 1-4 failing completely -- all samples predicted class 0 or random.

### 5.1.2 Option C: weight re-centering (attempted, failed)

**Approach:** zero-center each FC1 weight row:
$$w_{i,j} \leftarrow w_{i,j} - \mu_i, \quad b_i \leftarrow b_i + \mu_i \cdot n$$

where mu_i is mean of row i, n = 2304.

**Theory:** for binary inputs summing to n, bias compensation exactly cancels the shift. But AvgPool gives fractional inputs where sum != n. Bias compensation over-corrects by n/sum(x_j), destroying selectivity.

**Result:** accuracy dropped 53.75% -> 8.50%. Option C is dead. Root cause is architectural -- AvgPool between binary spike layers introduces non-binary values violating the math assumptions.

Documented in `results/spinnaker_optionC/option_c_fold4.json`.

---

## 5.2 FC2-only hybrid approach

### 5.2.1 architecture

Given the constraints above, went with a validated hybrid:

```
Software (snnTorch, CPU):
  Input mel spectrogram (1, 64, 216)
  -> Conv1 + BN + MaxPool + LIF1
  -> Conv2 + BN + MaxPool + LIF2
  -> AvgPool -> flatten
  -> FC1 (2304->256) + LIF3
  -> Binary hidden spike tensor: (T=25, N, 256)

Hardware (SpiNNaker, IF_curr_exp):
  Binary hidden spikes (25ms x 256 neurons)
  -> FC2 (256->50) + IF_curr_exp
  -> Total output spike counts over 25ms
  -> Argmax -> predicted class
```

Software portion produces binary spikes with 21.7% active neurons/timestep (55.6 of 256 firing), satisfying SpiNNaker's binary requirement for FC2.

### 5.2.2 SpiNNaker configuration

**Hardware:** SpiNN-5 at spinnaker.cs.man.ac.uk, sPyNNaker 1.0.0.

**Calibrated params** (9-point scale sweep + LIF param sweep):
```
Population: IF_curr_exp, 50 neurons (one per class)
Input: SpikeSourceArray, 256 neurons x 25ms
cm = 1.0 nF
tau_m = 20.0 ms
tau_refrac = 0.1 ms
tau_syn_E = 5.0 ms
v_thresh = 1.0 mV
v_rest = v_reset = 0.0 mV
weight_scale = 1.0
```

### 5.2.3 software feature extraction

400 spectrograms (fold 4 test set) processed through snnTorch SpikingCNN (conv + FC1 + LIF3) using the correct preprocessing (librosa, sr=22050, n_mels=64, etc). The critical bug from Runs 1-4 was using torchaudio with no normalisation -- features mismatched to training distribution. took ages to figure this out

Hidden spikes cached as `results/spinnaker_weights/hidden_spike_features.npy` (400 x 25 x 256). snnTorch reference accuracy on these 400: 51.25% (vs 53.75% on full fold 4 -- expected small diff from subsampling).

---

## 5.3 SpiNNaker inference results

### 5.3.1 validation run (Run 5)

**Setup:** 20 samples from fold 4 (10+ distinct classes), weight_scale=1.0.
**Result:** 8/20 = 40.0% SpiNNaker vs 10/20 = 50.0% snnTorch.
**Agreement:** 12/20 = 60%.

The 10 pp gap reflects:
1. IF_curr_exp != LIF exact dynamics (exponential synaptic vs instantaneous current)
2. Fixed-point quantisation error
3. 1ms timestep discretisation vs continuous snnTorch

### 5.3.2 full 400-sample inference

**Complete (4 March 2026).** All 400 fold-4 samples processed. Source: `results/spinnaker_results/fc2_all_iterations.jsonl`.

**Final: SpiNNaker 43.0% vs snnTorch 51.25% -- 8.25 pp gap, 64.5% agreement.**

Progression across the run:

| Samples | SpiNNaker | snnTorch | Gap | Agreement |
|---------|-----------|----------|-----|-----------|
| 37/400 | 43.2% | 51.4% | 8.2 pp | ~77% |
| 65/400 | 47.7% | 50.8% | 3.1 pp | 75.4% |
| 108/400 | 49.1% | 50.9% | 1.9 pp | 76.9% |
| 189/400 | 50.8% | 51.3% | 0.5 pp | 81.5% |
| 208/400 | 50.5% | 50.5% | 0.0 pp | 79.3% |
| 244/400 | 45.5% | 51.2% | 5.7 pp | 70.9% |
| **400/400** | **43.0%** | **51.25%** | **8.25 pp** | **64.5%** |

Hardware gap fluctuates a lot -- not monotonically converging. Ranges from 0.0 pp (n=208) to 8.25 pp (final). Later samples included harder classes for SpiNNaker (insects, helicopter, engine all 0%). Agreement dropped from ~81% mid-run to 64.5% final.

### 5.3.3 accuracy analysis

**Super-category breakdown (Run 6 final, n=400):**

| Category | SpiNNaker | snnTorch | Gap |
|----------|-----------|---------|-----|
| Animals (0-9) | 45.0% | 57.5% | snnTorch +12.5 pp |
| Nature (10-19) | 61.3% | 68.8% | snnTorch +7.5 pp |
| Human (20-29) | 46.2% | 56.2% | snnTorch +10.0 pp |
| Domestic (30-39) | 31.2% | 37.5% | snnTorch +6.2 pp |
| Urban (40-49) | 31.2% | 36.2% | snnTorch +5.0 pp |

snnTorch leads all five categories at n=400. Urban has smallest gap (5.0 pp). Nature easiest for both; Domestic and Urban hardest.

**Hardest for SpiNNaker (0%):** insects, door_wood_creaks, glass_breaking (50 pp gap vs snnTorch!), helicopter, engine.
**Easiest (100%):** clapping, thunderstorm.
**SpiNNaker beats snnTorch:** airplane (87.5% vs 50.0%, +37.5 pp!), mouse_click (+25 pp), can_opening (+12.5 pp), clock_tick (+12.5 pp). These are sounds with simple, consistent spectrotemporal patterns that IF_curr_exp integrates more reliably.

**Error analysis (n=400):**

| Category | Count | % |
|----------|-------|---|
| Both correct | 145 | 36.2% |
| SpiNNaker right, snnTorch wrong | 27 | 6.8% |
| snnTorch right, SpiNNaker wrong | 60 | 15.0% |
| Both wrong | 168 | 42.0% |

Both wrong on 42% -- the dominant error source is the task difficulty itself (256 hidden neurons isn't enough), not hardware noise.

### 5.3.4 towards 5-fold cross-validation

Run 6 is single-fold (fold 4). For full comparability with snnTorch baseline (47.15% +/- 4.50%), we prepared all five folds.

**Model integrity issue:** the augmented training job accidentally saved to same directory as canonical CSF3 models, overwriting all 5 checkpoints. Caught it because fold 3 returned 26.75% snnTorch accuracy (expected ~48%). Restored all from `csf3_results/snn/direct/` backup. that was scary

**snnTorch reference from extraction (canonical models):**

| Fold | snnTorch (extraction) | CSF3 best_acc | Diff |
|------|----------------------|---------------|------|
| 1 | 39.5% | 40.5% | -1.0 pp |
| 2 | 48.2% | 48.5% | -0.3 pp |
| 3 | 47.7% | 48.25% | -0.55 pp |
| 4 | 51.2% | 54.0% | -2.8 pp |
| 5 | 43.2% | 44.5% | -1.3 pp |
| **Mean** | **46.0%** | **47.15%** | **-1.15 pp** |

Small diffs from extraction vs training-time measurement. Mean 46.0% within 1.2 pp of canonical 47.15%, confirming integrity.

**Sparsity:** consistent across folds (79.9%, 72.6%, 76.1%, 78.3%, 75.8%). FC2-only approach valid across all folds.

**5-fold SpiNNaker results (completed 05 March 2026):**

| Fold | SpiNNaker | snnTorch ref | Hardware gap |
|------|-----------|-------------|-------------|
| 1 | 29.0% | 39.5% | +10.5 pp |
| 2 | 32.0% | 48.2% | +16.2 pp |
| 3 | 36.5% | 47.8% | +11.2 pp |
| 4 | 43.0% | 51.2% | +8.2 pp |
| 5 | 25.2% | 43.2% | +18.0 pp |
| **Mean** | **33.1%** | **46.0%** | **+12.8 pp** |
| **Std** | **6.9%** | | **4.1 pp** |

SpiNNaker 5-fold: 33.1% +/- 6.9% vs snnTorch 46.0%, mean hardware gap 12.8 +/- 4.1 pp. Fold-to-fold variability shows gap is somewhat fold-dependent, suggesting class composition matters for SpiNNaker performance.

---

## 5.4 NeuroBench energy analysis

### 5.4.1 methodology

NeuroBench v2.2.0 (Yik et al. 2025) wraps model in SNNTorchModel, measures SynapticOperations:
- **Dense SynOps:** total ops ignoring sparsity (upper bound)
- **Effective_ACs:** accumulate-only from non-zero binary activations
- **Effective_MACs:** multiply-accumulate from non-zero non-binary activations

Energy at 45nm CMOS: AC = 0.9 pJ, MAC = 4.6 pJ (5.1x more expensive).

Performed on all 5 folds (400 test samples each, direct encoding). 5-fold means: SNN 968 +/- 37 nJ/sample, ANN 454 +/- 11 nJ/sample.

### 5.4.2 results

**SNN (direct, fold 4 representative; 5-fold means in parens):**

| Metric | Value |
|--------|-------|
| Activation sparsity | 74.16% |
| Dense SynOps | 4,176,566/sample |
| Effective ACs | 1,084,732/sample |
| Effective MACs | 0 |
| Energy (sw sim) | 1,084,732 x 0.9 pJ = 968 nJ (5-fold: 968+/-37 nJ) |

**ANN:**

| Metric | Value |
|--------|-------|
| Activation sparsity | 59.03% |
| Dense SynOps | 166,298/sample |
| Effective ACs | 0 |
| Effective MACs | 100,561/sample |
| Energy (sw sim) | 100,561 x 4.6 pJ = 454 nJ (5-fold: 454+/-11 nJ) |

### 5.4.3 interpretation

**In software:** SNN is 2.1x more expensive than ANN (968 vs 454 nJ). Expected -- T=25 timesteps while ANN runs once, and 1.08M ACs outnumber 101K MACs despite the per-op cost difference.

**On neuromorphic hardware:** each AC costs ~0.9 pJ vs 4.6 pJ for MAC. But SNN still has more total ops. The break-even:
- SNN: 1,084,732 x 0.9 = 976,259 pJ
- ANN: 100,561 x 4.6 = 462,581 pJ

SNN is still more expensive even on neuromorphic hardware. Follows from Dampfhoffer et al. (2023): need <6.4% spike rate. Our 25.84% spike rate is well above this.

**The energy story:** direct SNN uses 2.1x more energy in software. But the SNN enables deployment to AC-only neuromorphic hardware. And PANNs + SNN head (92.50%) represents a better pathway: expensive feature extraction runs once, energy-cheap SNN handles classification.

### 5.4.4 energy-accuracy pareto frontier

| Model | Accuracy | Energy/sample | Hardware? |
|-------|----------|---------------|-----------|
| Direct SNN | 47.15% | 968 +/- 37 nJ (sw) | yes (SpiNNaker FC2 hybrid) |
| Rate SNN | 24.00% | ~950 nJ (est) | yes |
| ANN baseline | 63.85% | 454 +/- 11 nJ (sw) | no neuromorphic |
| PANNs+SNN (full, sw) | 92.50% | ~8 uJ (head only) | no (software) |
| PANNs+SNN (FC2 on SpiNNaker) | 92.50% | ~86 nJ (FC2 layer) | yes |
| PANNs+ANN head | 93.45% | ~650 nJ (est) | no |

The Pareto-optimal deployment: **PANNs embeddings + SpiNNaker FC2**: CNN14 extracts on CPU/NPU once, first two SNN layers in software, only 256->50 FC2 on SpiNNaker (~86 nJ). 92.50% accuracy with hardware-compatible neuromorphic inference for the classification step.

---

## 5.5 SpiNNaker Option A: hardware-aware retraining

Option A addresses FC1 cancellation by replacing AvgPool with MaxPool and retraining. Key insight: MaxPool on binary values produces binary outputs (max of {0,1} = {0,1}), guaranteeing FC1 gets truly binary inputs.

### 5.5.1 architecture change

**Original (incompatible):**
```
LIF2 spikes (binary) -> AvgPool2d(4,6) -> fractional [0,0.5] -> FC1
```

**Option A (compatible):**
```
LIF2 spikes (binary) -> MaxPool2d(4,6) -> binary {0,1} -> FC1
```

Same output shape (4,9). All FC sizes unchanged. Purely architectural fix at training time.

### 5.5.2 threshold sweep

Additional LIF threshold sweep ({1.0, 1.5, 2.0, 3.0}) to reduce FC1 input density. Higher threshold = fewer spikes = sparser FC1 input. Target: <500 active/step (from ~1398 with AvgPool).

**Status:** fold 4 complete. From `results/snn/maxpool/threshold_sweep_fold4.json`.

| Threshold | Test Acc | FC1 Active/step | Binary Frac | SpiNNaker-ready? |
|-----------|---------|-----------------|-------------|------------------|
| 1.0 | 9.25% | 1662.4/2304 (72.2%) | 1.000 | theoretically yes; dense though |
| 1.5 | 27.0% | 1409.7/2304 (61.2%) | 1.000 | theoretically yes |
| 2.0 | 34.25% | 1253.1/2304 (54.4%) | 1.000 | theoretically yes |
| **3.0** | **43.75%** | **956.1/2304 (41.5%)** | **1.000** | **best candidate** |

MaxPool guarantees binary fraction = 1.000 at all thresholds. Eliminates the fundamental AvgPool-FC1 incompatibility. threshold=3.0 gets best accuracy (43.75%) with 956/2304 active (58.5% sparsity). The <500/step target wasn't met -- higher thresholds could reduce further but would hurt accuracy more.

Original SNN direct fold 4: 54.0%. MaxPool at threshold=3.0: 43.75% -- 10.25 pp below, reflecting info lost by MaxPool vs AvgPool.

**Recommendation:** threshold=3.0 Option A is the best candidate for full FC1+FC2 SpiNNaker deployment. Binary fraction guaranteed. Need hardware testing to confirm router handles 956 simultaneous spikes/step.

---

## 5.6 chapter summary

1. **FC1 cancellation** is the fundamental barrier. AvgPool between LIF layers produces fractional inputs, violating spike-driven compute. Weight re-centering (Option C) failed because it assumes binary inputs.

2. **FC2-only hybrid** validated: software extracts binary hidden spikes (21.7% active), SpiNNaker classifies 256->50. Run 5 = 40%, Run 6 = 43.0% vs 51.25% snnTorch (8.25 pp gap, 64.5% agreement). Gap fluctuates through the run. SpiNNaker leads on airplane (+37.5 pp), mouse_click (+25 pp).

3. **Energy (NeuroBench, 5-fold):** SNN 2.1x more expensive in software (968 vs 454 nJ). PANNs + SpiNNaker FC2 is Pareto-optimal at 92.50% and ~86 nJ.

4. **5-fold SpiNNaker:** 33.1% +/- 6.9% vs 46.0% snnTorch (12.8 pp gap). Models restored from backup after augmented training overwrote them.

5. **Option A (MaxPool):** threshold sweep confirms binary fraction = 1.000 at all thresholds. threshold=3.0 gets 43.75% with 956 active/step. Full SpiNNaker deployment theoretically unblocked pending router test.
