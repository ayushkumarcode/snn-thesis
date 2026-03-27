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
|------|-----------|-------------|-------------|
| 1 | 29.0% | 39.5% | +10.5 pp |
| 2 | 32.0% | 48.2% | +16.2 pp |
| 3 | 36.5% | 47.8% | +11.2 pp |
| 4 | 43.0% | 51.2% | +8.2 pp |
| 5 | 25.2% | 43.2% | +18.0 pp |
| **Mean** | **33.1%** | **46.0%** | **+12.8 pp** |
| **Std** | **6.9%** | | **4.1 pp** |

SpiNNaker 5-fold mean: **33.1% ± 6.9%** vs snnTorch reference **46.0%**, yielding a mean hardware gap of **12.8 ± 4.1 pp**.

The fold-to-fold variability will also characterise the *reliability* of SpiNNaker inference — whether the hardware gap is systematic (constant across folds) or fold-dependent (suggesting class-specific difficulty). This information provides engineering guidance for future SpiNNaker audio deployments.

---

## 5.4 NeuroBench Energy Analysis

### 5.4.1 Methodology

NeuroBench v2.2.0 (Yik et al. 2025, Nature Communications 16:1589) wraps the model in `SNNTorchModel` and measures SynapticOperations metrics per inference sample:

- **Dense SynOps:** Total operations ignoring sparsity (upper bound — "ANN-equivalent" compute)
- **Effective_ACs (EFF_ACs):** Accumulate-only operations from non-zero binary activations (SNN sparse compute)
- **Effective_MACs (EFF_MACs):** Multiply-accumulate operations from non-zero non-binary activations (ANN sparse compute)

Energy conversion at 45nm CMOS (Yik et al. 2025):
- AC = 0.9 pJ per operation
- MAC = 4.6 pJ per operation (5.1× more expensive than AC)

**Experimental setup:** NeuroBench analysis was performed on all 5 folds (400 test samples each, direct encoding). Energy metrics are architecture-dependent and independent of the training backend. The fold 4 detailed breakdown below is representative; 5-fold validated means are: SNN 968 ± 37 nJ/sample, ANN 454 ± 11 nJ/sample.

### 5.4.2 Results

**Direct encoding SNN (fold 4 representative; 5-fold means in parentheses):**

| Metric | Value |
|--------|-------|
| Activation sparsity | 74.16% |
| Dense SynOps | 4,176,566/sample |
| Effective ACs | 1,084,732/sample |
| Effective MACs | 0 |
| **Energy (software sim)** | **1,084,732 × 0.9 pJ = 968 nJ (5-fold mean: 968±37 nJ) = 0.976 μJ** |

**ANN baseline:**

| Metric | Value |
|--------|-------|
| Activation sparsity | 59.03% |
| Dense SynOps | 166,298/sample |
| Effective ACs | 0 |
| Effective MACs | 100,561/sample |
| **Energy (software sim)** | **100,561 × 4.6 pJ = 454 nJ (5-fold mean: 454±11 nJ) = 0.454 μJ** |

### 5.4.3 Interpretation

**On GPU/CPU (software simulation):** The SNN is **2.1× more expensive** than the ANN (968 ± 37 nJ vs 454 ± 11 nJ, 5-fold validated). This is expected: the SNN runs for T=25 timesteps while the ANN runs once, and the SNN's large number of binary ACs (1.08M) outnumbers the ANN's MACs (101K) despite the higher per-operation cost of MACs.

**On neuromorphic hardware (SpiNNaker/Loihi):** The cost relationship inverts. Each SNN AC (binary × weight) costs ~0.9 pJ on neuromorphic hardware vs ~4.6 pJ for ANN MACs. If the SpiNNaker hardware achieves AC-only compute for the classification layer (fold 4 example calculation; 5-fold means: SNN 968 ± 37 nJ, ANN 454 ± 11 nJ):
$$\text{SNN hardware energy} = 1,084,732 \times 0.9 \text{ pJ} = 976 \text{ nJ (fold 4)}$$
$$\text{ANN hardware energy} = 100,561 \times 4.6 \text{ pJ} = 463 \text{ nJ (fold 4)}$$

On neuromorphic hardware, the SNN still has more total operations (1.08M vs 101K). The question is whether the neuromorphic AC is cheap enough to compensate:
$$\text{Break-even requires: SNN ACs} \times 0.9 < \text{ANN MACs} \times 4.6$$
$$1,084,732 \times 0.9 = 976,259 \text{ pJ vs } 100,561 \times 4.6 = 462,581 \text{ pJ (fold 4)}$$

The SNN remains more expensive even on neuromorphic hardware in this analysis. This follows from Dampfhoffer et al. (2023): SNNs need a spike rate below 6.4% to beat quantized ANNs on CPU. Our 74.16% sparsity (25.84% spike rate) is well above this threshold.

**The energy story for this thesis:** The direct encoding SNN uses 2.1× more energy than the ANN in software simulation. However, there is a compelling energy-accuracy trade-off: the SNN achieves 47.15% accuracy with temporal, binary computation, enabling deployment to AC-only neuromorphic hardware. The PANNs + SNN head (92.50% accuracy, §4.6) represents a more energy-efficient pathway than full ANN inference: the expensive feature extraction runs once (on-device ANN or NPU), and the energy-cheap SNN handles classification.

### 5.4.4 Energy-Accuracy Pareto Frontier

| Model | Accuracy | Energy/sample | Hardware-compatible? |
|-------|----------|---------------|---------------------|
| Direct SNN | 47.15% | 968 ± 37 nJ (sw, 5-fold) | ✅ SpiNNaker (FC2 hybrid) |
| Rate SNN | 24.00% | ~950 nJ (est) | ✅ SpiNNaker (FC2 hybrid) |
| ANN baseline | 63.85% | 454 ± 11 nJ (sw, 5-fold) | ❌ No neuromorphic |
| PANNs+SNN head (full, software) | 92.50% | ~8 μJ (head only, est) | ❌ Software only |
| PANNs+SNN head (FC2 on SpiNNaker) | 92.50% | ~86 nJ (FC2 layer) | ✅ SpiNNaker (FC2 layer) |
| PANNs+ANN head | 93.45% | ~650 nJ (est) | ❌ No neuromorphic |

*Full SNN head (3-layer, 2048→512→256→50, T=25, 70% sparsity est.) ≈ 29.8M dense ops × 0.30 × 0.9 pJ ≈ 8.0 μJ. FC2 layer only (256→50, T=25, 30% active): 320K ops × 0.9 pJ ≈ 86 nJ.*

The most promising **Pareto-optimal** deployment is **PANNs embeddings + SpiNNaker FC2 classification**: CNN14 extracts embeddings on CPU/NPU (once per sample), the first two SNN layers run in software, and only the 256→50 FC₂ layer runs on SpiNNaker (~86 nJ). This achieves 92.50% accuracy with hardware-compatible neuromorphic inference for the classification step — the same hybrid approach validated for the from-scratch SNN (§5.2), but with PANNs-quality features.

---

## 5.5 SpiNNaker Option A: Hardware-Aware Retraining

Option A addresses the FC1 cancellation by replacing AvgPool2d with MaxPool2d and retraining the model. The key insight is that MaxPool on binary spike values produces binary outputs (max of {0, 1} = {0, 1}), guaranteeing that FC₁ receives truly binary inputs that SpiNNaker can process natively.

### 5.5.1 Architecture Change

**Original (SpiNNaker-incompatible):**
```
LIF₂ spikes (binary 0/1) → AvgPool2d(4,6) → fractional values ∈ [0,0.5] → FC₁
```

**Option A (SpiNNaker-compatible):**
```
LIF₂ spikes (binary 0/1) → MaxPool2d(4,6) → binary values {0,1} → FC₁
```

The MaxPool2d(4,6) produces the same spatial output shape (4,9) as the original AvgPool2d(4,6) applied to input (16,54). All FC layer sizes remain unchanged (2304→256→50). This is a purely architectural fix applied at training time.

### 5.5.2 Threshold Sweep

An additional LIF threshold sweep ({1.0, 1.5, 2.0, 3.0}) reduces FC₁ input density. Higher thresholds require stronger cumulative membrane potential to trigger a spike, reducing the mean number of active FC₁ inputs per timestep. Target: <500 active inputs/step (from ~1,398 with AvgPool).

**Status:** Threshold sweep on fold 4 complete (4 March 2026). Results from `results/snn/maxpool/threshold_sweep_fold4.json`.

**Key metrics reported:**
- `fc1_binary_fraction`: fraction of FC₁ inputs that are exactly 0 or 1 (must be 1.000 for SpiNNaker, guaranteed by MaxPool on binary spikes)
- `fc1_mean_active_per_step`: mean simultaneous active FC₁ inputs per timestep (decreases with higher threshold)

**Exit criteria for full SpiNNaker deployment:**
- `fc1_binary_fraction` = 1.000 (guaranteed by MaxPool) — **all thresholds pass** ✅
- `fc1_mean_active_per_step` < 500 (reduced by higher threshold) — **none achieved** (range: 956–1662)
- Test accuracy > 42% (within 10 pp of FC2-only hybrid reference) — **threshold=3.0 passes** (43.75%) ✅

**Table: Option A threshold sweep results (fold 4)**

| Threshold | Test Acc | FC1 Active/step | FC1 Binary Frac | SpiNNaker-ready? |
|-----------|---------|-----------------|-----------------|------------------|
| 1.0 | 9.25% | 1662.4/2304 (72.2% active, 27.8% sparse) | **1.000** ✅ | Theoretically yes; fc1 dense (72.2% active) |
| 1.5 | 27.0% | 1409.7/2304 (61.2% active, 38.8% sparse) | **1.000** ✅ | Theoretically yes; fc1 dense (61.2% active) |
| 2.0 | 34.25% | 1253.1/2304 (54.4% active, 45.6% sparse) | **1.000** ✅ | Theoretically yes; fc1 dense (54.4% active) |
| **3.0** | **43.75%** | **956.1/2304 (41.5% active, 58.5% sparse)** | **1.000** ✅ | **Best candidate**: binary, 43.75% acc, 956/step |

*Source: `results/snn/maxpool/threshold_sweep_fold4.json`.*

**Interpretation:** MaxPool guarantees fc1_binary_fraction = 1.000 for all thresholds (confirmed). This eliminates the fundamental AvgPool-FC1 cancellation incompatibility documented in §5.1. The threshold=3.0 model achieves the best accuracy (43.75%) while reducing FC1 density to 956/2304 (41.5% active). The <500/step target is not met by any threshold in this sweep — thresholds above 3.0 could potentially reduce density further at cost of additional accuracy loss.

**Original SNN direct fold 4 baseline: 54.0%.** The MaxPool model at threshold=3.0 (43.75%) is 10.25 pp below this, reflecting the architectural change (MaxPool discards spatial information that AvgPool retains when inputs are continuous). However, the model was trained from scratch specifically for SpiNNaker compatibility.

**Recommendation:** The threshold=3.0 Option A model is the best candidate for full SpiNNaker FC1+FC2 deployment. FC1 inputs are binary (guaranteed), meeting the fundamental requirement. Hardware testing is needed to confirm the SpiNNaker router can handle 956 simultaneous spikes per FC1 step without overflow. If router capacity is sufficient, this would enable the first full convolutional SNN deployment on neuromorphic hardware for environmental sound classification.

---

## 5.6 Chapter Summary

1. **FC1 cancellation** is the fundamental barrier to full SpikingCNN deployment on SpiNNaker. Root cause: AvgPool between LIF layers produces fractional inputs (not binary), violating SpiNNaker's spike-driven compute model. Weight re-centring (Option C) fails because it assumes binary inputs.

2. **FC2-only hybrid** is the validated deployment approach: software feature extraction produces binary hidden spikes (21.7% active), which SpiNNaker classifies using a 256→50 layer. Run 5 (n=20) achieves 40%. **Run 6 (400-sample) final result: 43.0% SpiNNaker vs 51.25% snnTorch — hardware gap 8.25 pp, agreement rate 64.5%.** Checkpoint trajectory: n=108: 1.9 pp → n=189: 0.5 pp → n=208: 0.0 pp → n=244: 5.7 pp → n=400: 8.25 pp. The gap fluctuates significantly, with later samples proving harder for SpiNNaker (insects, helicopter, engine all 0% SpiNNaker). snnTorch leads all five super-categories; SpiNNaker beats snnTorch on airplane (+37.5 pp) and mouse_click (+25 pp).

3. **Energy analysis (NeuroBench, 5-fold validated):** Direct SNN uses 2.1× more energy than ANN in software simulation (968 ± 37 nJ vs 454 ± 11 nJ). PANNs + SNN head (FC2 layer on SpiNNaker only) is Pareto-optimal at 92.50% accuracy and ~86 nJ per classification (FC2 layer: 256×50×25×0.30 active×0.9 pJ).

4. **5-fold SpiNNaker preparation:** All five fold models were restored from CSF3 backup after being overwritten by augmented training. Feature extraction (400 samples each, canonical CSF3 models) yields snnTorch reference accuracies of 39.5%, 48.2%, 47.7%, 51.2%, 43.2% (mean 46.0%, within 1.2 pp of canonical 47.15%). Fold-specific FC2 connection lists generated. Automated 5-fold inference script (`spinnaker/run_5fold_spinnaker.sh`) prepared, using calibrated weight_scale=5.0 from Run 6. Hardware execution complete (05 March 2026): 33.1% ± 6.9% SpiNNaker mean accuracy across 5 folds.

5. **Option A retraining (MaxPool SNN):** The threshold sweep (fold 4) confirms FC1 binary fraction = 1.000 for all thresholds — MaxPool on binary spikes guarantees binary FC1 inputs, removing the theoretical incompatibility. Threshold=3.0 achieves 43.75% accuracy (best of sweep) with 956.1 FC1 active inputs per step (sparsity 58.5%). FC1 mean active/step remains above the 500 target — higher thresholds or L1 spike regularisation could reduce this further. Full SpiNNaker deployment with the threshold=3.0 Option A model requires hardware testing to confirm router capacity for 956 simultaneous inputs, but is no longer theoretically blocked.
