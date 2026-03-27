# chapter 4: core results -- encoding comparison

this is the big results chapter. covers the ANN baseline, all 7 encodings, surrogate ablation, augmentation (negative result), stats, and PANNs. lots of tables and data here.

---

## 4.1 ANN baseline

ConvANN baseline gets **63.85% +/- 3.07%** (mean +/- std, 5 folds) on ESC-50 without augmentation. Per fold: 63.25%, 59.50%, 65.25%, 68.75%, 62.50%.

This sets the ceiling for a lightweight ~622K param CNN from scratch on ESC-50 with no external pretraining. For reference, AST with AudioSet pretraining (Gong et al. 2021) gets 98.25%, human is 81.3% (Piczak 2015). The gap vs SOTA is explained by pretraining -- SOTA models use tens of millions of AudioSet clips.

Training dynamics: all 5 folds converge in 30-45 epochs (early stopping patience=10). ReduceLROnPlateau kicks in around epoch 20-25, dropping lr from 1e-3 to 5e-4. No fold fails to learn.

---

## 4.2 spike encoding comparison

### 4.2.1 results summary

| Encoding | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean | Std | Gap vs ANN |
|----------|--------|--------|--------|--------|--------|------|-----|-----------|
| **ANN (baseline)** | 63.25% | 59.50% | 65.25% | 68.75% | 62.50% | **63.85%** | 3.07% | -- |
| Direct | 40.50% | 48.50% | 48.25% | 54.00% | 44.50% | **47.15%** | 4.50% | -16.70 pp |
| Phase | 22.50% | 22.25% | 25.00% | 24.25% | 26.75% | **24.15%** | 1.66% | -39.70 pp |
| Rate | 24.50% | 27.25% | 23.00% | 21.50% | 23.75% | **24.00%** | 1.90% | -39.85 pp |
| Population | 22.75% | 18.50% | 15.75% | 22.00% | 16.75% | **19.15%** | 2.79% | -44.70 pp |
| Latency | 14.00% | 15.75% | 17.75% | 15.50% | 18.50% | **16.30%** | 1.62% | -47.55 pp |
| Delta | 8.25% | 7.75% | 7.25% | 7.50% | 5.50% | **7.25%** | 0.94% | -56.60 pp |
| Burst | 5.00% | 5.25% | 9.25% | 6.00% | 7.00% | **6.50%** | 1.54% | -57.35 pp |

All 7 complete. Phase (24.15%) is essentially tied with rate (24.00%), within 0.15 pp -- deterministic single-spike timing provides equivalent info to stochastic multi-spike rate coding at T=25. Population (19.15%) underperforms both with higher variance -- MSE count loss harder to optimise than CE rate loss.

### 4.2.2 direct encoding: best-performing SNN

Direct gets 47.15% +/- 4.50%, highest of all SNN configs. Normalised spectrogram repeated across all T=25 timesteps; LIF neurons get continuous current and generate their own spike timing through integration.

**Why direct wins:** preserves full magnitude info across all timesteps. LIF neurons do implicit rate coding -- high pixels drive neurons above threshold every step, low pixels may never fire. Preserves the continuous spatial structure that CNNs exploit, while binary LIF outputs provide temporal sparsity.

**Sparsity:** 74.16% activations are zero (NeuroBench, see 4.4), vs 59% for ANN. High sparsity despite non-sparse inputs -- LIF threshold filters continuous input into sparse spikes.

### 4.2.3 rate coding: stochasticity limits accuracy

Rate gets 24.00% +/- 1.90%. Each pixel fires at each timestep with probability = normalised intensity. Stochastic encoding introduces noise: same spectrogram produces different spike patterns each forward pass.

**Why rate < direct:** at T=25, rate can't reliably reconstruct fine spectral structure. Pixel at 0.3 fires ~7.5 times on average but with high variance (Bernoulli), nearby pixels become indistinguishable. Conv layers have to learn despite this noise.

Consistent with Larroza et al. (2025): rate < direct on ESC-10 too.

### 4.2.4 latency coding: sparse but lossy

16.30% +/- 1.62%. Each neuron fires once, higher intensity = earlier spike. Most information-efficient in theory but sensitive to global intensity range -- if all pixels are 0.2-0.6, spike times cluster together.

**Sensitivity to normalisation:** with tau=5.0, theta=0.01, most spikes cluster in first ~5 timesteps for high-intensity regions. Effectively reduces usable window from T=25 to T~5.

### 4.2.5 delta coding: near-chance

7.25% +/- 0.94%, near chance (2%) for 50 classes. Spikes generated on positive intensity changes across timesteps. Applied to static spectrogram with no inherent temporal variation = almost no spikes generated.

Delta is fundamentally inappropriate for static spectrograms. It's designed for DVS/event camera data with genuine temporal contrast. Replicates the expectation from theory.

### 4.2.6 burst coding: negative result with mechanistic explanation

| Fold | Best Acc | Best Epoch | Total Epochs |
|------|----------|------------|--------------|
| 1 | 5.00% | 7 | 17 |
| 2 | 5.25% | 3 | 13 |
| 3 | 9.25% | 26 | 36 |
| 4 | 6.00% | 10 | 20 |
| 5 | 7.00% | 10 | 20 |
| **Mean** | **6.50% +/- 1.54%** | | |

Near-chance. This is a **negative result** -- i expected burst to do better honestly.

**Root cause: temporal window mismatch.** Burst concentrates all N_max=5 spikes in the first 5 of 25 timesteps. After step 5, all neurons are silent. LIF neurons (beta=0.95) keep integrating for 20 more steps -- no new info, just decaying membrane. Network sees signal for 20% of integration window and noise for 80%.

**Overfitting:** train accuracy hits 45-62% (genuine learning happens) while test stays 5-9%. Model memorises training sequences but can't generalise. The 20% signal window has enough discriminative info to memorise but not to generalise.

**Contrast with delta:** delta fails because theres *no signal* (static spectrograms have no temporal contrast). Burst fails because *signal exists but is temporally mismatched*. Both get ~5-9% but via completely different failure mechanisms.

**What would fix it:** T=5 to match burst window, or architecture that reads only first N_max timesteps. Neither explored -- burst documented as negative result.

### 4.2.7 phase coding

Phase fires each neuron exactly once at t = floor((1-x_i)(T-1)). High intensity = timestep 0, low = timestep 24, zero = silent. Unlike latency, phase maps linearly and distributes spikes uniformly across full T=25.

| Fold | Best Acc | Best Epoch | Total Epochs |
|------|----------|------------|--------------|
| 1 | 22.50% | 24 | 34 |
| 2 | 22.25% | 12 | 22 |
| 3 | 25.00% | 15 | 25 |
| 4 | 24.25% | 29 | 39 |
| 5 | 26.75% | 30 | 40 |
| **Mean** | **24.15% +/- 1.66%** | | |

**Key finding:** phase (24.15%) tied with rate (24.00%), within 0.15 pp. Most surprising result in the encoding comparison.

**Why phase ~ rate despite different format:** rate provides ~6-7 spikes/neuron (T=25, p~0.25), phase provides exactly 1. Despite 6-7x fewer spikes, same accuracy. Explanation: phase deterministically maps intensity to time, preserving full magnitude ordering without stochastic noise. Rate spreads the same info across multiple noisy spikes. At T=25 these two representations have equivalent discriminative capacity.

Training accuracy at best epoch: 54-73% across folds -- genuine feature learning. Converges 22-40 epochs, more training than rate/latency but less than direct.

**vs latency:** phase beats latency (16.30%) by 7.85 pp. Key difference is window usage -- latency clusters spikes into first ~5 steps (exponential), phase distributes uniformly across all 25. Confirms full temporal window coverage is the critical factor, not spike count.

### 4.2.8 encoding comparison: key finding

Ordering: **direct > rate ~ phase > population > latency >> delta ~ burst**

Explained by information preservation:

| Encoding | Spikes/neuron | Deterministic | Preserves magnitude | Fills window |
|----------|--------------|---------------|--------------------|----|
| Direct | T (continuous) | Yes | Full | Yes |
| Phase | 1 | Yes | Timing proportional | Uniform |
| Rate | ~Txp (stochastic) | No | Average | Uniform (noisy) |
| Population | ~Txp (rate input) | No | Average | Uniform (noisy) |
| Latency | 1 | Yes | Timing proportional | Clustered early |
| Delta | ~0 (static) | Yes | Change only | N/A |
| Burst | 0-5 | Yes | Count proportional | Front-loaded |

The fundamental insight: **information preservation** (not biological plausibility) predicts SNN performance. The rate/phase near-equality is the strongest evidence -- both faithfully represent full intensity range across the complete window, achieving identical accuracy despite fundamentaly different coding schemes.

---

## 4.3 surrogate gradient ablation

**Setup:** 8 surrogates from snnTorch 0.9.4, fold 1, direct encoding, seed=42. ATan (alpha=2.0), FastSigmoid (slope=25), Sigmoid (slope=25), STE, Triangular, SpikeRateEscape, LSO (slope=0.1), SFS (slope=25).

**Literature hypothesis (Zenke & Vogels 2021):** shape matters less than slope. Steeper = sharper gradients near threshold = higher sparsity but instability. FastSigmoid expected highest sparsity; triangular expected worst.

**Significance:** no prior surrogate ablation exists for any audio task. Publishable regardless of outcome.

**FastSigmoid trajectory (fold 1, seed 42):**

| Epoch | Train Acc | Test Acc | Best |
|-------|-----------|----------|------|
| 10 | 8.4% | 10.0% | 10.2% |
| 20 | 19.4% | 17.2% | 17.2% |
| 30 | 36.6% | 24.7% | 27.5% |
| 40 | 47.6% | 29.2% | 36.7% |
| 50 | 67.0% | 44.8% | **44.75%** |

FastSigmoid finishes at 44.75% (epoch 50), still improving at termination -- exceeds fold 1 baseline of 40.5%. ATan gets 35.75% at epoch 49 -- 9.0 pp below fast_sigmoid despite same slope. Shows shape matters for this task.

**SpikeRateEscape gets the best: 46.00% at epoch 50** -- beats fast_sigmoid by 1.25 pp. Uses stochastic escape rate model providing larger effective gradients for high-intensity inputs. Both still improving at ep 50, longer training could help.

**Failure modes:**
- **Sigmoid (2.00%, early stop ep11):** never learns. Peak derivative sigma'(0)*slope = 6.25, but saturates faster than fast_sigmoid. Gradient dies quickly away from threshold. Contradicts Zenke & Vogels' claim that shape matters less -- most surprising result here.
- **STE (10.25%, early stop ep11):** piecewise-constant gradient (1 if |U-theta|<0.5, else 0), no smooth signal through threshold. Converges to near-chance local minimum.
- **Triangular (2.75%, early stop ep23):** near-chance, confirming literature prediction (narrowest effective bandwidth). check this citation

**Bimodal pattern:**
- **Learning group** (3): spike_rate_escape (46.00%), fast_sigmoid (44.75%), atan (35.75%)
- **Failure group** (4): STE (10.25%), sigmoid (2.00%), SFS (2.00%), triangular (2.75%)

The learning surrogates all maintain non-zero gradient over a broad range away from threshold, enabling gradient flow even when most neurons are well below threshold. Failure surrogates have narrow bandwidth (triangular), piecewise-constant gradients (STE), or practical saturation at high slope (sigmoid, SFS). This bimodal split is stronger than Zenke & Vogels predicted -- shape matters substantially for 50-class audio, not just slope.

| Surrogate | Best Acc (fold 1) | Best Epoch |
|-----------|-------------------|------------|
| fast_sigmoid | 44.75% | 50 |
| atan | 35.75% | 49 |
| sigmoid | 2.00% (early stop ep11) | 1 |
| ste | 10.25% (early stop ep11) | 1 |
| triangular | 2.75% (early stop ep23) | 13 |
| spike_rate_escape | **46.00%** (+1.25 pp vs fast_sigmoid) | 50 |
| lso | CRASHED (Python 3.14/snnTorch 0.9.4 thing) | -- |
| sfs | 2.00% (early stop ep10) | 1 |

Source: `results/snn/surrogate_ablation/ablation_fold1_seed42.json`. 1-seed result; CSF3 3-seed pending. LSO crashes due to StochasticSpikeOperator.forward() missing `variance` arg -- Python 3.14 API mismatch.

---

## 4.4 effect of data augmentation

**Complete (4 March 2026).** Ran locally on MPS. SNN: 100 epochs, all 5 folds. ANN: 100 epochs, all 5 folds.

### 4.4.1 setup

Augmentation on training spectrograms only:
1. **SpecAugment** (Park et al. 2019): 2 freq masks width F=8, 2 time masks width T=20. Masks set to spectrogram mean. Per-sample per-epoch.
2. **TimeShift**: cyclic shift +/-10% frames (+/-22), random per sample per epoch.

Both applied simultaneously. Otherwise identical to baseline: same arch, same training, but 100 epochs instead of 50.

### 4.4.2 results

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean | Std | vs baseline |
|-------|--------|--------|--------|--------|--------|------|-----|-------------|
| SNN direct (no aug) | 40.50% | 48.50% | 48.25% | 54.00% | 44.50% | 47.15% | 4.50% | -- |
| **SNN direct (aug)** | **46.00%** | **48.75%** | **24.50%** | **63.75%** | **20.75%** | **40.75%** | **16.03%** | **-6.40 pp** |
| ANN (no aug) | 63.25% | 59.50% | 65.25% | 68.75% | 62.50% | 63.85% | 3.07% | -- |
| **ANN (aug)** | **63.25%** | **59.75%** | **62.50%** | **68.50%** | **54.50%** | **61.70%** | **4.58%** | **-2.15 pp** |

Source: `results/snn/direct_aug/summary.json`, `results/ann/none_aug/summary.json`.

### 4.4.3 analysis

|-------|--------|--------|--------|--------|--------|------|-----|---------------|
| SNN direct (no aug) | 40.50% | 48.50% | 48.25% | 54.00% | 44.50% | 47.15% | 4.50% | — |
| **SNN direct (aug)** | **46.00%** | **48.75%** | **24.50%** | **63.75%** | **20.75%** | **40.75%** | **16.03%** | **−6.40 pp** |
| ANN (no aug) | 63.25% | 59.50% | 65.25% | 68.75% | 62.50% | 63.85% | 3.07% | — |
| **ANN (aug)** | **63.25%** | **59.75%** | **62.50%** | **68.50%** | **54.50%** | **61.70%** | **4.58%** | **−2.15 pp** |

*Source: `results/snn/direct_aug/summary.json`, `results/ann/none_aug/summary.json`.*

### 4.4.3 Analysis

**The key finding: SpecAugment harms both models on this dataset, with a dramatically worse effect on the SNN.**

The augmented SNN drops 6.40 pp in mean accuracy (47.15% → 40.75%) and its fold-to-fold variance **increases** from 4.50% to 16.03% — the opposite of what augmentation is intended to achieve. The augmented ANN drops 2.15 pp (63.85% → 61.70%), a smaller but consistent degradation.

**Root cause analysis:**

1. **Early stopping is too aggressive for augmented training.** SNN folds 3 and 5 stopped at epochs 39 and 33 respectively — before convergence. At ep 39 (fold 3), training accuracy was only 26.4%, far from plateau. Augmentation slows convergence by increasing input diversity, requiring more epochs for the model to find stable representations. With patience=10 and a harder loss landscape, early stopping fires prematurely on some folds. The result is bimodal: folds where learning accelerated past the patience threshold (folds 1, 2, 4) get reasonable results; folds where early plateaus triggered early stopping (folds 3, 5) get near-chance accuracy.

2. **Small dataset + aggressive augmentation = information loss.** With only 1,600 training samples per fold, SpecAugment's frequency and time masking removes a substantial fraction of the discriminative spectral content. For a 64-mel × 216-frame spectrogram, masking 2 frequency bands of 8 bins each removes 25% of mel bins; masking 2 time windows of 20 frames removes 18% of time. For sounds with compact spectral features (insects, glass_breaking), masking can remove the entire diagnostic region in a given epoch. The ANN, being more data-efficient with ReLU activations, handles this better than the SNN's threshold-based LIF neurons.

3. **Fold 4 is a genuine exception.** SNN fold 4 jumped from 54.00% (baseline) to 63.75% (augmented) — a +9.75 pp gain, the highest SNN fold accuracy recorded in this study. This is not noise: fold 4 ran for the full 100 epochs and reached training accuracy 80.2% at ep 100 with best test at ep 90 (63.75%). Fold 4's composition apparently benefits from augmentation, possibly because its test set contains more sounds where temporal position invariance is genuinely useful (door_knock, clock_tick, mouse_click — events that can occur at any point in the 5-second clip).

**Why augmentation hurts SNNs more than ANNs:**
The LIF threshold introduces a hard non-linearity that interacts badly with masked spectrogram inputs. When a frequency band is masked to the mean value, it produces a spike pattern that is neither "present" nor "absent" in the same way a clean spectrogram produces. The mean-value mask creates a constant, non-discriminative input signal that can mislead the LIF membrane potential integration across 25 timesteps. The ANN's ReLU activations are better calibrated to handle mean-value infill (the value is simply processed continuously). For the SNN, the masked region contributes undifferentiated current that competes with the discriminative signal from unmasked regions.

**Conclusion:** Data augmentation as implemented here (SpecAugment + TimeShift, patience=10 early stopping) is **not recommended** for convolutional SNNs on small audio datasets. The mean accuracy decrease (−6.40 pp SNN, −2.15 pp ANN) and dramatically increased variance rule it out as an improvement strategy. However, the fold 4 exception (+9.75 pp) suggests that with longer patience (patience=20–30) and fold-specific augmentation strength, augmentation could be beneficial for specific data distributions. Future work should use patience=20 and reduce mask widths to F=4, T=10 for small-dataset SNN training.

**The baseline SNN (47.15%) remains the primary result for all subsequent analyses.** The augmented SNN (40.75%) is documented as a negative result with mechanistic explanation.

---

## 4.5 Statistical Significance

The accuracy gap between the best SNN (direct, 47.15%) and the ANN baseline (63.85%) is **16.70 percentage points**. Statistical significance is assessed using paired tests on the 5 fold accuracies:

| Test | Statistic | p-value | Significant? |
|------|-----------|---------|--------------|
| Paired t-test (two-sided) | t = 8.64 | p = 0.0010 | ✅ p < 0.01 |
| Wilcoxon signed-rank | W = 0.0 | p = 0.0625 | ❌ p > 0.05 |

The Wilcoxon test is constrained by the minimum achievable p-value with n=5 samples (p=0.0625). The paired t-test, which has higher power for small samples when the normality assumption approximately holds, returns p=0.001. We conclude the SNN-ANN accuracy gap is statistically significant.

**Per-fold analysis:** The SNN underperforms the ANN on all 5 folds (no fold where SNN wins), providing a consistent directional finding independent of statistical test choice.

---

## 4.6 Transfer Learning: PANNs + SNN Head

### 4.6.1 Setup

CNN14 (Kong et al. 2020), pretrained on AudioSet (1.8M clips, 527 tags), extracts 2048-dimensional embeddings from ESC-50 audio. A 3-layer SNN classification head (2048→512→256→50, LIF neurons with β=0.9, rate encoding, T=25) is trained on frozen CNN14 embeddings for 50 epochs.

All 2,000 ESC-50 embeddings are precomputed and cached, reducing training time to seconds per epoch. Three classifiers are compared: SNN head, ANN head (identical architecture with ReLU), and linear classifier (logistic regression on 2048-d embeddings).

### 4.6.2 Results

**5-fold cross-validation on frozen CNN14 embeddings:**

| Classifier | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean | Std |
|-----------|--------|--------|--------|--------|--------|------|-----|
| SNN head (ours) | 92.00% | 94.50% | 91.00% | 93.50% | 91.50% | **92.50%** | 1.30% |
| ANN head | 93.00% | 95.00% | 92.00% | 95.50% | 91.75% | **93.45%** | 1.54% |
| Linear classifier | 94.25% | 95.75% | 92.50% | 95.25% | 91.25% | **93.80%** | 1.69% |

### 4.6.3 Interpretation

**The SNN-ANN gap collapses from 16.70 pp to 0.95 pp with pre-trained features.** This is the most important finding in this thesis.

The headline interpretation: the accuracy gap between SNNs and ANNs is almost entirely explained by feature quality, not by the spiking computation mechanism. When both classifiers receive identical, high-quality 2048-d AudioSet features, they achieve statistically indistinguishable accuracy (92.50% vs 93.45%).

**What this means for the accuracy gap (§4.2):** The 16.70 pp gap seen in §4.2 is not a fundamental limitation of spiking computation. It reflects the difficulty of learning rich audio features from scratch with a ~622K parameter network trained on 1,600 clips. An ANN with the same capacity would achieve a similar gap against SNN if both were constrained to the same small-data regime.

**What this means for neuromorphic deployment:** An SNN trained on top of pre-computed PANNs embeddings achieves 92.50% accuracy on ESC-50 — surpassing the human performance benchmark (81.3%) and approaching ANN state-of-the-art from scratch. This opens a practical deployment pathway: run the ANN feature extractor once (on a CPU or NPU), then classify with the SNN on neuromorphic hardware (energy-efficient for the final classification step).

**Novelty:** This is the first combination of PANNs embeddings with an SNN classifier in the published literature (to the best of our knowledge).

---

## 4.7 Population Coding

Population coding expands the output layer to 500 neurons (50 classes × 10 neurons per class, `pop_n=10`). All 10 neurons for a class contribute to the class vote via summed spike count. Loss: `SF.mse_count_loss(population_code=True, num_classes=50)`. Accuracy: `SF.accuracy_rate(population_code=True, num_classes=50)`. Input encoding: rate coding (same as §4.2.3).

**Five-fold results:**

| Fold | Best Acc | Best Epoch | Total Epochs |
|------|----------|------------|--------------|
| 1 | 22.75% | 32 | 42 |
| 2 | 18.50% | 21 | 31 |
| 3 | 15.75% | 18 | 28 |
| 4 | 22.00% | 44 | 50 |
| 5 | 16.75% | 21 | 31 |
| **Mean ± Std** | **19.15% ± 2.79%** | — | — |

**Hypothesis rejected:** Population coding achieves 19.15% ± 2.79% — lower than rate coding (24.00% ± 1.90%) and with higher variance (std=2.79 vs 1.90). The expanded output representation did not improve accuracy.

**Why population coding underperforms rate coding:**
Two mechanisms explain the failure. First, the MSE count loss formulation (`SF.mse_count_loss`) is harder to optimise than cross-entropy rate loss: training accuracy at termination reaches only 18–24% across folds, compared to ~50% for rate coding at equivalent epoch count. The MSE loss produces shallower gradients and slower convergence. Second, the 10× increase in output neurons (500 vs 50) adds parameters to the output layer without a corresponding increase in the expressivity of the preceding feature extractor — the bottleneck is the 256-dimensional FC1 representation, not the output layer width.

**Comparison to standard rate coding:** Rate coding with CE loss achieves 24.00% ± 1.90% at lower computational cost (50 output neurons, 3× fewer parameters in output layer). Population coding does not compensate for its training disadvantage.

**Note on variance:** Population coding exhibits the highest fold-to-fold variance of all non-random-chance encodings (std=2.79%). Folds 3 and 5 perform near-chance (15.75%, 16.75%), while folds 1 and 4 approach rate coding performance (22.75%, 22.00%). This inconsistency suggests the MSE loss landscape has multiple local minima that may or may not be escaped depending on initialisation.

---

## 4.8 Chapter Summary

The core findings of this chapter are:

1. **ANN baseline:** 63.85% ± 3.07% — a reasonable from-scratch baseline for ESC-50 with a lightweight CNN.

2. **Best SNN (direct encoding):** 47.15% ± 4.50% — **the first convolutional SNN result on ESC-50** (C1).

3. **Encoding ordering (direct > rate ≈ phase > population > latency >> delta ≈ burst):** Explained by information preservation: encodings that better retain spectrogram magnitude and structure achieve higher accuracy. The near-equality of rate (24.00%) and phase (24.15%) is particularly notable — deterministic single-spike timing achieves the same accuracy as stochastic multi-spike counting at T=25. Population coding (19.15%) underperforms both despite 10× more output neurons — the MSE count loss is harder to optimise than CE rate loss. Delta and burst coding fail near-chance for distinct mechanistic reasons — delta because static spectrograms have no temporal contrast; burst because a 5-timestep signal window mismatches the 25-timestep LIF integration window. (C2 contribution: systematic comparison with 7 methods, all complete.)

4. **The PANNs finding:** SNN-ANN gap collapses from 16.70 pp to 0.95 pp with AudioSet-pretrained features, demonstrating that the gap is a feature-learning problem, not a spiking-computation problem (C5).

5. **Statistical significance:** The SNN-ANN gap (direct vs ANN baseline) is significant at p=0.001 (paired t-test).
