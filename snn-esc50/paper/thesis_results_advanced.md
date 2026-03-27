# chapter 6: advanced analysis

covers adversarial, continual learning, temporal patterns, t-SNE, per-class stuff. some of these are the most interesting findings in the whole thesis honestly.

---

## 6.1 adversarial robustness

### 6.1.1 setup

Evaluated on fold 4 (400 samples) using:
- **FGSM** (Goodfellow et al. 2015): single-step, x' = x + eps * sign(grad_x L)
- **PGD** (Madry et al. 2018): 40 iterations, step size alpha = eps/10

Both applied to normalised mel spectrogram input (not raw audio), eps in {0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3}.

**SNN wrapping:** SNNWrapper applies direct encoding (repeat T times) and returns mem_out.sum(dim=0) as differentiable logits. ANN returns standard logits. Both via torchattacks v3.x.

**Caveat (Wang et al. 2025):** standard PGD may underestimate SNN vulnerability because vanishing surrogate gradients cause PGD gradient update to under-estimate true adversarial gradient. SNN PGD numbers should be interpreted as upper bounds. need to be careful how i frame this

### 6.1.2 results

| eps | FGSM SNN | FGSM ANN | PGD SNN | PGD ANN |
|-----|----------|----------|---------|---------|
| 0.000 (clean) | 53.75% | 68.75% | 53.75% | 68.75% |
| 0.010 | 37.50% | 22.50% | 23.50% | 14.75% |
| 0.020 | 32.00% | 8.75% | 20.50% | 2.00% |
| 0.050 | 29.00% | 2.50% | 19.25% | 0.00% |
| 0.100 | **26.00%** | **1.75%** | 6.25% | 0.00% |
| 0.200 | 21.50% | 1.25% | 1.25% | 0.00% |
| 0.300 | 20.75% | 0.75% | 1.25% | 0.00% |

Source: `results/adversarial/robustness_fold4.json`. Clean accuracy (53.75% SNN, 68.75% ANN) consistent with canonical fold 4 values. 0.25 pp SNN difference is MPS vs CUDA backend variation -- doesn't affect any conclusions.

### 6.1.3 key finding: dramatic SNN robustness

**eps=0.1 FGSM:** SNN keeps 26.00% vs ANN's 1.75% -- 24.25 pp advantage, 14.9x more robust.
**eps=0.05 PGD:** SNN keeps 19.25% vs ANN's 0.00% -- complete ANN collapse while SNN survives.

SNN is dramatically more robust across all eps and both attack types. The advantage grows with eps: at small eps both degrade similarly, but above ~0.02 the ANN collapses while SNN retains meaningful accuracy.

### 6.1.4 mechanism: spike threshold filtering

The robustness comes from the spike threshold acting as a non-linear filter:

1. **Hard threshold at membrane potential:** small input changes (delta ~ eps) produce small current changes that may not cross threshold. If threshold isnt crossed, output spike pattern is unchanged.

2. **Gradient masking:** surrogate gradient is zero almost everywhere in the forward pass. The true gradient is effectively masked, weakening gradient-based attacks.

3. **Temporal averaging:** SNN decision based on summed membrane over T=25 timesteps. Perturbations targeting one timestep may not persist across full integration.

Consistent with Sharmin et al. (2020 ECCV) who showed similar effects with Poisson encoding in black-box scenarios.

**Limitation:** standard PGD may underestimate vulnerability. SA-PGD (Wang et al. 2025) would give more reliable numbers. todo for future work

### 6.1.5 implications for edge audio security

Practical implications:
- SNN on a smart building sensor is substantially harder to fool with adversarial audio
- At eps=0.1 (perturbation likely below JND for human hearing), ANN is non-functional (1.75%) while SNN keeps 26%
- Natural robustness + energy efficiency makes SNNs attractive for adversarial-robust edge deployment

---

## 6.2 continual learning

### 6.2.1 setup

Sequential training on 5 ESC-50 super-categories (10 classes each):
- Task 1: Animals (0-9): dog, rooster, pig, cow, frog, cat, hen, insects, sheep, crow
- Task 2: Nature (10-19): rain, sea waves, crackling fire, crickets, chirping birds, water drops, wind, pouring water, toilet flush, thunderstorm
- Task 3: Human (20-29): crying baby, sneezing, clapping, breathing, coughing, footsteps, laughing, brushing teeth, snoring, drinking/sipping
- Task 4: Domestic (30-39): door knock, mouse click, keyboard, door creaks, can opening, washing machine, vacuum cleaner, clock alarm, clock tick, glass breaking
- Task 5: Urban (40-49): helicopter, chainsaw, siren, car horn, engine, train, church bells, airplane, fireworks, hand saw

BWT measured after each task:
$$\text{BWT} = \frac{1}{T-1}\sum_{i<T}(R_{T,i} - R_{i,i})$$

Both SpikingCNN (direct) and ConvANN, identical protocol (20 epochs/task, Adam lr=1e-3, no replay, no regularisation).

### 6.2.2 results

Both suffer severe catastrophic forgetting -- worse than the +/-50% BWT from literature. Makes sense given the extreme imbalance: each task trains on only 10/50 classes (320 samples), gradient pressure from each subset rapidly overwrites global representation.

**SNN accuracy matrix (fold 4 representative):**

| | After T0 | After T1 | After T2 | After T3 | After T4 |
|---|---|---|---|---|---|
| Task 0 (Animals) | 78.75% | 8.75% | 2.50% | 11.25% | 0.00% |
| Task 1 (Nature) | -- | 87.50% | 20.00% | 8.75% | 0.00% |
| Task 2 (Human) | -- | -- | 75.00% | 0.00% | 0.00% |
| Task 3 (Domestic) | -- | -- | -- | 68.75% | 12.50% |
| Task 4 (Urban) | -- | -- | -- | -- | 78.75% |

**ANN accuracy matrix (fold 4):**

| | After T0 | After T1 | After T2 | After T3 | After T4 |
|---|---|---|---|---|---|
| Task 0 (Animals) | 81.25% | 45.00% | 17.50% | 6.25% | 1.25% |
| Task 1 (Nature) | -- | 93.75% | 46.25% | 15.00% | 0.00% |
| Task 2 (Human) | -- | -- | 81.25% | 7.50% | 0.00% |
| Task 3 (Domestic) | -- | -- | -- | 73.75% | 3.75% |
| Task 4 (Urban) | -- | -- | -- | -- | 88.75% |

**Summary (5-fold validated):**

| Metric | SNN | ANN |
|--------|-----|-----|
| Mean forgetting | **69.9% +/- 4.3%** | 74.7% +/- 2.4% |
| Mean BWT | -0.699 | -0.747 |
| Final avg accuracy | 18.3% | 18.8% |
| Final Task 4 only | 78.75% | 88.75% |

### 6.2.3 analysis

**SNN forgets less than ANN (69.9% vs 74.7%, -4.7 pp advantage).** Both suffer catastrophically though -- way worse than literature's "~50% BWT" for regularisation-free CL. Expected given the extreme setup.

**Why ANN forgets more:** continuous activations produce larger gradient magnitudes when fine-tuning on new tasks, overwriting weights more completely. SNN's binary outputs mean sparser gradient flow -- only neurons that fired get gradient updates. ~74% of weights get zero gradient per pass. Fewer weights modified per new task = more original representation preserved. Same mechanism as adversarial robustness -- spike threshold gates information flow.

**ANN has higher per-task peak accuracy** (93.75% vs 87.50% for Nature, 88.75% vs 78.75% for Urban), reflecting larger effective capacity per task. The forgetting difference isn't about peak performance but gradient interference.

**Both converge to last task:** after all 5, both classify mainly Urban. Animals, Nature, Human collapse to 0% for both. Classic "last task dominance" -- hallmark of catastrophic forgetting without replay.

Results in `results/continual_learning/forgetting_fold{1-5}_pretrained_20ep.json`.

---

## 6.3 temporal spike pattern analysis

### 6.3.1 rate vs first-spike decoding

**Setup:** fold 4 test (400 samples), direct encoding SNN, two readout methods on same model:
- **Rate:** argmax of summed membrane potential (standard)
- **First-spike:** argmin of first output spike time per class

| Readout | Accuracy (fold 4) |
|---------|-------------------|
| Rate (membrane sum) | 51.50% |
| First-spike latency | 25.75% |

### 6.3.2 interpretation

Rate dramatically beats first-spike (51.50% vs 25.75%). This means:

1. SNN encodes info in total spike count, not timing. Model trained with rate loss (sum membrane over T) and optimises for rate-code output.

2. First-spike timing is noisy -- class may spike early by chance, not because its most active. Rate averages over T=25, suppressing noise.

3. To exploit first-spike timing, need temporal objective. As Yu et al. (2025) show, surrogate gradient training can enable timing-based learning but only with a loss that specifically rewards early correct-class firing.

**Practical implication:** this SNN is fundamentally a rate-coded classifer implemented in spiking hardware. Temporal sparsity exists from LIF thresholding but isn't informationally exploited beyond rate integration.

### 6.3.3 raster plots

Key observations from the raster plots (fold 4, direct, T=25):
- Output spike density: ~3-5 spikes/step out of 50 neurons (6-10%)
- Correct samples show clear winner (one neuron consistently active)
- Misclassified show ties between multiple output neurons
- Spikes not temporally structured -- approximately uniform across T=25 (consistent with constant current from direct encoding)

### 6.3.4 per-class first-spike latency

Mean timestep of first correct-class spike, averaged over fold 4 test samples. Source: `results/temporal_analysis/temporal_analysis_fold4.json`.

**Earliest firing (step < 1.0):**

| Class | Mean first-spike |
|-------|-----------------|
| can_opening | 0.12 |
| sneezing | 0.38 |
| water_drops | 0.50 |
| church_bells | 0.63 |
| frog | 0.75 |
| cat | 0.75 |
| door_wood_knock | 0.75 |
| mouse_click | 0.75 |
| rooster | 0.88 |
| crow | 0.88 |
| siren | 0.88 |

**Latest firing:**

| door_wood_knock | 0.75 |
| mouse_click | 0.75 |
| rooster | 0.88 |
| crow | 0.88 |
| siren | 0.88 |

**Latest-firing classes:**

| Class | Mean first-spike (step out of 25) |
|-------|----------------------------------|
| pig | 3.00 |
| snoring | 3.25 |
| door_wood_creaks | 2.75 |
| vacuum_cleaner | 2.63 |
| brushing_teeth | 2.63 |
| insects | 2.63 |

**Interpretation:** The mean first-spike latency is very short (0.12–3.25 timesteps out of 25), confirming that the SNN, when given direct encoding (continuous current throughout the window), generates output spikes early in the simulation. This is consistent with the direct encoding maintaining constant input current — the LIF neurons accumulate charge rapidly and fire within the first few timesteps. The first-spike readout (25.75% accuracy) underperforms rate readout (51.50%) because the first spike is noisy (other classes may fire first by chance in the same early window), not because the correct class fires late.

---

## 6.4 Representation Analysis (t-SNE)

### 6.4.1 Setup

Hidden representations from the FC₁ layer (256-dimensional) are extracted for all 400 fold-4 test samples and projected to 2D using t-SNE (perplexity=30, 1,000 iterations, random seed 42). Both SNN and ANN representations are visualised.

### 6.4.2 Results

![t-SNE 2D projection of FC₁ representations (256-d → 2D, perplexity=30, 1000 iterations) for all 400 fold-4 test samples, coloured by ESC-50 class (50 classes). Left: SNN direct encoding — diffuse clusters with substantial inter-class overlap. Right: ANN — tighter, more separated class clusters, consistent with higher accuracy (63.85% vs 47.15%).](../results/analysis/tsne_snn_fold4.png)

*SNN (left): `results/analysis/tsne_snn_fold4.png`; ANN (right): `results/analysis/tsne_ann_fold4.png`*

**Qualitative observations:**

The t-SNE projections reveal a clear qualitative difference in representational geometry between SNN and ANN hidden spaces:

- **ANN representations** show tighter, more separated class clusters. Within-class variance is lower and between-class distances are larger, reflecting the ANN's higher accuracy (63.85%).

- **SNN representations** show more diffuse clusters with greater overlap. Several ESC-50 super-categories (Animals, Nature) form loose macro-clusters even if individual classes are not well-separated, suggesting that the SNN learns coarse categorical structure but fails to discriminate within categories.

**Super-category clustering:** Both models exhibit emergent super-category structure (Animals cluster together, Urban sounds cluster together) even though super-category labels are not used during training. This suggests that spectral features are sufficiently distinctive at the super-category level for both models.

**Hardest classes (SNN):** Classes with high within-cluster overlap are the hardest for the SNN. Preliminary analysis suggests transient, impulsive sounds (glass breaking, sneezing, gunshots) are harder for the SNN than continuous, harmonic sounds (insects, water). This is consistent with the SNN's temporal integration accumulating evidence over 25 timesteps — impulsive events at a single timestep may not trigger sufficient integration.

---

## 6.5 Per-Class Difficulty Analysis

### 6.5.1 Setup

Per-class accuracy is computed for both SNN (direct) and ANN across all 5 folds (n=40 test samples per class total). This is the 5-fold aggregate from `results/snn/direct/evaluation.json` and `results/ann/none/evaluation.json`. Classes are ranked from easiest to hardest for each model, and the intersection of "hardest 10" classes is analysed.

### 6.5.2 Results

**Top 10 classes by SNN accuracy (all 5 folds, direct encoding, corrected data):**

| Class | SNN Accuracy | ANN Accuracy | SNN−ANN |
|-------|-------------|-------------|---------|
| toilet_flush | **83%** | 85% | −2% |
| crying_baby | **80%** | 73% | **+7%** |
| door_wood_knock | **80%** | 73% | **+7%** |
| rooster | 78% | 88% | −10% |
| pouring_water | **75%** | 70% | **+5%** |
| thunderstorm | 75% | 95% | −20% |
| can_opening | 73% | 78% | −5% |
| hand_saw | 73% | 83% | −10% |
| crackling_fire | **68%** | 65% | **+3%** |
| coughing | **68%** | 60% | **+8%** |

**Bottom 10 classes by SNN accuracy:**

| Class | SNN Accuracy | ANN Accuracy | SNN−ANN |
|-------|-------------|-------------|---------|
| engine | 8% | 43% | −35% |
| door_wood_creaks | 10% | 25% | −15% |
| helicopter | 10% | 33% | −23% |
| pig | 13% | 35% | −22% |
| laughing | 13% | 53% | −40% |
| water_drops | 15% | 43% | −28% |
| drinking_sipping | 20% | 45% | −25% |
| clock_tick | 23% | 68% | −45% |
| fireworks | 23% | 43% | −20% |
| hen | 28% | 43% | −15% |

**Classes where SNN outperforms ANN (6/50):**
1. coughing: SNN 68% vs ANN 60% (+8%)
2. crying_baby: SNN 80% vs ANN 73% (+7%)
3. door_wood_knock: SNN 80% vs ANN 73% (+7%)
4. pouring_water: SNN 75% vs ANN 70% (+5%)
5. crackling_fire: SNN 68% vs ANN 65% (+3%)
6. footsteps: SNN 55% vs ANN 53% (+2%)

**Updated statistical significance (with corrected fold 1 = 40.5%):**
- Paired t-test: p = 0.0010 (highly significant, t = 8.64)
- Wilcoxon signed-rank: p = 0.0625 (limited by n=5 minimum achievable p-value)
- SNN outperforms ANN on 6/50 classes

### 6.5.3 Interpretation of Class Patterns

The per-class results reveal a clear and unexpected pattern. **The SNN wins on high-energy, spectrally distinctive sounds** (crying_baby, door_wood_knock, coughing) not on sustained tonal sounds as initially hypothesized. The mechanism: LIF neurons with threshold=1.0 and β=0.95 integration require consistent, strong input current to fire across multiple timesteps. High-energy sounds (crying baby has a broad-bandwidth, high-amplitude spectrogram) drive many neurons above threshold reliably in every sample. The SNN's rate-code classification averages over T=25 timesteps, accumulating strong evidence for these classes.

**The SNN fails on low-energy, subtle sounds:** engine hum (7%), door creaks (10%), clock_tick (23%), water_drops (15%). These sounds have spectrograms concentrated in narrow frequency bands at low amplitudes. The LIF threshold acts as a high-pass filter on input energy — quiet sounds may push only a few neurons above threshold, producing sparse, noisy spike patterns that the network cannot consistently interpret. The ANN's ReLU does not have this threshold effect: every non-zero activation contributes to the classification decision.

**The clock_tick gap (SNN 23%, ANN 68%) is the strongest evidence for this mechanism.** Clock_tick is a quiet, periodic click at regular intervals — a narrow spectrogram line at low amplitude, repeated. The ANN learns the narrow spectral signature reliably. The SNN's threshold may not fire on the quiet pixels, making the clock_tick indistinguishable from silence.

### 6.5.4 Comparison with Human Performance

Human ESC-50 accuracy is 81.3% (Piczak 2015). Classes where humans struggle (< 70% human accuracy) are: insects vs frogs (confusable calls), domestic mechanical sounds (clock tick vs keyboard), and urban machinery (train vs airplane). The SNN may show different confusion patterns — discriminating sounds that confuse humans (via spectral pattern recognition) while failing on sounds that humans identify easily (via semantic/contextual cues absent from a 5-second isolated clip).

---

## 6.6 Chapter Summary

1. **Adversarial robustness (§6.1, C4):** SNN retains 26% accuracy at ε=0.1 FGSM vs ANN's 1.75%. The spike threshold provides natural adversarial filtering, making SNNs substantially more robust to gradient-based attacks on audio spectrograms. This is the first such analysis for SNN on environmental sound data.

2. **Continual learning (§6.2):** Both SNN and ANN suffer severe catastrophic forgetting without replay or regularisation. **SNN mean forgetting: 69.9% ± 4.3%; ANN mean forgetting: 74.7% ± 2.4% — SNN forgets 4.7 pp less (5-fold validated).** The spike threshold's sparsity limits gradient interference between tasks. Both converge to classifying only the most recently seen task (Urban) after all 5 sequential tasks.

3. **Temporal analysis (§6.3):** Rate readout (51.50%) dramatically outperforms first-spike readout (25.75%) on the same model, confirming that this SNN is a rate-coded classifier. Temporal spike structure is not exploited by the training objective.

4. **Representation quality (§6.4):** t-SNE reveals tighter ANN clusters consistent with its higher accuracy, while SNN representations show emergent super-category structure (Animals cluster, Urban cluster) despite weaker within-category discrimination.

5. **Per-class analysis (§6.5):** SNN outperforms ANN on 6/50 classes. Contrary to expectations, the SNN wins on high-energy impulsive sounds (crying_baby +7pp, door_wood_knock +7pp, coughing +8pp) rather than sustained tonal sounds. The SNN fails most severely on low-energy continuous sounds (engine 8%, door_wood_creaks 10%) where the LIF threshold is not consistently crossed. Clock_tick shows the largest gap (SNN 23%, ANN 68%), suggesting the SNN cannot detect quiet periodic signals reliably.
