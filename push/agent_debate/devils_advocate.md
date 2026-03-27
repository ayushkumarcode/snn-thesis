# Playing Devil's Advocate on Our ICONS Submission

putting myself in the shoes of the harshest possible reviewer. going through each contribution and finding the strongest objections.

---

## C1: First Convolutional SNN on ESC-50

### the strongest objection
being first to apply a well-established method (convolutional SNN with surrogate gradients, technique from 2018-2019) to a well-established benchmark (ESC-50, published 2015) is a weak form of novelty. the architecture is completely standard -- Conv2d, BatchNorm, MaxPool, LIF neurons. there's nothing architecturally new.

a hostile reviewer might say: "nobody did this because ESC-50 is considered solved for ANNs (98-99% SOTA) and the SNN community correctly identified that the dataset is too small (2000 samples) to be neuromorphically interesting."

also, Larroza et al. was submitted to ICASSP 2026, a more prestigious venue. if ICASSP reviewers accept Larroza with ESC-10 only, will ICONS reviewers view ESC-50 as meaningful extension or just "more classes"?

### my assessment
**medium risk.** survives because the prior work vacuum is genuinely confirmed by multiple surveys. but the paper needs to deliver scientific insight from those results, not just rely on being first. the encoding hierarchy, PANNs collapse, adversarial results are what make it worth publishing.

---

## C2: 7-Encoding Comparison

### the strongest objections

**problem 1:** three of the seven encodings never had a realistic chance of working. delta encoding on static mel spectrograms repeated across timesteps has literally zero temporal variation to detect. burst encoding front-loads spikes in 5/25 timesteps -- obviously creates temporal mismatch. including obviously ill-suited encodings inflates appearance of comprehensiveness.

**problem 2:** we don't compare against the most relevant recent encodings. Larroza uses TAE, SF, MW -- none of which we test. TAE gets 69.0% on ESC-10. a reviewer from Larroza's group will immediately ask: why was TAE not included?

**problem 3:** all seven encodings share the same hyperparameters (Adam lr=1e-3, patience=10, 50 epochs). rate, latency, and phase have different temporal dynamics and may need different learning rates. the comparison may penalize some encodings because of suboptimal shared hyperparameters, not because they're fundamentally inferior.

**problem 4:** the statistical significance claim is questionable. we report "paired t-test: p=0.001; Wilcoxon: p=0.0625." the Wilcoxon is ABOVE the 0.05 threshold. reporting this as if it supports significance while acknowledging it doesn't is cherry-picking.

### my assessment
**medium risk.** the comparison IS the largest in audio SNN literature, and that's well-supported. but a reviewer may accept the claim while questioning whether it's the most *informative* comparison. survivable with good rebuttal.

---

## C3: SpiNNaker Deployment -- This Is the Highest Risk

### the strongest objections

**problem 1:** we only deploy FC2 (256->50) on SpiNNaker. the conv layers, pooling, and FC1 all run in software. deploying a single 256->50 linear layer is trivial -- it's within undergraduate SpiNNaker tutorial scope. Dominguez-Morales (2016) deployed a full multilayer SNN. we actually deploy LESS of the network.

**problem 2:** the hardware gap (12.8 +/- 4.1pp) with 17.8pp range across folds suggests unreliable implementation, not just quantization issues. 64.5% agreement rate means 35.5% of samples are classified differently by hardware vs software.

**problem 3:** Dominguez-Morales classified audio on SpiNNaker. our distinction is "environmental" vs "pure tones." a reviewer could argue this is a semantic distinction trying to carry the weight of a major novelty claim.

**problem 4:** energy numbers for SpiNNaker aren't measured. wall-clock energy per sample is "left for future measurement."

**problem 5:** SpiNNaker 1 is antiquated by 2026 standards. SpiNNaker 2 has 10x better capacity per watt. demonstrating on 2012 hardware with 130nm process has limited value.

### my assessment
**HIGH risk.** combination of (a) only deploying 1/4 layers, (b) large variable accuracy gap, (c) semantic distinction with Dominguez-Morales, (d) no measured energy makes this the most vulnerable claim. needs to be framed as characterisation, not performance claim.

---

## C4: Adversarial Robustness

### the strongest objections

**problem 1:** Wang et al. (2025) explicitly warns that FGSM/PGD may underestimate SNN vulnerability due to surrogate gradient inaccuracies. we acknowledge this but still present 14.9x as a validated result. if the robustness is primarily gradient masking, the reported numbers aren't meaningful.

**problem 2:** originally single-fold (now 5-fold -- corrected). but at 400 samples and 50 classes, that's 8 samples per class. statistical reliability of per-epsilon numbers from such small per-class counts is questionable.

**problem 3:** clean accuracy differential complicates interpretation. SNN starts 15pp lower. the ANN's catastrophic drop may partly be because it had more to lose.

**problem 4:** "first on audio spectrograms" is a narrow novelty claim. SNN adversarial robustness is established for images (Sharmin ECCV 2020). this is domain transfer, not fundamental discovery.

### my assessment
**medium risk.** the novelty of "first on audio" is defensible. bigger risk is the result being challenged as methodological artifact. survivable if Wang et al. caveat is foregrounded rather than buried.

---

## C5: PANNs+SNN Gap Collapse

### the strongest objections

**problem 1:** the result is trivially expected. freezing CNN14 (trained on 2M AudioSet clips) and attaching a tiny SNN head that achieves 92.5% -- of course a reasonable classifier works on excellent features. the interesting question "does the SNN head learn different representations?" is not addressed.

**problem 2:** the 0.95pp gap between SNN and ANN heads may not be significant. confidence intervals overlap substantially with n=5 folds.

**problem 3:** PANNs+Linear (93.80%) beats both ANN and SNN heads. so the SNN head is actually the worst of three options. "why bother with spiking?"

**problem 4:** this is ANN-to-SNN transfer in frozen embedding form, a well-studied approach in vision. the paper doesn't distinguish itself adequately from conversion literature.

### my assessment
**medium risk.** the "gap collapse" finding IS genuinely useful framing. but needs to defend against "trivially expected" criticism by arguing why it wasn't obvious a priori.

---

## C6: NeuroBench Energy Analysis

### the strongest objections

this is not a novelty contribution -- it's using an existing tool on a new model. NeuroBench itself provides the framework. the SNN is 2.1x MORE expensive than the ANN, which undercuts the energy narrative. the "5.1x per-operation advantage on neuromorphic hardware" is from 45nm theoretical values, not measured SpiNNaker energy. SpiNNaker 1 costs ~5.8 uJ/SOP at 130nm -- orders of magnitude worse than theoretical 0.9 pJ/AC.

### my assessment
**HIGH risk as a standalone contribution.** should be absorbed into deployment section, not listed separately.

---

## Overall Assessment

| Claim | Challenge Severity | Novelty Risk | Fatal? |
|-------|-------------------|-------------|--------|
| C1: First SNN on ESC-50 | Moderate | MEDIUM | No -- dataset novelty is real but thin |
| C2: 7-encoding comparison | Significant | MEDIUM | No -- survivable with rebuttal on TAE |
| C3: SpiNNaker deployment | Severe | HIGH | Potentially -- partial deployment + semantic distinction |
| C4: Adversarial robustness | Significant | MEDIUM | Potentially -- gradient masking must be addressed |
| C5: PANNs+SNN gap collapse | Moderate | MEDIUM | No -- "trivially expected" is manageable |
| C6: NeuroBench energy | Severe | HIGH | Yes as standalone -- absorb into C3 |

### most likely reviewer split
- **one reviewer:** enthusiastic. appreciates comprehensive evaluation, recognizes gap analysis insight, values hardware attempt
- **one reviewer:** skeptical. raises SpiNNaker concerns, demands TAE comparison
- **one reviewer:** hostile. argues no novel method, no competitive results, partial deployment doesn't count

**the outcome depends on the rebuttal and framing.** if rejected, it'll be for the SpiNNaker partial deployment or the adversarial gradient masking concern, not the accuracy numbers.

the paper is borderline publishable at ICONS 2026, with ICONS being the correct venue precisely because it explicitly welcomes benchmark and application papers with modest accuracy results.
