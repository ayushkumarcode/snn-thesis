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

