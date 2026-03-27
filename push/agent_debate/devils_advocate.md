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

