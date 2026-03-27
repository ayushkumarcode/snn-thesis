# Final Verdict: Is This Publishable at ICONS 2026?

after going back and forth on all the arguments for and against, here's where i've landed on each contribution.

---

## What's Genuinely Novel (assessed honestly)

### Claim 1: First convolutional SNN on full ESC-50
**strong.** no prior publication reports SNN accuracy on full 50-class ESC-50. confirmed by Larroza et al. explicitly stating "no state-of-the-art solution has yet encoded environmental sound datasets using spike-based methods" (and they only evaluate ESC-10), plus two independent 2025 surveys finding nothing. ESC-50 is a meaningful benchmark -- predefined 5-fold CV, standard since 2015, substantially harder than ESC-10. filling this gap creates the reference point for all future SNN audio work.

### Claim 2: 7-encoding comparison
**strong.** the most comprehensive encoding comparison in SNN audio. next best is Yarga (ICONS 2022) with 4 on speech digits. the ranking (direct >> rate ~ phase > population > latency >> delta ~ burst) is internally consistent with mechanistic explanations for each. the rate-phase tie (equivalent accuracy at 7x different spike counts) is a genuine insight about temporal vs rate representations. the burst coding failure analysis (front-loading -> temporal window mismatch -> severe overfitting) is interpretable and useful. this is not just a horse race -- it's a controlled experiment with causal stories.

### Claim 3: PANNs+SNN gap collapse
**strong.** no prior work combines frozen AudioSet features with an SNN classifier. the 17.6x collapse ratio (16.7pp -> 0.95pp) exceeds the ~7x reported in vision (Spikformer V2). the interpretation -- spiking computation is not the bottleneck, feature learning is -- is a claim with field-level implications. it reframes the research agenda from "how to make spiking better" to "how to give SNNs better features." one honest caveat: the pattern is visible in vision conversion literature, so it doesn't shock the theory. but it's the first audio-domain empirical evidence, clearest quantification, and cleanest experimental design.

### Claim 4: First SpiNNaker deployment for environmental sound
**strong at ICONS specifically.** the FC1 cancellation root-cause analysis (AvgPool -> fractional outputs incompatible with SpiNNaker's binary input requirement), the FC2-only hybrid validation at 5-fold, and the Option A path to full deployment are all engineering contributions that help future researchers. the 12.8pp gap quantification across 5 folds is itself a scientific measurement -- first of its kind for audio on SpiNNaker. at a general ML venue, 33% accuracy would be a weakness. at ICONS, which values systematic hardware characterization, it's a valid contribution.

### Claim 5: First adversarial robustness on audio spectrograms
**moderate-to-strong.** genuinely new in the audio domain. the magnitude (14.9x at FGSM eps=0.1) is striking, and the mechanistic explanation (binary thresholding as natural high-frequency noise filter) is sound. but Wang et al. (2025) warning about FGSM/PGD overestimating SNN robustness due to surrogate gradient inaccuracies adds a real caveat. the finding remains meaningful even with uncertainty about exact numbers -- ANN at 1.75% and SNN at 26% are qualitatively different regimes regardless of gradient masking concerns.

### Claim 6: NeuroBench energy analysis
**moderate.** correct application of an existing tool with honest reporting. not a primary novelty claim -- using NeuroBench on a new model isn't a research contribution, it's benchmarking. valuable but should be absorbed into the deployment section, not listed separately.

### Claim 7: Surrogate gradient ablation
**weak-to-moderate.** useful practical result (don't use sigmoid or STE for audio SNNs), but surrogate comparisons exist in vision. the bimodal failure pattern IS interesting and challenges Zenke 2021's "shape doesn't matter" claim. but single-seed single-fold limits confidence.

### Claim 8: Continual learning
**weak.** novel domain (first for audio SNNs), but 6.9pp difference in catastrophic forgetting is modest. no CL baselines (EWC, PackNet). descriptive result, not mechanistic. supporting finding at best, not headline.

---

## What's NOT Novel or Risky

- **47.15% absolute accuracy** -- doesn't advance SOTA. the contribution is the comparative analysis, not the number.
- **architecture** -- completely standard Conv2d-BN-MaxPool-LIF. no architectural innovation.
- **SpiNNaker accuracy gap may be questioned** -- FC2-only is honest but some reviewers will argue it's not real deployment.
- **adversarial robustness caveat** -- gradient masking means exact numbers are uncertain.
- **continual learning lacks baselines** -- no EWC, PackNet, replay comparison.
- **augmented training negative result** -- honest but doesn't add novelty.
- **PANNs+SNN at 92.5% is driven by CNN14** -- the scientific contribution is the gap analysis, not the headline number.

---

## The Scientific Story

### the single most important finding

**the SNN-ANN accuracy gap on audio classification is a feature-learning problem, not a spiking computation problem.**

scratch SNN: 47.15%. scratch ANN: 63.85%. gap: 16.7pp. identical architectures except LIF vs ReLU.

PANNs+SNN head: 92.50%. PANNs+ANN head: 93.45%. gap: 0.95pp. same architecture, same features, only activation differs.

