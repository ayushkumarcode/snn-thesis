# ICONS Expert Assessment -- What the Conference Actually Values

thinking through what ICONS reviewers actually care about, based on analyzing 50+ papers from 2022-2025.

---

## What ICONS Rewards (in rough order of weight)

**1. Hardware-grounded work.** papers that actually touch neuromorphic hardware get immediate credibility. SpiNNaker, Loihi, BrainScaleS, Xylo, Akida papers have been in both full-talk and lightning-talk slots consistently. a paper without hardware can still be accepted, but hardware = strong positive signal.

**2. First-time benchmark establishment.** ICONS explicitly welcomes "benchmark tasks for neuromorphic computing" in its CFP. establishing that a domain has never been benchmarked under SNN conditions is a recognized contribution.

**3. Algorithmic novelty with neuromorphic motivation.** methods papers need a clear story about why the approach matters for neuromorphic execution. the 2025 best paper (turbulence modeling) had no accuracy metric at all -- accepted for novel connection between neuron dynamics and physical systems.

**4. Systematic ablation or comparison studies.** papers rigorously comparing design choices are valued as reference works. Yarga et al. ICONS 2022 (4-encoding speech comparison) is the canonical example.

**5. Honest negative results with mechanistic explanation.** ICONS accepts papers where the SNN doesn't win, provided the authors explain why. the community is mature enough to value "we tried this and it failed, here is why."

**NOT primarily rewarded:** raw accuracy vs ANN SOTA, parameter efficiency in isolation, leaderboard-beating, applications with no neuromorphic angle.

---

## What Reviewers Will Respond To Positively

**A. "First SNN on full ESC-50" claim.** clean, verifiable. reviewers can check this. Larroza only covers ESC-10. Dominguez-Morales uses pure tones. the claim is watertight.

**B. 7-encoding comparison.** most comprehensive in SNN audio. Yarga (ICONS 2022) did 4 on speech digits. reviewers who know Yarga will see this as meaningful extension.

**C. SpiNNaker deployment.** hardware work gets immediate credibility at ICONS. the validated FC2-only hybrid with documented root-cause analysis of FC1 cancellation is actually better than a naive full deployment -- it shows understanding of what hardware requires. 5-fold CV on SpiNNaker is methodologically stronger than most hardware papers.

**D. Adversarial robustness.** 14.9x advantage is striking. the growing ICONS security subgroup ("Do Spikes Protect Privacy?" at 2025, "Neuromorphic Cybersecurity") will find this compelling.

**E. PANNs gap collapse.** reframes the narrative from "SNN underperforms" to "bottleneck is feature learning, not spiking." novel and clean insight.

**F. NeuroBench compliance.** signals alignment with community standards.

---

## What Reviewers Will Flag

**A. FC2-only deployment.** reviewers will ask "if most compute is in software, what's the hardware contribution?" need to pre-empt by explaining FC1 cancellation root cause as a design insight.

**B. 12.8pp hardware gap.** need to contextualize against DYNAP-SE (7.1pp on simpler task) and Loihi 2 (near-zero with QAT).

**C. Only 1,600 training samples.** reviewer may argue 47.15% reflects data scarcity not fundamental SNN property. PANNs partially addresses this.

**D. Surrogate ablation single-seed.** needs labeling as preliminary if 3-seed results aren't ready.

**E. Energy framing.** SNN is 2.1x MORE expensive in software. reviewers who know Dampfhoffer will recognize 25.8% spike rate exceeds break-even. must not claim software energy efficiency.

**F. Continual learning thin.** 6.9pp advantage is modest, one fold. supporting result, not headline.

---

## Comparison to Yarga et al. ICONS 2022

