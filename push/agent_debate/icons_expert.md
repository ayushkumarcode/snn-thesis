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

| Dimension | Yarga 2022 | Our Paper |
|-----------|-----------|-----------|
| Encodings | 4 | 7 |
| Task difficulty | 10-class digits | 50-class environmental sounds |
| Hardware deployment | None | SpiNNaker 5-fold |
| Energy analysis | None | NeuroBench |
| Adversarial | None | FGSM + PGD, 7 eps values |
| Transfer learning | None | PANNs+SNN, gap-collapse |
| Surrogate ablation | None | 8 surrogates |
| Continual learning | None | 5-task sequential |
| Negative results documented | Partial | Full |

we are substantially more comprehensive on every dimension. Yarga was accepted as a full paper. by the ICONS 2022 bar, we're stronger. by 2025 bar (conference has grown, quality may have risen), still competitive if hardware story is well-framed.

---

## Is 47.15% "Significant" at ICONS?

**yes.** ICONS reviewers understand 47.15% on 50 classes (random=2%) represents 45.15pp improvement over chance. they won't compare to 98.25% ANN SOTA without context. they'll ask: given the architecture, dataset size, and training approach, what does this tell us about current SNN capability?

the ICONS community's reference is other SNN papers, not ANN leaderboards. 47.15% as the first SNN result on this task -- there's no prior bar to beat. ICONS 2025 best paper was about turbulence modeling with no accuracy metric at all.

the PANNs result rehabilitates: 92.5% proves the 47.15% gap isn't inherent to spiking. the gap-collapse insight is a cleaner and more interesting story than just achieving high accuracy.

frame the 47.15% as "establishing a baseline for future SNN audio work" and the collapse as "demonstrating SNN formalism is not the bottleneck." don't apologize for the number.

---

