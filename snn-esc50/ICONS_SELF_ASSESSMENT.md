# ICONS 2026 Simulated Reviewer Assessment

Paper: "SpiNNaker Deployment of Pruned Spiking Neural Networks for 50-Class Sound Classification"

This document simulates a strict ICONS reviewer assessment, drawing on the standards observed across ICONS 2022--2024 proceedings and the broader neuromorphic computing publication landscape.

---

## Review Criteria Scores (1--5 scale)

| Criterion | Score | Justification |
|-----------|-------|---------------|
| Novelty | 4/5 | First SNN on ESC-50 is verifiable and genuine. Pruning-as-regularization for hardware deployment is a fresh angle. 7-encoding comparison extends ICONS 2022 work meaningfully. Docked 1 point because the Rhythm SNN contribution is thin (limited ablation in this 6-page version) and the hybrid deployment is partial. |
| Technical Soundness | 4/5 | Controlled ANN-SNN comparison is methodologically rigorous. 5-fold CV on predefined ESC-50 folds follows standard protocol. NeuroBench energy methodology is appropriate. Statistical tests are reported. Docked 1 point for: (a) FC2-only deployment weakens the "SpiNNaker deployment" claim, (b) adversarial analysis uses standard PGD rather than SNN-adapted attacks, (c) energy is estimated not measured. |
| Significance | 3/5 | The pruning-as-regularization finding is interesting but the practical impact is limited by the partial deployment and moderate accuracy. ESC-50 is a useful benchmark but the community already has well-established SNN benchmarks (SHD, DVS-Gesture) that would strengthen the case. The gap-collapse finding (16.7 pp to 0.95 pp) is the most significant result but is somewhat peripheral to the paper's stated focus on pruning. |
| Presentation | 4/5 | Paper is well-organized with clear section structure. Figures are informative (architecture diagram, pipeline, bar chart, Pareto curve). Tables use booktabs formatting and are easy to read. Writing is concise and formal. Docked 1 point for: the paper attempts to cover too many results (7 encodings, Rhythm SNN, pruned deployment, adversarial robustness, gap collapse, temporal efficiency, continual learning) in 6 pages, risking a "laundry list" impression. |
| Reproducibility | 4/5 | Code and models are linked (GitHub). All hyperparameters are specified. SpiNNaker deployment parameters are documented. 5-fold CV protocol is standard. Docked 1 point because: SpiNNaker hardware access is institution-specific (not everyone can reproduce the hardware experiments), and some implementation details (core splitting, membrane initialization) require familiarity with sPyNNaker. |

**Aggregate Score: 19/25 (76%)**

---

## Overall Recommendation

**Weak Accept (3.5/5)**

The paper presents genuine novelty (first SNN on ESC-50, first neuromorphic audio deployment beyond pure tones) and a technically sound experimental methodology. The pruning-as-regularization finding for hardware deployment is the most interesting contribution and is well-supported by 22,000 hardware inferences across 10 sparsity levels. However, the partial deployment (FC2 only), the breadth-over-depth presentation, and the moderate absolute accuracy weaken the overall impact. The paper is above the typical ICONS acceptance bar but not by a large margin.

---

## Detailed Strengths

### S1: Verified Novelty
The claim of "first SNN on ESC-50" is verifiable and appears to be accurate. The most recent comparable work (Larroza et al., arXiv March 2025) evaluates only ESC-10 with FC networks. The extension to 50 classes with convolutional architectures and hardware deployment is non-trivial and fills a genuine gap in the literature. This alone meets ICONS's stated interest in "domain-specific implementations."

### S2: Comprehensive Hardware Sweep
22,000 hardware inferences across 10 pruning levels with 5-fold cross-validation is unusually thorough for a neuromorphic deployment paper. Most ICONS deployment papers report a single configuration on a single dataset split. The systematic sweep enables the pruning-as-regularization finding, which would not be visible from a single deployment.

### S3: Honest Failure Documentation
The root-cause analysis of the full-deployment failure (AvgPool fractional outputs causing excitatory-inhibitory cancellation) is valuable. The paper does not hide this failure; it documents it, explains why it occurs, and presents a validated fix (MaxPool replacement, verified at 43.75% on fold 4). This transparency is scientifically responsible and practically useful for the community.

### S4: Community Integration
The paper builds explicitly on prior ICONS work (Yarga et al. 2022 encoding comparison, Seekings et al. 2024 hybrid deployment). The 7-encoding comparison is a direct extension of the 4-encoding ICONS 2022 paper. This lineage shows awareness of the ICONS community and positions the work as a natural next step.

### S5: Controlled Experimental Design
The matched ANN-SNN comparison (identical architecture, 622K params, same training protocol) isolates the effect of spiking computation. This is the correct methodology that many papers in the field fail to follow. The PANNs gap-collapse experiment further isolates the feature-learning vs. spiking-classification question.

### S6: Multiple Quantified Findings
The paper produces several concrete, numbered findings: +3.1 pp pruning improvement on SpiNNaker, 0.6 pp hardware gap at 85% sparsity, transfer ratio of 0.255, 6.0x adversarial robustness, 16.7 pp to 0.95 pp gap collapse. Each is supported by 5-fold validation and (where appropriate) statistical tests.

---

## Detailed Weaknesses

### W1: Partial Hardware Deployment (Major)
Only FC2 (12,800 of 621,906 parameters, ~2% of the model) runs on SpiNNaker. The convolutional feature extraction, which accounts for 98% of computation, runs on conventional hardware. While the paper is transparent about this and the hybrid approach follows Seekings et al. (ICONS 2024), the title "SpiNNaker Deployment" may overstate the contribution. A reviewer unfamiliar with the hybrid paradigm may view this as a significant gap between the claim and the reality.

**Impact on score:** -0.5 on Novelty, -0.5 on Technical Soundness

### W2: Breadth Over Depth (Moderate)
The 6-page paper covers: 7-encoding comparison, Rhythm SNN, pruned SpiNNaker deployment (10 levels), adversarial robustness (FGSM + PGD), gap collapse with PANNs, temporal efficiency, continual learning, surrogate gradient ablation, and energy analysis. Each individual contribution receives only 1-2 paragraphs. A more focused paper (e.g., just pruning-as-regularization with deep analysis) might be more impactful.

The continual learning result (4.8 pp less forgetting) and surrogate gradient ablation are interesting but feel like they belong in a different paper. Including them dilutes the central message.

**Impact on score:** -0.5 on Presentation, -0.5 on Significance

### W3: Moderate Absolute Accuracy (Minor)
61.10% on ESC-50 (vs. 81.3% human, 99%+ SOTA ANN) may give pause to reviewers from the audio ML community, though this is less concerning for the ICONS neuromorphic audience. The gap-collapse result (92.50% with CNN14 features) mitigates this concern but is a separate experimental setup from the SpiNNaker deployment.

**Impact on score:** -0.3 on Significance

### W4: Energy Analysis Limitations (Minor)
NeuroBench operation counting, while standard, does not capture SpiNNaker's actual power characteristics (idle power, communication overhead, core management). The claimed 3.5x energy reduction is for the FC2 layer only; the full-system energy including the GPU-based feature extraction is not reported.

**Impact on score:** -0.3 on Technical Soundness

### W5: Statistical Power (Minor)
With n=5 folds, statistical tests have limited power. The negative hardware gaps (-0.45 pp and -0.30 pp) are within the standard deviation of measurements. While the paper does not overclaim these results, a skeptical reviewer may question whether the pruning-as-regularization effect is robust or partly a statistical artifact of the fold structure.

**Impact on score:** -0.2 on Technical Soundness

---

## Questions for Authors (Rebuttal Phase)

1. **Pruning mechanism.** You claim pruning acts as "implicit regularization for hardware deployment." Can you quantify the quantization error (mean absolute error between float32 and integer weights) as a function of sparsity level? This would directly test whether the improvement comes from reduced quantization noise, as hypothesized.

2. **Rhythm SNN ablation.** What is the accuracy of a standard SNN (no oscillatory modulation) at T=3? This number is needed to separate the contribution of reduced timesteps from the oscillatory threshold.

3. **Full deployment feasibility.** You verified MaxPool replacement on fold 4 (43.75%). Can you provide the 5-fold result, and how does this compare to the hybrid deployment (57.35%)?

4. **Negative gap reproducibility.** Are the negative hardware gaps at 60% and 80% consistent across individual folds, or driven by 1-2 folds? Reporting per-fold SpiNNaker vs. snnTorch differences would clarify this.

5. **Energy fairness.** The SNN energy (4,706 nJ unpruned) is 10x higher than the ANN (470 nJ). The paper frames pruning as reducing this gap, but even at 95% sparsity (747 nJ), the SNN is still more energy-expensive than the ANN. How do you reconcile the energy motivation with these numbers?

6. **Encoding comparison scope.** You evaluate 7 encodings but deploy only direct-encoding models on SpiNNaker. Have you deployed any other encoding on SpiNNaker? The interaction between encoding method and hardware quantization could be informative.

7. **Continual learning and adversarial robustness.** These sections feel disconnected from the main pruning-and-deployment narrative. Would the paper be stronger without them, using the freed space for deeper analysis of the pruning mechanism?

---

## Comparison with Typical ICONS Papers

### Based on ICONS 2022--2024 Proceedings Review

**Paper quality range at ICONS:**
ICONS accepts papers across a wide quality spectrum, from undergraduate research projects to national laboratory results. The conference has historically had a ~60% acceptance rate (13/22 in 2018; recent data not public). Papers range from 4-page short papers to 8-page full papers.

**Typical ICONS paper characteristics:**
- Novel hardware deployment or simulation result (often a single platform, single task)
- 1-2 main experiments, not 7+ as in our paper
- Modest datasets and modest accuracy numbers are acceptable if the hardware contribution is clear
- Strong preference for papers that deploy on real hardware (SpiNNaker, Loihi, analog, memristor)
- Application papers (audio, vision, robotics, materials science) are welcomed
- Benchmarking papers (encoding comparisons, algorithm comparisons) have a home here (Yarga 2022)

**How our paper compares:**

| Dimension | Typical ICONS Paper | Our Paper | Assessment |
|-----------|-------------------|-----------|------------|
| Hardware deployment | Single config, 1 dataset | 10 pruning levels, 22K inferences | Above average |
| Dataset complexity | Simple (MNIST, digits, gestures) | ESC-50 (50 classes) | Above average |
| Number of experiments | 2-4 | 7+ distinct analyses | Well above average (risk: too many) |
| Statistical rigor | Often minimal (no p-values) | 5-fold CV, paired t-tests, effect sizes | Above average |
| Novelty claim | Incremental improvement | First SNN on ESC-50 + first neuromorphic audio deployment | Above average |
| Absolute accuracy | Varies widely | 61.10% (moderate) | Average |
| Writing quality | Variable | Professional, concise | Above average |
| Page count | 4-8 pages | 6 pages | Standard |

**Positioning relative to ICONS award-winning papers:**
- ICONS 2024 best papers (Schone et al., Hassan et al., Primavera et al.) are typically from established labs with deeper single contributions. Our paper's breadth is a stylistic mismatch with best-paper expectations, but the hardware deployment novelty and comprehensive sweep are competitive for a standard acceptance.

**Estimated acceptance probability: 65-75%**

The paper is above the median ICONS submission in terms of novelty, experimental rigor, and hardware deployment scale. The main risks are: (1) a reviewer who views the partial FC2 deployment as disqualifying, (2) a reviewer who wants deeper analysis of fewer results, or (3) a reviewer from the audio ML community who focuses on the 61% accuracy number.

---

## Verdict Summary

| Aspect | Rating |
|--------|--------|
| Novelty | 4/5 -- Genuine firsts (ESC-50, neuromorphic audio deployment) |
| Technical Soundness | 4/5 -- Rigorous methodology with acknowledged limitations |
| Significance | 3/5 -- Interesting findings but moderate practical impact |
| Presentation | 4/5 -- Well-written but tries to cover too much |
| Reproducibility | 4/5 -- Code available, parameters documented, hardware access limited |
| **Overall** | **Weak Accept (3.5/5)** |

**Recommendation:** Accept as a full paper. The first-SNN-on-ESC-50 novelty and the 22,000 hardware inference sweep are the strongest selling points. Recommend the authors focus the narrative more tightly on the pruning-as-regularization finding and move the adversarial/continual learning results to supplementary material or a separate paper.

---

## Suggested Revisions for Camera-Ready (if Accepted)

1. **Tighten the narrative.** Lead with the pruning-as-regularization finding as the central thesis. Move encoding comparison and adversarial robustness to supporting roles.

2. **Add the T=3 standard SNN baseline.** This ablation separates the Rhythm SNN's oscillatory contribution from the timestep reduction effect.

3. **Report per-fold SpiNNaker results.** A supplementary table with per-fold accuracy for each pruning level would strengthen the statistical claims.

4. **Quantify FC2's share of computation.** Stating that FC2 is ~2% of parameters but receives ~X% of synaptic events would contextualize the partial deployment.

5. **Remove or compress continual learning.** The 4.8 pp forgetting reduction is preliminary and disconnected from the main narrative. A single sentence mentioning it as an auxiliary finding would suffice.

6. **Add confidence intervals to Table 4.** The pruning sweep results should show 95% confidence intervals alongside standard deviations, especially for the negative gap claims.

7. **Soften the adversarial robustness framing.** Change "6.0x more robust" to "6.0x more robust under standard FGSM attack" and acknowledge SA-PGD as future work more prominently.
