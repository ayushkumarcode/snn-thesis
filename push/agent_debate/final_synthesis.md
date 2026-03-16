# Final Verdict: Publishable Novelty Assessment for ICONS 2026

**Reviewer role:** Neutral senior researcher
**Date of assessment:** 9 March 2026
**Subject:** UoM COMP30040 undergraduate thesis — Convolutional SNN for ESC-50 Environmental Sound Classification
**Target venue:** ICONS 2026 (ACM International Conference on Neuromorphic Systems), deadline April 1 2026

---

## Part 1: What Is Genuinely Novel (Line by Line)

### Claim 1: First convolutional SNN on the full ESC-50 benchmark (50 classes, 5-fold CV)

**The claim.** No prior publication reports SNN accuracy on the full 50-class ESC-50 dataset. The only comparable work, Larroza et al. (arXiv:2503.11206, submitted March 2025), explicitly states "no state-of-the-art solution has yet encoded environmental sound datasets using spike-based methods" — and themselves evaluate only ESC-10 (10 classes) with fully-connected networks. Two independent 2025 survey papers (Basu et al. arXiv:2502.15056; Baek & Lee, Biomedical Engineering Letters 14:5) confirm no ESC-50 SNN results in the literature.

**Why this is novel, not just first.** First results are not automatically interesting; what matters is whether the benchmark is meaningful. ESC-50 is: it has predefined 5-fold CV making results directly reproducible and comparable, it has been a standard ANN benchmark since Piczak 2015, and it is substantially harder than ESC-10 (5× more classes, more acoustic diversity). Filling this gap is a real service to the field: it establishes a baseline from which all future SNN audio work can measure progress. The use of a convolutional architecture (rather than FC-only) also represents a step up in architectural ambition that matches what the ANN field has known to be necessary for this task since 2015.

**Novelty strength: Strong.**

---

### Claim 2: Systematic comparison of 7 spike encoding methods on the same architecture and dataset

**The claim.** This thesis compares 7 encodings (direct, rate, phase, population, latency, delta, burst) on a single fixed architecture and fixed dataset. The next most comprehensive study is Larroza et al. (2025) with 3 encodings on ESC-10. Yarga et al. (ICONS 2022) compared 4 on speech digits. No prior paper has compared 7 encodings for any audio SNN task.

**Why this is novel.** The encoding hierarchy found — direct (47.15%) >> rate (24.00%) ≈ phase (24.15%) > population (19.15%) > latency (16.30%) >> delta (7.25%) ≈ burst (6.50%) — is not something the field could have predicted in advance. The result that phase coding matches rate coding with exactly 1 deterministic spike per neuron vs approximately 7 stochastic spikes is a genuine insight about the information content of temporal vs rate representations in this setting. The finding that burst coding catastrophically fails (6.5%, near random) and the mechanistic explanation (front-loaded spikes create temporal window mismatch with LIF integration) are both interpretable and useful to the field. This is not a "horse race" contribution; it is a controlled experiment with a clear causal story for each outcome. The ICONS 2022 Yarga paper is the direct precursor, and this work meaningfully extends it in scope, dataset difficulty, and depth of analysis.

**Novelty strength: Strong.**

---

### Claim 3: The PANNs + SNN head transfer learning experiment and the gap-collapse finding

**The claim.** No prior paper combines a frozen AudioSet-pretrained audio model (PANNs CNN14) with an SNN classifier head. After exhaustive search, zero prior works were found using any audio foundation model (PANNs, VGGish, wav2vec, HuBERT, CLAP, BEATs, AST) as frozen features for an SNN classifier. The result — 92.50% SNN vs 93.45% ANN on the same features, a 0.95 percentage point gap — is the first quantified demonstration of the feature-learning bottleneck hypothesis in audio.

**Why this is novel.** The scientific value here is not the 92.50% number. It is the 17.6× collapse ratio: scratch gap is 16.70 pp, features-equal gap is 0.95 pp. This ratio exceeds what has been reported in vision (Spikformer V2 achieves roughly 7× collapse going from scratch to SSL pretraining on ImageNet). The interpretation — that spiking computation itself is not the bottleneck, that LIF neurons can classify just as well as ReLU neurons when both get equal-quality representations — is a claim with implications for how the entire field should think about SNN accuracy gaps. It reframes 47.15% vs 63.85% as a data-scale and feature-learning problem, not a fundamental limitation of the spiking formalism. The closest parallel in the literature is SAFE (ICLR 2025 submission, later withdrawn), which uses CNN features plus an SNN classifier for fake audio detection, but SAFE does not measure or discuss the gap collapse. This thesis is the first to frame it explicitly as a scientific finding and quantify the ratio.

One honest caveat: the gap-collapse finding is consistent with a pattern already visible in vision (ANN-to-SNN conversion literature, Spikformer V2), so it does not shock the theory. What it adds is the first audio-domain empirical evidence, the clearest quantification, and the cleanest experimental design (same architecture, only activation type differs). That is enough.

**Novelty strength: Strong.**

---

### Claim 4: First neuromorphic hardware deployment for environmental sound classification beyond pure tones

**The claim.** The only prior SNN-on-SpiNNaker audio paper is Dominguez-Morales et al. (ICANN 2016), which classifies 8 pure synthetic tones (130–1397 Hz) — a task that is trivially simple compared to 50-class environmental sound. This thesis deploys an SNN on SpiNNaker for ESC-50, achieves 33.1% ± 6.9% across 5 folds, characterises the hardware-software accuracy gap (12.8 ± 4.1 pp), and documents the root cause of that gap (FC1 input activation distribution incompatibility with integer weights and binary-threshold SpiNNaker neurons).

**Why this is novel.** Hardware deployment papers at ICONS are first-class contributions, not ancillary ones. The community values honest characterisation of what works and what does not on real hardware. The fact that the result is 33.1% rather than matching the 47.15% software result is not a weakness — it is the scientific content. The FC1 cancellation root-cause analysis (near-zero-mean weights across 1398 simultaneous binary inputs producing net negative current) and the FC2-only hybrid solution are exactly the kind of insight that accelerates subsequent work by others attempting SpiNNaker deployment. The Option A and Option C experiments further document what does and does not transfer from software to hardware in a systematic way. There is nothing comparable in the literature for environmental sound on any neuromorphic chip.

**Novelty strength: Strong at ICONS specifically.** For a general ML venue, hardware deployment at 33% accuracy would be a weakness. At ICONS, which explicitly includes systematic hardware characterisation as a category of valued contribution, this is well within the normal profile of accepted hardware papers.

---

### Claim 5: First adversarial robustness analysis of SNNs on audio spectrograms

**The claim.** All prior adversarial SNN robustness work is in the vision domain (Sharmin et al. ECCV 2020; Wang et al. arXiv:2512.22522 2025). No paper has studied FGSM or PGD attacks on audio SNNs. The results here are striking: at FGSM ε=0.1, SNN retains 26.00% accuracy while the matched ANN collapses to 1.75%. At PGD ε=0.05, SNN retains 19.25% while ANN reaches 0%.

**Why this is novel.** The quantitative gap (14.8× advantage at ε=0.1 FGSM) is larger than what has been reported in vision. The interpretive explanation — that binary thresholding in LIF neurons acts as a natural high-frequency noise filter, attenuating gradient-based perturbations that do not reach threshold — is mechanistically sound and has not been stated in the audio context before. The caveat that Wang et al. (2025) warns that FGSM/PGD may underestimate SNN vulnerability due to surrogate gradient inaccuracies must be acknowledged and is acknowledged, but the qualitative finding of robustness advantage remains meaningful even if the exact numbers are uncertain. This is a novel empirical result with a clear practical implication: SNNs may offer a robustness advantage for safety-critical audio sensing applications without any adversarial training.

**Novelty strength: Moderate-to-Strong.** The result is genuinely new in the audio domain, but the vision-domain context means a savvy reviewer will immediately contextualise it. The uncertainty about surrogate gradient robustness evaluation also adds a caveat that reduces confidence in the exact numbers.

---

### Claim 6: NeuroBench-compliant energy analysis with honest framing

**The claim.** The thesis applies NeuroBench (Yik et al., Nature Communications 2025) to measure SNN energy (976 nJ/sample, 1.08M ACs) and ANN energy (463 nJ/sample, 101K MACs) and explicitly acknowledges the SNN uses more energy in software simulation.

**Why this is notable.** Most SNN papers either avoid energy comparison or cherry-pick metrics that favour SNNs. This paper does neither. It correctly identifies that the SNN is 4× above the break-even spike rate for software simulation, and that the energy advantage only materialises on hardware where ACs are 5.1× cheaper than MACs. The honest framing — "on dedicated neuromorphic hardware, ACs cost 5.1× less than MACs, making the SNN hardware-compatible for AC-only execution — though total energy remains higher due to T=25× more operations" — is more rigorous than typical published claims. The use of NeuroBench as a standardised framework rather than a custom metric also makes the results directly comparable to other NeuroBench-reported systems.

**Novelty strength: Moderate.** This is not a new scientific finding; NeuroBench itself provides the framework. The contribution is correct application and honest reporting, which is genuinely valuable but is not a primary novelty claim.

---

### Claim 7: Surrogate gradient ablation across 8 surrogate functions for audio SNN training

**The claim.** This thesis tests 8 surrogate gradient functions (spike_rate_escape, fast_sigmoid, atan, STE, sigmoid, sfs, triangular, LSO) on fold 1 of ESC-50. Key finding: bimodal split — 3 learning surrogates (sre=46%, fast_sigmoid=44.75%, atan=35.75%) vs 4 failing surrogates (STE=10.25%, sigmoid=2%, sfs=2%, triangular=2.75%). Sigmoid failure is surprising given Zenke (2021) claims shape matters less than slope.

**Why this is notable.** The bimodal failure pattern is unexpected and the sigmoid failure specifically contradicts prior theoretical predictions. No prior paper has done this comparison for audio SNNs. However, surrogate gradient comparisons for vision SNNs do exist (e.g., Li et al. IJCAI 2023), so the methodological approach is not novel — the application to audio is the novelty. The finding is useful to future audio SNN practitioners.

**Novelty strength: Weak-to-Moderate.** Useful result, but surrogate gradient ablation is established methodology and the audio-domain application does not change the interpretation substantially.

---

### Claim 8: Continual learning forgetting analysis for audio SNNs

**The claim.** SNN forgetting across 5 ESC-50 super-categories: 74.4%. ANN forgetting: 81.3%. SNN forgets 6.9 pp less. No prior work studies continual learning for audio SNNs.

**Why this is notable.** The result is positive for SNNs and the domain is genuinely unexplored. However, a 6.9 pp difference in catastrophic forgetting on a small dataset is not a dramatic finding by continual learning standards, and the experimental setup (5 sequential tasks, 10 classes each, fold 4 only, 20 epochs per task) is modest. The absence of standard continual learning baselines (EWC, PackNet, replay) means the 6.9 pp advantage cannot be attributed to any specific mechanism. It is a descriptive result, not a mechanistic one.

**Novelty strength: Weak.** Domain is novel (first for audio SNNs), but the result is not striking enough to stand alone, and the methodology lacks baselines that would allow attribution.

---

## Part 2: What Is NOT Novel or Is Risky

### 2.1 The 47.15% absolute accuracy number

This will be the first thing adversarial reviewers look at. ESC-50 ANN SOTA is 99.1% (OmniVec2, CVPR 2024). Human performance is 81.3%. The matched ANN baseline is 63.85%. The scratch SNN achieves 47.15%. None of these numbers advance the state of the art for ESC-50. The paper does not claim they do, but reviewers from outside the neuromorphic community may apply the wrong evaluation frame. This is not a novelty problem — the novelty is the encoding comparison and analysis, not the absolute number — but it is a framing risk. The paper must be submitted to a venue (ICONS) that evaluates methodology and hardware, not accuracy rank tables.

### 2.2 The architecture itself is not novel

Conv2d → BN → MaxPool → LIF, repeated twice, then FC layers is a standard SNN architecture pattern from the snnTorch tutorial literature. There is no architectural innovation here. The contribution is the systematic evaluation and analysis, not the design of a new network. This is acceptable for ICONS but would be a rejection criterion at ICLR or NeurIPS.

### 2.3 The SpiNNaker accuracy gap may be questioned

33.1% SpiNNaker vs 47.15% software (12.8 pp gap) will prompt questions about whether the FC2-only hybrid approach is a fair evaluation of "SpiNNaker deployment." Technically, the SNN is not fully deployed on SpiNNaker — FC1 and LIF layers 1-3 run in software. The paper documents this honestly, but a reviewer could argue this is not a real hardware result. The counter-argument — that the hybrid approach is itself the novel contribution, documenting what can and cannot run on SpiNNaker and why — is correct, but requires clear framing in the paper.

### 2.4 The adversarial robustness result has a known methodological caveat

Wang et al. (arXiv:2512.22522, 2025) argue that standard FGSM/PGD attacks may underestimate SNN vulnerability due to the gradient approximation through the surrogate function. This means the SNN's apparent robustness advantage (26% vs 1.75% at ε=0.1) could partially reflect gradient masking rather than genuine robustness. The paper should acknowledge this explicitly and note it as a limitation. Failure to do so would invite a sharp reviewer comment. The finding is still worth reporting as a first-of-its-kind result, but the caveat must be foregrounded.

### 2.5 The continual learning experiment lacks baselines

The forgetting comparison (SNN 74.4%, ANN 81.3%) is reported without any standard continual learning baselines. There is no EWC, no PackNet, no replay buffer comparison. The result describes what happens, not why. For a dedicated continual learning paper, this would be a fatal flaw. At ICONS, in an 8-page paper where it appears as one of several secondary results, it is acceptable as a descriptive finding, but it should be framed explicitly as preliminary.

### 2.6 Augmented training is a negative result, not a novel one

Data augmentation hurting SNN training on small datasets is consistent with known SNN sensitivity to training instability. It is an honest result to report, but it does not add novelty to the paper.

### 2.7 The PANNs+SNN result at 92.5% is driven almost entirely by CNN14

CNN14 was trained on 2 million AudioSet clips. The SNN head trained on its frozen embeddings only needs to classify 50 classes from already-excellent representations. The 92.5% is real and reproducible, but it should not be presented as "the SNN achieving 92.5% accuracy." It is more precisely "an SNN classifier head trained on CNN14 features achieving 92.5%." The scientific contribution is the gap analysis, not the 92.5% headline. Reviewers who misread this framing will object.

---

## Part 3: The Scientific Story

### The single most important finding

**The SNN-ANN accuracy gap on audio classification is a feature-learning problem, not a spiking computation problem.**

This is the one finding worth remembering from this thesis. Everything else — the 47.15% encoding result, the SpiNNaker deployment, the adversarial robustness — is individually interesting but parochial. The gap-collapse finding has implications beyond this paper and beyond ESC-50.

The argument, spelled out explicitly:

Scratch SNN: 47.15%. Scratch ANN: 63.85%. Gap: 16.7 pp. Both architectures are identical except for LIF vs ReLU.

PANNs+SNN head: 92.50%. PANNs+ANN head: 93.45%. Gap: 0.95 pp. Same architecture, same features, only activation type differs.

The 17.6× collapse in the gap when features are equalised means that 16.7 - 0.95 = 15.75 pp of the scratch gap is attributable to features, and only 0.95 pp is attributable to the spiking formalism itself. In other words, approximately 94% of the gap is about what the network learns, not how it computes.

**Why does this matter to the field?** The dominant narrative in SNN audio research has implicitly attributed the performance gap to the LIF non-linearity, the surrogate gradient approximation, or the binary spike constraint. This finding says: no, on a small dataset with limited expressive power, the bottleneck is the quality of the learned representations, not the spiking mechanism. This reframes the research agenda. Instead of asking "how do we make spiking computation better," the more productive question becomes "how do we give SNNs access to better features." The PANNs result is one answer; ANN-to-SNN conversion from large pretrained models is another; self-supervised pre-training for SNNs is a third.

The vision domain literature (Spikformer V2, ANN-to-SNN conversion papers) shows the same pattern at a 7× collapse ratio. The audio domain shows it at a 17.6× ratio. This suggests audio may be even more feature-starved than vision in the low-data regime — a plausible finding given that acoustic representations are less intuitive and harder to learn from scratch than visual ones.

This is publishable at ICONS regardless of what else is in the paper. The encoding comparison and hardware work add depth, but the gap-collapse finding is the scientific contribution that will age well.

---

## Part 4: Does It Meet Dr. Rhodes' Bar?

Dr. Rhodes' specific bar: "results with significance and novelty are required. Consider how results/methods advance state of the art or propose novel/original approaches."

**On novelty:** The paper has multiple confirmed first-in-literature claims. The "first SNN on full ESC-50" claim is backed by two independent 2025 surveys and an explicit statement in the closest competitor paper. The "first PANNs+SNN combination for audio" claim is backed by an exhaustive search finding zero prior work. The "first SpiNNaker deployment for environmental sound" claim is backed by the only prior audio SpiNNaker paper being 2016 pure tone work. These are original approaches by any reasonable definition.

**On advancing state of the art:** This is where the answer is nuanced, and honesty matters.

The paper does NOT advance the accuracy state of the art on ESC-50. The ANN SOTA is 99.1%. The PANNs+SNN result (92.5%) does not set a new SNN record because there was no prior SNN record on this dataset. The scratch SNN result (47.15%) establishes the first baseline, which future work can beat. These are establishing contributions, not surpassing contributions.

What the paper DOES advance:

1. It establishes the first rigorous benchmark for SNN audio classification on a standard dataset, creating a reference point the field lacked.
2. It proposes and validates the "frozen audio features + SNN head" paradigm as a practical architecture for neuromorphic audio deployment, demonstrating that the gap is bridgeable with pre-trained features.
3. It identifies and characterises the feature-learning bottleneck hypothesis empirically, providing the field with a theoretically significant finding that redirects the research agenda.
4. It documents, with root-cause analysis, what happens when a convolutional SNN is deployed on SpiNNaker, providing engineering knowledge the field needs for future deployments.

Dr. Rhodes' bar is met. The novelty is genuine and multi-dimensional. The significance of the gap-collapse finding is defensible. The "advance" is not in accuracy terms — it is in establishing groundwork and generating new insight. That is a legitimate form of scientific contribution, and it is explicitly the kind ICONS values.

---

## Part 5: Publication Verdict

### Is it publishable at ICONS 2026?

**Yes. With specific conditions.**

ICONS is the correct venue for this paper. The conference's explicit scope — neuromorphic algorithms, hardware deployment, benchmark tasks, energy analysis — matches every major contribution this paper makes. The acceptance rate (historically around 59%, likely lower now but still in the range of 40-50%) is achievable for work with this profile. The ICONS 2025 program includes a SpiNNaker2 hardware paper accepted as a lightning talk (Arfa et al.), a vibration-based application benchmark paper (Vasilache et al.), and a continual learning SNN paper (Bouhadjar et al.) — all of which are less novel than what this thesis contributes. The ICONS 2022 Yarga encoding comparison paper (4 encodings, speech digits) is the direct precursor and this work is substantially more comprehensive. The community will recognise this.

### Strongest framing

The paper should lead with the gap-collapse finding, not the 47.15% number.

The current abstract opens with "our scratch-trained convolutional SNN achieves 47.15%." This is the wrong hook. The paper should open with: "We demonstrate that the SNN-ANN accuracy gap on environmental sound classification collapses 17.6× — from 16.7 pp to under 1 pp — when both networks receive equal-quality pre-trained features, revealing that the bottleneck is feature learning, not spiking computation." Everything else is then presented as supporting evidence for that central claim.

Secondary framing: "Despite lower accuracy than conventional deep learning, scratch-trained SNNs exhibit 14.8× greater adversarial robustness at FGSM ε=0.1, suggesting a natural noise-filtering property with practical implications for edge audio sensing."

The hardware story frames as: "We characterise for the first time what SNN environmental sound classification looks like on real neuromorphic hardware, documenting the root cause of the 12.8 pp hardware-software accuracy gap and providing actionable deployment guidance."

### The 2-3 things that make or break the submission

**1. The gap-collapse analysis must be the centrepiece, not a result among others.**

If the paper presents the gap-collapse as a subsection buried after encoding results and SpiNNaker tables, reviewers may miss its significance. It needs to be foregrounded in the abstract, introduction, and conclusion. The 17.6× collapse ratio needs to be explicitly compared to the vision-domain evidence (7× in Spikformer V2) to show this is not a trivial replication but a domain-specific finding. If this framing is absent, the paper reads as "here are many experiments on ESC-50 with a SNN" rather than "here is a scientific insight about why SNNs underperform on audio."

**2. The SpiNNaker deployment must be framed as characterisation, not as a performance claim.**

The 33.1% SpiNNaker result will draw scrutiny. A reviewer who sees "SpiNNaker accuracy: 33.1%" without the surrounding context will immediately conclude the hardware deployment is poor. The paper needs to make clear that the contribution is not the 33.1% number but the root-cause analysis: why FC1 cannot run on SpiNNaker (near-zero-mean weights + 1398 binary inputs + integer weight representation), what the FC2-only hybrid approach achieves, and what the 12.8 pp gap tells us about the engineering challenges of deploying convolutional SNNs on SpiNNaker 1. If reviewers understand this is a characterisation paper, not a performance paper, the hardware section is a contribution. If they read it as a performance claim, it is a weakness.

**3. The adversarial robustness result must be accompanied by the Wang et al. caveat or it will draw a pointed reviewer comment.**

Wang et al. (arXiv:2512.22522, 2025) specifically warn that FGSM/PGD attacks may underestimate SNN vulnerability due to gradient estimation inaccuracies introduced by the surrogate function. This paper is known enough in the community that at least one reviewer is likely to raise it. The correct response is not to suppress the adversarial result but to acknowledge the limitation and argue that: (a) the finding is the first audio-domain adversarial SNN result regardless of the exact magnitude, (b) the qualitative advantage of SNN over ANN is robust to methodological concerns at ε=0.1 (even if the numbers overestimate by some factor, an ANN at 1.75% and an SNN at 26% are qualitatively in different regimes), and (c) this is a direction for future work to validate with adaptive attacks. Without this framing, a reviewer familiar with Wang et al. will reject the adversarial section as methodologically naive.

---

## Summary Table

| Contribution | Novelty Strength | Publication Risk | Notes |
|---|---|---|---|
| First SNN on full ESC-50 | Strong | Low | Watertight; confirmed by 2 surveys + competitor paper |
| 7-encoding comparison | Strong | Low | Most comprehensive in audio SNN literature |
| PANNs+SNN gap-collapse finding | Strong | Low | Central scientific contribution; must be centrepiece |
| SpiNNaker hardware deployment | Strong (at ICONS) | Moderate | Framing as characterisation, not performance claim, is critical |
| Adversarial robustness | Moderate-to-Strong | Moderate | Wang et al. (2025) caveat must be addressed explicitly |
| NeuroBench energy analysis | Moderate | Low | Correct application; honest framing; no new methodology |
| Surrogate gradient ablation | Weak-to-Moderate | Low | Useful but not primary; fits in supplementary if space is tight |
| Continual learning | Weak | Low | Novel domain, modest finding, include as secondary result only |

**Overall verdict: Submit. The work is publishable at ICONS 2026. The novelty is genuine across multiple dimensions, the literature search is rigorous, and the central scientific finding is significant. The principal risk is framing: if the paper leads with 47.15% rather than the gap-collapse, it will read below its actual quality. Reframe and submit.**
