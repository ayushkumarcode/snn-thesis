# Devil's Advocate: ICONS 2026 Reviewer Analysis
*Prepared as a hard-nosed critical review of the ICONS 2026 submission draft*
*Date: 9 March 2026*

---

## Reviewer Preamble

This paper presents a convolutional SNN evaluation on ESC-50 with seven spike encodings, SpiNNaker deployment, adversarial robustness analysis, PANNs transfer learning, and NeuroBench energy benchmarking. The author claims six novelty contributions (C1-C6). As a reviewer with background in neuromorphic computing and SNN audio processing, I will assess each claim with maximum scrutiny before offering an overall verdict.

---

## C1: First Convolutional SNN on ESC-50

### The Strongest Challenge

The novelty here is not architectural novelty — it is dataset novelty by exclusion. The paper is saying "nobody has done this exact thing on this exact dataset." That is a weak form of novelty. A reviewer will argue: being first to apply a well-established method (convolutional SNN with surrogate gradients, a technique dating to 2018-2019) to a well-established benchmark (ESC-50, published 2015) does not constitute a research contribution in 2026. The architecture is completely standard — Conv2d, BatchNorm, MaxPool, LIF neurons, snnTorch surrogate gradients. There is nothing architecturally new here. The paper is not proposing a novel SNN architecture; it is running a benchmark experiment on an untried dataset.

Furthermore, the reason nobody has done ESC-50 with a convolutional SNN before is almost certainly not that the community lacked interest — it is that ESC-50 is considered a solved problem for ANNs (98.25-99.1% ANN SOTA) and the SNN community has focused on harder, more deployment-relevant tasks. The paper may have found an open niche not because it is scientifically important, but because the SNN community correctly identified that the dataset is too small (2000 samples) and not neuromorphically interesting enough to pursue.

The closest prior work, Larroza et al. (arXiv:2503.11206), explicitly states that no prior SNN has encoded full ESC-50 — but that paper itself was submitted to ICASSP 2026, a more prestigious venue. If the ICASSP reviewers accepted Larroza et al. with ESC-10 only, there is a real question of whether ICONS reviewers will view ESC-50 as a meaningful extension or just as "more classes."

### What a Reviewer Would Specifically Say

"The claimed novelty of C1 is tautological. Any method applied to any dataset not previously evaluated on that dataset constitutes 'first application.' The convolutional SNN architecture is entirely standard (snnTorch + LIF + Conv2d), and the authors have contributed no architectural innovation. The scientific question 'what happens when you apply a standard convolutional SNN to ESC-50' is not a priori interesting — it becomes interesting only if the results reveal something non-obvious. The 47.15% result is in fact fully predictable: it is comparable to what FC-only SNNs achieve on ESC-10 (69% on 10 classes ~ a 6.9% per-class average; the authors achieve 47.15% on 50 classes ~ 0.94% per class, which is actually worse in relative terms). The authors should explain what specific scientific hypothesis this benchmark tests, rather than framing dataset novelty as a research contribution."

### Novelty Risk: MEDIUM RISK

It survives as a novelty claim because the prior work vacuum is genuinely confirmed by multiple surveys. However, a reviewer will correctly identify that dataset novelty alone is insufficient — the paper must also deliver scientific insight from those results, which it does (PANNs collapse, encoding hierarchy, adversarial robustness). The risk is that this contribution will be downgraded from a primary novelty to a "we establish a baseline" framing.

---

## C2: Most Comprehensive Spike Encoding Comparison for Audio (7 Methods)

### The Strongest Challenge

The 7-encoding comparison is the paper's strongest contribution on the surface, but it has serious methodological vulnerabilities that a specialist reviewer will identify immediately.

**Problem 1: Three of the seven encodings never had a realistic chance of working.**

Delta encoding is defined as spiking on positive temporal intensity changes. Applied to a static mel-spectrogram that is repeated across T=25 timesteps (as in direct encoding) or converted from a fixed image, there is literally zero temporal variation to detect between timesteps. Of course delta encoding fails at 7.25%. This was predictable from the encoding definition and the static nature of the input. Similarly, burst encoding front-loads all spikes in 5 of 25 timesteps — concentrating signal in 20% of the simulation window is a design choice that obviously creates the temporal window mismatch described. These are not meaningful comparative failures; they are encoding-dataset mismatches that the paper itself acknowledges. A reviewer will argue that including obviously ill-suited encodings inflates the appearance of a comprehensive comparison while not testing genuinely competitive methods.

**Problem 2: The paper does not compare against the most relevant recent encodings in the SNN audio literature.**

The Larroza et al. paper (2025) — the paper's own most-cited competitor — uses Threshold Adaptive Encoding (TAE), Step Forward (SF), and Moving Window (MW). None of these three encodings are tested in the paper under review. Larroza et al.'s TAE achieves 69.0% on ESC-10. The paper under review does not evaluate TAE. A reviewer from the Larroza group (or a reviewer who read Larroza) will immediately ask: why was TAE not included? Similarly, the DCLS-Delays approach (learnable delays, ICLR 2024), which achieves SOTA on speech commands, is not evaluated. The encoding comparison therefore covers seven encodings that are either standard (rate, direct, latency) or known to be problematic for static inputs (delta, burst), while omitting the encodings that are specifically designed for environmental sound.

**Problem 3: The comparison is not controlled for hyperparameters per encoding.**

All seven encodings are evaluated with the same training configuration (Adam lr=1e-3, early stopping patience=10, 50 epochs). Rate coding, latency coding, and phase coding have different temporal dynamics and may require different learning rates, different timesteps, or different threshold settings to perform optimally. The paper does not perform any encoding-specific hyperparameter search. This means the comparison may be penalising some encodings not because they are fundamentally inferior but because the shared hyperparameters are suboptimal for them. This is a methodological weakness that reviewers at any serious SNN venue will flag.

**Problem 4: The statistical significance claim is questionable.**

The paper claims the 16.7 pp SNN-ANN gap is "statistically significant (paired t-test: p = 0.001; Wilcoxon: p = 0.0625)." The Wilcoxon p-value of 0.0625 does NOT meet the conventional 0.05 threshold. The paper reports this as if it supports significance while simultaneously acknowledging it is above the threshold ("the minimum achievable with n=5 folds"). This is cherry-picking: if paired t-test gives p=0.001 but Wilcoxon gives p=0.0625, the appropriate conclusion is not "statistically significant" but "the result is significant under parametric assumptions that may not hold with n=5." A reviewer will flag this as a statistical methodological concern.

### What a Reviewer Would Specifically Say

"The 7-encoding comparison is incomplete in a critical way: it omits Threshold Adaptive Encoding (TAE) and Moving Window (MW) encoding, the two methods shown by Larroza et al. (2025) to be most effective for environmental sound classification. Including delta and burst encodings that are obviously incompatible with static mel-spectrograms while omitting task-relevant alternatives creates the false impression of comprehensiveness. The comparison should be either truly comprehensive (including TAE, SF, MW, and DCLS-Delays) or explicitly scoped to classical encoding families with clear justification for the exclusions. Additionally, the lack of per-encoding hyperparameter optimisation means the reported ranking may reflect optimisation gap rather than fundamental encoding properties."

### Novelty Risk: MEDIUM RISK

The 7-encoding comparison is genuinely the largest such comparison in SNN audio literature, and this fact is well-supported by the literature review. The risk is not rejection of the claim but downgrading — a reviewer may accept that it is "the most encodings compared so far" while questioning whether it is "the most informative comparison." This is survivable with good rebuttal.

---

## C3: First SNN on SpiNNaker for Environmental Sound

### The Strongest Challenge

This is the most problematic novelty claim from a technical perspective, and it faces the most severe combined challenges of both novelty and results quality.

**Problem 1: The SpiNNaker deployment is not a full network deployment.**

The paper deploys only FC2 (256→50) on SpiNNaker. The convolutional layers (Conv1, Conv2), the pooling layers, and FC1 all run in software on a CPU. The paper then argues this is a "hybrid approach" and a "novel co-design insight." A reviewer with SpiNNaker expertise will be unimpressed: deploying a single 256→50 linear layer with 50 output neurons on SpiNNaker is trivial — it is well within the capability of undergraduate student SpiNNaker tutorials. The Dominguez-Morales et al. (2016) work, which the paper cites as the only prior SpiNNaker audio work, deployed a full multilayer SNN. The paper under review actually deploys less of the network on hardware.

**Problem 2: The hardware accuracy gap (12.8 ± 4.1 pp across 5 folds) suggests the deployment is not properly validated.**

SpiNNaker=33.1% vs snnTorch=46.0%. This is a 12.8 pp gap with 4.1 pp standard deviation. The paper explains this as weight quantization and timing issues. However, a gap this large with this much variance suggests the SpiNNaker deployment is not a reliable implementation — it is a demonstration that the approach approximately works some of the time. The agreement rate of 64.5% (Run 6, fold 4) means 35.5% of samples are classified differently by hardware vs software. A reviewer evaluating deployment quality will ask: is this deployment scientifically useful or merely demonstrative? Per-fold variation (F1=29.0%, F2=32.0%, F3=36.5%, F4=43.0%, F5=25.2%) is enormous — a 17.8 pp range across folds. This suggests the hardware behaviour is not stable.

**Problem 3: Dominguez-Morales et al. (2016) already establishes the precedent more cleanly.**

The paper claims "first SNN on SpiNNaker for environmental sound classification." Dominguez-Morales et al. classified audio samples on SpiNNaker. The paper distinguishes itself on the grounds that pure tones are not "environmental sounds." This distinction will not survive determined reviewer scrutiny: pure tone classification IS a form of acoustic/sound classification on hardware. The word "environmental" is doing a lot of work in the novelty claim, and a reviewer could reasonably classify it as an attempt to make a narrow, potentially semantic distinction carry the weight of a major novelty claim.

**Problem 4: The energy numbers for SpiNNaker are not measured.**

The paper claims "86 nJ/sample" for SpiNNaker (mentioned in the SOTA document) but the actual paper abstract presents NeuroBench simulation energy (976 nJ SNN, 463 nJ ANN) based on software operation counting, not real SpiNNaker measurement. Wall-clock energy per sample on SpiNNaker is explicitly stated as "left for future measurement." The energy argument that motivates SpiNNaker deployment is therefore not validated by actual hardware measurement.

**Problem 5: SpiNNaker 1 is an antiquated platform by 2026 standards.**

SpiNNaker 2 (2024) offers 10x better neural simulation capacity per watt, 22nm FDSOI process, 0.292 pJ/SOP. The E-prop on SpiNNaker 2 paper (Yan et al. 2022) achieved 91.12% on speech commands with online learning. SpiNNaker 1 (130nm, ~5.8 μJ/SOP) is comparatively inefficient. A reviewer may argue that demonstrating deployment on an antiquated platform with poor energy efficiency has limited scientific value in 2026.

### What a Reviewer Would Specifically Say

"The SpiNNaker deployment is limited to a single FC layer (256→50) running on hardware, with all convolutional and first FC layers running in software. This is not a meaningful neuromorphic deployment — it is a partial hardware execution with the computationally expensive parts remaining in software. The resulting accuracy gap (12.8 ± 4.1 pp) is too large to attribute entirely to quantization; it suggests the hardware execution is unreliable. Furthermore, Dominguez-Morales et al. (2016) deployed a full multilayer SNN on SpiNNaker for audio, making the 'first SNN on SpiNNaker for environmental sound' claim dependent on a semantic distinction between 'pure tones' and 'environmental sounds' that may not be scientifically meaningful. The SpiNNaker 1 platform's energy characteristics (5.8 μJ/SOP, 130nm process) are not competitive with modern neuromorphic hardware (SpiNNaker 2: 0.292 pJ/SOP). The claimed 5.1x per-operation energy advantage is a theoretical calculation based on 45nm operation costs, not a measured result on the actual hardware used."

### Novelty Risk: HIGH RISK

This is the highest-risk claim in the paper. The combination of: (a) only deploying 1 of 4 layers on hardware, (b) a large and variable accuracy gap, (c) the Dominguez-Morales semantic distinction argument, and (d) no measured hardware energy creates a highly vulnerable claim. A determined reviewer can argue this is not a true neuromorphic deployment. The paper needs either a much stronger rebuttal of the Dominguez-Morales comparison or a frank acknowledgement that this is a partial deployment with significant hardware gap.

---

## C4: First Adversarial Robustness Analysis of SNNs on Audio Spectrograms

### The Strongest Challenge

**Problem 1: The dramatic result (14.9x robustness at eps=0.1) may be an artefact of gradient masking rather than genuine adversarial robustness.**

Wang et al. (2025, arXiv:2512.22522) — cited in the paper's own reference list — explicitly warn that standard FGSM and PGD attacks may underestimate SNN vulnerability due to surrogate gradient inaccuracies. The paper under review acknowledges this in the text: "SA-PGD should be used for stronger reliability guarantees." However, the paper then presents the dramatic 14.9x robustness finding as if it is a validated result. If the robustness is primarily gradient masking (the adversarial examples computed against the surrogate gradient approximation are not valid attacks against the true spiking discontinuity), then the reported robustness numbers are not meaningful measures of genuine adversarial robustness — they measure how well the surrogate gradient approximation misleads the attacker. A reviewer who has read Wang et al. (2025) will flag this prominently.

**Problem 2: The result is on only one fold (fold 4, 400 samples).**

The adversarial robustness analysis is conducted on fold 4 only. With 400 samples and 50 classes, this is 8 samples per class on average. The statistical reliability of per-epsilon accuracy numbers from such small per-class sample counts is questionable. The finding that ANN accuracy drops to 1.75% at eps=0.1 (7 correct out of 400) has enormous statistical uncertainty.

**Problem 3: The baseline clean accuracy differential complicates interpretation.**

At eps=0.0, SNN=53.75% and ANN=68.75%. The SNN starts 15 pp lower. If we normalise adversarial accuracy as a fraction of clean accuracy: SNN at eps=0.1 retains 26/53.75 = 48.4% of its clean performance; ANN retains 1.75/68.75 = 2.5% of its clean performance. This relative comparison still favours the SNN dramatically. But a reviewer might argue that the ANN's catastrophic drop is partly because it had more to lose — higher clean accuracy means more information that adversarial perturbations can destroy. The robustness comparison is somewhat confounded by the accuracy differential.

**Problem 4: The novelty claim is specifically "first on audio spectrograms" — which is very narrow.**

Prior work (Sharmin et al. ECCV 2020, cited) already established SNN adversarial robustness on image inputs. The contribution is applying a known phenomenon (SNN robustness from binary thresholding) to audio spectrogram inputs. This is a domain transfer, not a fundamental discovery. A reviewer may accept it as a contribution but downgrade it from "novel finding" to "corroborating evidence in a new modality."

### What a Reviewer Would Specifically Say

"The adversarial robustness analysis is undermined by a critical methodological concern: Wang et al. (2025) demonstrate that standard FGSM/PGD attacks may not correctly evaluate SNN robustness due to the surrogate gradient approximation mismatch with the true spiking discontinuity. The paper acknowledges this limitation but then presents the 14.9x robustness advantage as a meaningful result without addressing whether the result reflects genuine adversarial robustness or gradient masking. The authors must either (a) use SA-PGD as Wang et al. recommend, (b) empirically verify that surrogate gradient attacks produce valid adversarial examples for their specific architecture, or (c) prominently reframe the finding as 'resistance to surrogate-gradient-based attacks' rather than 'adversarial robustness.' As presented, the 14.9x figure may be an optimistic artifact of the evaluation methodology rather than a genuine property of the SNN."

### Novelty Risk: MEDIUM RISK

The novelty claim ("first on audio spectrograms") is defensible but thin. The bigger risk is that the result itself may be challenged as a methodological artifact. This is survivable if the paper strongly foregrounds the Wang et al. (2025) caveat rather than burying it in a subordinate clause.

---

## C5: First PANNs + SNN Combination

### The Strongest Challenge

**Problem 1: The result is trivial in hindsight and may not survive "why didn't you just use an ANN?" scrutiny.**

The experiment shows that freezing CNN14 (a massive ANN trained on 2 million AudioSet clips) and attaching a tiny 3-layer SNN head achieves 92.50%, while attaching a tiny 3-layer ANN head achieves 93.45%. The 0.95 pp gap is the claimed scientific insight (gap collapses). However, a reviewer will observe: the tiny SNN head is a near-trivial classifier applied to already-excellent 2048-dimensional features. The fact that a tiny SNN can classify pre-extracted features at nearly the same accuracy as a tiny ANN is not scientifically surprising — it is what you would expect from any reasonably trained linear-ish classifier applied to good features. The interesting question is not "can a small SNN head classify CNN14 features?" (trivially yes) but "does the SNN head learn fundamentally different representations?" The paper does not address this.

**Problem 2: The 0.95 pp advantage for ANN over SNN is not significant.**

The SNN achieves 92.50% ± 1.30% and the ANN achieves 93.45% ± 1.54%. The difference is 0.95 pp. The confidence intervals overlap substantially. A t-test on these five-fold results would almost certainly not be significant. The paper claims this "demonstrates the SNN architecture can learn competitive classifiers from rich features" — but the tiny performance gap could equally be attributed to random variation from the 5-fold split.

**Problem 3: This is ANN-to-SNN knowledge transfer in frozen embedding form, which is a well-studied approach.**

The ANN-to-SNN conversion literature (Bu et al., CVPR 2025; multiple ICLR submissions) covers scenarios where pretrained ANN representations are used to initialise or guide SNN training. The paper's approach (freeze ANN embeddings, train SNN head) is a specific instance of this general paradigm. The paper frames it as novel but does not adequately distinguish it from transfer learning approaches in the vision domain (Stanojevic et al. 2024: 0.3 spikes/neuron from pretrained ANN conversion). A reviewer familiar with ANN-to-SNN conversion literature will question whether the contribution is truly novel or whether it is a straightforward application of a known approach to audio.

**Problem 4: The PANNs Linear Probe (93.80%) outperforms both the ANN head (93.45%) and the SNN head (92.50%).**

A linear probe on CNN14 embeddings outperforms the SNN head. This could be interpreted as: the 3-layer SNN head is actually slightly worse than a linear classifier on the same features. If the PANNs embeddings are so good that even linear separation works better than the SNN head, the contribution of the SNN is questionable.

### What a Reviewer Would Specifically Say

"The PANNs+SNN experiment demonstrates that a small SNN classification head can match a small ANN classification head when applied to 2048-dimensional embeddings from CNN14. This is not scientifically surprising: any reasonable classifier (including logistic regression, as the linear probe result shows) performs well on such high-quality representations. The claimed insight — that the SNN-ANN gap collapses from 16.7 pp to <1 pp — reflects not a property of the SNN formalism but the overwhelming quality of the CNN14 features. The experiment does not demonstrate that SNNs are capable feature learners; it demonstrates that the CNN14 embeddings are excellent. Additionally, the 0.95 pp gap between SNN and ANN heads is within the range of random variation given n=5 folds and overlapping confidence intervals. The novelty of using PANNs embeddings with an SNN head, while not previously done in exactly this form, is a straightforward combination of existing techniques without architectural innovation."

### Novelty Risk: MEDIUM RISK

The claim survives on the grounds that this specific combination has not been done before, and the "gap collapse" finding does provide a useful scientific framing. However, the paper will need to defend itself against the "trivially expected result" criticism. The stronger framing — "the bottleneck is feature learning, not spiking computation" — is genuinely useful but needs to be argued more carefully, with reference to why this was not obvious a priori.

---

## C6: NeuroBench-Compliant Energy Analysis

### The Strongest Challenge

**Problem 1: This is not a novelty contribution — it is a tool application.**

NeuroBench is a published framework (Yik et al., Nature Communications 2025). Using NeuroBench on a new model does not constitute a research contribution; it constitutes using a tool. The paper is not proposing a new energy analysis methodology, not extending NeuroBench, and not discovering anything new about energy modelling. It is running an existing tool on a new model and reporting the numbers. The claim of "NeuroBench-compliant energy analysis" as a novelty contribution (C6) is the weakest of all six claims and should not be listed alongside genuine contributions.

**Problem 2: The energy analysis demonstrates that the SNN is MORE expensive than the ANN.**

The SNN uses 976 nJ vs 463 nJ for the ANN — a 2.1x penalty. The paper then constructs an argument that this would be reversed on neuromorphic hardware (because ACs cost 5.1x less than MACs). However, the 5.1x per-operation advantage comes from 45nm CMOS energy values (Horowitz, ISSCC 2014), not from measured SpiNNaker energy. SpiNNaker 1 uses 130nm CMOS and costs approximately 5.8 μJ/SOP — orders of magnitude more expensive than the theoretical 0.9 pJ/AC at 45nm. The paper is using theoretical hardware numbers to argue for hardware efficiency while the actual hardware used (SpiNNaker 1) is not energy-efficient by any modern measure. The "5.1x per-operation advantage on neuromorphic hardware" claim is not validated by SpiNNaker measurement and would be falsified if applied to SpiNNaker 1's actual energy cost.

**Problem 3: The activation sparsity (74.16%) is below the break-even threshold for software energy efficiency.**

The paper correctly cites Dampfhoffer et al. (2023) and notes that the spike rate (25.8%) is above the 6.4% threshold. However, it does not state clearly enough that this means the energy argument for SNNs is currently unconvincing. Yang et al. (2024) show that for T>16 timesteps (the paper uses T=25), sparsity must exceed 97% for energy efficiency — the paper's 74.16% sparsity falls far short. On the paper's chosen hardware (SpiNNaker 1), which is not an efficient neuromorphic chip by modern standards, the energy argument is even weaker.

### What a Reviewer Would Specifically Say

"Contribution C6 (NeuroBench-compliant energy analysis) is not a novelty claim — it is a routine benchmarking exercise using an existing tool. The energy analysis reveals that the SNN is 2.1x more expensive than the ANN in software simulation and does not reach the sparsity threshold (<6.4% spike rate required; paper's spike rate is 25.8%) for energy efficiency. The paper argues that neuromorphic hardware would reverse this penalty via the 5.1x AC/MAC cost ratio, but this claim is based on 45nm theoretical values, not on measured SpiNNaker 1 energy (which is ~5.8 μJ/SOP at 130nm — far worse than the 0.9 pJ/AC theoretical value). The energy narrative as presented is misleading: the theoretical hardware advantage is presented as if it applies to the actual hardware used, when in fact the real hardware measurement is missing."

### Novelty Risk: HIGH RISK (as a contribution claim)

C6 is essentially the most likely to be rejected as a contribution claim by reviewers — it may be accepted as a validation/benchmarking exercise but not as a novelty contribution in its own right. This claim should be absorbed into the SpiNNaker deployment section rather than listed as a standalone contribution.

---

## Overarching Questions: Significance and Publishability

### Is 47.15% on ESC-50 "Significant"?

In absolute terms: no. Human performance is 81.3%, ANN SOTA is 98.25-99.1%. The SNN achieves 47.15%, which is 34 pp below human performance and 51 pp below ANN SOTA. Against the matched ANN, the SNN is 16.7 pp worse. If a student reported that their model achieves 47.15% on a task where humans score 81.3% and ANNs score 98.25%, a typical machine learning reviewer would view this as a failed model, not a publishable result.

The paper's argument is that this is "the first SNN on ESC-50, so any result is informative." This is true as a benchmark statement, but it is a weak justification for ICONS specifically. ICONS values hardware-neuromorphic systems work, not pure benchmarking. The 47.15% becomes defensible only in the context of the PANNs experiment (gap collapses) and the encoding comparison (seven methods, systematic analysis). Even then, a reviewer may ask: if the PANNs+SNN achieves 92.50%, why not submit a paper primarily about PANNs+SNN, with the scratch SNN as a baseline comparison, rather than leading with a 47.15% result?

### Does This "Advance State of the Art"?

It does not advance ANN SOTA on ESC-50. It does not advance SNN SOTA on any benchmark (the paper is the only SNN on ESC-50, so it cannot be claimed to advance SNN SOTA — it establishes a baseline). It does not advance SpiNNaker deployment capability (the deployment is partial). The strongest case for "advancing state of the art" is:
- Establishing the first SNN ESC-50 baseline (C1)
- The encoding hierarchy finding (C2)
- The "gap collapse with pretrained features" insight from PANNs (C5)
- The adversarial robustness finding in audio domain (C4)

None of these individually constitute a major advance in state of the art. Collectively, they constitute a comprehensive benchmark study — which is a legitimate and publishable contribution at ICONS if framed correctly.

### Is the Approach "Novel/Original"?

The approach is not novel in any technical sense:
- Convolutional SNN with surrogate gradients: standard since 2019
- Spike encoding comparison: done before (Larroza 2025, Yarga 2022, Bian 2024)
- SpiNNaker deployment: done before (Dominguez-Morales 2016, multiple SpiNNaker papers)
- Adversarial robustness: done before in vision (Sharmin 2020)
- Transfer learning to SNN: done before in vision (Stanojevic 2024, conversion literature)
- NeuroBench energy analysis: tool application

The originality is entirely in the combination and the dataset. This is a "first comprehensive evaluation" paper, not a "novel method" paper. Whether ICONS reviewers find this sufficient depends on how it is framed and how strong the empirical findings are.

### What Would Make This Publishable vs Not?

**Arguments for acceptance at ICONS:**
- ICONS explicitly welcomes benchmark and application papers (per their CFP)
- Prior ICONS papers with modest accuracy and application-focused results have been accepted (ICONS 2025 "SNN for Vibration-Based Predictive Maintenance"; ICONS 2024 "Continuous Learning for Auditory Source Separation")
- The seven-encoding comparison is genuinely the most comprehensive in audio SNN literature
- The PANNs gap-collapse finding is a useful scientific result
- The SpiNNaker deployment, despite its limitations, demonstrates real hardware execution
- The adversarial robustness finding, while potentially confounded by gradient masking, is dramatic enough to be interesting

**Arguments for rejection:**
- No novel method or architecture is proposed
- The best accuracy result (47.15%) is far from competitive
- The SpiNNaker deployment is partial (FC2 only) with a large, variable hardware gap
- C6 is not a contribution
- The adversarial result may be a methodological artefact
- The surrogate ablation is single-seed, single-fold
- The continual learning result is not in the ICONS draft at all (it is in the thesis appendix)
- Eight-page limit forces extreme compression of seven major experiments, resulting in shallow coverage of each

**The core tension:** The paper is trying to be a comprehensive evaluation paper with six major claims in eight pages. Each claim gets ~half a page of coverage. At that density, none of the claims can be developed with enough depth to be fully convincing. A reviewer may conclude that the paper is "a lot of interesting experiments with no single result strong enough to carry an ICONS paper."

---

## Summary Risk Table

| Claim | Challenge Severity | Novelty Risk | Fatal Flaw? |
|-------|-------------------|-------------|-------------|
| C1: First Conv SNN on ESC-50 | Moderate | MEDIUM | No — dataset novelty is real but thin |
| C2: Most comprehensive encoding comparison | Significant | MEDIUM | No — survivable with rebuttal on TAE omission |
| C3: First SNN on SpiNNaker for env. sound | Severe | HIGH | Potentially — partial deployment + semantic distinction |
| C4: First adversarial robustness on audio spectrograms | Significant | MEDIUM | Potentially — gradient masking concern must be addressed |
| C5: First PANNs + SNN | Moderate | MEDIUM | No — "trivially expected" criticism is manageable |
| C6: NeuroBench energy analysis | Severe | HIGH | Yes as standalone — must be absorbed into C3/deployment |

---

## Recommended Responses to Each Challenge

**On C1:** Accept the framing critique. Lead with the scientific insight (gap analysis, encoding hierarchy) rather than dataset novelty. Frame as "comprehensive evaluation" not "first application."

**On C2:** Address the TAE omission explicitly. Either add TAE as an encoding (computationally feasible) or explain in the limitations why it was not included. Acknowledge that delta and burst encodings were included as contrastive negative controls, not competitive alternatives. Fix the Wilcoxon p=0.0625 framing.

**On C3:** This is the hardest challenge. Options: (a) Reframe as "hardware characterisation study" rather than a deployment novelty claim, explicitly acknowledging the FC2-only scope and documenting the root cause of FC1 failure as the primary technical contribution; (b) the Option A (MaxPool SNN threshold sweep) result showing fc1_binary_fraction=1.000 should be emphasised as the path forward; (c) remove the "first" claim and simply document the hardware gap as a novel measurement.

**On C4:** Immediately front-load the Wang et al. (2025) caveat. Consider reframing as "resistance to surrogate-gradient-based adversarial attacks" and proposing SA-PGD as future work. The finding is still interesting even with the caveat.

**On C5:** Emphasise the scientific insight (feature learning bottleneck) rather than the method novelty. The 0.95 pp gap significance issue should be addressed with explicit confidence interval reporting.

**On C6:** Do not list as a separate contribution. Merge energy analysis into the SpiNNaker deployment section.

---

## Bottom Line Verdict

The paper is **borderline publishable at ICONS 2026**, with ICONS being the correct venue precisely because it explicitly welcomes benchmark and application-focused papers with modest accuracy results. The paper will not be accepted trivially — it requires a strong rebuttal capacity on C3 (SpiNNaker partial deployment) and C4 (gradient masking concern). The most likely reviewer outcome is:

- **One reviewer:** Enthusiastic. Appreciates the comprehensive evaluation, recognises the gap analysis insight, values the hardware deployment attempt.
- **One reviewer:** Skeptical. Raises C3 hardware gap concerns, C4 methodology concerns, demands TAE comparison.
- **One reviewer:** Hostile. Argues no novel method, no competitive results, partial hardware deployment does not count.

The outcome will depend heavily on the rebuttal phase and on how the paper is framed in its final submission. The largest single risk is if a reviewer with deep SpiNNaker expertise (or the Dominguez-Morales group) reviews the paper and dismantles the C3 claim. The second largest risk is a reviewer who has read Wang et al. (2025) and flags the adversarial robustness as gradient masking.

**If the paper is rejected, it will be for one of these two reasons, not for the accuracy numbers.**

---

*This review represents the most adversarial reading of the submission. Actual ICONS reviewers may be more sympathetic to benchmark contributions. The goal of this document is to identify every vulnerability before submission, not to predict rejection.*
