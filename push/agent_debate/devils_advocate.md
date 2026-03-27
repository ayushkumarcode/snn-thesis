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

