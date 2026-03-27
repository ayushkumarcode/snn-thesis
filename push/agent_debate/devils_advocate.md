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

