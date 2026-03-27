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
