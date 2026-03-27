# Defense Counsel: The Case for Publishing This Work at ICONS 2026

**Role:** Senior neuromorphic computing researcher and conference paper author
**Date:** 9 March 2026
**Task:** Provide the strongest honest case for each of the six contributions; address the supervisor's concern directly

---

## Prefatory Note on Standards

ICONS is the premier dedicated venue for neuromorphic systems research. It is not NeurIPS. It is not ICLR. Its own call for papers explicitly welcomes "benchmark tasks for neuromorphic computing," "hardware deployment," and "algorithms and training" work. The historical acceptance rate has been ~59% (ICONS 2018 data), and both ICONS 2024 and 2025 accepted multiple papers whose scientific value rested on first-ever demonstrations and systematic methodology rather than competitive accuracy numbers. The ICONS 2022 most directly comparable paper (Yarga et al.) benchmarked 4 encoding schemes on speech digit recognition and was accepted. The ICONS 2025 best paper was about turbulence modeling using neuron random walks — no classification accuracy metric whatsoever.

The question is not whether this paper is publishable at a top ML venue. It is not. The question is whether it is publishable at ICONS, which explicitly serves the community this work is designed for. The answer, argued contribution by contribution below, is yes.

---

## C1: First Convolutional SNN Evaluation on Full ESC-50

### Core Novelty Claim

This is the first evaluation of any spiking neural network architecture on the ESC-50 benchmark in its full 50-class, 5-fold cross-validation configuration.

### Why It Is Genuinely Novel

The claim is not contested by any paper in existence. Larroza et al. (arXiv:2503.11206, March 2025) — the closest competitor in the world, posted simultaneously with this thesis — explicitly state in their own abstract: "no state-of-the-art solution has yet encoded environmental sound datasets using spike-based methods." They then evaluate only ESC-10 (10 classes, a curated subset), using a fully-connected 3-layer architecture, testing 3 encoding schemes, with no hardware deployment. The full ESC-50 benchmark — 50 classes, 2,000 recordings, the standard benchmark used by every ANN paper in the field since Piczak (2015) — has never been addressed by any SNN paper. Research agents confirmed this across arXiv, IEEE Xplore, ACM DL, Semantic Scholar, and Google Scholar. The survey paper by Basu et al. (arXiv:2502.15056, February 2025), a 24-page dedicated survey of neuromorphic audio classification, reaches the same conclusion without finding a single full-ESC-50 SNN result.

