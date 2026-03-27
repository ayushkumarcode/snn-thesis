# SOTA SNN Audio Classification — Deep Research Report
*Generated: 5 March 2026*

## Executive Summary

**Our novelty claim stands firm.** As of March 2026, there is ZERO prior SNN work on the full 50-class ESC-50. Confirmed across arXiv, IEEE, ACM, Semantic Scholar, Google Scholar.

**Key update:** ESC-50 ANN SOTA has moved beyond 98.25% → **OmniVec2 achieves 99.1% (CVPR 2024)**. Must update thesis.

---

## 1. The ONLY Direct Competitor: Larroza et al. (2025)

| Field | Detail |
|-------|--------|
| Authors | Larroza, Naranjo-Alcazar, Ortiz, Cobos, Zuccarello |
| Venue | arXiv:2503.11206 (submitted to ICASSP 2026) |
| Dataset | **ESC-10 only** (NOT ESC-50) |
| Architecture | 4-layer FC SNN (no convolutions) |
| Encodings | TAE, Step Forward, Moving Window (3 total) |
| Best result | F1=0.661 on ESC-10 |
| Hardware | None |
| Key quote | "no state-of-the-art solution has yet encoded environmental sound datasets using spike-based methods" |

**Our advantages over Larroza:** Full ESC-50 (50 classes vs 10), convolutional architecture, 7 encodings (vs 3), SpiNNaker hardware deployment, adversarial robustness, transfer learning, continual learning, energy analysis.

---

## 2. Other SNN Environmental Sound Work

| Paper | Year | Dataset | Accuracy | Notes |
|-------|------|---------|----------|-------|
