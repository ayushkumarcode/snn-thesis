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
| Wu et al. | 2018 | RWCP / TIDIGITS | 99.60% / 97.4% | SOM-SNN, not ESC-50 |
| Yu et al. | 2019 | Environmental sound (unclear dataset) | - | Sparse key-point encoding |
| S-CMRL | 2025 | UrbanSound8K-AV | - | Audio-VISUAL (not comparable) |
| Dominguez-Morales | 2016 | Pure tones | >85% | SpiNNaker, but trivial sounds |

**Bottom line:** Nobody has done SNN on ESC-50. Nobody has done SNN on any environmental sound dataset with >10 classes.

---

## 3. SNN Keyword Spotting (Google Speech Commands) — Much More Active

| Paper | Year | GSC Accuracy | SHD | Architecture | Energy |
|-------|------|-------------|-----|--------------|--------|
| **SpikCommander** | 2025 | **96.92%** | 96.41% | Spiking Transformer + MSTASA | 0.042 mJ |
| SpikeSCR | 2024 | 95.60% | - | Spike-driven attention | 54.8% reduction |
| DCLS-Delays | 2024 (ICLR) | 95.35% | 95.07% | Learnable delays | - |
| Spiking-LEAF | 2024 (ICASSP) | SOTA | - | IHC-LIF + learnable filterbank | - |
| Speech2Spikes | 2023 (NICE) | 88.5% (35-class) | - | FF SNN | Deployed on Loihi |
| SE-adLIF | 2024 | - | 95.81% | Adaptive LIF | 0.45M params |

**Keyword spotting is the most mature SNN audio field.** Environmental sound classification is severely underexplored by comparison.

---

## 4. Neuromorphic Audio Hardware Deployments

### SpiNNaker
| System | Year | Task | Result |
