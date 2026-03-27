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
|--------|------|------|--------|
| Dominguez-Morales | 2016 | Pure tone classification | >85%, 4-chip SpiNNaker |
| Sound source localization | 2023 | SSL | Comparable to traditional |
| **Our work** | 2026 | **ESC-50 classification** | **FIRST on SpiNNaker** |

### Intel Loihi / Loihi 2
| System | Year | Task | Result |
|--------|------|------|--------|
| Blouw et al. | 2019 | Single-phrase KWS | Best energy vs CPU/GPU/Jetson |
| Speech2Spikes | 2023 | GSC 35-class KWS | 88.5%, 109x lower energy than GPU |
| Efficient Audio | 2024 (ICASSP) | Denoising + KWS on Loihi 2 | Orders of magnitude EDP improvement |
| EventProp | 2025 | SHD/SSC on Loihi 2 | 18x faster, 200-250x less energy than Jetson |

### SynSense Xylo Audio 2 (Commercial!)
- **95% accuracy, 291 μW dynamic power, 6.6 μJ/inference** for "Aloha" KWS
- TSMC 40nm, integrated audio front-end
- This is a real commercial product

### BrainChip Akida
- KWS demos at CES 2024, edge learning with few-shot

### FPGA
- NEUROSEC (2024): Adversarial audio security, 94% detection, FGSM/PGD resilient

---

## 5. Transfer Learning + SNN (Our PANNs+SNN Approach)

**Our approach appears novel.** No paper found that specifically:
1. Uses frozen pre-trained audio model (PANNs/CNN14) to extract embeddings
2. Trains a separate SNN head on those embeddings
3. Reports the gap between SNN head and ANN head

Closest:
- Three-stage hybrid SNN (2025): ANN→conversion→SNN fine-tuning (different paradigm)
- Knowledge distillation approaches: Transfer during training, not frozen features
- SAFE (ICLR 2025): CNN+SNN for fake audio detection (different task)

**Our gap-collapse finding (17pp → 1pp) appears to be genuinely novel.**

---

## 6. Encoding Comparisons in Literature

| Paper | Year | Encodings | Dataset |
|-------|------|-----------|---------|
| Larroza et al. | 2025 | TAE, SF, MW (3) | ESC-10 |
| Yarga et al. | 2022 (ICONS) | Send-on-Delta, TTFS, LIF, BSA (4) | Speech digits |
| Spike encoding for IoT | 2022 | Rate, binary, temporal, delta, MT-delta (5) | IoT signals |
| **Our work** | 2026 | **direct, rate, phase, population, latency, delta, burst (7)** | **ESC-50** |

**We have the most comprehensive encoding comparison for audio SNNs.** Nobody else has compared 7 encodings on the same audio dataset.

---

## 7. ESC-50 Overall SOTA (Any Model)
