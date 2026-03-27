# SNN Audio Classification -- Deep Dive (March 2026)

**our novelty claim holds up.** as of March 2026, there is ZERO prior SNN work on the full 50-class ESC-50. checked arXiv, IEEE, ACM, Semantic Scholar, Google Scholar.

important update: ESC-50 ANN SOTA has moved -- **OmniVec2 achieves 99.1% (CVPR 2024)**. need to update the thesis from the old 98.25% figure.

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

our advantages: full ESC-50 (50 vs 10 classes), convolutional architecture, 7 encodings (vs 3), SpiNNaker deployment, adversarial robustness, transfer learning, continual learning, energy analysis.

---

## 2. Other SNN Environmental Sound Work

| Paper | Year | Dataset | Accuracy | Notes |
|-------|------|---------|----------|-------|
| Wu et al. | 2018 | RWCP / TIDIGITS | 99.60% / 97.4% | SOM-SNN, not ESC-50 |
| Yu et al. | 2019 | Environmental sound (unclear dataset) | - | Sparse key-point encoding |
| S-CMRL | 2025 | UrbanSound8K-AV | - | Audio-VISUAL (not comparable) |
| Dominguez-Morales | 2016 | Pure tones | >85% | SpiNNaker, but trivial sounds |

bottom line: nobody has done SNN on ESC-50. nobody has done SNN on any environmental sound dataset with >10 classes.

---

## 3. SNN Keyword Spotting -- Much More Active

| Paper | Year | GSC Accuracy | SHD | Architecture | Energy |
|-------|------|-------------|-----|--------------|--------|
| **SpikCommander** | 2025 | **96.92%** | 96.41% | Spiking Transformer + MSTASA | 0.042 mJ |
| SpikeSCR | 2024 | 95.60% | - | Spike-driven attention | 54.8% reduction |
| DCLS-Delays | 2024 (ICLR) | 95.35% | 95.07% | Learnable delays | - |
| Spiking-LEAF | 2024 (ICASSP) | SOTA | - | IHC-LIF + learnable filterbank | - |
| Speech2Spikes | 2023 (NICE) | 88.5% (35-class) | - | FF SNN | Deployed on Loihi |
| SE-adLIF | 2024 | - | 95.81% | Adaptive LIF | 0.45M params |

keyword spotting is by far the most mature SNN audio field. environmental sound classification is way behind.

---

## 4. Neuromorphic Audio Hardware Deployments

### SpiNNaker
| System | Year | Task | Result |
|--------|------|------|--------|
| Dominguez-Morales | 2016 | Pure tone classification | >85%, 4-chip SpiNNaker |
| Sound source localization | 2023 | SSL | Comparable to traditional |
| **our work** | 2026 | **ESC-50 classification** | **first on SpiNNaker** |

### Intel Loihi / Loihi 2
| System | Year | Task | Result |
|--------|------|------|--------|
| Blouw et al. | 2019 | Single-phrase KWS | Best energy vs CPU/GPU/Jetson |
| Speech2Spikes | 2023 | GSC 35-class KWS | 88.5%, 109x lower energy |
| Efficient Audio | 2024 (ICASSP) | Denoising + KWS on Loihi 2 | Orders of magnitude EDP improvement |
| EventProp | 2025 | SHD/SSC on Loihi 2 | 18x faster, 200-250x less energy |

### SynSense Xylo Audio 2
- 95% accuracy, 291 uW dynamic power, 6.6 uJ/inference for "Aloha" KWS
- TSMC 40nm, integrated audio front-end
- actual commercial product

### BrainChip Akida
- KWS demos at CES 2024, edge learning with few-shot

### FPGA
- NEUROSEC (2024): adversarial audio security, 94% detection, FGSM/PGD resilient

---

## 5. Transfer Learning + SNN (Our PANNs+SNN Approach)
