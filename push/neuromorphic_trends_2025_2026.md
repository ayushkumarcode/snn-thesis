# Neuromorphic Research Trends 2025-2026
*Generated: 15 March 2026*

## Key Finding: We're Perfectly Positioned

### ZERO audio papers at ICONS 2025
31 presentations, dominated by hardware deployment and novel applications. No audio whatsoever. Wide-open gap for us.

### SNN explosion at top venues
- ICLR 2026: 29 SNN papers
- NeurIPS 2025: 23 SNN papers
- ICML 2025: 11 SNN papers
- Hot topics: spiking transformers, SNN-LLM intersection, adversarial robustness, ANN-to-SNN conversion

### Our adversarial result aligns with Nature Communications (Nov 2025)
A major paper on SNN adversarial robustness was published in Nature Communications. Our audio-domain result directly extends this to a new modality.

---

## Hardware Landscape

| Player | Status | Key Number |
|--------|--------|-----------|
| Intel Loihi 2 | Research platform | 75x lower latency, 1000x energy vs Jetson |
| IBM NorthPole | Not truly spiking | 25x energy on ResNet-50 |
| SpiNNaker 2 | Deployed at Sandia (June 2025) | SpiNNcloud selling 5M-core systems |
| BrainChip Akida | AKD1500 shipping | $25M raised |
| **Innatera Pulsar** | **Mass-produced** | **Audio classification at 400μW** |

**Innatera is validating the audio neuromorphic market.** Our work on SNN audio classification is directly aligned with commercial hardware trends.

---

## Most Impactful SNN Papers

1. **Eshraghian et al.** — "Training SNNs Using Lessons From Deep Learning" — **2024 Proceedings of IEEE Best Paper Award** (this is the snnTorch paper we use!)
2. **NeuroBench** (Yik et al.) — Nature Communications Feb 2025 (we use this)
3. **Nature Jan 2025** — "Neuromorphic computing at a pivotal moment"
4. **SNN adversarial robustness** — Nature Communications Nov 2025

---

## Grand Challenges (Unsolved)

1. Software ecosystem gap (no PyTorch equivalent for SNNs)
2. Scaling SNNs to large models
3. Accuracy gap (best SNN: 83.73% ImageNet vs ANN: 90%+)
4. Finding a killer application
5. Verifying energy claims on real hardware

**We directly address #4 (novel application) and #5 (NeuroBench + SpiNNaker deployment).**

---

## The Narrative Shift

The community has matured: **SNNs are complementary specialized accelerators, not ANN replacements.** Energy advantage is conditional (>93% sparsity). The gap is smallest with pretrained features (exactly our PANNs finding). Hybrid ANN+SNN is the pragmatic path.

**Our insight — "the gap is feature-learning, not spiking computation" — is exactly what the community is converging on.** We have the first empirical demonstration of this for audio.

---

## Neuromorphic Audio Community (Small but Growing)

| Group | Focus |
|-------|-------|
| Seville (Dominguez-Morales) | SpiNNaker audio (pure tones only) |
| Spain (Larroza et al.) | Spike encoding for ESC-10 (2025) |
| Zenke Lab | SHD dataset (speech digits) |
| Innatera | Commercial audio on neuromorphic |
| SynSense | Xylo Audio chip |

**Nobody has:** full ESC-50 with SNNs, 7-encoding comparison for audio, PANNs+SNN transfer, adversarial robustness for SNN audio, or SpiNNaker deployment for environmental sound.

---

## How to Make the Paper Stand Out

1. Real hardware deployment (we have this)
2. Honest energy analysis (we do this)
3. Bridge SNN and mainstream ML (PANNs transfer does this)
4. Clear memorable insight ("gap is feature-learning, not spiking")
5. Address the AI energy narrative (topical, relevant to sustainability)

---

## Sources
- ICONS 2025 schedule analysis
- ICLR 2026 / NeurIPS 2025 / ICML 2025 proceedings
- Nature Communications (Nov 2025) — SNN adversarial robustness
- Nature (Jan 2025) — "Neuromorphic computing at a pivotal moment"
- Innatera product specs
- Open Neuromorphic community
