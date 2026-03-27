# Neuromorphic Research Trends 2025-2026

## we're well positioned

### zero audio papers at ICONS 2025
31 presentations, dominated by hardware deployment and novel applications. no audio whatsoever. wide-open gap for us.

### SNN explosion at top venues
- ICLR 2026: 29 SNN papers
- NeurIPS 2025: 23 SNN papers
- ICML 2025: 11 SNN papers
- hot topics: spiking transformers, SNN-LLM intersection, adversarial robustness, ANN-to-SNN conversion

### our adversarial result aligns with Nature Communications (Nov 2025)
a major paper on SNN adversarial robustness was published in Nature Comms. our audio-domain result directly extends this to a new modality.

---

## Hardware Landscape

| Player | Status | Key Number |
|--------|--------|-----------|
| Intel Loihi 2 | Research platform | 75x lower latency, 1000x energy vs Jetson |
| IBM NorthPole | Not truly spiking | 25x energy on ResNet-50 |
| SpiNNaker 2 | Deployed at Sandia (June 2025) | SpiNNcloud selling 5M-core systems |
| BrainChip Akida | AKD1500 shipping | $25M raised |
| **Innatera Pulsar** | **Mass-produced** | **Audio classification at 400 uW** |

Innatera is basically validating the audio neuromorphic market. our work on SNN audio classification is directly aligned with commercial hardware trends.

---

## Most Impactful SNN Papers

1. **Eshraghian et al.** -- "Training SNNs Using Lessons From Deep Learning" -- **2024 Proceedings of IEEE Best Paper Award** (this is the snnTorch paper we use!)
2. **NeuroBench** (Yik et al.) -- Nature Communications Feb 2025 (we use this)
3. **Nature Jan 2025** -- "Neuromorphic computing at a pivotal moment"
4. **SNN adversarial robustness** -- Nature Communications Nov 2025

---

## Grand Challenges (Unsolved)

1. software ecosystem gap (no PyTorch equivalent for SNNs)
2. scaling SNNs to large models
3. accuracy gap (best SNN: 83.73% ImageNet vs ANN: 90%+)
4. finding a killer application
5. verifying energy claims on real hardware

**we directly address #4 (novel application) and #5 (NeuroBench + SpiNNaker deployment).**

---

## The Narrative Shift

the community has matured: SNNs are complementary specialized accelerators, not ANN replacements. energy advantage is conditional (>93% sparsity). the gap is smallest with pretrained features (exactly our PANNs finding). hybrid ANN+SNN is the pragmatic path.

our insight -- "the gap is feature-learning, not spiking computation" -- is exactly what the community is converging on. we have the first empirical demonstration for audio.

---

## Neuromorphic Audio Community (Small but Growing)

| Group | Focus |
|-------|-------|
| Seville (Dominguez-Morales) | SpiNNaker audio (pure tones only) |
| Spain (Larroza et al.) | Spike encoding for ESC-10 (2025) |
| Zenke Lab | SHD dataset (speech digits) |
| Innatera | Commercial audio on neuromorphic |
| SynSense | Xylo Audio chip |

nobody has done: full ESC-50 with SNNs, 7-encoding comparison for audio, PANNs+SNN transfer, adversarial robustness for SNN audio, or SpiNNaker deployment for environmental sound.

---

## How to Make the Paper Stand Out

1. real hardware deployment (we have this)
2. honest energy analysis (we do this)
3. bridge SNN and mainstream ML (PANNs transfer does this)
4. clear memorable insight ("gap is feature-learning, not spiking")
5. address the AI energy narrative (topical, relevant to sustainability)

---
