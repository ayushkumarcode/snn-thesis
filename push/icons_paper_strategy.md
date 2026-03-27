# ICONS 2026 Paper Strategy
*Generated: 15 March 2026*

## Core Conclusion
**Our paper is significantly stronger than the median ICONS paper, but it tries to say too much.** The current draft reads like a thesis summary. The #1 action is ruthless narrative focus.

---

## THE MAIN STORY

**"From Software to Silicon: The First Convolutional SNN for Environmental Sound Classification with Neuromorphic Hardware Deployment"**

Arc: No one has tried SNNs on ESC-50 → We built it, compared 7 encodings → Deployed on SpiNNaker → Gap collapses with transfer learning → SNNs offer natural adversarial robustness

**Make SpiNNaker the star.** Hardware deployment is what makes this an ICONS paper, not a generic ML paper. Papers with hardware get full talks; simulation-only papers get posters.

---

## 4 CONTRIBUTIONS (cut from 6)

1. First convolutional SNN on ESC-50 with 7 encoding comparison
2. SpiNNaker deployment (first for environmental sound) with root-cause analysis
3. PANNs+SNN transfer demonstrating gap is feature-learning, not spiking
4. First adversarial robustness analysis of SNNs on audio spectrograms

**CUT:** surrogate ablation, continual learning, augmentation, t-SNE, temporal analysis, per-class analysis. These go in the thesis.

---

## PAPER STRUCTURE (8 pages)

```
Title + Abstract (150-200 words)                ~0.3 pages
1. Introduction                                  ~1.0 pages
2. Background & Related Work                     ~0.8 pages
3. Methods                                       ~1.5 pages
   3.1 Architecture (Figure 1: arch diagram)
   3.2 Spike Encoding Methods (compact table)
   3.3 Training Protocol
   3.4 SpiNNaker Deployment (Figure 2: pipeline)
   3.5 NeuroBench Energy Methodology
4. Results                                       ~2.5 pages
