# ICONS 2026 Paper Strategy

## the core problem

**our paper is significantly stronger than the median ICONS paper, but it tries to say too much.** the current draft reads like a thesis summary. the #1 action is ruthless narrative focus.

---

## the main story

**"From Software to Silicon: The First Convolutional SNN for Environmental Sound Classification with Neuromorphic Hardware Deployment"**

arc: nobody has tried SNNs on ESC-50 -> we built it, compared 7 encodings -> deployed on SpiNNaker -> gap collapses with transfer learning -> SNNs offer natural adversarial robustness

**make SpiNNaker the star.** hardware deployment is what makes this an ICONS paper, not a generic ML paper. papers with hardware get full talks; simulation-only papers get posters.

---

## 4 contributions (cut from 6)

1. first convolutional SNN on ESC-50 with 7 encoding comparison
2. SpiNNaker deployment (first for environmental sound) with root-cause analysis
3. PANNs+SNN transfer demonstrating gap is feature-learning, not spiking
4. first adversarial robustness analysis of SNNs on audio spectrograms

**cut:** surrogate ablation, continual learning, augmentation, t-SNE, temporal analysis, per-class analysis. those go in the thesis.

---
