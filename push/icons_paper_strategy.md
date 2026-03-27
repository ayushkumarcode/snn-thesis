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
   4.1 Encoding Comparison (Table 1)
   4.2 SpiNNaker Hardware Results (Table 2)
   4.3 Transfer Learning: Gap Collapse (Table 3)
   4.4 Adversarial Robustness (Table 4)
   4.5 Energy Analysis (Table 5)
5. Discussion                                    ~0.8 pages
6. Conclusions & Future Work                     ~0.3 pages
References                                       ~0.8 pages
```

**Target: 5 tables, 3 figures**

---

## 3 ESSENTIAL FIGURES

1. **Architecture diagram** — SpikingCNN with encoding input, Conv-BN-Pool-LIF blocks, output
2. **SpiNNaker pipeline** — hybrid: snnTorch (conv+FC1) → binary spikes → SpiNNaker (FC2). Show FC1 cancellation as crossed-out path
3. **Encoding bar chart** — 7 encodings + ANN baseline with error bars

---

## TITLE OPTIONS

| Option | Title |
|--------|-------|
| A (recommended) | Spiking Neural Networks for Environmental Sound Classification: From Seven Encodings to SpiNNaker Deployment |
| B (short) | First Convolutional SNN on ESC-50: Encoding Comparison and SpiNNaker Deployment |
