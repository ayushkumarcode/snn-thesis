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

## paper structure (8 pages)

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

target: 5 tables, 3 figures.

---

