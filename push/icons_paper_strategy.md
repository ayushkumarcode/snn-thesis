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
| C (finding) | Bridging the SNN-ANN Gap in Environmental Sound Classification |
| D (hardware-forward) | SpiNNaker Deployment of Spiking Neural Networks for 50-Class Environmental Sound Classification |

---

## REVIEWER OBJECTIONS — PREPARED RESPONSES

### "47% accuracy is low"
Frame as baseline datum, not final word. PANNs+SNN (92.5%) proves the architecture works when features are good. The 47% identifies the bottleneck (feature learning from small data).

### "SNN uses MORE energy than ANN"
Three-part honest framing:
1. Quantify: SNN 976 nJ vs ANN 463 nJ due to T=25 timesteps
2. Path to efficiency: spike rate 25.8% vs threshold <6.4%. Reducing T and increasing sparsity closes gap
3. Position: first quantified energy baseline for SNN audio. SNN is AC-only → deployable on neuromorphic hardware without multiplier circuits

### "SpiNNaker 33% with high variance"
Dominguez-Morales (2016) only classified 8 pure tones. Our 50-class task is 6.25x harder. 12.8pp gap has documented root cause. First quantified hardware gap for SNN audio.

### "Only ESC-50, no cross-dataset"
Acknowledge. ESC-50's 5-fold CV is the standard. Propose UrbanSound8K as future work. Better: run 1-fold on UrbanSound8K in next 2 weeks.

### "PANNs+SNN is not really neuromorphic"
Frame as hybrid edge deployment: CNN14 in cloud, SNN head on neuromorphic edge. Cite Seekings et al. (ICONS 2024) as precedent for hybrid approaches.

### "Standard PGD, not SA-PGD"
FGSM results (single-step, no SNN adaptation) provide clean lower bound. Cite Wang et al. (2025) and acknowledge SA-PGD as future work.

---

## EXPERIMENTS THAT WOULD STRENGTHEN THE PAPER (2 weeks)

### Must do:
1. **SpiNNaker latency measurement** — wall-clock ms per inference. Free to measure.
2. **SpiNNaker energy from provenance data** — real hardware energy, not theoretical
3. **Statistical significance tests** — Wilcoxon for all key comparisons

### Nice to have:
4. Reduce SpiNNaker hardware gap (tune tau_syn, weight quantization)
5. 1-fold on UrbanSound8K (kills "single dataset" objection)

---

## FULL vs SHORT PAPER

**FULL PAPER (8 pages).** No question.
- We have MORE than enough content
- Full = 20-min talk (more visibility)
- Full carries more academic weight on CV
- If not accepted as full, automatically considered for poster (safety net)

---

## HOW WE COMPARE TO TYPICAL ICONS PAPERS

| Dimension | Typical ICONS | Us | |
|-----------|--------------|----|-|
| Novelty | Incremental | First SNN on ESC-50 | **Above average** |
| Hardware | Often sim only | SpiNNaker 5-fold (2000 inferences) | **Above average** |
| Eval rigor | 1-2 folds | 5-fold CV, 7 encodings | **Well above average** |
| Dataset | 10-35 classes | 50 classes | **Above average** |
| Contributions | 1-2 | 4 focused | **Above average** |
| Accuracy | Often >90% | 47% scratch, 92.5% PANNs | **Needs framing** |
| Energy analysis | Absent/hand-wavy | NeuroBench, honest | **Above average** |

**Overall: Top 20-30% of ICONS submissions in scientific content.**

---

## TOP 5 ACTIONS (Priority Order)

1. **Cut to 4 contributions.** Remove surrogate, CL, augmentation, t-SNE, temporal
2. **Make SpiNNaker the star.** Expand deployment section, add pipeline figure, add latency/energy measurements
3. **Create 3 polished figures.** Architecture, SpiNNaker pipeline, encoding bar chart
4. **Rewrite abstract to 150-200 words.** Tight narrative arc
5. **Frame energy honestly.** Three-part argument, no overclaiming

---

## Sources
- ICONS 2023-2025 proceedings analysis (50+ papers)
- [Seekings et al. ICONS 2024 — Hybrid SNN](https://arxiv.org/abs/2407.08704)
- [An et al. ICONS 2025 — DYNAP-SE deployment](https://arxiv.org/abs/2509.21345)
- [Schone et al. ICONS 2024 — Best Paper](https://arxiv.org/abs/2404.18508)
