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

## 3 essential figures

1. **architecture diagram** -- SpikingCNN with encoding input, Conv-BN-Pool-LIF blocks, output
2. **SpiNNaker pipeline** -- hybrid: snnTorch (conv+FC1) -> binary spikes -> SpiNNaker (FC2). show FC1 cancellation as crossed-out path
3. **encoding bar chart** -- 7 encodings + ANN baseline with error bars

---

## title options

| Option | Title |
|--------|-------|
| A (recommended) | Spiking Neural Networks for Environmental Sound Classification: From Seven Encodings to SpiNNaker Deployment |
| B (short) | First Convolutional SNN on ESC-50: Encoding Comparison and SpiNNaker Deployment |
| C (finding) | Bridging the SNN-ANN Gap in Environmental Sound Classification |
| D (hardware-forward) | SpiNNaker Deployment of Spiking Neural Networks for 50-Class Environmental Sound Classification |

---

## reviewer objections -- prepared responses

### "47% accuracy is low"
frame as baseline datum, not final word. PANNs+SNN (92.5%) proves the architecture works when features are good. the 47% identifies the bottleneck (feature learning from small data).

### "SNN uses MORE energy than ANN"
three-part honest framing:
1. quantify: SNN 976 nJ vs ANN 463 nJ due to T=25 timesteps
2. path to efficiency: spike rate 25.8% vs threshold <6.4%. reducing T and increasing sparsity closes gap
3. position: first quantified energy baseline for SNN audio. SNN is AC-only -> deployable on neuromorphic hardware without multiplier circuits

### "SpiNNaker 33% with high variance"
Dominguez-Morales (2016) only classified 8 pure tones. our 50-class task is 6.25x harder. 12.8pp gap has documented root cause. first quantified hardware gap for SNN audio.

### "Only ESC-50, no cross-dataset"
acknowledge. ESC-50's 5-fold CV is the standard. propose UrbanSound8K as future work. better yet: run 1-fold on UrbanSound8K in next 2 weeks.

### "PANNs+SNN is not really neuromorphic"
frame as hybrid edge deployment: CNN14 in cloud, SNN head on neuromorphic edge. cite Seekings et al. (ICONS 2024) as precedent for hybrid approaches.

### "Standard PGD, not SA-PGD"
FGSM results (single-step, no SNN adaptation) provide clean lower bound. cite Wang et al. (2025), acknowledge SA-PGD as future work.

---

## experiments to strengthen the paper (next 2 weeks)

### must do:
1. SpiNNaker latency measurement -- wall-clock ms per inference. free to measure.
2. SpiNNaker energy from provenance data -- real hardware energy, not theoretical
3. statistical significance tests -- Wilcoxon for all key comparisons

### nice to have:
4. reduce SpiNNaker hardware gap (tune tau_syn, weight quantization)
5. 1-fold on UrbanSound8K

---

## full vs short paper

**full paper (8 pages).** no question.
- we have more than enough content
- full = 20-min talk (more visibility)
- carries more academic weight
- if not accepted as full, automatically considered for poster (safety net)

---

## how we compare to typical ICONS papers

| Dimension | Typical ICONS | Us | |
|-----------|--------------|----|-|
| Novelty | Incremental | First SNN on ESC-50 | above average |
| Hardware | Often sim only | SpiNNaker 5-fold (2000 inferences) | above average |
| Eval rigor | 1-2 folds | 5-fold CV, 7 encodings | well above average |
| Dataset | 10-35 classes | 50 classes | above average |
| Contributions | 1-2 | 4 focused | above average |
| Accuracy | Often >90% | 47% scratch, 92.5% PANNs | needs framing |
| Energy analysis | Absent/hand-wavy | NeuroBench, honest | above average |

i'd put us in the top 20-30% of ICONS submissions in terms of scientific content.

---

## top 5 actions (priority order)

1. cut to 4 contributions. remove surrogate, CL, augmentation, t-SNE, temporal
2. make SpiNNaker the star. expand deployment section, add pipeline figure, add latency/energy measurements
3. create 3 polished figures
4. rewrite abstract to 150-200 words with tight narrative arc
5. frame energy honestly with three-part argument

---

## Sources
- ICONS 2023-2025 proceedings analysis (50+ papers)
- [Seekings et al. ICONS 2024](https://arxiv.org/abs/2407.08704)
- [An et al. ICONS 2025](https://arxiv.org/abs/2509.21345)
- [Schone et al. ICONS 2024 Best Paper](https://arxiv.org/abs/2404.18508)
