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
