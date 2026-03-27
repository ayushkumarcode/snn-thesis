# Spiking Neural Networks for ECG / Heartbeat Anomaly Detection

Looking into whether SNNs for ECG classification could work as a thesis topic. Covering what exists, datasets, performance vs conventional DL, open-source tools, novelty angles, and whether it's actually feasible for an undergrad project.

---

## Has Anyone Done This Before?

Yeah, it's an active but still emerging field. Between 2020 and 2025, roughly 15-20 peer-reviewed papers have directly addressed SNN-based ECG classification. That's not a lot compared to the hundreds using conventional deep learning (CNNs, LSTMs, Transformers). Most of the work is motivated by energy efficiency -- SNNs consume orders of magnitude less power than traditional DNNs, which is great for wearable and edge-deployed cardiac monitors.

Best SNN accuracy on MIT-BIH reaches 98.29% (SparrowSNN, 2024) at 31.39 nanojoules per inference, which is competitive with CNN baselines (97-99%). But most SNN-ECG work focuses narrowly on MIT-BIH with single-lead signals and 5-class AAMI classification. Big gaps exist in 12-lead classification (PTB-XL), spike encoding comparison, interpretability, and continual/few-shot learning -- all potentially viable novelty angles.

The natural fit here is actually a strong argument. ECG signals are temporal, quasi-periodic, and have spike-like QRS complexes. R-peaks and QRS complexes map naturally to spike trains, and delta modulation encoding can convert ECG signals into sparse spike representations with minimal information loss.

### Key papers and results:

| Paper / System | Year | Dataset | Classes | Accuracy | Energy | Key Innovation |
|---|---|---|---|---|---|---|
| Energy Efficient ECG (Corradi et al.) | 2020 | MIT-BIH | 5 (AAMI) | ~95% | Low (estimated) | First dedicated SNN-ECG work; delta modulation encoding |
| SNN + Attention (Deng et al.) | 2022 | MIT-BIH | 5 (AAMI) | 98.26% | 346.33 uJ/beat | Channel-wise attentional module in SNN |
| Deep SNN from CNN Conversion (Hu et al.) | 2022 | MIT-BIH | 4 | 84.41% | -- | DNN-to-SNN conversion with ReLU, 14-layer deep SNN |
| SNN + STDP Learning (various) | 2023 | MIT-BIH | 4 | 97.9% | 1.78 uJ/beat | Unsupervised STDP training; real-time inference |
| sCCfC (Spiking ConvLSTM + CfC) | 2024 | PTB-XL / CPSC | Multiple | Competitive | 4.68 uJ/Inf (neuromorphic) vs 450 uJ/Inf (CPU) | On-device edge learning |
| SparrowSNN | 2024 | MIT-BIH | 5 (AAMI) | 98.29% | 31.39 nJ/inference | SOTA SNN accuracy; ASIC co-design |
| LIF-based ANN-Inspired SNN | 2024 | MIT-BIH | 5 | ~93.8% | -- | LIF neurons within ANN-inspired framework |
| Neuromorphic Arrhythmia Detection (Kolhar) | 2025 | MIT-BIH | Multiple | 94.4% overall | <8ms inference, 1.28M FLOPs, 2.59 MB model | Lightweight for wearable deployment |
| AF Detection on Wearable Edge | 2024 | PhysioNet AF | 2 (AF/Normal) | High (>95%) | Minimal | Feed-forward SNN with custom encoder |

