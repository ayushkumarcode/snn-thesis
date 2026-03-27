# Spiking Neural Networks for ECG / Heartbeat Anomaly Detection
## Comprehensive Research Report

**Date:** 2026-02-25
**Research Scope:** SNN-based ECG classification, datasets, performance benchmarks, open-source tools, novelty angles, and undergraduate thesis feasibility.

---

## 1. Executive Summary

Spiking Neural Networks (SNNs) applied to ECG classification and heartbeat anomaly detection represent a rapidly growing but still under-explored research area with significant untapped potential. Between 2020 and 2025, approximately 15-20 peer-reviewed papers have directly addressed SNN-based ECG classification, compared to hundreds using conventional deep learning (CNNs, LSTMs, Transformers). The field is dominated by energy-efficiency motivations -- SNNs consume orders of magnitude less power than traditional DNNs, making them ideal for wearable and edge-deployed cardiac monitors. State-of-the-art SNN accuracy on the MIT-BIH benchmark reaches 98.29% (SparrowSNN, 2024) at 31.39 nanojoules per inference, competitive with CNN baselines (97-99%). However, most SNN-ECG work focuses narrowly on the MIT-BIH dataset with single-lead signals and 5-class AAMI classification. Major gaps exist in 12-lead classification (PTB-XL), spike encoding method comparison, interpretability, and continual/few-shot learning -- all of which represent viable novelty angles for an undergraduate thesis.

The natural fit between ECG signals (temporal, quasi-periodic, spike-like QRS complexes) and SNNs (event-driven, temporal processing) is a core strength of this research direction. ECG R-peaks and QRS complexes map naturally to spike trains, and delta modulation encoding can convert ECG signals into sparse spike representations with minimal information loss.

---

## 2. Has Anyone Done SNN-Based ECG Classification? What Results Did They Get?

### Yes -- this is an active but still emerging field. Key papers and results:

#### Landmark Papers (Chronological)

| Paper / System | Year | Dataset | Classes | Accuracy | Energy | Key Innovation |
|---|---|---|---|---|---|---|
| Energy Efficient ECG (Corradi et al.) | 2020 | MIT-BIH | 5 (AAMI) | ~95% | Low (estimated) | First dedicated SNN-ECG work; delta modulation encoding |
| SNN + Attention (Deng et al.) | 2022 | MIT-BIH | 5 (AAMI) | 98.26% | 346.33 uJ/beat | Channel-wise attentional module in SNN |
| Deep SNN from CNN Conversion (Hu et al.) | 2022 | MIT-BIH | 4 | 84.41% | -- | DNN-to-SNN conversion with ReLU, 14-layer deep SNN |
| SNN + STDP Learning (various) | 2023 | MIT-BIH | 4 | 97.9% | 1.78 uJ/beat | Unsupervised STDP training; real-time inference |
| sCCfC (Spiking ConvLSTM + CfC) | 2024 | PTB-XL / CPSC | Multiple | Competitive | 4.68 uJ/Inf (neuromorphic) vs 450 uJ/Inf (CPU) | On-device edge learning; bio-inspired architecture |
| SparrowSNN (Hardware/Software Co-design) | 2024 | MIT-BIH | 5 (AAMI) | 98.29% | 31.39 nJ/inference | SOTA SNN accuracy; ASIC co-design; minimal timesteps |
| LIF-based ANN-Inspired SNN | 2024 | MIT-BIH | 5 | ~93.8% | -- | LIF neurons within ANN-inspired framework |
| Neuromorphic Arrhythmia Detection (Kolhar) | 2025 | MIT-BIH | Multiple | 94.4% overall | <8ms inference, 1.28M FLOPs, 2.59 MB model | Lightweight for real-time wearable deployment |
| AF Detection on Wearable Edge | 2024 | PhysioNet AF | 2 (AF/Normal) | High (>95%) | Minimal | Feed-forward SNN with custom encoder |
