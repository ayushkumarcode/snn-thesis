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

Things to notice:
- Accuracy range for SNNs on MIT-BIH: 84% to 98.29%, depending on architecture and training
- Three main training paradigms: (1) ANN-to-SNN conversion, (2) surrogate gradient backprop, (3) unsupervised STDP
- ANN-to-SNN conversion suffers ~1-15% accuracy drop vs original ANN
- Surrogate gradient training (direct SNN training) gives the best results

Sources:
- SparrowSNN (arXiv 2024): https://arxiv.org/html/2406.06543
- SNN + Attention (MDPI Electronics 2022): https://www.mdpi.com/2079-9292/11/12/1889
- sCCfC On-device Edge Learning (APL Machine Learning 2024): https://pubs.aip.org/aip/aml/article/2/2/026109/3282738
- Review on SNN-based ECG Classification (Biomedical Engineering Letters 2024): https://link.springer.com/article/10.1007/s13534-024-00391-2
- Neuromorphic Arrhythmia Detection (Scientific Reports 2025): https://www.nature.com/articles/s41598-025-23248-9
- LIF-based SNN Framework (Sensors 2024): https://www.mdpi.com/1424-8220/24/11/3426

---

## Available Datasets

### Tier 1: Primary Benchmarks (most used in SNN-ECG research)

| Dataset | Records | Leads | Sampling Rate | Classes | Size | Access |
|---|---|---|---|---|---|---|
| **MIT-BIH Arrhythmia Database** | 48 recordings (47 subjects) | 2-lead | 360 Hz | 5 AAMI classes (N,S,V,F,Q) | ~100 MB | Free on PhysioNet |
| **PTB-XL** | 21,799 ECGs (18,869 patients) | 12-lead | 500 Hz (+ 100 Hz) | 71 SCP-ECG statements, 5 super-classes | ~7.7 GB | Free on PhysioNet |
| **CPSC 2018 (ICBEB)** | 6,877 training + 2,954 test | 12-lead | 500 Hz | 9 classes (1 normal + 8 abnormal) | ~1 GB | Free on PhysioNet |

### Tier 2: Supplementary

| Dataset | Description | Access |
|---|---|---|
| **Chapman-Shaoxing** | 45,152 patients, 12-lead, 500 Hz | Free on PhysioNet |
| **St Petersburg INCART** | 32 Holter records, 12-lead, annotated | Free on PhysioNet |
| **PhysioNet/CinC Challenge 2020** | Multi-database 12-lead ECG classification | Free on PhysioNet |
| **QTDB** | QT interval annotations, used in some SNN studies | Free on PhysioNet |
| **Icentia11k** | 11,000 patients, single-lead, 7 days continuous | Free (large download) |
| **Kaggle MIT-BIH (CSV format)** | Pre-processed MIT-BIH in accessible CSV format | Free on Kaggle |

All major datasets are freely available through PhysioNet (https://physionet.org). MIT-BIH is the de facto standard for SNN-ECG work (~90% of papers use it). PTB-XL is the gold standard for 12-lead but has NOT been used with SNNs -- that's a major gap. Kaggle versions of MIT-BIH give you ready-to-use CSV/numpy arrays.

---

## SNN Performance vs Conventional Deep Learning

### Accuracy Comparison (MIT-BIH, 5-class AAMI)

| Method | Architecture | Accuracy | F1 Score | Energy per Inference |
|---|---|---|---|---|
| CNN (conventional) | 1D-CNN | 97.4-99.5% | 95-98% | ~450 uJ (CPU) |
| CNN-LSTM hybrid | CNN + BiLSTM + Attention | 99.2% | 98.3% | High (GPU) |
| CNN-LSTM-SE | CNN + LSTM + Squeeze-Excite | 98.5% | >97% | High (GPU) |
| SNN (SparrowSNN) | Co-designed SNN + ASIC | 98.29% | ~97% | **31.39 nJ** |
| SNN + Attention | SNN + Channel-wise Attention | 98.26% | 89.09% | 346.33 uJ |
| SNN (STDP) | Unsupervised STDP | 97.9% | -- | 1.78 uJ |
| SNN (ANN-to-SNN) | Converted 14-layer CNN | 84.41% | -- | Low |
| SNN (Neuromorphic 2025) | Lightweight SNN | 94.4% | >88% | 1.28M FLOPs |

