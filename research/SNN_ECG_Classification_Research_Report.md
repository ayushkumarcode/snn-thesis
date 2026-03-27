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

#### Key Observations
- **Accuracy range for SNNs on MIT-BIH**: 84% to 98.29%, depending on architecture and training method
- **Best performing**: SparrowSNN (98.29%) and SNN+Attention (98.26%) are near-SOTA
- **Training approaches**: Three main paradigms -- (1) ANN-to-SNN conversion, (2) surrogate gradient backpropagation, (3) unsupervised STDP
- **ANN-to-SNN conversion** suffers ~1-15% accuracy drop vs. original ANN
- **Surrogate gradient training** (direct SNN training) yields the best results

### Sources
- [SparrowSNN (arXiv 2024)](https://arxiv.org/html/2406.06543)
- [SNN + Attention (MDPI Electronics 2022)](https://www.mdpi.com/2079-9292/11/12/1889)
- [sCCfC On-device Edge Learning (APL Machine Learning 2024)](https://pubs.aip.org/aip/aml/article/2/2/026109/3282738/On-device-edge-learning-for-cardiac-abnormality)
- [Review on SNN-based ECG Classification (Biomedical Engineering Letters 2024)](https://link.springer.com/article/10.1007/s13534-024-00391-2)
- [Neuromorphic Arrhythmia Detection (Scientific Reports 2025)](https://www.nature.com/articles/s41598-025-23248-9)
- [LIF-based SNN Framework (Sensors 2024)](https://www.mdpi.com/1424-8220/24/11/3426)

---

## 3. Available Datasets

### Tier 1: Primary Benchmark Datasets (Most Used in SNN-ECG Research)

| Dataset | Records | Leads | Sampling Rate | Classes | Size | Access |
|---|---|---|---|---|---|---|
| **MIT-BIH Arrhythmia Database** | 48 recordings (47 subjects) | 2-lead | 360 Hz | 5 AAMI classes (N,S,V,F,Q) | ~100 MB | Free on PhysioNet |
| **PTB-XL** | 21,799 ECGs (18,869 patients) | 12-lead | 500 Hz (+ 100 Hz) | 71 SCP-ECG statements, 5 super-classes | ~7.7 GB | Free on PhysioNet |
| **CPSC 2018 (ICBEB)** | 6,877 training + 2,954 test | 12-lead | 500 Hz | 9 classes (1 normal + 8 abnormal) | ~1 GB | Free on PhysioNet |

### Tier 2: Supplementary Datasets

| Dataset | Description | Access |
|---|---|---|
| **Large Scale 12-Lead ECG for Arrhythmia (Chapman-Shaoxing)** | 45,152 patients, 12-lead, 500 Hz | Free on PhysioNet |
| **St Petersburg INCART** | 32 Holter records, 12-lead, annotated | Free on PhysioNet |
| **PhysioNet/CinC Challenge 2020** | Multi-database 12-lead ECG classification | Free on PhysioNet |
| **QTDB** | QT interval annotations, used in some SNN studies | Free on PhysioNet |
| **Icentia11k** | 11,000 patients, single-lead, 7 days continuous | Free (large download) |
| **Kaggle MIT-BIH (CSV format)** | Pre-processed MIT-BIH in accessible CSV format | Free on Kaggle |

### Access Notes
- **All major datasets are freely available** through PhysioNet (https://physionet.org)
- MIT-BIH is the de facto standard for SNN-ECG work (used in ~90% of papers)
- PTB-XL is the gold standard for 12-lead classification but has NOT been used with SNNs (major gap)
- CPSC 2018 is used by some SNN papers (sCCfC) but under-explored
- Kaggle versions of MIT-BIH provide ready-to-use CSV/numpy arrays

### Sources
- [MIT-BIH Arrhythmia Database (PhysioNet)](https://physionet.org/content/mitdb/1.0.0/)
- [PTB-XL Dataset (PhysioNet)](https://physionet.org/content/ptb-xl/1.0.3/)
- [CPSC 2018 Challenge](http://2018.icbeb.org/Challenge.html)
- [PhysioNet Database List](https://physionet.org/about/database/)

---

## 4. SNN Performance vs. Conventional Deep Learning

### Accuracy Comparison (MIT-BIH, 5-class AAMI)

| Method | Architecture | Accuracy | F1 Score | Energy per Inference |
|---|---|---|---|---|
| **CNN (conventional)** | 1D-CNN | 97.4-99.5% | 95-98% | ~450 uJ (CPU) |
| **CNN-LSTM hybrid** | CNN + BiLSTM + Attention | 99.2% | 98.3% | High (GPU) |
| **CNN-LSTM-SE** | CNN + LSTM + Squeeze-Excite | 98.5% | >97% | High (GPU) |
| **SNN (SparrowSNN)** | Co-designed SNN + ASIC | 98.29% | ~97% | **31.39 nJ** |
| **SNN + Attention** | SNN + Channel-wise Attention | 98.26% | 89.09% | 346.33 uJ |
| **SNN (STDP)** | Unsupervised STDP | 97.9% | -- | 1.78 uJ |
| **SNN (ANN-to-SNN)** | Converted 14-layer CNN | 84.41% | -- | Low |
| **SNN (Neuromorphic 2025)** | Lightweight SNN | 94.4% | >88% | 1.28M FLOPs |

### Key Takeaways

1. **Accuracy gap is narrowing**: Best SNNs (98.29%) are within 1% of best CNNs (99.5%) on MIT-BIH
2. **Energy advantage is massive**: SNNs are 100x to 10,000x more energy efficient
   - SparrowSNN: 31.39 nJ vs. CNN on CPU: ~450 uJ (a factor of ~14,000x)
   - SNN on Loihi: ~30 mW vs. LSTM on GPU: ~15W (a factor of 500x)
3. **F1 score gap exists**: SNN F1 scores (89-97%) trail CNNs (95-98%), especially on minority classes (S, F)
4. **Model size advantage**: SNN models are typically 2-10 MB vs. 50-200+ MB for CNN/Transformers
5. **Latency advantage**: SNN inference in <8ms enables true real-time classification
6. **The trade-off is clear**: SNNs sacrifice 1-5% accuracy for 100-10,000x energy savings

### When SNNs Win
- Edge/wearable deployment where power is constrained
- Real-time continuous monitoring (latency matters)
- Battery-powered devices (smartwatches, patches)
- Always-on cardiac monitoring

### When CNNs/Transformers Win
- Server-side batch processing where power is unlimited
- Maximum accuracy is the only priority
- 12-lead ECG analysis (SNNs have not been validated here)

### Sources
- [SparrowSNN Energy Results (arXiv 2024)](https://arxiv.org/html/2406.06543)
- [CNN-LSTM-SE Comparison (MDPI Sensors 2024)](https://www.mdpi.com/1424-8220/24/19/6306)
- [Systematic Review: ECG Arrhythmia Classification (arXiv 2025)](https://arxiv.org/abs/2503.07276)

---

## 5. Natural Fit Between ECG Signals and SNNs

### This is one of the strongest arguments for this research direction.

#### Why ECG and SNNs are a Natural Match

1. **Temporal/Event-Driven Nature**
   - ECG signals are inherently temporal and quasi-periodic
   - QRS complexes are sharp, spike-like events (naturally map to SNN spikes)
   - The R-peak is the dominant "event" in each heartbeat cycle
   - P-waves, T-waves are secondary events with precise timing
   - SNNs process information through precisely timed spikes -- a direct analogy

2. **Sparse Representation**
