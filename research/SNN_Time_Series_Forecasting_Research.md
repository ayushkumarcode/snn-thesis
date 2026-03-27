# Spiking Neural Networks for Time-Series Prediction and Forecasting
## Comprehensive Research Report -- February 2026

---

## 1. Executive Summary

Spiking Neural Networks (SNNs) for time-series forecasting is a rapidly emerging research direction that has gained significant momentum in 2024-2026, with landmark papers appearing at ICML 2024, NeurIPS 2024, and ICLR 2025. The field is at an inflection point: SNNs have been demonstrated to achieve **comparable or superior forecasting accuracy** to traditional ANNs (LSTMs, Transformers) on standard benchmarks while consuming **60-99% less energy**. However, critical gaps remain -- particularly in long-range dependency modelling, standardised benchmarks specifically designed for SNN temporal evaluation, and application to domains beyond traffic and electricity data. This makes the direction both timely and rich with opportunities for an undergraduate thesis contribution.

Key findings:
- **Financial time series**: Applied but limited; NeuCube-based multimodal approaches (Scientific Reports, 2023) and VMD-SNN hybrids (2024-2025) show promise but the area is under-explored with SNNs.
- **Weather/sensor data**: Very early stage; polychronous SNNs and binarized SNNs have been applied but no systematic benchmarking exists.
- **Standard forecasting benchmarks**: SNNs now competitive on Metr-LA, Pems-Bay, Solar, Electricity datasets (ICML 2024, ICLR 2025).
- **Energy efficiency**: 60-99% energy reduction compared to Transformers demonstrated across multiple papers.
- **Novelty**: High -- the first major venue papers on SNN time-series forecasting appeared only in 2024. Many application domains remain untouched.

---

## 2. Has Anyone Applied SNNs to Specific Time-Series Domains?

### 2.1 Financial Time Series

| Paper | Year | Venue | Approach | Key Result |
|-------|------|-------|----------|------------|
| Kasabov et al. -- NeuCube multimodal | 2023 | Scientific Reports | SNN + news integration for stock prediction | Demonstrated explainable multimodal streaming data modelling; revealed dynamic interaction between stock variables and news |
| ICS-SNN (Improved Cuckoo Search) | 2025 | MDPI Algorithms | SNN optimised by meta-heuristic for futures price prediction | 13.82% MAE reduction, 21.27% MSE reduction, 15.21% MAPE reduction vs baselines |
| VMD-SNN | 2024 | Journal of Applied Statistics | Variational Mode Decomposition + SNN for stock market index | Integrated VMD preprocessing with SNN for carbon/stock forecasting |
| TCN-LSTM-SNN hybrid | 2025 | PMC | Hybrid architecture for stock market index prediction | Combined feature extraction with SNN temporal processing |
| Reid & Hussain | 2014 | PLOS ONE | Early SNN for financial time series | Demonstrated feasibility; favourable annualised returns for 1-step and 5-step predictions |

**Assessment**: Financial time-series with SNNs is a **niche but growing** area. Most work uses older SNN architectures or hybrid approaches. Applying modern SNN frameworks (iSpikformer, TS-LIF) to financial data would be **novel**.

### 2.2 Weather Prediction

| Paper | Year | Approach | Key Result |
|-------|------|----------|------------|
| Polychronous SNN for weather signals | 2015 | PSNN with axonal delay encoding | Inherent characteristics well-suited to complex weather signal processing and prediction |
| Binarized SNN (SWP-AAFT-BSNN) | 2024 | Knowledge and Information Systems | Enhanced Smart Weather Prediction using binarized SNNs for atmospheric analysis |

**Assessment**: Weather prediction with SNNs is **very under-explored**. Only a handful of papers exist, and none use modern SNN architectures or standard weather benchmarks (e.g., the Jena Weather dataset used in mainstream forecasting).

### 2.3 Sensor Data / IoT Forecasting

| Paper | Year | Approach | Key Result |
|-------|------|----------|------------|
| Online spiking reservoir | 2022 | Neurocomputing | Spiking reservoir-based network for online time series forecasting with temporal spike encoding |
| SNN for ECG/EEG classification | 2020-2025 | Multiple | Extensive work on wearable sensor time-series classification (not forecasting) |
| Vacuum Spiker | 2025 | arXiv | SNN-based anomaly detection in time series |
| Evolving SNN anomaly detection | 2022 | Machine Learning (Springer) | Unsupervised anomaly detection in multivariate time series with online evolving SNNs |
| Enhanced quantile regression SNN | 2025 | arXiv | 92.3% accuracy in component failure prediction, 90-hour advance warning |

**Assessment**: Sensor data classification with SNNs is well-explored (especially ECG/EEG). Sensor data **forecasting** with SNNs is **under-explored** and represents a clear gap.

### 2.4 Standard Forecasting Benchmarks (Traffic, Electricity, Solar)

This is where the most rigorous SNN time-series forecasting work exists:

| Paper | Year | Venue | Datasets | Key Models |
|-------|------|-------|----------|------------|
| SeqSNN (Lv et al.) | 2024 | ICML | Metr-LA, Pems-Bay, Solar, Electricity | Spike-TCN, Spike-RNN, Spike-GRU, iSpikformer |
| CPG-PE (Lv et al.) | 2024 | NeurIPS | Multiple including time-series forecasting | Central Pattern Generator positional encoding for SNNs |
| TS-LIF (Feng et al.) | 2025 | ICLR | Metr-LA, Pems-Bay, Solar, Electricity | Temporal Segment LIF with dual-compartment architecture |
| SpikySpace (Tang et al.) | 2026 | arXiv (Jan 2026) | Metr-LA, Pems-Bay, Solar, Electricity | First full spiking state-space model |
| SpikeSTAG (Hu et al.) | 2025 | arXiv | Metr-LA, Pems-Bay, Solar, Electricity | GNN-SNN collaboration for spatial-temporal forecasting |
| Derivative spike encoding | 2024 | MDPI Computers | Electricity load forecasting | SLAYER-trained SNN with novel encoding |

---

## 3. Results Compared to LSTMs and Transformers

### 3.1 SeqSNN (ICML 2024) -- Microsoft Research

**Metrics**: RSE (Root Relative Squared Error) and R-squared

| Model Type | Energy Reduction vs ANN Equivalent | Performance |
|------------|-----------------------------------|-------------|
| Spike-TCN vs TCN | 63.60% energy reduction | Comparable accuracy |
| Spike-GRU vs GRU | 75.05% energy reduction | Comparable accuracy |
| iSpikformer vs iTransformer | 66.30% energy reduction | Lowest average RSE; R-squared only 0.001 below iTransformer |

**Average energy savings**: ~70.33% across all SNN variants on 45nm neuromorphic hardware.

Key finding: **iSpikformer achieves the lowest average RSE compared to ALL other methods (including ANNs)** and nearly matches iTransformer R-squared with only a marginal decrease of 0.001.

### 3.2 TS-LIF (ICLR 2025)

**Key comparisons (Electricity dataset, 96-step prediction)**:

| Model | R-squared | RSE |
|-------|-----------|-----|
| TS-former (proposed) | 0.977 | 0.261 |
| iSpikformer | 0.963 | 0.348 |
| Spike-GRU | 0.959 | 0.317 |
| TS-GRU (proposed) | 0.976 | 0.240 |

**Overall rankings**: TS-former achieved best average ranking (3.3), outperforming iTransformer (4.4) and iSpikformer (4.6).
