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
