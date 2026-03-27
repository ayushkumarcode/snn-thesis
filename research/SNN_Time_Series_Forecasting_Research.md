# SNNs for Time-Series Prediction and Forecasting

this is a pretty exciting direction that's blown up in 2024-2026. landmark papers at ICML 2024, NeurIPS 2024, and ICLR 2025. SNNs have been shown to get **comparable or better forecasting accuracy** than LSTMs/Transformers on standard benchmarks while using **60-99% less energy**. but there are still critical gaps -- long-range dependency modelling, standardized SNN benchmarks, and application to domains beyond traffic and electricity data.

key findings:
- **Financial time series**: limited SNN work; NeuCube-based multimodal approaches (Scientific Reports, 2023) and VMD-SNN hybrids show promise but area is under-explored
- **Weather/sensor data**: very early stage; polychronous SNNs and binarized SNNs exist but no systematic benchmarking
- **Standard benchmarks**: SNNs now competitive on Metr-LA, Pems-Bay, Solar, Electricity (ICML 2024, ICLR 2025)
- **Energy**: 60-99% reduction vs Transformers across multiple papers
- **Novelty**: HIGH -- first major venue papers on SNN time-series forecasting appeared only in 2024

---

## Has Anyone Applied SNNs to Specific Domains?

### Financial Time Series

| Paper | Year | Venue | Approach | Key Result |
|-------|------|-------|----------|------------|
| Kasabov et al. -- NeuCube multimodal | 2023 | Scientific Reports | SNN + news for stock prediction | Explainable multimodal streaming data; dynamic stock-news interaction |
| ICS-SNN | 2025 | MDPI Algorithms | SNN + meta-heuristic for futures | 13.82% MAE reduction, 21.27% MSE reduction |
| VMD-SNN | 2024 | J. Applied Statistics | VMD preprocessing + SNN | Carbon/stock forecasting |
| TCN-LSTM-SNN hybrid | 2025 | PMC | Hybrid architecture | Combined feature extraction with SNN temporal processing |
| Reid & Hussain | 2014 | PLOS ONE | Early SNN for financial | Favorable returns for 1-step and 5-step |

niche but growing. most work uses older SNN architectures or hybrids. applying modern frameworks (iSpikformer, TS-LIF) to financial data would be novel.

### Weather Prediction

| Paper | Year | Approach | Key Result |
|-------|------|----------|------------|
| Polychronous SNN | 2015 | PSNN with axonal delay encoding | Well-suited to complex weather signals |
| Binarized SNN | 2024 | Knowledge & Info Systems | Enhanced weather prediction using binarized SNNs |

very under-explored. only a handful of papers, none using modern architectures or standard weather benchmarks (like the Jena Weather dataset).

### Sensor / IoT Forecasting

| Paper | Year | Approach | Key Result |
|-------|------|----------|------------|
| Online spiking reservoir | 2022 | Neurocomputing | Online time series forecasting with temporal spike encoding |
| SNN for ECG/EEG | 2020-2025 | Multiple | Lots of work on sensor classification (not forecasting) |
| Vacuum Spiker | 2025 | arXiv | SNN-based anomaly detection in time series |
| Evolving SNN | 2022 | Machine Learning (Springer) | Unsupervised anomaly detection in multivariate time series |
| Enhanced quantile regression SNN | 2025 | arXiv | 92.3% accuracy in component failure prediction, 90-hour advance warning |

sensor classification is well-explored (especially ECG/EEG). sensor **forecasting** with SNNs is under-explored -- clear gap.

### Standard Forecasting Benchmarks (Traffic, Electricity, Solar)

this is where the most rigorous SNN forecasting work lives:

| Paper | Year | Venue | Datasets | Models |
|-------|------|-------|----------|--------|
| SeqSNN (Lv et al.) | 2024 | ICML | Metr-LA, Pems-Bay, Solar, Electricity | Spike-TCN, Spike-RNN, Spike-GRU, iSpikformer |
| CPG-PE (Lv et al.) | 2024 | NeurIPS | Multiple including forecasting | Central Pattern Generator positional encoding |
