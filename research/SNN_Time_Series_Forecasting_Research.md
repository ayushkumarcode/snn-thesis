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
| TS-LIF (Feng et al.) | 2025 | ICLR | Metr-LA, Pems-Bay, Solar, Electricity | Temporal Segment LIF with dual-compartment |
| SpikySpace (Tang et al.) | 2026 | arXiv (Jan 2026) | Metr-LA, Pems-Bay, Solar, Electricity | First full spiking state-space model |
| SpikeSTAG (Hu et al.) | 2025 | arXiv | Same four | GNN-SNN collaboration |
| Derivative spike encoding | 2024 | MDPI Computers | Electricity load | SLAYER-trained SNN with novel encoding |

---

## Results vs LSTMs and Transformers

### SeqSNN (ICML 2024) -- Microsoft Research

| Model Type | Energy Reduction vs ANN | Performance |
|------------|------------------------|-------------|
| Spike-TCN vs TCN | 63.60% less energy | Comparable accuracy |
| Spike-GRU vs GRU | 75.05% less energy | Comparable accuracy |
| iSpikformer vs iTransformer | 66.30% less energy | Lowest avg RSE; R-squared only 0.001 below iTransformer |

average energy savings: ~70% on 45nm neuromorphic hardware. the key thing: **iSpikformer achieves the lowest average RSE compared to ALL methods including ANNs** while nearly matching R-squared.

### TS-LIF (ICLR 2025)

on Electricity dataset, 96-step prediction:

| Model | R-squared | RSE |
|-------|-----------|-----|
| TS-former (proposed) | 0.977 | 0.261 |
| iSpikformer | 0.963 | 0.348 |
| Spike-GRU | 0.959 | 0.317 |
| TS-GRU (proposed) | 0.976 | 0.240 |

TS-former got best average ranking (3.3), beating iTransformer (4.4) and iSpikformer (4.6).

### SpikySpace (January 2026)

| Comparison | Energy Reduction | Accuracy |
|-----------|-----------------|----------|
| vs iTransformer | 98.73% (78.9x reduction) | Competitive; beat on Electricity (R^2 0.994 vs 0.983) |
| vs iSpikformer | 96.24% (26.6x reduction) | Up to 3.0% better than previous best SNN |
| Parameters | 53-55% of baseline | -- |

consumed only **0.17 mJ** on Electricity dataset with T=3. that's impressively low.

### Summary

| Metric | SNNs vs LSTMs/RNNs | SNNs vs Transformers |
|--------|--------------------|--------------------|
| Accuracy (short horizon) | Comparable to slightly better | Comparable (within 0.1-1%) |
| Accuracy (long horizon) | Gap exists for very long sequences | Significant gap on 2400+ timestep tasks |
| Energy | 60-75% less | 66-99% less |
| Parameters | 45-55% fewer (SpikySpace) | 45-55% fewer |
| Training difficulty | Harder (surrogate gradients) | Harder |

### Being Honest About Limitations

from the survey "Spiking Neural Networks for Temporal Processing: Status Quo and Future Prospects" (arXiv, Feb 2025):
- on Binary Adding task with T=2400: LSTM/SSM/Transformer get ~100%, while advanced SNNs **degrade substantially** and some "fail to learn any meaningful temporal information"
- **significant gap still exists for long-range dependencies**
- many SNN benchmarks (CIFAR10-DVS, N-MNIST) lack genuine temporal dependencies and mask limitations
- SNNs get 5-45x better energy despite lower accuracy on demanding temporal tasks

---

## Why SNNs Might Be Good for Temporal Data (Theory)

### Inherent Temporal Processing

unlike ANNs where neurons are constant regardless of time, SNN neurons change over time. when membrane potential hits threshold, the neuron fires. this means:
- process temporal info **natively** without special mechanisms (unlike LSTMs needing gating)
- encode information in spike timing, not just amplitude
- operate as **inherently stateful** with rich neuronal dynamics

### Temporal Coding Efficiency

a single spiking neuron with temporal coding can theoretically replace hundreds of hidden units in a conventional NN. fewer spikes = fewer computations, and data arrives over time / spikes process over time -- natural alignment.

### Event-Driven Computation

binary sparse outputs, asynchronous. reduces communication, lowers energy up to **1000x** on neuromorphic processors, naturally aligns with streaming/real-time data.

### CPG Analogy (NeurIPS 2024)

the CPG-PE paper showed that sinusoidal positional encoding is mathematically a specific solution to membrane potential dynamics of a particular Central Pattern Generator. direct theoretical link between biological rhythm generation and sequence model positional encoding. cool.

### Counterarguments
