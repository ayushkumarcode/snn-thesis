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

- surrogate gradients introduce approximation errors
- non-differentiable spiking makes optimization harder
- long-range dependencies remain fundamentally challenging
- current SNN architectures often just adopt ANN architectures directly, which is probably sub-optimal for spikes

---

## Datasets and Benchmarks

### Used in SNN Forecasting Papers

| Dataset | Domain | Variables | Samples | Frequency | Used In |
|---------|--------|-----------|---------|-----------|---------|
| Metr-LA | Traffic (LA) | 207 | 34,272 | 5 min | SeqSNN, TS-LIF, SpikySpace, SpikeSTAG |
| Pems-Bay | Traffic (Bay Area) | 325 | 52,116 | 5 min | Same |
| Solar-Energy | Solar power | 137 | 52,560 | Hourly | Same |
| Electricity | Consumption | 321 | 26,304 | Hourly | Same |

### Standard Benchmarks NOT Yet Used with SNNs

| Dataset | Domain | Variables | Frequency | Notes |
|---------|--------|-----------|-----------|-------|
| ETTh1 / ETTh2 | Transformer temperature | 7 | Hourly | Standard long-term forecasting benchmark |
| ETTm1 / ETTm2 | Same | 7 | 15-min | Same |
| Weather | Meteorological | 21 | 10-min | Max Planck Institute, Jena (2020) |
| Traffic | Highway occupancy | 862 | Hourly | CalTrans PEMS |
| ILI | Influenza-like illness | 7 | Weekly | CDC data |

**important observation**: most SNN papers use Metr-LA, Pems-Bay, Solar, Electricity. the ETTh/ETTm, Weather, and ILI datasets used by every Transformer forecasting paper have **NOT been benchmarked with SNNs**. that's a clear gap.

### Evaluation Protocol

- split: 6:2:2 for ETT, 7:1:2 for others
- horizons: {96, 192, 336, 720} most datasets; {24, 36, 48, 60} for ILI
- metrics: MSE and MAE standard for Transformer papers; RSE and R-squared used in SNN papers (this mismatch is annoying)

### Dataset Sources

- ETT: https://github.com/zhouhaoyi/ETDataset
- Collection: https://github.com/juyongjiang/TimeSeriesDatasets
- SeqSNN data: Google Drive via SeqSNN repo

---

## Open-Source Implementations

### SNN Forecasting Specific

| Repo | Paper/Venue | Framework | URL |
|------|-------------|-----------|-----|
| SeqSNN (Microsoft) | ICML 2024 + NeurIPS 2024 | PyTorch | https://github.com/microsoft/SeqSNN |
| TS-LIF | ICLR 2025 | PyTorch (builds on SeqSNN) | https://github.com/kkking-kk/TS-LIF |
| Lvchangze/SeqSNN | ICML + NeurIPS | PyTorch | https://github.com/Lvchangze/SeqSNN |

### General Frameworks

| Framework | Strengths | URL |
|-----------|-----------|-----|
| **snnTorch** | Regression tutorials, beginner-friendly | https://github.com/jeshraghian/snntorch |
| **SpikingJelly** | Fastest (0.26s forward+backward), deployment | https://github.com/fangwei123456/spikingjelly |
| **Norse** | Bio-inspired, flexible custom neurons | https://github.com/norse/norse |
| **Lava** (Intel) | Loihi 2 deployment | https://github.com/lava-nc/lava |

### Getting Started

1. **start with snnTorch** -- tutorials for regression exist:
   - Tutorial 1: Spike Encoding
   - Tutorial 5: Training SNNs
   - Regression tutorials Part I and II

2. **then look at SeqSNN** for time-series specific architectures:
   ```
   conda create -n SeqSNN python=3.10
   git clone https://github.com/microsoft/SeqSNN/
   cd SeqSNN && pip install .
   python -m SeqSNN.entry.tsforecast exp/forecast/ispikformer/ispikformer_electricity.yml
   ```

### Paper Lists with Code

- https://github.com/zhouchenlin2096/Awesome-Spiking-Neural-Networks
- https://github.com/yfguo91/Awesome-Spiking-Neural-Networks
- https://github.com/SpikingChen/SNN-Daily-Arxiv (daily arXiv updates)
