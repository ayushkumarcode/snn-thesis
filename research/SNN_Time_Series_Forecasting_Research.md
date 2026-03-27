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

### 3.3 SpikySpace (January 2026)

| Comparison | Energy Reduction | Accuracy |
|-----------|-----------------|----------|
| SpikySpace vs iTransformer | 98.73% (78.9x reduction) | Competitive; outperformed on Electricity (R-squared 0.994 vs 0.983) |
| SpikySpace vs iSpikformer | 96.24% (26.6x reduction) | Up to 3.0% better than previous best SNN |
| Parameter efficiency | 53.1%-55.4% of baseline parameters | -- |

Consumed only **0.17 mJ** on Electricity dataset with T=3 timesteps.

### 3.4 Summary Comparison Table

| Metric | SNNs vs LSTMs/RNNs | SNNs vs Transformers |
|--------|--------------------|--------------------|
| Accuracy (short horizon) | Comparable to slightly better | Comparable (within 0.1-1%) |
| Accuracy (long horizon) | Gap exists for very long sequences | Significant gap on 2400+ timestep tasks |
| Energy efficiency | 60-75% less energy | 66-99% less energy |
| Parameter count | 45-55% fewer parameters (SpikySpace) | 45-55% fewer parameters |
| Training difficulty | Harder (surrogate gradients) | Harder (surrogate gradients) |

### 3.5 Honest Assessment of Limitations

From the survey "Spiking Neural Networks for Temporal Processing: Status Quo and Future Prospects" (arXiv, Feb 2025):
- On the Binary Adding task with T=2400: LSTM/SSM/Transformer achieve ~100% accuracy, while advanced SNNs (LTC, CELIF, PMSN) **degrade substantially** and some "fail to learn any meaningful temporal information"
- **A significant gap still exists between SNNs and ANNs in modelling long-range dependencies**
- Many existing SNN benchmarks (CIFAR10-DVS, N-MNIST) lack genuine temporal dependencies and mask the limitations
- SNNs achieve 5-45x better energy efficiency despite lower accuracy on demanding temporal tasks

---

## 4. Theoretical Arguments for SNNs on Temporal Data

### 4.1 Inherent Temporal Processing

Unlike ANNs where neurons maintain constant state regardless of time, SNN neurons change over time in response to stimuli. When membrane potential reaches a threshold, the neuron "fires." This discrete firing mechanism enables SNNs to:
- Process temporal information **natively** without additional architectural mechanisms (unlike LSTMs which need gating)
- Encode information in spike timing, not just amplitude
- Operate as **inherently stateful models** with rich neuronal dynamics

### 4.2 Temporal Coding Efficiency

A single spiking neuron with temporal coding can theoretically replace hundreds of hidden units in a conventional neural network. Temporal coding represents complex temporal patterns with relatively few spikes, providing:
- **Information density**: Spike timing encodes more information per event than rate coding
- **Energy efficiency**: Fewer spikes = fewer computations
- **Natural time-series alignment**: Data arrives over time; spikes process data over time

### 4.3 Event-Driven Computation

SNNs transmit binary outputs sparsely and asynchronously as spikes. This event-based transmission:
- Reduces communication channels
- Lowers energy requirements by up to **1,000x** on neuromorphic processors vs traditional processors
- Naturally aligns with streaming/real-time time-series data

### 4.4 Biological Plausibility (STDP)

Spike-Timing-Dependent Plasticity (STDP) provides a biologically-plausible learning mechanism where long-term changes in synapse strength are modulated by temporal relationships between pre- and post-synaptic spikes. Neurons learn sequences over long timescales and shift their spikes towards the first inputs in a sequence -- a form of **efficient coding** that naturally captures temporal patterns.

### 4.5 Central Pattern Generator Analogy (NeurIPS 2024)

The CPG-PE paper demonstrated that the commonly used sinusoidal positional encoding is mathematically a specific solution to the membrane potential dynamics of a particular CPG. This establishes a direct theoretical link between biological rhythm generation and positional encoding in sequence models.

### 4.6 Counterarguments and Caveats

- Training SNNs via surrogate gradients introduces approximation errors
- The non-differentiable spiking function makes optimisation harder than in ANNs
- Long-range dependency modelling remains a fundamental challenge
- Current SNN architectures often **adopt ANN architectures directly**, which are likely sub-optimal for spike-based computation

---

## 5. Available Datasets and Benchmarks

### 5.1 Datasets Used in SNN Time-Series Forecasting Papers

| Dataset | Domain | Variables | Samples | Frequency | Used In |
|---------|--------|-----------|---------|-----------|---------|
| Metr-LA | Traffic speed (Los Angeles) | 207 | 34,272 | 5 min | SeqSNN, TS-LIF, SpikySpace, SpikeSTAG |
| Pems-Bay | Traffic speed (Bay Area) | 325 | 52,116 | 5 min | SeqSNN, TS-LIF, SpikySpace, SpikeSTAG |
| Solar-Energy | Solar power production | 137 | 52,560 | Hourly | SeqSNN, TS-LIF, SpikySpace |
| Electricity | Electricity consumption | 321 | 26,304 | Hourly | SeqSNN, TS-LIF, SpikySpace |

### 5.2 Standard Time-Series Forecasting Benchmarks (Not Yet Widely Used with SNNs)

| Dataset | Domain | Variables | Frequency | Notes |
|---------|--------|-----------|-----------|-------|
| ETTh1 / ETTh2 | Electricity Transformer Temperature | 7 | Hourly | Standard long-term forecasting benchmark |
| ETTm1 / ETTm2 | Electricity Transformer Temperature | 7 | 15-minute | Standard long-term forecasting benchmark |
| Weather | Meteorological measurements | 21 | 10-minute | Max Planck Institute, Jena, Germany (2020) |
| Traffic | California highway occupancy | 862 | Hourly | CalTrans PEMS |
| ILI (Illness) | Influenza-like illness | 7 | Weekly | CDC data |

**Key observation**: Most SNN papers use Metr-LA, Pems-Bay, Solar, Electricity. The ETTh/ETTm, Weather, and ILI datasets widely used in Transformer-based forecasting have **NOT been systematically benchmarked with SNNs**. This is a clear research gap.

### 5.3 Standard Evaluation Protocol

- Train/val/test split: 6:2:2 for ETT datasets, 7:1:2 for others
- Prediction horizons: {96, 192, 336, 720} for most datasets; {24, 36, 48, 60} for ILI
- Metrics: MSE and MAE are standard for Transformer comparisons; RSE and R-squared used in SNN papers (complicates direct comparison)

### 5.4 Dataset Sources

- ETT datasets: https://github.com/zhouhaoyi/ETDataset
- Comprehensive collection: https://github.com/juyongjiang/TimeSeriesDatasets
- SeqSNN datasets: Available via Google Drive from the SeqSNN repository

---

## 6. Open-Source Implementations

### 6.1 SNN Time-Series Forecasting Specific

| Repository | Paper/Venue | Framework | URL | Stars | Key Features |
|-----------|-------------|-----------|-----|-------|-------------|
| SeqSNN (Microsoft) | ICML 2024 + NeurIPS 2024 | PyTorch | https://github.com/microsoft/SeqSNN | -- | iSpikformer, Spike-TCN, Spike-GRU; YAML config-driven |
| TS-LIF | ICLR 2025 | PyTorch (builds on SeqSNN) | https://github.com/kkking-kk/TS-LIF | -- | Dual-compartment TS-LIF neuron model |
| Lvchangze/SeqSNN | ICML 2024 + NeurIPS 2024 | PyTorch | https://github.com/Lvchangze/SeqSNN | -- | Author's version with additional features |

### 6.2 General SNN Frameworks (for Building Custom Models)

| Framework | Language | Backend | Key Strengths | URL |
|-----------|----------|---------|---------------|-----|
| **snnTorch** | Python | PyTorch | Regression tutorials, beginner-friendly, extensive docs | https://github.com/jeshraghian/snntorch |
| **SpikingJelly** | Python | PyTorch + CuPy | Fastest (0.26s forward+backward), full-stack including neuromorphic chip deployment | https://github.com/fangwei123456/spikingjelly |
| **Norse** | Python | PyTorch | Bio-inspired primitives, flexible custom neuron models, torch.compile support | https://github.com/norse/norse |
| **Lava** (Intel) | Python | Custom | Loihi 2 deployment, SLAYER learning rule, neuromorphic hardware | https://github.com/lava-nc/lava |

### 6.3 Recommended Starting Points

**For an undergraduate thesis, the recommended path is**:

1. **Start with snnTorch** -- it has the most beginner-friendly tutorials:
   - Tutorial 1: Spike Encoding (https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html)
   - Tutorial 5: Training SNNs (https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html)
   - Tutorial Regression Part I: Membrane potential trajectory training (https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_regression_1.html)
   - Tutorial Regression Part II: Recurrent feedback for regression (https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_regression_2.html)

2. **Then examine SeqSNN** for reference implementations of time-series specific architectures:
   ```
   conda create -n SeqSNN python=3.10
   conda activate SeqSNN
   git clone https://github.com/microsoft/SeqSNN/
   cd SeqSNN
   pip install .
   python -m SeqSNN.entry.tsforecast exp/forecast/ispikformer/ispikformer_electricity.yml
   ```

### 6.4 Curated Paper Lists with Code

- https://github.com/zhouchenlin2096/Awesome-Spiking-Neural-Networks
- https://github.com/yfguo91/Awesome-Spiking-Neural-Networks
- https://github.com/AXYZdong/awesome-snn-conference-paper
- https://github.com/SpikingChen/SNN-Daily-Arxiv (daily arXiv updates)

---

## 7. How Novel is This Direction?

### 7.1 Publication Timeline

| Year | Milestone |
|------|-----------|
| 2010 | First SNN for electricity price forecasting (IEEE, temporal encoding) |
| 2014 | Financial time series with SNN (PLOS ONE) |
| 2015 | Polychronous SNN for weather (Springer) |
| 2022 | Online spiking reservoir for time series (Neurocomputing) |
| 2023 | NeuCube for multimodal financial time series (Scientific Reports) |
| **2024** | **SeqSNN: First top-venue paper (ICML) -- the breakthrough year** |
| **2024** | **CPG-PE for sequential SNN modelling (NeurIPS)** |
| **2024** | **Derivative spike encoding for forecasting (MDPI)** |
| **2024** | **Univariate time-series methodology (Neural Networks journal)** |
| **2025** | **TS-LIF: Dual-compartment SNN for forecasting (ICLR)** |
| **2025** | **ICS-SNN: Financial forecasting with meta-heuristic optimisation (MDPI)** |
| **2025** | **SpikeSTAG: GNN-SNN collaboration for spatial-temporal (arXiv)** |
| **2026** | **SpikySpace: First full spiking state-space model (arXiv, Jan 2026)** |

### 7.2 Novelty Assessment

**Overall field maturity**: EMERGING -- The first major conference paper on SNN time-series forecasting appeared only in February 2024 (ICML). This is a field that is approximately **2 years old** at the top-venue level.

**Under-explored sub-areas** (potential thesis directions):

| Sub-area | Current State | Novelty Level |
|----------|--------------|---------------|
| SNNs on ETTh/ETTm/Weather benchmarks | Not systematically done | **HIGH** |
| SNNs for financial time series (modern architectures) | Only hybrid/legacy approaches | **HIGH** |
| SNNs for weather forecasting | 2 papers, neither using modern SNN architectures | **VERY HIGH** |
| SNNs for IoT sensor forecasting | Mostly classification, not forecasting | **HIGH** |
| SNN vs Transformer on long-horizon prediction | Limited comparison | **HIGH** |
| Encoding strategy comparison for time series | Fragmented across papers | **MEDIUM-HIGH** |
| Energy efficiency benchmarking on real hardware | Mostly theoretical calculations | **HIGH** |
| SNN for anomaly detection in time series | Early stage (2-3 papers) | **HIGH** |
| Explainability of SNN temporal predictions | Only NeuCube approach | **VERY HIGH** |

### 7.3 Why Now?

