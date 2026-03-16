# SNN Novel Application Domains: Comprehensive Research Assessment

**Date:** 2026-02-25
**Purpose:** Identify underexplored application domains for Spiking Neural Networks, assess existing literature, natural SNN advantages, and undergraduate feasibility.

---

## Executive Summary

After exhaustive searching across arxiv, Google Scholar, IEEE, Springer, and GitHub, the 10 proposed SNN application domains vary dramatically in their maturity. **Music generation**, **astronomy transient detection**, and **drug discovery** represent the most genuinely underexplored frontiers with the fewest papers. **Wearable sensor data**, **radar/sonar**, and **industrial anomaly detection** are moderately explored with clear SNN advantages. **NLP/sentiment**, **game playing/RL**, and **financial fraud** have emerging but growing literature. **Environmental monitoring** sits in a middle ground with a handful of pioneering papers using evolving SNNs.

The most promising domains for an undergraduate thesis that balances novelty, feasibility, and natural SNN advantage are: **(1) SNN for music generation**, **(2) SNN for environmental monitoring**, **(3) SNN for wearable sensor data**, and **(4) SNN for anomaly detection in industrial IoT**.

---

## Domain-by-Domain Assessment

---

### 1. SNN for Music Generation / Audio Synthesis

**Existing Literature: SPARSE (5-8 papers total)**

| Paper | Year | Venue | Key Contribution |
|-------|------|-------|------------------|
| Stylistic Composition of Melodies Based on Brain-Inspired SNN (NeuCube) | 2021 | Frontiers in Systems Neuroscience | First SNN melody composition using STDP and sequential memory |
| Musical Pattern Recognition in SNNs | ~2018 | Thesis/Report | First-layer note differentiation in monophonic sequences |
| Multilayer SNN for Audio Classification (SpiNNaker) | ~2019 | Published paper | 3-layer LIF network for pure tone classification on SpiNNaker |
| Mode-conditioned music learning and composition | 2024 | arxiv (2411.14773) | Tonality-aware SNN for musical mode and key representation |
| MuSpike: Benchmark for Symbolic Music Generation with SNNs | 2025 | arxiv (2508.19251) | First comprehensive benchmark; tests 5 SNN architectures across 5 datasets |
| Spiking Vocos: Energy-Efficient Neural Vocoder | 2025 | arxiv (2509.13049) | Spiking vocoder for audio synthesis |

**GitHub Repositories:**
- `mrahtz/musical-pattern-recognition-in-spiking-neural-networks` - Note differentiation
- `jpdominguez/Multilayer-SNN-for-audio-samples-classification-using-SpiNNaker` - Audio classification on SpiNNaker

**Natural SNN Advantage: HIGH**
- Music is inherently temporal and spike-like (note onsets, rhythmic patterns)
- MIDI events are discrete, event-driven data -- naturally suited to spike encoding
- Biological auditory processing uses spike-timing codes
- STDP learning mirrors associative musical memory
- Energy efficiency matters for real-time embedded music applications

**Undergraduate Feasibility: HIGH**
- MIDI datasets are abundant and well-structured (JSB Chorales, POP909, Lakh MIDI)
- MuSpike benchmark (2025) provides a ready-made evaluation framework
- snnTorch/SpikingJelly provide accessible Python frameworks
- A focused project on single-instrument melody generation is well-scoped
- Can compare against simple RNN/LSTM baselines easily
- No need for neuromorphic hardware -- can simulate in software

**Novelty Assessment: VERY HIGH**
- Only ~5-8 papers exist in total, with the field only gaining traction in 2024-2025
- MuSpike (2025) explicitly notes the field is "significantly underexplored"
- Enormous creative space for novel contributions

**Verdict: EXCELLENT thesis candidate -- high novelty, natural SNN fit, achievable scope**

---

### 2. SNN for Anomaly Detection in Industrial IoT Sensor Data

**Existing Literature: MODERATE (10-20 papers)**

| Paper | Year | Venue | Key Contribution |
|-------|------|-------|------------------|
| Unsupervised Anomaly Detection in Multivariate Time Series with Online Evolving SNNs | 2022 | Machine Learning (Springer) | Evolving SNN for streaming time series |
| Deep Spiking Neural Network Anomaly Detection Method | 2022 | PMC/Sensors | Vibration analysis for oil infrastructure |
| Vacuum Spiker: SNN-Based Anomaly Detection in Time Series | 2025 | arxiv (2510.06910) | Efficient anomaly detection model |
| Toward End-to-End Bearing Fault Diagnosis with SNNs | 2024 | KDD 2025 | Industrial bearing diagnosis |
| Multi-modal multi-sensor SNN for bearing weak fault diagnosis | 2024 | Engineering Applications of AI | Multi-sensor fusion |
| Hybrid Recurrent + SNN for IoT Network Security | 2025 | PMC | IoT intrusion detection |
| Convolutional SNN for Intrusion Detection | 2024 | Nature Scientific Reports | Network anomaly detection |
| Brain-Inspired SNNs for Industrial Fault Diagnosis (Survey) | 2024 | Survey paper | Comprehensive survey |

**GitHub Repositories:**
- `iago-creator/Vacuum_Spiker_experimentation` - Time series anomaly detection
- `TheBrainLab/Awesome-Spiking-Neural-Networks` - Curated list with anomaly detection papers

**Natural SNN Advantage: HIGH**
- Industrial sensor data is temporal and often event-driven (anomalies are transients)
- Spike-based processing naturally detects threshold-crossing events
- Low power consumption critical for IoT edge deployment
- Real-time processing requirement matches SNN's event-driven nature
- Online/streaming learning (evolving SNNs) enables adaptation without retraining

**Undergraduate Feasibility: MODERATE-HIGH**
- Public datasets available: CWRU Bearing Dataset, NASA Bearing Dataset, SMD, SMAP
- Frameworks: snnTorch, SpikingJelly support temporal processing
- Challenge: encoding continuous sensor data into spikes requires careful design
- Well-scoped project: single-sensor anomaly detection (e.g., bearing vibration)
- Good baselines exist (autoencoders, LSTM-based methods)

**Novelty Assessment: MODERATE**
- Active but not saturated field
- Novel angles: specific industrial domains (e.g., CNC machining, HVAC systems)
- Combining online learning with anomaly detection still underexplored

**Verdict: STRONG thesis candidate -- practical impact, good SNN fit, moderate novelty**

---

### 3. SNN for Financial Fraud Detection

**Existing Literature: SPARSE-MODERATE (3-6 papers)**

| Paper | Year | Venue | Key Contribution |
|-------|------|-------|------------------|
| Reinforcement-Guided Hyper-Heuristic for SNN-Based Financial Fraud Detection | 2025 | arxiv (2508.16915) | CSNPC model, 90.8% recall at 5% FPR on BAF dataset |
| SNN for Financial Time Series Prediction | 2014 | PLOS ONE | Foundational work on SNN for financial data |
| High-Frequency Trading with SNNs | 2021 | Published paper | SNN for HFT prediction |
| SNN for Financial Data Prediction | 2013 | IEEE Conference | Early exploration |
| VMD-SNNs for Stock Market Index Prediction | 2025 | PMC | Hybrid TCN-LSTM-SNN for stock prediction |
| ICS-SNN for Financial Time Series Forecasting | 2025 | Algorithms (MDPI) | Optimized SNN for financial forecasting |

**Natural SNN Advantage: MODERATE**
- Financial transactions are event-driven (discrete events in time)
- Temporal patterns in fraud (velocity checks, time-of-day patterns) match SNN dynamics
- Energy efficiency less critical here (server-side processing)
- Spike-based threshold detection could naturally flag anomalous transactions
- However: tabular data (not temporal sequences) reduces SNN advantage for many fraud tasks

**Undergraduate Feasibility: MODERATE**
- Public datasets: Kaggle Credit Card Fraud, Bank Account Fraud (BAF) dataset
- Challenge: class imbalance is extreme (~0.17% fraud in typical datasets)
- Encoding tabular features into spikes is non-trivial
- Comparison with well-established baselines (XGBoost, Random Forest) may be unfavorable
- The 2025 CSNPC paper sets a high bar

**Novelty Assessment: MODERATE-HIGH**
- Very few SNN papers specifically on fraud detection
- Most financial SNN work focuses on time series prediction, not classification
- Novel angle: real-time streaming fraud detection with online learning

**Verdict: MODERATE thesis candidate -- novel but encoding challenge is significant**

---

### 4. SNN for Natural Language Tasks (Sentiment, Classification)

**Existing Literature: MODERATE-ESTABLISHED (15-25 papers)**

| Paper | Year | Venue | Key Contribution |
|-------|------|-------|------------------|
| SpikeGPT: Generative Pre-trained Language Model with SNNs | 2023 | arxiv (2302.13939) | 216M parameter SNN language model, 32.2x fewer operations |
| SNNLP: Energy-Efficient NLP Using SNNs | 2024 | arxiv (2401.17911) | Spike encoding methods, 32x energy reduction |
| Spiking Convolutional NNs for Text Classification | 2024 | arxiv (2406.19230) | Conversion + fine-tuning for text classification |
| SNN-BERT: Training-efficient Spiking BERT | 2024 | Neural Networks | Energy-efficient BERT conversion |
| SpikingMiniLM: Spiking Transformer for NLU | 2024 | Science China Information Sciences | Spiking transformer for language understanding |
| Neuromorphic Sentiment Analysis Using SNNs | 2023 | PMC/Sensors | Sentiment on SpiNNaker, 100% accuracy on reviews |
| Sentence-level Sentiment with Spiking Neural P Systems | 2024 | ScienceDirect | Multi-attention bidirectional gated SNN |
| Efficient Aspect Term Extraction Using SNN | 2025 | arxiv (2601.06637) | Fine-grained sentiment analysis |

**Natural SNN Advantage: LOW-MODERATE**
- Text is not naturally temporal/event-driven (unlike audio or sensor data)
- Main advantage is energy efficiency, not representation quality
- Text-to-spike encoding is an active research challenge
- SNNs trail ANNs significantly in NLP accuracy
- SpikeGPT demonstrates feasibility but at lower performance

**Undergraduate Feasibility: MODERATE**
- Many standard NLP datasets available (SST-2, IMDB, AG News)
- SpikeGPT and SNNLP provide reference implementations
- Challenge: achieving competitive accuracy is difficult
- Could focus on energy efficiency comparison rather than accuracy
- snnTorch tutorials cover text encoding basics

**Novelty Assessment: LOW-MODERATE**
- Growing body of work (15+ papers)
- Rapidly becoming an established subfield
- Novel angles: specific languages, domain-specific text, multimodal text+image

**Verdict: MODERATE thesis candidate -- feasible but less novel, energy efficiency angle needed**

---

### 5. SNN for Game Playing / Simple RL Tasks

**Existing Literature: MODERATE (10-15 papers)**

| Paper | Year | Venue | Key Contribution |
|-------|------|-------|------------------|
| Deep Spiking Q-learning (DSQN) | 2022 | arxiv (2201.09754) | SNN Q-network outperforms DQN on 17 Atari games |
| PopSAN: Population-coded Spiking Actor Network | 2021 | CoRL/PMLR | 140x energy reduction on Loihi, continuous control |
| BrainQN: Enhanced Robustness with SNNs | 2024 | Advanced Intelligent Systems | Improved robustness in spiking DRL |
| SNN RL for Atari Breakout (conversion) | 2019 | Neural Networks | ANN-to-SNN conversion for RL |
| Fully Spiking Actor Network for RL | 2024 | arxiv (2401.05444) | Intra-layer connections for RL |
| Adaptive Surrogate Gradients for Sequential RL in SNNs | 2025 | arxiv (2510.24461) | Improved training for sequential RL |
| SpikeGym Comparison | 2024 | Published paper | 1-layer SNN-PPO outperforms PopSAN by 4.4x |
| Exploring SNNs for Deep RL in Robotic Tasks | 2024 | Scientific Reports (Nature) | Comprehensive comparison in robotics |

**Natural SNN Advantage: MODERATE**
- RL environments are sequential and temporal -- SNN's temporal dynamics help
- Energy efficiency matters for embedded agents (robotics)
- DSQN showed SNNs can outperform ANNs in robustness to adversarial attacks
- Membrane potential as Q-value is a natural representation
- Biological plausibility argument for reward-modulated STDP

**Undergraduate Feasibility: MODERATE**
- CartPole, LunarLander, simple Atari games are well-defined environments
- DSQN, PopSAN have published code
- Challenge: training SNN-based RL agents is still finicky
- Well-scoped project: SNN-DQN on CartPole with energy comparison
- Gymnasium (formerly OpenAI Gym) provides excellent environment API

**Novelty Assessment: LOW-MODERATE**
- Reasonably established subfield
- Novel angles: specific game environments, multi-agent SNN, or spike-based exploration

**Verdict: MODERATE thesis candidate -- good learning experience but limited novelty**

---

### 6. SNN for Environmental Monitoring (Pollution, Wildlife)

**Existing Literature: SPARSE (4-7 papers)**

| Paper | Year | Venue | Key Contribution |
|-------|------|-------|------------------|
| Air Pollution Prediction with Clustering-based Ensemble of Evolving SNNs | 2019 | Environmental Modelling & Software | CEeSNN for air pollution prediction in London |
| Evolving SNN for PM2.5 Prediction (Staging-eSNN) | 2021 | Aerosol and Air Quality Research | Seasonal PM2.5 prediction in Beijing/Shanghai |
| Forest Fire Detection Using SNNs | 2018 | ACM Computing Frontiers | SNN for sensor-based fire detection |
| NeuCube for environmental data | Various | Various | General SNN architecture applied to environmental data |

**No GitHub repositories found specifically for SNN + environmental monitoring**

**Natural SNN Advantage: MODERATE-HIGH**
- Environmental sensor data is continuous temporal streams -- good for evolving SNNs
- Edge deployment in remote sensors requires ultra-low power (strong SNN advantage)
- Event-driven nature suits anomaly/threshold detection (pollution spikes, fire events)
- Wildlife acoustic monitoring involves temporal audio patterns
- Real-time alerting naturally maps to spike-based threshold crossing

**Undergraduate Feasibility: HIGH**
- Public datasets: EPA air quality data, UCI Air Quality, Kaggle environmental datasets
- Forest fire datasets: UCI Forest Fires, FIRMS satellite data
- NeuCube framework exists for evolving SNN approaches
- Well-scoped: predict air quality index from sensor readings using SNN
- Clear practical motivation and societal impact

**Novelty Assessment: HIGH**
- Very few papers (4-7) specifically on SNN + environmental monitoring
- Wildlife acoustic monitoring with SNN: essentially unexplored
- Pollution prediction with modern SNN frameworks (snnTorch): unexplored
- Combining IoT edge deployment narrative with environmental monitoring is novel

**Verdict: EXCELLENT thesis candidate -- high novelty, practical impact, good feasibility**

---

### 7. SNN for Wearable Device Data (Accelerometer, Gyroscope)

**Existing Literature: MODERATE (10-20 papers)**

| Paper | Year | Venue | Key Contribution |
|-------|------|-------|------------------|
| Evaluating SNN on Neuromorphic Platform for HAR | 2023 | ACM ISWC | Systematic evaluation of encoding methods for IMU data |
| SNNs for Ubiquitous Computing (Survey) | 2025 | arxiv (2506.01737) | Comprehensive survey including wearable applications |
| Efficient HAR with Spatio-temporal SNNs | 2023 | Frontiers in Neuroscience | Spatio-temporal SNN for activity recognition |
| SNN for EMG Gesture Recognition on Loihi | 2023 | IEEE Conference | Low-power gesture recognition on neuromorphic chip |
| Spiking-IMU Dataset and SNN for HAR | 2023 | GitHub released | Benchmark dataset and direct-trained SNN |
| Multi-threshold Delta Encoding for IMU | Various | Various | Encoding scheme for wearable sensor data |

**GitHub Repository:**
- `zhaxidele/HAR` - Spiking-IMU dataset and direct-trained SNN for HAR

**Natural SNN Advantage: VERY HIGH**
- Wearable sensors generate continuous temporal data -- perfect for SNNs
- Ultra-low power is critical for battery-powered wearables (core SNN advantage)
- Event-driven processing: only process data when movement occurs (spike-based)
- Neuromorphic chips (Loihi, Xylo) specifically target wearable edge computing
- Latency requirements (real-time gesture recognition) favor SNN's temporal processing
- Multi-threshold delta encoding naturally converts sensor data to spikes

**Undergraduate Feasibility: HIGH**
- Public datasets: UCI HAR, WISDM, PAMAP2, Spiking-IMU
- snnTorch provides straightforward training pipeline
- Well-scoped: classify 6-10 activities from accelerometer data
- Clear energy efficiency narrative for edge deployment
- Can compare with CNN/LSTM baselines easily
- No neuromorphic hardware needed for proof-of-concept

**Novelty Assessment: MODERATE**
- Growing field but still many unexplored angles
- Novel angles: specific activities (fall detection, sports), multi-sensor fusion
- Deployment on specific neuromorphic chips is still underexplored
- Combining with on-device learning (continual learning) is novel

**Verdict: STRONG thesis candidate -- strongest SNN advantage, very practical, well-scoped**

---

### 8. SNN for Radar/Sonar Signal Processing

**Existing Literature: MODERATE (10-15 papers)**

| Paper | Year | Venue | Key Contribution |
|-------|------|-------|------------------|
| Imec SNN Chip for Radar Signal Processing | 2020 | Industry (imec) | First SNN-based chip for radar, 100x power reduction |
| Automotive Radar Processing with SNNs | 2022 | Frontiers in Neuroscience | Concepts and challenges for automotive radar |
| Radar Emitter Recognition Based on SNNs | 2024 | Remote Sensing (MDPI) | Radar emitter classification |
| Spiking Neural Resonators for FMCW Radar | 2025 | arxiv (2503.00898) | Range/angle estimation, 0.02% data transmission |
| Sparse SNNs for Radar Signal Processing | 2024 | TU Munich thesis | Sparse radar processing |
| Radar-Based Hand Gesture Recognition Using SNNs | 2021 | Electronics (MDPI) | Gesture recognition via radar + SNN |
| SNN with delay-lines for sonar echo classification | Various | Various | 93.5% accuracy on sonar pattern recognition |

**Natural SNN Advantage: VERY HIGH**
- Radar/sonar signals are inherently temporal and oscillatory
- Resonate-and-fire neuron models naturally match radar frequency analysis
- Extreme low power requirements for embedded radar systems (automotive, IoT)
- Real-time processing with minimal latency is critical
- Imec's chip demonstrates 100x power reduction -- compelling industrial case
- Spiking neural resonators outperform FFT-based approaches on efficiency

**Undergraduate Feasibility: LOW-MODERATE**
- Radar/sonar datasets are harder to obtain (often restricted/military)
- Signal processing knowledge required (FFT, Doppler analysis, beamforming)
- More specialized domain knowledge needed
- Some synthetic radar data could be generated
- Public sonar dataset: UCI Sonar (Mines vs. Rocks) is simple but classic
- Micro-Doppler gesture recognition is more accessible

**Novelty Assessment: MODERATE**
- Active research area, especially automotive radar
- Sonar classification with SNNs is genuinely sparse
- Novel angles: specific radar applications (weather, drone detection)

**Verdict: MODERATE thesis candidate -- strong SNN advantage but high domain barrier**

---

### 9. SNN for Astronomy Data (Transient Detection)

**Existing Literature: SPARSE-MODERATE (5-8 papers)**

| Paper | Year | Venue | Key Contribution |
|-------|------|-------|------------------|
| SNNs for RFI Detection in Radio Astronomy | 2024 | arxiv (2412.06124) / Nature Comms Physics | First from-scratch trained SNN on real radio astronomy data |
| Neuromorphic Astronomy: End-to-End SNN Pipeline for RFI on Hardware | 2025 | arxiv (2511.16060) | Full pipeline on SynSense Xylo chips at 100mW |
| Potential Impact of Neuromorphic Computing on Radio Telescopes | 2025 | arxiv (2601.07130) | Vision paper for neuromorphic radio astronomy |
| SNN for Anomaly Detection at CERN (particle physics) | 2021 | IRIS-HEP | SNN for particle physics anomaly detection |
| UWA PhD Project: SNNs for Transient Event Detection | Active | University of Western Australia | Ongoing PhD project |

**Natural SNN Advantage: HIGH**
- Radio telescope data is inherently temporal (time-domain astronomy)
- RFI events are transient -- naturally suited to spike-based detection
- Data volumes are enormous (SKA: ~1 TB/s) -- energy efficiency is critical
- Real-time processing requirements match SNN's event-driven nature
- Time-varying SNN dynamics naturally process visibility data
- Demonstrated 100mW power consumption on neuromorphic hardware

**Undergraduate Feasibility: LOW-MODERATE**
- Radio astronomy data requires domain knowledge
- LOFAR data is publicly available but complex
- Simulated RFI data can be generated
- The 2024 RFI detection paper provides a clear methodology to follow/extend
- Narrow scope possible: RFI detection on simulated data
- Astronomical transient detection (FRBs, pulsars) requires deeper expertise

**Novelty Assessment: HIGH**
- Only 5-8 papers in the entire SNN + astronomy space
- Gravitational wave detection with SNNs: essentially zero papers
- Fast radio burst detection with SNNs: zero papers
- Huge opportunity for first-mover contributions

**Verdict: MODERATE thesis candidate -- very novel but requires astronomy domain knowledge**

---

### 10. SNN for Drug Discovery / Molecular Property Prediction

**Existing Literature: VERY SPARSE (2-3 papers)**

| Paper | Year | Venue | Key Contribution |
|-------|------|-------|------------------|
| Screening Bioactivity of P450 Enzyme by SNNs | 2025 | Springer (LNCS) | SNN for enzyme bioactivity prediction using molecular fingerprints |
| Discovery of Covalent Lead Compounds with Lateral Interactions SNN | 2024 | J. Chem. Inf. Model. | SNN for covalent drug discovery |
| Spiking Graph Neural Networks (SpikingGCN) | 2022-2025 | Various arxiv | General framework -- not yet applied to molecules |

**Natural SNN Advantage: LOW**
- Molecular data is primarily structural/graph-based, not temporal
- No natural spike encoding for molecular fingerprints or SMILES strings
- Graph Neural Networks (GNNs) dominate this space for good reason
- Energy efficiency is not a primary concern in drug discovery (offline computation)
- Spiking GNNs exist but are early-stage and not molecule-specific
- Only advantage: potential for novel representations via temporal encoding of molecular dynamics

**Undergraduate Feasibility: LOW**
- Requires chemistry/bioinformatics domain knowledge
- Molecular datasets exist (MoleculeNet, ZINC, QM9) but preprocessing is complex
- Encoding molecules into spikes is a significant research challenge
- Competing with well-established GNN baselines (SchNet, DimeNet) is extremely difficult
- No existing SNN frameworks specifically designed for molecular data

**Novelty Assessment: VERY HIGH**
- Only 2-3 papers in the entire space
- Spiking GNNs for molecules: zero papers
- First-mover advantage is enormous but the gap exists for a reason

**Verdict: POOR thesis candidate -- very novel but low SNN advantage and high barrier**

---

## Comparative Summary Table

| Domain | Papers Exist | SNN Advantage | Undergrad Feasibility | Novelty | Overall Rating |
|--------|-------------|---------------|----------------------|---------|----------------|
| 1. Music Generation | 5-8 | HIGH | HIGH | VERY HIGH | EXCELLENT |
| 2. Industrial IoT Anomaly | 10-20 | HIGH | MODERATE-HIGH | MODERATE | STRONG |
| 3. Financial Fraud | 3-6 | MODERATE | MODERATE | MODERATE-HIGH | MODERATE |
| 4. NLP/Sentiment | 15-25 | LOW-MODERATE | MODERATE | LOW-MODERATE | MODERATE |
| 5. Game Playing/RL | 10-15 | MODERATE | MODERATE | LOW-MODERATE | MODERATE |
| 6. Environmental Monitoring | 4-7 | MODERATE-HIGH | HIGH | HIGH | EXCELLENT |
| 7. Wearable Sensor Data | 10-20 | VERY HIGH | HIGH | MODERATE | STRONG |
| 8. Radar/Sonar | 10-15 | VERY HIGH | LOW-MODERATE | MODERATE | MODERATE |
| 9. Astronomy Transients | 5-8 | HIGH | LOW-MODERATE | HIGH | MODERATE |
| 10. Drug Discovery | 2-3 | LOW | LOW | VERY HIGH | POOR |

---

## Top Recommendations for Undergraduate Thesis

### Tier 1: Strongly Recommended

**1. SNN for Music Generation (Symbolic/MIDI)**
- Why: Fewest papers, natural SNN advantage, excellent frameworks and datasets available
- Suggested scope: Implement SNN-based melody generation on JSB Chorales using snnTorch, compare energy efficiency and musical quality with LSTM baseline
- Key reference: MuSpike benchmark (2025) provides evaluation methodology
- Risk: Low -- well-scoped, accessible tools

**2. SNN for Environmental Monitoring / Air Quality Prediction**
- Why: Very few papers, strong practical motivation, accessible datasets
- Suggested scope: SNN-based air quality index prediction from EPA sensor data, compare with LSTM and evolving SNN (NeuCube) approaches
- Key reference: CEeSNN (2019) and Staging-eSNN (2021) provide methodology
- Risk: Low -- standard regression/classification task with temporal data

### Tier 2: Recommended

**3. SNN for Wearable Sensor Human Activity Recognition**
- Why: Strongest natural SNN advantage, very practical, good scope
- Suggested scope: SNN for HAR on UCI HAR dataset with multi-threshold delta encoding, compare energy/accuracy tradeoff with CNN/LSTM
- Key reference: Spiking-IMU (2023) and ISWC evaluation paper
- Risk: Low -- but slightly more existing literature reduces novelty

**4. SNN for Industrial IoT Anomaly Detection**
- Why: Practical impact, good SNN fit, available datasets
- Suggested scope: SNN-based bearing fault detection on CWRU dataset, compare with autoencoder baseline
- Key reference: MRA-SNN (2024) and Vacuum Spiker (2025)
- Risk: Low-moderate -- requires understanding time series encoding

### Tier 3: Feasible but With Caveats

**5. SNN for Financial Fraud Detection**
- Why: Novel but encoding tabular data into spikes is challenging
- Suggested scope: SNN for streaming transaction classification on Kaggle credit card dataset
- Risk: Moderate -- class imbalance and encoding challenges

**6. SNN for Game Playing / Simple RL**
- Why: Good learning experience but less novel
- Suggested scope: SNN-DQN on CartPole, compare energy with standard DQN
- Risk: Moderate -- training instability in SNN-RL

---

## Key Frameworks and Tools for Implementation

| Framework | Language | Best For | Difficulty |
|-----------|----------|----------|------------|
| snnTorch | Python/PyTorch | General SNN training, tutorials | Beginner-friendly |
| SpikingJelly | Python/PyTorch | Research-grade SNN, many examples | Intermediate |
| Brian2 | Python | Neuroscience simulation, STDP | Beginner-friendly |
| Norse | Python/PyTorch | Bio-inspired SNNs | Intermediate |
| NEST | Python/C++ | Large-scale simulation | Advanced |
| Lava (Intel) | Python | Loihi deployment | Advanced |

**Recommended for undergraduate:** snnTorch (best tutorials, PyTorch-based, active community)

---

## Key Datasets by Domain

| Domain | Dataset | Size | Access |
|--------|---------|------|--------|
| Music | JSB Chorales | 382 chorales | Public |
| Music | POP909 | 909 pop songs | Public |
| Music | Lakh MIDI | 176k MIDI files | Public |
| IoT Anomaly | CWRU Bearing | ~480k samples | Public |
| IoT Anomaly | SMD (Server Machine) | 28 machines | Public |
| Fraud | Kaggle Credit Card | 284k transactions | Public |
| Fraud | BAF (Bank Account Fraud) | 1M+ transactions | Public |
| NLP | SST-2 | 67k sentences | Public |
| NLP | IMDB Reviews | 50k reviews | Public |
| RL | Gymnasium (CartPole) | Simulated | Public |
| Environmental | UCI Air Quality | 9358 instances | Public |
| Environmental | EPA AQS Data | Millions of records | Public |
| Wearable | UCI HAR | 10,299 samples | Public |
| Wearable | WISDM | 1M+ samples | Public |
| Wearable | Spiking-IMU | Custom | GitHub |
| Radar | UCI Sonar | 208 samples | Public |
| Astronomy | LOFAR RFI data | Large | Public |

---

## Research Gaps and Opportunities (Most Novel Combinations)

The following specific combinations have essentially ZERO papers and represent genuine research frontiers:

1. **SNN + wildlife acoustic monitoring** (e.g., bird call classification with spikes)
2. **SNN + water quality prediction** from IoT sensors
3. **Spiking GNN + molecular property prediction** (combining two emerging fields)
4. **SNN + gravitational wave detection** from LIGO data
5. **SNN + fast radio burst classification** from radio telescope data
6. **SNN + speech emotion recognition** (audio sentiment, not text)
7. **SNN + soil moisture prediction** for precision agriculture
8. **SNN + sports activity recognition** from wearable IMUs
9. **SNN + music emotion recognition** (classifying mood of music)
10. **SNN + drone/UAV radar detection** with FMCW radar

---

## Sources

### Music Generation
- [Mode-conditioned music learning and composition (2024)](https://arxiv.org/abs/2411.14773)
- [MuSpike Benchmark (2025)](https://arxiv.org/abs/2508.19251)
- [Stylistic Composition of Melodies (2021)](https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/fnsys.2021.639484/full)
- [Musical Pattern Recognition GitHub](https://github.com/mrahtz/musical-pattern-recognition-in-spiking-neural-networks)
- [SpiNNaker Audio Classification GitHub](https://github.com/jpdominguez/Multilayer-SNN-for-audio-samples-classification-using-SpiNNaker)

### Industrial IoT Anomaly Detection
- [Vacuum Spiker (2025)](https://arxiv.org/html/2510.06910)
- [Evolving SNN for Time Series Anomaly (2022)](https://link.springer.com/article/10.1007/s10994-022-06129-4)
- [End-to-End Bearing Fault Diagnosis with SNN (2024)](https://arxiv.org/abs/2408.11067)
- [Multi-modal Sensor Fusion SNN (2024)](https://www.sciencedirect.com/science/article/abs/pii/S0952197624020049)

### Financial Fraud
- [CSNPC for Financial Fraud Detection (2025)](https://arxiv.org/abs/2508.16915)
- [VMD-SNNs for Stock Prediction (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11873965/)
- [SNN for Financial Time Series (2014)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0103656)

### NLP/Sentiment
- [SpikeGPT (2023)](https://arxiv.org/abs/2302.13939)
- [SNNLP: Energy-Efficient NLP (2024)](https://arxiv.org/abs/2401.17911)
- [Spiking CNN for Text Classification (2024)](https://arxiv.org/abs/2406.19230)
- [SNN-BERT (2024)](https://dl.acm.org/doi/10.1016/j.neunet.2024.106630)
- [SpikingMiniLM (2024)](https://link.springer.com/article/10.1007/s11432-024-4101-6)

### Game Playing / RL
- [Deep Spiking Q-learning (2022)](https://arxiv.org/abs/2201.09754)
- [PopSAN (2021)](https://proceedings.mlr.press/v155/tang21a.html)
- [Exploring SNNs for Deep RL in Robotic Tasks (2024)](https://www.nature.com/articles/s41598-024-77779-8)
- [BrainQN (2024)](https://advanced.onlinelibrary.wiley.com/doi/10.1002/aisy.202400075)

### Environmental Monitoring
- [Air Pollution Prediction with Evolving SNNs (2019)](https://www.sciencedirect.com/science/article/abs/pii/S1364815218307448)
- [PM2.5 Prediction with Staging-eSNN (2021)](https://aaqr.org/articles/aaqr-20-05-oa-0247)
- [Forest Fire Detection with SNN (2018)](https://dl.acm.org/doi/10.1145/3203217.3203231)

### Wearable Sensor Data
- [Evaluating SNN on Neuromorphic Platform for HAR (2023)](https://dl.acm.org/doi/10.1145/3594738.3611369)
- [SNNs for Ubiquitous Computing Survey (2025)](https://arxiv.org/html/2506.01737v1)
- [Spiking-IMU Dataset GitHub](https://github.com/zhaxidele/HAR)

### Radar/Sonar
- [Imec SNN Chip for Radar](https://www.imec-int.com/en/articles/imec-builds-world-s-first-spiking-neural-network-based-chip-for-radar-signal-processing)
- [Spiking Neural Resonators for FMCW Radar (2025)](https://arxiv.org/abs/2503.00898)
- [Automotive Radar with SNNs (2022)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.851774/full)
- [Radar Emitter Recognition with SNNs (2024)](https://www.mdpi.com/2072-4292/16/14/2680)

### Astronomy
- [SNN for RFI Detection in Radio Astronomy (2024)](https://arxiv.org/abs/2412.06124)
- [Neuromorphic Astronomy Pipeline on Hardware (2025)](https://arxiv.org/html/2511.16060)
- [Impact of Neuromorphic Computing on Radio Telescopes (2025)](https://arxiv.org/html/2601.07130)
- [UWA PhD Project: SNNs for Transient Detection](https://researchdegrees.uwa.edu.au/projects/172184/spiking-neural-networks-for-fast-and-efficient-transient-event-detection-in-astronomy)

### Drug Discovery
- [P450 Bioactivity Screening with SNNs (2025)](https://link.springer.com/chapter/10.1007/978-3-031-90714-2_20)
- [Spiking Graph Neural Networks Survey](https://arxiv.org/abs/2205.02767)
- [SGNNBench (2025)](https://arxiv.org/html/2509.21342v1)

### Frameworks
- [snnTorch Documentation](https://snntorch.readthedocs.io/en/latest/)
- [SpikingJelly GitHub](https://github.com/fangwei123456/spikingjelly)
- [Brian2 Documentation](https://brian2.readthedocs.io/)
- [Awesome-Spiking-Neural-Networks](https://github.com/TheBrainLab/Awesome-Spiking-Neural-Networks)
