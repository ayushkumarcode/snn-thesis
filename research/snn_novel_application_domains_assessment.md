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
