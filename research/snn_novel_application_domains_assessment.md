# SNN Novel Application Domains: What's Actually Underexplored?

went through a bunch of potential application domains for SNNs to figure out what's genuinely underexplored vs what's already been done to death. searched arxiv, Google Scholar, IEEE, Springer, and GitHub pretty thoroughly.

the tl;dr: **music generation**, **astronomy transient detection**, and **drug discovery** have the fewest papers and are the most genuinely novel. **wearable sensor data**, **radar/sonar**, and **industrial anomaly detection** are moderately explored with clear SNN advantages. **NLP/sentiment**, **game playing/RL**, and **financial fraud** have growing but established literature. **environmental monitoring** is in a middle ground with a handful of pioneering papers.

for an undergrad thesis balancing novelty, feasibility, and natural SNN advantage, i'd say the best candidates are: **(1) music generation**, **(2) environmental monitoring**, **(3) wearable sensor data**, and **(4) industrial IoT anomaly detection**.

---

## Domain-by-Domain Breakdown

---

### 1. SNN for Music Generation / Audio Synthesis

**papers that exist: about 5-8 total**

| Paper | Year | Key Contribution |
|-------|------|------------------|
| Stylistic Composition of Melodies (NeuCube) | 2021 | First SNN melody composition using STDP and sequential memory |
| Musical Pattern Recognition in SNNs | ~2018 | First-layer note differentiation in monophonic sequences |
| SpiNNaker Audio Classification | ~2019 | 3-layer LIF for pure tones on SpiNNaker |
| Mode-conditioned music learning | 2024 | Tonality-aware SNN for musical mode/key |
| MuSpike: Benchmark for Symbolic Music Generation | 2025 | First comprehensive benchmark; 5 SNN architectures across 5 datasets |
| Spiking Vocos | 2025 | Spiking vocoder for audio synthesis |

**why SNNs make sense here:** music is inherently temporal and spike-like (note onsets, rhythmic patterns). MIDI events are discrete, event-driven data. biological auditory processing uses spike-timing codes. STDP mirrors associative musical memory. energy efficiency matters for real-time embedded music.

**feasibility:** HIGH. MIDI datasets are abundant (JSB Chorales, POP909, Lakh MIDI). MuSpike benchmark (2025) provides evaluation methodology. snnTorch/SpikingJelly work fine. single-instrument melody generation is well-scoped. easy LSTM baselines.

**novelty:** VERY HIGH. MuSpike (2025) explicitly notes the field is "significantly underexplored." enormous creative space.

**verdict: excellent thesis candidate**

---

### 2. SNN for Industrial IoT Anomaly Detection

**papers: about 10-20**

| Paper | Year | Key Contribution |
|-------|------|------------------|
| Evolving SNN for Multivariate Time Series | 2022 | Evolving SNN for streaming data |
| Deep SNN Anomaly Detection | 2022 | Vibration analysis for oil infrastructure |
| Vacuum Spiker | 2025 | Efficient anomaly detection model |
| End-to-End Bearing Fault Diagnosis | 2024 | Industrial bearing diagnosis (KDD 2025) |
| Multi-modal Sensor Fusion | 2024 | Multi-sensor bearing fault detection |
| Hybrid Recurrent + SNN for IoT Security | 2025 | IoT intrusion detection |
| Brain-Inspired SNNs for Fault Diagnosis (Survey) | 2024 | Survey of the field |

**why SNNs make sense:** sensor data is temporal and event-driven (anomalies are transients). spike-based threshold detection naturally catches events. low power critical for IoT edge. real-time processing fits event-driven nature. online/evolving learning enables adaptation.

**feasibility:** MODERATE-HIGH. public datasets exist (CWRU Bearing, NASA Bearing, SMD, SMAP). encoding continuous sensor data into spikes takes some thought. well-scoped: single-sensor anomaly detection (e.g., bearing vibration).

**novelty:** MODERATE. active but not saturated. novel angles: specific industrial domains (CNC machining, HVAC), combining online learning with anomaly detection.

**verdict: strong thesis candidate**

---

### 3. SNN for Financial Fraud Detection

**papers: about 3-6**

| Paper | Year | Key Contribution |
|-------|------|------------------|
| Reinforcement-Guided SNN for Fraud | 2025 | 90.8% recall at 5% FPR on BAF dataset |
| SNN for Financial Time Series | 2014 | Foundational work |
| High-Frequency Trading with SNNs | 2021 | SNN for HFT prediction |
| ICS-SNN | 2025 | Optimized SNN for financial forecasting |

**why SNNs might work:** financial transactions are event-driven (discrete events in time). temporal patterns in fraud match SNN dynamics. however, energy efficiency is less critical here (server-side processing), and tabular data (not temporal sequences) reduces the advantage for many fraud tasks.

**feasibility:** MODERATE. public datasets exist (Kaggle Credit Card Fraud, BAF). but class imbalance is extreme (~0.17% fraud), encoding tabular features into spikes is non-trivial, and comparing with XGBoost/Random Forest may be unfavorable.

**novelty:** MODERATE-HIGH. very few SNN papers specifically on fraud detection. most financial SNN work is time series prediction, not classification.

**verdict: moderate -- novel but the encoding challenge is real**

---

### 4. SNN for NLP/Sentiment

