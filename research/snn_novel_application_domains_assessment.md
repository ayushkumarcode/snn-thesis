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

**papers: about 15-25**

| Paper | Year | Key Contribution |
|-------|------|------------------|
| SpikeGPT | 2023 | 216M param SNN language model, 32.2x fewer operations |
| SNNLP | 2024 | Spike encoding methods, 32x energy reduction |
| Spiking CNN for Text Classification | 2024 | Conversion + fine-tuning for text |
| SNN-BERT | 2024 | Energy-efficient BERT conversion |
| SpikingMiniLM | 2024 | Spiking transformer for NLU |
| Neuromorphic Sentiment on SpiNNaker | 2023 | 100% accuracy on reviews |

**why SNNs might work:** mainly energy efficiency, not representation quality. text isn't naturally temporal/event-driven like audio or sensors. text-to-spike encoding is an active research challenge. SNNs trail ANNs significantly in NLP accuracy.

**feasibility:** MODERATE. standard datasets available (SST-2, IMDB). SpikeGPT and SNNLP provide reference implementations. challenging to get competitive accuracy.

**novelty:** LOW-MODERATE. rapidly becoming established (15+ papers). growing subfield.

**verdict: moderate -- feasible but less novel, need energy efficiency angle**

---

### 5. SNN for Game Playing / RL

**papers: about 10-15**

| Paper | Year | Key Contribution |
|-------|------|------------------|
| Deep Spiking Q-learning (DSQN) | 2022 | Outperforms DQN on 17 Atari games |
| PopSAN | 2021 | 140x energy reduction on Loihi |
| BrainQN | 2024 | Improved robustness in spiking DRL |
| Fully Spiking Actor Network | 2024 | Intra-layer connections for RL |
| SpikeGym Comparison | 2024 | 1-layer SNN-PPO outperforms PopSAN by 4.4x |
| Exploring SNNs for Deep RL in Robotics | 2024 | Nature Scientific Reports comparison |

**why SNNs might work:** RL is sequential/temporal. energy matters for embedded agents. membrane potential as Q-value is natural. biological plausibility angle with reward-modulated STDP. DSQN showed robustness advantages.

**feasibility:** MODERATE. CartPole, LunarLander, simple Atari are well-defined. DSQN and PopSAN have code. training SNN-RL is still finicky though. well-scoped: SNN-DQN on CartPole with energy comparison.

**novelty:** LOW-MODERATE. reasonably established subfield.

**verdict: moderate -- good learning experience but limited novelty**

---

### 6. SNN for Environmental Monitoring (Pollution, Wildlife)

**papers: about 4-7**

| Paper | Year | Key Contribution |
|-------|------|------------------|
| Air Pollution Prediction with Evolving SNNs | 2019 | CEeSNN for London air pollution |
| Evolving SNN for PM2.5 | 2021 | Seasonal PM2.5 prediction Beijing/Shanghai |
| Forest Fire Detection | 2018 | Sensor-based fire detection |
| NeuCube for environmental data | Various | General evolving SNN architecture |

no GitHub repos found for SNN + environmental monitoring specifically.

**why SNNs make sense:** environmental sensors generate continuous temporal streams. edge deployment in remote sensors needs ultra-low power (strong SNN case). event-driven processing suits anomaly detection (pollution spikes, fire events). wildlife acoustic monitoring involves temporal audio patterns.

**feasibility:** HIGH. public datasets: EPA air quality, UCI Air Quality, Kaggle environmental datasets, UCI Forest Fires, FIRMS satellite data. NeuCube framework exists. well-scoped: predict air quality index from sensor readings. clear practical motivation.

**novelty:** HIGH. very few papers. wildlife acoustic monitoring with SNN: essentially unexplored. pollution prediction with modern frameworks (snnTorch): unexplored.

**verdict: excellent thesis candidate**

---

### 7. SNN for Wearable Sensor Data (Accelerometer, Gyroscope)

**papers: about 10-20**

| Paper | Year | Key Contribution |
|-------|------|------------------|
| Evaluating SNN for HAR | 2023 | Systematic evaluation of encoding for IMU data (ACM ISWC) |
| SNNs for Ubiquitous Computing (Survey) | 2025 | Survey including wearable applications |
| Efficient HAR with Spatio-temporal SNNs | 2023 | Activity recognition |
| SNN for EMG Gesture on Loihi | 2023 | Low-power gesture recognition on neuromorphic chip |
| Spiking-IMU Dataset and SNN | 2023 | Benchmark dataset and direct-trained SNN |

**why SNNs make sense:** VERY HIGH advantage here. wearable sensors generate continuous temporal data. ultra-low power is critical for battery-powered devices. event-driven: only process data when movement occurs. neuromorphic chips (Loihi, Xylo) specifically target wearable edge. real-time latency requirements. multi-threshold delta encoding naturally converts sensor data to spikes.

**feasibility:** HIGH. public datasets: UCI HAR, WISDM, PAMAP2, Spiking-IMU. well-scoped: classify 6-10 activities from accelerometer data. clear energy efficiency narrative. easy CNN/LSTM baselines.

**novelty:** MODERATE. growing but many unexplored angles: specific activities (fall detection, sports), multi-sensor fusion, on-device continual learning.

**verdict: strong thesis candidate -- best natural SNN advantage, very practical**

---

### 8. SNN for Radar/Sonar

**papers: about 10-15**

| Paper | Year | Key Contribution |
|-------|------|------------------|
| Imec SNN Chip for Radar | 2020 | First SNN chip for radar, 100x power reduction |
| Automotive Radar with SNNs | 2022 | Concepts for automotive radar |
| Radar Emitter Recognition | 2024 | Radar emitter classification |
| Spiking Neural Resonators for FMCW | 2025 | Range/angle estimation, 0.02% data transmission |
| Radar-Based Gesture Recognition | 2021 | Gesture recognition via radar + SNN |

**why SNNs make sense:** VERY HIGH advantage. radar/sonar signals are inherently temporal and oscillatory. resonate-and-fire neurons naturally match radar frequency analysis. extreme low power for embedded radar (automotive, IoT). real-time processing with minimal latency. imec demonstrated 100x power reduction.

**feasibility:** LOW-MODERATE. radar/sonar datasets are harder to get (often restricted). signal processing knowledge required (FFT, Doppler, beamforming). more specialized domain knowledge. UCI Sonar dataset exists but is basic. micro-Doppler gesture recognition is more accessible.

**novelty:** MODERATE. active for automotive radar. sonar classification genuinely sparse.

**verdict: moderate -- great SNN fit but high domain barrier for undergrads**

---

### 9. SNN for Astronomy (Transient Detection)

**papers: about 5-8**

| Paper | Year | Key Contribution |
|-------|------|------------------|
| SNN for RFI Detection in Radio Astronomy | 2024 | First from-scratch SNN on real radio astronomy data |
| Neuromorphic Astronomy Pipeline | 2025 | Full pipeline on SynSense Xylo chips at 100mW |
| Impact of Neuromorphic on Radio Telescopes | 2025 | Vision paper |
| SNN for Anomaly Detection at CERN | 2021 | Particle physics anomaly detection |
| UWA PhD Project: SNNs for Transients | Active | Ongoing PhD project |

**why SNNs make sense:** radio telescope data is temporal (time-domain astronomy). RFI events are transient -- natural for spike-based detection. data volumes are enormous (SKA: ~1 TB/s), so energy efficiency matters. demonstrated 100mW on neuromorphic hardware.

**feasibility:** LOW-MODERATE. requires domain knowledge. LOFAR data is public but complex. RFI detection paper provides a methodology to follow. narrow scope possible: RFI detection on simulated data.

**novelty:** HIGH. only 5-8 papers total. gravitational wave detection with SNNs: zero papers. fast radio burst detection: zero papers.

**verdict: moderate -- very novel but needs astronomy knowledge**

---

### 10. SNN for Drug Discovery / Molecular Property Prediction

**papers: about 2-3**

| Paper | Year | Key Contribution |
|-------|------|------------------|
| P450 Bioactivity Screening with SNNs | 2025 | SNN for enzyme bioactivity from molecular fingerprints |
