# Spiking Neural Networks for Cybersecurity / Network Intrusion Detection
## Comprehensive Research Report -- Thesis Feasibility Assessment
### Date: 2026-02-25

---

## 1. EXECUTIVE SUMMARY

Spiking Neural Networks (SNNs) applied to Network Intrusion Detection Systems (NIDS) is a rapidly growing research area with substantial activity in 2023-2026. The field has progressed from early proof-of-concept work to sophisticated architectures achieving 98-99%+ accuracy on standard benchmarks (NSL-KDD, CICIDS, UNSW-NB15), often matching or exceeding traditional deep learning while consuming dramatically less energy (70-90% reductions) and offering 20-500x faster inference on neuromorphic hardware.

The natural argument for SNNs in this domain is compelling: network intrusions are sparse, temporal, event-driven phenomena -- precisely the type of data SNNs are biologically optimized to process. Combined with the push toward edge/IoT deployment where power constraints are critical, SNNs offer a genuine architectural advantage rather than being a mere substitution for CNNs or LSTMs.

As a thesis topic, this is well-positioned: the field is active enough to provide a solid literature foundation and reproducible baselines, yet new enough that genuine contributions are achievable at the undergraduate level -- particularly in areas like novel encoding schemes, new dataset evaluations, hybrid architectures, or edge deployment demonstrations. The topic carries a strong narrative ("brain-inspired AI for real-time cybersecurity") that resonates with both academic and industry audiences.

---

## 2. HAS ANYONE APPLIED SNNs TO NETWORK INTRUSION DETECTION?

**Yes -- extensively, and with accelerating interest since 2023.** This is not a hypothetical application; it is an active research front with dozens of published papers. Key milestones:

### 2.1 Landmark and Foundational Papers

| Year | Paper | Venue | Key Contribution |
|------|-------|-------|------------------|
| 2017 | "Network intrusion detection for cyber security on neuromorphic computing system" | IEEE IJCNN | First major demonstration of neuromorphic IDS; deployed on IBM TrueNorth |
| 2020 | "Spiking Neural Networks with Single-Spike Temporal-Coded Neurons for Network Intrusion Detection" (Zhou & Li) | arXiv 2010.07803 | Temporal coding approach on NSL-KDD and AWID; 99.0% accuracy on NSL-KDD |
| 2022 | "Cyber-Neuro RT: Real-time Neuromorphic Cybersecurity" | Procedia Computer Science | Proof-of-concept for real-time HPC-scale IDS on Loihi and BrainChip Akida; 98.4% accuracy (9-class) |
| 2023 | "Binarized SNN with blockchain-based intrusion detection" | Knowledge-Based Systems | Combined binarized SNNs with blockchain for cloud IDS |

### 2.2 Recent Papers (2024-2026) -- The Current Wave

| Year | Paper | Venue | Key Contribution |
|------|-------|-------|------------------|
| 2024 (Mar) | Wang et al. "An efficient intrusion detection model based on convolutional spiking neural network" | Scientific Reports | Lightweight ConvSNN; 98.82% on CSE-IDS2018; 99.86% on DDoS2019; model only 0.034 MB |
| 2024 (Jun) | "SURFS: Sustainable IntrUsion Detection with HieraRchical Federated Spiking Neural Networks" | IEEE ICC 2024 | Federated learning + SNN for distributed IDS |
| 2024 | "A revolutionary approach to use convolutional spiking neural networks for robust intrusion detection" | Cluster Computing (Springer) | 23% accuracy improvement, 28% energy reduction over prior SNN methods |
| 2024 | "An Intrusion Detection System for 5G SDN Networks Utilizing Binarized Deep Spiking Capsule Fire Hawk Neural Networks" | Future Internet (MDPI) | SNNs for 5G/SDN-specific threats |
| 2024 (Nov) | Zivadinovic et al. "Resource efficient IoT intrusion detection with spiking neural networks" | FedCSIS 2024 | F1=0.957 with 240 hidden neurons, 10K samples |
| 2024 | "Analyzing darknet traffic through ML and NeuCube SNNs" | Intelligent and Converged Networks | NeuCube SNN for darknet traffic; 84.31% SNN accuracy |
| 2025 | "Event-Driven Intrusion Detection Systems using Spiking Neural Networks for Edge and IoT Security" | IEEE Conference | STDP-based unsupervised SNN for IoT edge IDS |
| 2025 | Vishwanath et al. "Feature-Optimized Intrusion Detection Based on a Hybrid SNN for IoT" | JAIT | LOA-BHLESNN; 99.96% on ToN-IoT, 99.94% on BoT-IoT |
| 2025 (Aug) | Mia et al. "Neuromorphic Cybersecurity with Semi-supervised Lifelong Learning" | ACM ICONS 2025 | Lifelong learning SNN with Ad-STDP; Intel Lava framework; 85.3% on UNSW-NB15 with continual learning |
| 2025 | "Hybrid recurrent with spiking neural network model (HRSNN) for enhanced anomaly prediction in IoT networks security" | PMC/Nature | RNN+SNN hybrid; 99.60% and 99.16% accuracy |
| 2026 (Feb) | "Energy-efficient intrusion detection with a protocol-aware transformer-spiking hybrid model (TASNN)" | Scientific Reports | Transformer+SNN; Macro-F1=0.93, AUC=0.98 on NSL-KDD; cross-dataset generalization |

### 2.3 Related Application Areas

- **Encrypted traffic classification**: SNN used to classify encrypted internet traffic using only packet size and inter-arrival times, beating state of the art on precision/recall (Rouxelin et al., Neurocomputing 2023)
- **Automotive cybersecurity**: SNN conversion for car hacking/CAN bus intrusion detection (IEEE, 2020)
- **Darknet traffic analysis**: NeuCube SNN applied to CIC-Darknet2020 dataset (2024)
- **Malware detection**: Cyber-SN P systems (spiking neural P systems) for Android malware and phishing detection (Journal of Membrane Computing, 2024)

---

## 3. DATASETS

### 3.1 Primary Benchmark Datasets

| Dataset | Year | Records | Features | Attack Types | Availability |
|---------|------|---------|----------|--------------|--------------|
| **NSL-KDD** | 2009 | 125,973 (train) / 22,544 (test) | 41 | DoS, Probe, R2L, U2R | Free: https://www.unb.ca/cic/datasets/nsl.html and Kaggle |
| **CICIDS-2017** | 2017 | ~2.8M | 79 | Brute Force, DoS, DDoS, Web Attack, Infiltration, Botnet, Heartbleed | Free: UNB CIC website, Kaggle, IEEE DataPort |
| **CSE-CIC-IDS2018** | 2018 | ~16.2M | 79 | Same as 2017 + expanded variants | Free: UNB CIC website |
| **UNSW-NB15** | 2015 | 2,540,044 | 49 | Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms | Free for academic use: UNSW Research website, Kaggle |
| **CIC-DDoS2019** | 2019 | Large-scale | 79 | 12 DDoS attack types | Free: UNB CIC |
| **ToN-IoT** | 2020 | ~461K | 44 | 9 IoT-specific attack types | Free: UNSW Canberra |
| **BoT-IoT** | 2018 | ~73M | 46 | DDoS, DoS, Reconnaissance, Theft | Free: UNSW |
| **AWID** | 2015 | ~1.7M | 155 | WiFi attacks (injection, impersonation, flooding) | Free: Aegean WiFi Intrusion Dataset |

### 3.2 Dataset Characteristics for SNN Work

**NSL-KDD** remains the most commonly used benchmark in SNN-IDS papers due to its manageable size and widespread adoption. However, it is aging (2009 vintage, derived from 1999 KDD Cup data).

**CICIDS-2017/2018** are preferred by recent papers for their modern attack types and flow-based features extracted by CICFlowMeter. Wang et al. (2024) used CSE-CIC-IDS2018 with their ConvSNN.

**UNSW-NB15** is increasingly used as a more challenging and realistic benchmark. It is the primary dataset in the lifelong learning SNN paper (Mia et al., 2025).

**Recommendation for thesis**: Use UNSW-NB15 as primary (modern, challenging, well-documented) and NSL-KDD as secondary for comparison with literature. CICIDS-2017 as optional third.

---

## 4. ACCURACY AND PERFORMANCE: SNNs vs. TRADITIONAL ML/DL

### 4.1 Summary Performance Table (SNN-based models on IDS benchmarks)

| Model | Dataset | Accuracy | F1/AUC | Energy | Source |
|-------|---------|----------|--------|--------|--------|
| ConvSNN (Wang 2024) | CSE-CIC-IDS2018 | 98.82% | -- | 1.775 x10^-4 kWh/10K | Scientific Reports |
| ConvSNN (Wang 2024) | CIC-DDoS2019 | 99.86% | -- | (same model) | Scientific Reports |
