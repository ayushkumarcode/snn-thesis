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
| Single-Spike SNN (Zhou 2020) | NSL-KDD | 99.0% | AUC=1.00 | -- | arXiv |
| Single-Spike SNN (Zhou 2020) | UNSW-NB15 | 96.80% | -- | -- | arXiv |
| Single-Spike SNN (Zhou 2020) | CICIDS-2017 | 99.53% | -- | -- | arXiv |
| TASNN (2026) | NSL-KDD | -- | F1=0.93, AUC=0.98 | Low (spiking) | Scientific Reports |
| LOA-BHLESNN (2025) | ToN-IoT | 99.96% | -- | -- | JAIT |
| LOA-BHLESNN (2025) | BoT-IoT | 99.94% | -- | -- | JAIT |
| HRSNN (2025) | IoT datasets | 99.60% | -- | -- | PMC |
| Lifelong SNN (Mia 2025) | UNSW-NB15 (continual) | 85.3% | -- | Very low (Lava) | ACM ICONS |
| Cyber-Neuro RT (2022) | NSL-KDD | 98.4% (9-class) | -- | Neuromorphic SWaP-C | Procedia CS |
| NeuCube SNN (2024) | CIC-Darknet2020 | 84.31% | -- | -- | ICN |
| FedCSIS SNN (2024) | IoT dataset | -- | F1=0.957 | -- | FedCSIS |

### 4.2 Comparison with Traditional ML/DL Baselines

| Method | NSL-KDD (typical) | UNSW-NB15 (typical) | CICIDS-2017 (typical) |
|--------|-------------------|---------------------|------------------------|
| Random Forest | 95-99% | 93-97% | 97-99% |
| SVM | 92-97% | 88-93% | 95-98% |
| Decision Tree | 93-97% | 90-95% | 96-99% |
| CNN | 97-99% | 95-98% | 98-99% |
| LSTM/RNN | 96-99% | 93-97% | 97-99% |
| **SNN (best reported)** | **98.4-99.0%** | **85.3-96.8%** | **99.53%** |

### 4.3 Key Insights

1. **Accuracy is competitive but not the main selling point.** SNNs match traditional DL on most benchmarks. On some datasets (CICIDS), they exceed. On harder tasks like continual learning (UNSW-NB15), accuracy drops to 85.3% but this is expected -- the comparison should be against other continual learning methods, not static models.

2. **The real advantage is efficiency.** Wang et al. (2024) demonstrated this decisively:
   - 70-90% energy reduction compared to CNNs
   - 0.034 MB model size vs. 68.77 MB for equivalent CNN
   - 7,482 parameters vs. 17.2M for CNN
   - 5,333 samples/second vs. 264 for CNN
   - 204,800 FLOPs vs. 149.5M for CNN

3. **TASNN (2026) showed robust cross-dataset generalization** -- GAR above 0.93 across NSL-KDD, KDDTest+21, and CICIDS-2017, demonstrating that SNN models can generalize, not just memorize.

---

## 5. THE NATURAL ARGUMENT FOR SNNs IN THIS DOMAIN

### 5.1 Why SNNs Are Architecturally Suited to Network Intrusion Detection

The argument for SNNs in cybersecurity is not just theoretical -- it is one of the strongest domain-fit arguments in the SNN literature. Here is why:

**A. Network attacks are inherently sparse and temporal.**
Network traffic is a stream of discrete events (packets) occurring at specific times. Attacks represent rare, anomalous patterns within this stream. This maps directly to SNN's event-driven, spike-based processing -- neurons fire only when meaningful events occur, naturally ignoring the vast majority of benign traffic.

**B. Real-time detection is critical.**
Intrusion detection must operate at line speed. SNNs on neuromorphic hardware achieve inference latencies of 2-3 ms (NeuEdge framework), far below the requirements for real-time packet inspection.

**C. Edge and IoT deployment demands low power.**
Modern networks are increasingly distributed, with security needing to operate at edge gateways, routers, and IoT devices. Power budgets at the edge can be as low as milliwatts. SNNs on neuromorphic chips achieve:
- Up to 15x energy improvement over ARM Cortex-M7 ANN implementations
- 847 GOp/s/W energy efficiency (NeuEdge)
- Intel Loihi: 128 cores, 128M synapses in a single chip

**D. Packet inter-arrival times carry temporal information.**
The timing between packets, burst patterns, and flow durations contain critical discriminative information. SNNs can encode these temporal features natively via spike timing, whereas CNNs and MLPs must artificially flatten or window this data.

**E. Unsupervised/adaptive learning for novel attack detection.**
STDP and other biologically-plausible learning rules allow SNNs to learn patterns without labels -- crucial for detecting zero-day attacks. The lifelong learning work (Mia et al., 2025) demonstrated this: the network can incrementally learn new attack types without forgetting old ones.

**F. Adversarial robustness.**
TASNN (2026) demonstrated resilience to noise, class imbalance, and adversarial perturbations -- important because attackers actively try to evade detection.

### 5.2 Argument Structure for Thesis

The thesis narrative writes itself:

> "Traditional IDS approaches face a trilemma: they must be (1) accurate, (2) real-time, and (3) deployable at scale on resource-constrained edge devices. Deep learning achieves accuracy but demands significant computational resources. Rule-based systems are fast but brittle. SNNs, with their event-driven computation and temporal coding, offer a biologically-inspired resolution to this trilemma -- matching DL accuracy while consuming orders of magnitude less energy and enabling real-time edge deployment."

---

## 6. OPEN-SOURCE IMPLEMENTATIONS AND TOOLS

### 6.1 Direct SNN-IDS Implementations

| Repository | Description | Datasets | Framework |
|-----------|-------------|----------|-----------|
| [zbs881314/Intrusion-detection](https://github.com/zbs881314/Intrusion-detection) | SNN with single-spike temporal coding for IDS | NSL-KDD, AWID | Custom Python |
| [zbs881314/Temporal-Coded-Deep-SNN](https://github.com/zbs881314/Temporal-Coded-Deep-SNN) | Companion temporal coding implementation | NSL-KDD | Custom Python |

### 6.2 SNN Frameworks (for building your own IDS)

| Framework | Repository | PyTorch | Key Strengths | Stars |
|-----------|-----------|---------|---------------|-------|
| **snnTorch** | [jeshraghian/snntorch](https://github.com/jeshraghian/snntorch) | Yes | Excellent tutorials, surrogate gradient training, Colab notebooks | 1.5K+ |
| **SpikingJelly** | [fangwei123456/spikingjelly](https://github.com/fangwei123456/spikingjelly) | Yes | Full-stack toolkit, CuPy acceleration, published in Science Advances | 2K+ |
| **Norse** | [norse/norse](https://github.com/norse/norse) | Yes | Bio-plausible models, PyTorch native | 700+ |
| **BindsNET** | [BindsNET/bindsnet](https://github.com/BindsNET/bindsnet) | Yes | STDP learning, used in neuromorphic cybersecurity paper (Mia 2025) | 1.3K+ |
| **Brian2** | [brian-team/brian2](https://github.com/brian-team/brian2) | No (standalone) | Equation-based modeling, gold standard for neuroscience | 900+ |
| **Intel Lava** | [lava-nc/lava](https://github.com/lava-nc/lava) | No (standalone) | Official Intel framework for Loihi deployment | 500+ |

### 6.3 IDS/Dataset Tools

| Tool | Purpose | Link |
|------|---------|------|
| CICFlowMeter | Extract flow-based features from pcap files | UNB CIC GitHub |
| Awesome-SNN | Curated paper list with codes | [TheBrainLab/Awesome-Spiking-Neural-Networks](https://github.com/TheBrainLab/Awesome-Spiking-Neural-Networks) |
| SNN-Daily-Arxiv | Daily tracking of new SNN papers | [SpikingChen/SNN-Daily-Arxiv](https://github.com/SpikingChen/SNN-Daily-Arxiv) |

### 6.4 Recommended Stack for Thesis

**Primary framework**: snnTorch (best tutorials, easiest learning curve, PyTorch-based)
**Alternative**: SpikingJelly (more features, CuPy acceleration for large datasets)
**For STDP/unsupervised**: BindsNET
**Dataset loading**: scikit-learn + pandas for CSV datasets; CICFlowMeter for raw pcap

---

## 7. NOVELTY ASSESSMENT: HOW NOVEL WOULD THIS BE AS AN UNDERGRADUATE THESIS?

### 7.1 Honest Assessment

**The intersection of SNNs and IDS is NOT novel in itself** -- there are now 20+ papers in this space. However, this does NOT diminish its viability as a thesis topic. The field is young enough that meaningful gaps exist.

### 7.2 What HAS Been Done

- Pure SNN classification on NSL-KDD (well-explored)
- ConvSNN for IDS (Wang et al., 2024)
- Transformer+SNN hybrid for IDS (TASNN, 2026)
- Federated SNN-IDS (SURFS, 2024)
- Lifelong/continual learning SNN-IDS (Mia et al., 2025)
- Encrypted traffic classification with SNN (Rouxelin et al., 2023)
- STDP-based unsupervised IDS (Event-driven SNN, 2025)

### 7.3 What HAS NOT Been Done (Novel Thesis Angles)

The following represent genuine research gaps an undergraduate could address:

**Gap 1: Systematic encoding scheme comparison for IDS.**
No paper has systematically compared rate coding, temporal coding, latency coding, delta coding, and phase coding on the same IDS datasets with the same SNN architecture. This is a clean, achievable, and publishable contribution.

**Gap 2: SNN-IDS on the latest CICIDS datasets (CIC-IDS2024 if available).**
Most SNN papers use NSL-KDD (outdated) or CSE-CIC-IDS2018. Applying SNNs to newer datasets with modern attack patterns would be a contribution.

**Gap 3: Interpretability/explainability of SNN-IDS decisions.**
No work has explored what the spiking patterns mean -- i.e., can we visualize or explain which spike patterns correspond to which attack types? This connects to the broader XAI (Explainable AI) movement.

**Gap 4: Practical edge deployment demonstration.**
While papers claim edge suitability, very few have actually deployed an SNN-IDS on actual edge hardware (e.g., Raspberry Pi, Jetson Nano, or a real neuromorphic chip). A working demonstration would be highly valued.

**Gap 5: SNN-IDS for specific emerging threats.**
Applying SNNs to specific attack categories that are underexplored: encrypted traffic attacks, DNS tunneling, or supply chain attacks.

**Gap 6: Hybrid SNN + traditional ML pipeline.**
Using SNN as a fast first-stage filter (binary: normal/anomalous) followed by a traditional classifier for attack-type classification. This mirrors the two-stage approach in Mia et al. (2025) but with a simpler, more practical design.

**Gap 7: Transfer learning across IDS datasets with SNNs.**
Testing whether an SNN trained on one dataset generalizes to another without retraining -- important for real-world deployment.

### 7.4 Recommended Thesis Angle

For an achievable yet novel undergraduate thesis, I recommend:

**Title concept**: "Evaluating Spike Encoding Strategies for Energy-Efficient Network Intrusion Detection Using Spiking Neural Networks"

**Scope**:
1. Implement a baseline SNN-IDS using snnTorch on UNSW-NB15 and NSL-KDD
2. Compare 3-4 different spike encoding methods (rate, temporal, latency, delta)
3. Benchmark against standard ML baselines (RF, SVM, CNN, LSTM)
4. Measure not just accuracy but also energy proxy metrics (spike count, synaptic operations)
5. Analyze which encoding best captures the temporal features of network traffic

**Why this works**:
- Achievable in one semester with existing frameworks
- Provides a systematic contribution the field lacks
- Natural comparison structure makes for a clean thesis
- Multiple datasets strengthen claims
- Energy analysis adds practical relevance
- Results are publishable regardless of which encoding "wins"

---

## 8. COMPREHENSIVE PAPER CATALOG (2023-2026)

### 2023

1. **"Binarized Spiking Neural Network with blockchain based intrusion detection framework"** -- Rajagopalan & Rethinam, Knowledge-Based Systems, 2023. Combines binarized SNNs with blockchain consensus for cloud IDS.

2. **"Encrypted internet traffic classification using a supervised spiking neural network"** -- Rouxelin et al., Neurocomputing, 2023 (published). SNN using packet size/timing for encrypted traffic classification. Beats SOTA on precision/recall.

3. **"A Homomorphic Encryption Framework for Privacy-Preserving Spiking Neural Networks"** -- Nikfam et al., Information, 2023. Compares SNNs and DNNs under fully homomorphic encryption.

### 2024

4. **"An efficient intrusion detection model based on convolutional spiking neural network"** -- Wang et al., Scientific Reports 14:7054, March 2024. *Key paper*. ConvSNN achieving 98.82% accuracy with 0.034 MB model. 70-90% energy reduction.

5. **"SURFS: Sustainable IntrUsion Detection with HieraRchical Federated Spiking Neural Networks"** -- Aouedi & Piamrat, IEEE ICC 2024, June 2024. Three-tier federated SNN architecture.

6. **"A revolutionary approach to use convolutional spiking neural networks for robust intrusion detection"** -- Cluster Computing (Springer), 2024. 23% accuracy improvement over prior SNN methods.

7. **"An Intrusion Detection System for 5G SDN Network Utilizing Binarized Deep Spiking Capsule Fire Hawk Neural Networks"** -- Future Internet (MDPI), October 2024.

8. **"Analyzing darknet traffic through ML and NeuCube spiking neural networks"** -- Intelligent and Converged Networks, 2024. NeuCube SNN for darknet; 84.31% accuracy.

9. **"Resource efficient Internet-of-Things intrusion detection with spiking neural networks"** -- Zivadinovic et al., FedCSIS 2024, November 2024. F1=0.957.

10. **"Applications of spiking neural P systems in cybersecurity"** -- Journal of Membrane Computing, 2024. Cyber-SN P systems for malware/phishing.

### 2025

11. **"Neuromorphic Cybersecurity with Semi-supervised Lifelong Learning"** -- Mia et al., ACM ICONS 2025, August 2025. *Significant paper*. Two-stage SNN with Ad-STDP, GWR plasticity. Intel Lava. UNSW-NB15. 85.3% with continual learning.

12. **"Event-Driven Intrusion Detection Systems using Spiking Neural Networks for Edge and IoT Security"** -- IEEE Conference, 2025. STDP-based unsupervised SNN-IDC framework.

13. **"Feature-Optimized Intrusion Detection Based on a Hybrid Spiking Neural Network for the Internet of Things"** -- Vishwanath et al., JAIT 6:52-63, 2025. LOA-BHLESNN. 99.96% on ToN-IoT.

14. **"Hybrid recurrent with spiking neural network model (HRSNN) for enhanced anomaly prediction in IoT networks security"** -- PMC/Scientific Reports, 2025. RNN+SNN hybrid; 99.60% accuracy.

15. **"Efficacy of Spiking Neural Networks for Intrusion Detection Systems"** -- IEEE Conference, 2025.

16. **"Towards the neuromorphic Cyber-Twin: architecture for cognitive defense in digital twin ecosystems"** -- Frontiers in Big Data, 2025.

### 2026

17. **"Energy-efficient intrusion detection with a protocol-aware transformer-spiking hybrid model (TASNN)"** -- Scientific Reports, February 2026. Transformer+SNN. Macro-F1=0.93 on NSL-KDD. Cross-dataset generalization.

---

## 9. NETWORK / RELATIONSHIP MAP

```
                    NEUROMORPHIC CYBERSECURITY ECOSYSTEM
                    =====================================

    HARDWARE                    FRAMEWORKS                  APPLICATIONS
    --------                    ----------                  ------------
    Intel Loihi 2  --------->  Intel Lava  ------------>  Lifelong NIDS
    IBM TrueNorth                                          (Mia 2025)
    BrainChip Akida -------->  Cyber-Neuro RT --------->  HPC-scale IDS
    SpiNNaker                                              (Penn State)
                               snnTorch  -----+
                               SpikingJelly ---+--------->  ConvSNN-IDS
                               BindsNET  ------+            (Wang 2024)
                               Norse  ---------+
                                               |
    DATASETS                                   v
    --------                           SNN-IDS RESEARCH
    NSL-KDD  ----+                     ================
    CICIDS-2017 -+---> Benchmarks ---> Encoding: rate, temporal, latency
    UNSW-NB15 ---+                     Learning: STDP, surrogate gradient,
    ToN-IoT -----+                               backprop-through-time
    BoT-IoT -----+                     Architecture: ConvSNN, HRSNN, TASNN
    AWID --------+                     Deployment: edge, IoT, federated

    KEY RESEARCH GROUPS
    -------------------
    - Penn State (Sengupta group) -- Neuromorphic cybersecurity, lifelong learning
    - UTM Malaysia (Wang, Ghaleb, Zainal) -- ConvSNN efficiency
    - Various IEEE/Springer authors -- Hybrid architectures
    - Quantum Ventura -- Cyber-Neuro RT commercial application
```

---

## 10. RESEARCH GAPS AND LIMITATIONS

### What I could NOT find:

1. **No published SNN-IDS work on Intel Loihi 2 with full deployment metrics.** The Mia 2025 paper used Lava framework simulation but not actual chip deployment for IDS specifically.

2. **No comprehensive benchmark comparing all SNN-IDS papers on identical experimental settings.** Each paper uses different subsets, preprocessing, and train/test splits, making direct comparison unreliable.

3. **No undergraduate thesis specifically on SNN-IDS found.** This supports novelty at the thesis level -- the existing work is from research groups, not student projects.

4. **Limited explainability work.** No paper explains what the spiking patterns mean in terms of network security semantics.

5. **No published work on SNN-IDS for zero-day attacks using truly unsupervised detection** beyond the STDP proof-of-concept.

---

## 11. CONFIDENCE ASSESSMENT

| Finding | Confidence | Basis |
|---------|------------|-------|
| SNNs have been applied to IDS | Very High | 15+ peer-reviewed papers found |
| SNNs achieve 98-99% on standard benchmarks | High | Multiple independent papers confirm |
| SNNs offer 70-90% energy savings over CNNs | High | Wang et al. 2024 provides detailed measurements |
| Edge deployment is feasible | Medium-High | Demonstrated in simulation; limited real hardware results |
| The topic is suitable for undergraduate thesis | High | Active but young field with clear gaps |
| A novel contribution is achievable | High | Multiple unexplored angles identified |
| The existing open-source code is usable | Medium | zbs881314 repo exists but limited documentation; frameworks are well-documented |

---

## 12. RECOMMENDED FOLLOW-UPS

1. **Read in full**: Wang et al. 2024 (Scientific Reports) -- the most detailed SNN-IDS paper with reproducibility information
2. **Read in full**: Mia et al. 2025 (ACM ICONS) -- most sophisticated SNN-IDS architecture
3. **Try the code**: Clone zbs881314/Intrusion-detection and run on NSL-KDD
4. **Work through snnTorch tutorials**: https://snntorch.readthedocs.io/ -- 10 tutorial notebooks from basics to advanced
5. **Download UNSW-NB15**: https://research.unsw.edu.au/projects/unsw-nb15-dataset
6. **Check with supervisor**: Discuss whether encoding comparison, edge deployment, or hybrid approach best fits the program's expectations

---

## 13. SOURCES AND REFERENCES

### Key Papers (Direct Links)
- [Wang et al. 2024 - ConvSNN IDS (Scientific Reports)](https://www.nature.com/articles/s41598-024-57691-x)
- [Wang et al. 2024 - PMC Full Text](https://pmc.ncbi.nlm.nih.gov/articles/PMC10963367/)
- [TASNN 2026 - Transformer-Spiking Hybrid (Scientific Reports)](https://www.nature.com/articles/s41598-026-37367-4)
- [Mia et al. 2025 - Neuromorphic Cybersecurity Lifelong Learning (arXiv)](https://arxiv.org/abs/2508.04610)
- [SURFS - Federated SNN IDS (IEEE)](https://ieeexplore.ieee.org/document/10622560/)
- [Event-Driven SNN-IDS (IEEE)](https://ieeexplore.ieee.org/document/11171294/)
- [Hybrid HRSNN for IoT (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12546366/)
- [ConvSNN Robust IDS (Springer)](https://link.springer.com/article/10.1007/s10586-024-04603-3)
- [Feature-Optimized SNN IDS (JAIT)](https://ojs.istp-press.com/jait/article/view/848)
- [Resource Efficient IoT SNN-IDS (ResearchGate)](https://www.researchgate.net/publication/386152689_Resource_efficient_Internet-of-Things_intrusion_detection_with_spiking_neural_networks)
- [SNN Darknet Traffic (SciOpen)](https://www.sciopen.com/article/10.23919/ICN.2024.0022)
- [Cyber-Neuro RT (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S1877050922017938)
- [Low-Power Cybersecurity on Neuromorphic Tech (CSIAC)](https://csiac.dtic.mil/articles/low-power-cybersecurity-attack-detection-using-deep-learning-on-neuromorphic-technologies/)
- [SNN Encrypted Traffic (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0925231222007895)
- [Binarized SNN + Blockchain IDS (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S156849462301236X)
- [5G SDN SNN IDS (MDPI)](https://www.mdpi.com/1999-5903/16/10/359)
- [Cyber-SN P Systems for Cybersecurity (Springer)](https://link.springer.com/article/10.1007/s41965-024-00166-9)

### GitHub Repositories
- [zbs881314/Intrusion-detection](https://github.com/zbs881314/Intrusion-detection)
- [snnTorch](https://github.com/jeshraghian/snntorch) / [Docs](https://snntorch.readthedocs.io/)
- [SpikingJelly](https://github.com/fangwei123456/spikingjelly)
- [Norse](https://github.com/norse/norse)
- [BindsNET](https://github.com/BindsNET/bindsnet)
- [Awesome-SNN Paper List](https://github.com/TheBrainLab/Awesome-Spiking-Neural-Networks)
- [SNN-Daily-Arxiv](https://github.com/SpikingChen/SNN-Daily-Arxiv)

### Datasets
- [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) / [Kaggle](https://www.kaggle.com/datasets/hassan06/nslkdd)
- [CICIDS-2017](https://www.unb.ca/cic/datasets/ids-2017.html) / [Kaggle](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset)
- [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) / [Kaggle](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15)
- [UNSW-NB15 (Papers With Code)](https://paperswithcode.com/dataset/unsw-nb15)

---

*Report generated through exhaustive multi-vector web research across academic databases, GitHub, arXiv, IEEE Xplore, Springer, Nature, PMC, ResearchGate, and specialized neuromorphic computing resources.*
