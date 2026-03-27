# spiking neural networks for cybersecurity / network intrusion detection

i wanted to see if SNNs have been applied to network intrusion detection systems (NIDS), and it turns out this is a surprisingly active area. there's been a lot of work since 2023 -- people are getting 98-99%+ accuracy on standard benchmarks (NSL-KDD, CICIDS, UNSW-NB15), often matching traditional deep learning while using way less energy (70-90% reductions) and getting 20-500x faster inference on neuromorphic hardware.

the natural argument here is actually pretty strong: network intrusions are sparse, temporal, event-driven -- exactly the kind of data SNNs should be good at. and with the push toward edge/IoT deployment where power matters, SNNs offer a genuine architectural advantage over just using CNNs or LSTMs.

as a thesis topic it's well-positioned -- active enough for solid baselines and literature, but new enough that an undergrad can actually contribute something real, especially around encoding schemes, new datasets, hybrid architectures, or edge deployment demos. plus "brain-inspired AI for real-time cybersecurity" is a strong pitch for both academic and industry audiences.

---

## has anyone actually done this?

**yes -- a lot of people, and it's been accelerating since 2023.** not hypothetical at all. here's the timeline:

### foundational papers

| Year | Paper | Venue | What it did |
|------|-------|-------|------------|
| 2017 | "Network intrusion detection for cyber security on neuromorphic computing system" | IEEE IJCNN | First major neuromorphic IDS demo, deployed on IBM TrueNorth |
| 2020 | "Spiking Neural Networks with Single-Spike Temporal-Coded Neurons for Network Intrusion Detection" (Zhou & Li) | arXiv 2010.07803 | Temporal coding approach; 99.0% on NSL-KDD |
| 2022 | "Cyber-Neuro RT: Real-time Neuromorphic Cybersecurity" | Procedia Computer Science | Real-time HPC-scale IDS on Loihi and BrainChip Akida; 98.4% (9-class) |
| 2023 | "Binarized SNN with blockchain-based intrusion detection" | Knowledge-Based Systems | Combined binarized SNNs with blockchain for cloud IDS |

### recent papers (2024-2026) -- where things really picked up

| Year | Paper | Venue | What it did |
|------|-------|-------|------------|
| 2024 (Mar) | Wang et al. "An efficient intrusion detection model based on convolutional spiking neural network" | Scientific Reports | Lightweight ConvSNN; 98.82% on CSE-IDS2018; 99.86% on DDoS2019; model only 0.034 MB |
| 2024 (Jun) | "SURFS: Sustainable IntrUsion Detection with HieraRchical Federated Spiking Neural Networks" | IEEE ICC 2024 | Federated learning + SNN for distributed IDS |
| 2024 | "A revolutionary approach to use convolutional spiking neural networks for robust intrusion detection" | Cluster Computing (Springer) | 23% accuracy improvement, 28% energy reduction over prior SNN methods |
| 2024 | "An Intrusion Detection System for 5G SDN Networks Utilizing Binarized Deep Spiking Capsule Fire Hawk Neural Networks" | Future Internet (MDPI) | SNNs for 5G/SDN-specific threats |
| 2024 (Nov) | Zivadinovic et al. "Resource efficient IoT intrusion detection with spiking neural networks" | FedCSIS 2024 | F1=0.957 with just 240 hidden neurons, 10K samples |
| 2024 | "Analyzing darknet traffic through ML and NeuCube SNNs" | Intelligent and Converged Networks | NeuCube SNN for darknet; 84.31% accuracy |
| 2025 | "Event-Driven Intrusion Detection Systems using Spiking Neural Networks for Edge and IoT Security" | IEEE Conference | STDP-based unsupervised SNN for IoT edge IDS |
| 2025 | Vishwanath et al. "Feature-Optimized Intrusion Detection Based on a Hybrid SNN for IoT" | JAIT | LOA-BHLESNN; 99.96% on ToN-IoT, 99.94% on BoT-IoT |
| 2025 (Aug) | Mia et al. "Neuromorphic Cybersecurity with Semi-supervised Lifelong Learning" | ACM ICONS 2025 | Lifelong learning SNN with Ad-STDP; Intel Lava framework; 85.3% on UNSW-NB15 with continual learning |
| 2025 | "Hybrid recurrent with spiking neural network model (HRSNN) for enhanced anomaly prediction in IoT networks security" | PMC/Nature | RNN+SNN hybrid; 99.60% and 99.16% accuracy |
| 2026 (Feb) | "Energy-efficient intrusion detection with a protocol-aware transformer-spiking hybrid model (TASNN)" | Scientific Reports | Transformer+SNN; Macro-F1=0.93, AUC=0.98 on NSL-KDD; cross-dataset generalization |

### related areas

- **encrypted traffic classification**: SNN classifying encrypted traffic using just packet size and inter-arrival times, beating SOTA on precision/recall (Rouxelin et al., Neurocomputing 2023)
- **automotive cybersecurity**: SNN conversion for car hacking/CAN bus intrusion detection (IEEE, 2020)
- **darknet traffic**: NeuCube SNN on CIC-Darknet2020 (2024)
- **malware detection**: Cyber-SN P systems for Android malware and phishing detection (Journal of Membrane Computing, 2024)

---

## datasets

### the main benchmarks

| Dataset | Year | Records | Features | Attack Types | Availability |
|---------|------|---------|----------|-------------|-------------|
| **NSL-KDD** | 2009 | 125,973 train / 22,544 test | 41 | DoS, Probe, R2L, U2R | Free: https://www.unb.ca/cic/datasets/nsl.html and Kaggle |
| **CICIDS-2017** | 2017 | ~2.8M | 79 | Brute Force, DoS, DDoS, Web Attack, Infiltration, Botnet, Heartbleed | Free: UNB CIC, Kaggle, IEEE DataPort |
| **CSE-CIC-IDS2018** | 2018 | ~16.2M | 79 | Same as 2017 + expanded | Free: UNB CIC |
| **UNSW-NB15** | 2015 | 2,540,044 | 49 | Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Recon, Shellcode, Worms | Free for academic: UNSW Research, Kaggle |
| **CIC-DDoS2019** | 2019 | Large-scale | 79 | 12 DDoS attack types | Free: UNB CIC |
| **ToN-IoT** | 2020 | ~461K | 44 | 9 IoT-specific attack types | Free: UNSW Canberra |
| **BoT-IoT** | 2018 | ~73M | 46 | DDoS, DoS, Recon, Theft | Free: UNSW |
| **AWID** | 2015 | ~1.7M | 155 | WiFi attacks (injection, impersonation, flooding) | Free: Aegean WiFi Intrusion Dataset |

### notes on which datasets to use

NSL-KDD is the most used benchmark in SNN-IDS papers because it's manageable and widely adopted. but it's getting old (derived from 1999 KDD Cup data).

CICIDS-2017/2018 are preferred by newer papers for modern attack types and flow-based features from CICFlowMeter. Wang et al. (2024) used CSE-CIC-IDS2018 for their ConvSNN.

UNSW-NB15 is increasingly used as a harder, more realistic benchmark. it's the primary dataset in Mia et al. (2025).

for a thesis, i'd use UNSW-NB15 as primary (modern, challenging, well-documented), NSL-KDD as secondary for literature comparison, and maybe CICIDS-2017 as an optional third.

---

## accuracy: how do SNNs compare?

### SNN results on IDS benchmarks

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
| Cyber-Neuro RT (2022) | NSL-KDD | 98.4% (9-class) | -- | Neuromorphic | Procedia CS |
| NeuCube SNN (2024) | CIC-Darknet2020 | 84.31% | -- | -- | ICN |
| FedCSIS SNN (2024) | IoT dataset | -- | F1=0.957 | -- | FedCSIS |

### vs traditional ML/DL baselines

| Method | NSL-KDD (typical) | UNSW-NB15 (typical) | CICIDS-2017 (typical) |
|--------|-------------------|---------------------|----------------------|
| Random Forest | 95-99% | 93-97% | 97-99% |
| SVM | 92-97% | 88-93% | 95-98% |
| Decision Tree | 93-97% | 90-95% | 96-99% |
| CNN | 97-99% | 95-98% | 98-99% |
| LSTM/RNN | 96-99% | 93-97% | 97-99% |
| **SNN (best reported)** | **98.4-99.0%** | **85.3-96.8%** | **99.53%** |

### what to take from this

1. **accuracy is competitive but not the main selling point.** SNNs match traditional DL on most benchmarks. on CICIDS they actually beat it. the continual learning result (85.3% on UNSW-NB15) looks lower but you should compare it against other continual learning methods, not static models.

2. **the real advantage is efficiency.** Wang et al. (2024) showed this convincingly:
   - 70-90% energy reduction vs CNNs
   - 0.034 MB model vs 68.77 MB for equivalent CNN
   - 7,482 parameters vs 17.2M for CNN
   - 5,333 samples/second vs 264 for CNN
   - 204,800 FLOPs vs 149.5M for CNN

3. **cross-dataset generalization works.** TASNN (2026) got GAR above 0.93 across NSL-KDD, KDDTest+21, and CICIDS-2017. not just memorizing one dataset.

---

## why SNNs actually make sense here

honestly the domain fit argument for SNNs in cybersecurity is one of the strongest in the whole SNN literature. here's why:

**network attacks are inherently sparse and temporal.** network traffic is a stream of discrete events (packets) at specific times. attacks are rare anomalous patterns in this stream. this maps directly to SNN event-driven processing -- neurons fire only when meaningful events happen, naturally ignoring the majority of benign traffic.

**real-time detection is critical.** IDS must operate at line speed. SNNs on neuromorphic hardware get inference latencies of 2-3 ms (NeuEdge framework).

**edge/IoT deployment needs low power.** security needs to run at edge gateways, routers, IoT devices. power budgets at the edge can be milliwatts. SNNs on neuromorphic chips get up to 15x energy improvement over ARM Cortex-M7 ANN implementations, 847 GOp/s/W efficiency (NeuEdge), and Intel Loihi packs 128 cores with 128M synapses in a single chip.

**packet inter-arrival times carry temporal info.** timing between packets, burst patterns, flow durations -- these have discriminative value. SNNs encode temporal features natively via spike timing. CNNs and MLPs have to artificially flatten or window this data.

**unsupervised/adaptive learning for novel attacks.** STDP lets SNNs learn without labels -- useful for detecting zero-day attacks. Mia et al. (2025) showed the network can incrementally learn new attack types without forgetting old ones.

**adversarial robustness.** TASNN (2026) demonstrated resilience to noise, class imbalance, and adversarial perturbations -- matters because attackers actively try to evade detection.

### the thesis narrative basically writes itself

traditional IDS faces a trilemma: be accurate, be real-time, and deploy on resource-constrained edge devices. DL gets accuracy but needs big compute. rule-based systems are fast but brittle. SNNs offer a resolution -- matching DL accuracy while consuming orders of magnitude less energy and enabling real-time edge deployment.

---

## open-source code and tools

### SNN-IDS specific repos

| Repository | What it does | Datasets | Framework |
|-----------|-------------|----------|-----------|
| [zbs881314/Intrusion-detection](https://github.com/zbs881314/Intrusion-detection) | SNN with single-spike temporal coding for IDS | NSL-KDD, AWID | Custom Python |
| [zbs881314/Temporal-Coded-Deep-SNN](https://github.com/zbs881314/Temporal-Coded-Deep-SNN) | Companion temporal coding implementation | NSL-KDD | Custom Python |

### SNN frameworks for building your own

| Framework | Repo | PyTorch? | Strengths | Stars |
|-----------|------|----------|-----------|-------|
| **snnTorch** | [jeshraghian/snntorch](https://github.com/jeshraghian/snntorch) | Yes | Great tutorials, surrogate gradients, Colab notebooks | 1.5K+ |
| **SpikingJelly** | [fangwei123456/spikingjelly](https://github.com/fangwei123456/spikingjelly) | Yes | Full-stack, CuPy acceleration, Science Advances publication | 2K+ |
| **Norse** | [norse/norse](https://github.com/norse/norse) | Yes | Bio-plausible models, PyTorch native | 700+ |
| **BindsNET** | [BindsNET/bindsnet](https://github.com/BindsNET/bindsnet) | Yes | STDP learning, used in Mia 2025 | 1.3K+ |
| **Brian2** | [brian-team/brian2](https://github.com/brian-team/brian2) | No (standalone) | Equation-based modeling, gold standard for neuroscience | 900+ |
| **Intel Lava** | [lava-nc/lava](https://github.com/lava-nc/lava) | No (standalone) | Official Intel framework for Loihi | 500+ |

### IDS/dataset tools

| Tool | Purpose | Link |
|------|---------|------|
| CICFlowMeter | Extract flow-based features from pcap files | UNB CIC GitHub |
| Awesome-SNN | Curated paper list with codes | [TheBrainLab/Awesome-Spiking-Neural-Networks](https://github.com/TheBrainLab/Awesome-Spiking-Neural-Networks) |
| SNN-Daily-Arxiv | Daily new SNN papers tracking | [SpikingChen/SNN-Daily-Arxiv](https://github.com/SpikingChen/SNN-Daily-Arxiv) |

### what i'd use for a thesis

- **primary framework**: snnTorch (best tutorials, easiest learning curve, PyTorch)
- **alternative**: SpikingJelly (more features, CuPy acceleration for large datasets)
- **for STDP/unsupervised**: BindsNET
- **data loading**: scikit-learn + pandas for CSV datasets; CICFlowMeter for raw pcap

---

## how novel would this be as an undergrad thesis?

### honest answer

the intersection of SNNs and IDS is not novel in itself -- there's 20+ papers now. but that doesn't mean it's not a viable thesis topic. the field is young enough that meaningful gaps exist.

### what's been done

- pure SNN classification on NSL-KDD (well-explored)
- ConvSNN for IDS (Wang et al., 2024)
- Transformer+SNN hybrid (TASNN, 2026)
- federated SNN-IDS (SURFS, 2024)
- lifelong/continual learning SNN-IDS (Mia et al., 2025)
- encrypted traffic classification with SNN (Rouxelin et al., 2023)
- STDP-based unsupervised IDS (2025)

### what hasn't been done -- actual gaps

**gap 1: encoding scheme comparison.** nobody has compared rate, temporal, latency, delta, and phase coding on the same IDS datasets with the same SNN architecture. clean, achievable, publishable.

**gap 2: newer datasets.** most SNN papers use NSL-KDD (outdated) or CSE-CIC-IDS2018. applying SNNs to newer datasets with modern attack patterns would be a contribution.

**gap 3: interpretability.** nobody's explored what the spiking patterns actually mean -- can you visualize which spike patterns correspond to which attacks? connects to the broader XAI movement.

**gap 4: actual edge deployment.** papers claim edge suitability but very few have deployed an SNN-IDS on real edge hardware (Raspberry Pi, Jetson Nano, neuromorphic chip). a working demo would be highly valued.

**gap 5: specific emerging threats.** applying SNNs to underexplored attack categories: encrypted traffic attacks, DNS tunneling, supply chain attacks.

**gap 6: hybrid SNN + traditional ML pipeline.** using SNN as a fast first-stage filter (normal/anomalous) followed by traditional classifier for attack-type classification. mirrors Mia et al. (2025) but simpler and more practical.

**gap 7: transfer learning across datasets.** does an SNN trained on one dataset generalize to another without retraining? important for real deployment.

### suggested thesis angle

something like: "evaluating spike encoding strategies for energy-efficient network intrusion detection using spiking neural networks"

scope:
1. implement baseline SNN-IDS using snnTorch on UNSW-NB15 and NSL-KDD
2. compare 3-4 different spike encoding methods (rate, temporal, latency, delta)
3. benchmark against standard ML baselines (RF, SVM, CNN, LSTM)
4. measure not just accuracy but also energy proxies (spike count, synaptic operations)
5. analyze which encoding best captures temporal features of network traffic

why this works:
- achievable in one semester
- the field actually lacks this kind of study
- comparison structure makes for a clean thesis
- multiple datasets strengthen claims
- energy analysis adds practical relevance
- results are publishable regardless of which encoding "wins"

---

## full paper catalog (2023-2026)

### 2023

1. **"Binarized Spiking Neural Network with blockchain based intrusion detection framework"** -- Rajagopalan & Rethinam, Knowledge-Based Systems, 2023. Binarized SNNs + blockchain consensus for cloud IDS.

2. **"Encrypted internet traffic classification using a supervised spiking neural network"** -- Rouxelin et al., Neurocomputing, 2023. SNN using packet size/timing for encrypted traffic. Beats SOTA on precision/recall.

3. **"A Homomorphic Encryption Framework for Privacy-Preserving Spiking Neural Networks"** -- Nikfam et al., Information, 2023. Compares SNNs and DNNs under fully homomorphic encryption.

### 2024

4. **"An efficient intrusion detection model based on convolutional spiking neural network"** -- Wang et al., Scientific Reports 14:7054, March 2024. *Key paper*. ConvSNN, 98.82%, 0.034 MB model, 70-90% energy reduction.

5. **"SURFS: Sustainable IntrUsion Detection with HieraRchical Federated Spiking Neural Networks"** -- Aouedi & Piamrat, IEEE ICC 2024, June 2024.

6. **"A revolutionary approach to use convolutional spiking neural networks for robust intrusion detection"** -- Cluster Computing (Springer), 2024. 23% accuracy improvement over prior SNN methods.

7. **"An Intrusion Detection System for 5G SDN Network Utilizing Binarized Deep Spiking Capsule Fire Hawk Neural Networks"** -- Future Internet (MDPI), October 2024.

8. **"Analyzing darknet traffic through ML and NeuCube spiking neural networks"** -- Intelligent and Converged Networks, 2024. NeuCube SNN for darknet; 84.31%.

9. **"Resource efficient Internet-of-Things intrusion detection with spiking neural networks"** -- Zivadinovic et al., FedCSIS 2024, November 2024. F1=0.957.

10. **"Applications of spiking neural P systems in cybersecurity"** -- Journal of Membrane Computing, 2024. Cyber-SN P systems for malware/phishing.

### 2025

11. **"Neuromorphic Cybersecurity with Semi-supervised Lifelong Learning"** -- Mia et al., ACM ICONS 2025, August 2025. *Important paper*. Two-stage SNN with Ad-STDP, GWR plasticity. Intel Lava. UNSW-NB15. 85.3% with continual learning.

12. **"Event-Driven Intrusion Detection Systems using Spiking Neural Networks for Edge and IoT Security"** -- IEEE Conference, 2025. STDP-based unsupervised SNN-IDC framework.

13. **"Feature-Optimized Intrusion Detection Based on a Hybrid Spiking Neural Network for the Internet of Things"** -- Vishwanath et al., JAIT 6:52-63, 2025. LOA-BHLESNN. 99.96% on ToN-IoT.

14. **"Hybrid recurrent with spiking neural network model (HRSNN) for enhanced anomaly prediction in IoT networks security"** -- PMC/Scientific Reports, 2025. RNN+SNN hybrid; 99.60%.

15. **"Efficacy of Spiking Neural Networks for Intrusion Detection Systems"** -- IEEE Conference, 2025.

16. **"Towards the neuromorphic Cyber-Twin: architecture for cognitive defense in digital twin ecosystems"** -- Frontiers in Big Data, 2025.

### 2026

17. **"Energy-efficient intrusion detection with a protocol-aware transformer-spiking hybrid model (TASNN)"** -- Scientific Reports, February 2026. Transformer+SNN. Macro-F1=0.93 on NSL-KDD. Cross-dataset generalization.

---

## research ecosystem map
