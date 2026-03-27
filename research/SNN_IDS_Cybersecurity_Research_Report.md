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
