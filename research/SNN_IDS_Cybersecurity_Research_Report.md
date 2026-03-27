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
