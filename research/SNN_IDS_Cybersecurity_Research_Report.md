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
