# SNNs for Audio Classification: SHD & SSC Benchmarks

so i've been looking into whether SNNs are actually competitive for audio classification, specifically on the Spiking Heidelberg Digits (SHD) and Spiking Speech Commands (SSC) benchmarks. turns out they are -- like genuinely beating ANNs, which is kind of surprising honestly.

the best SNN methods on SHD are hitting around 96.4% accuracy, which blows past the best ANN baselines (92.4% CNN, 90.4% GRU). on the larger SSC benchmark, best SNNs get 83.5-86% vs the GRU baseline of 79%. the field moved fast since 2022 -- key innovations include learnable synaptic delays (DCLS-Delays, ICLR 2024), adaptive neuron models (adLIF, RadLIF, SE-adLIF), parameter-free attention (Pfa-SNN), and spiking transformers (SpikCommander). most of these have open-source code and train on a single GPU in minutes to hours. this seems really feasible for a 3rd year thesis.

---

## 1. The Datasets

### 1.1 Spiking Heidelberg Digits (SHD)

| Property | Value |
|----------|-------|
| Task | Spoken digit classification (0-9 in English and German) |
| Classes | 20 |
| Training samples | 8,156 |
| Test samples | 2,264 |
| Input channels | 700 (artificial cochlea model) |
| Data format | Spike trains with timestamps |
| Created by | Cramer et al. (2020), Zenke Lab, Heidelberg |
| Reference | [arXiv:1910.07407](https://arxiv.org/abs/1910.07407) |

SHD encodes spoken digit recordings into spike trains using "Lauscher," an artificial cochlea model that mimics the human inner ear. each sample is spike events across 700 frequency channels with precise temporal information.

### 1.2 Spiking Speech Commands (SSC)

| Property | Value |
