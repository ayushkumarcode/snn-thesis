# SNN research gaps -- achievable thesis opportunities

the SNN field is in a weird spot right now: mature enough that good tools and datasets exist, but immature enough that huge gaps remain in basic empirical coverage. most SNN papers focus on image classification (MNIST, CIFAR-10, ImageNet) with surrogate gradient training. entire application domains, datasets, and framework comparisons remain untouched or have like 1-2 papers. that's actually great for an undergrad thesis because there's plenty of room to contribute something genuinely new without needing PhD-level ambition.

the single easiest path to a real contribution is: **take an existing SNN architecture and apply it to a dataset or domain where nobody has tried it.** second easiest: **run the same experiment across multiple frameworks and report the differences.** both are basically "engineering" contributions -- running experiments and reporting results -- not "invention" contributions. but they're genuinely useful to the community and count as novel work.

---

## 1. domains where SNNs haven't been tried (or barely tried)

### completely untouched or near-untouched

| Domain | Status | Why SNNs could work | Effort |
|--------|--------|-------------------|--------|
| **plant disease detection from leaf images** | zero SNN papers found. whole agricultural CV field uses CNNs/transformers. | standard image classification; direct transfer of existing SNN architectures. | LOW |
| **wildlife camera trap classification** | nothing found. | sparse, event-like data (animals appear briefly). SNNs could exploit temporal sparsity. | LOW-MEDIUM |
| **satellite/remote sensing land cover** | one paper (SNN4Space, ESA) on EuroSAT and UC Merced. no follow-ups. | standard image classification with big datasets. energy efficiency argument strong for satellite edge computing. | LOW |
| **document/OCR classification** | nothing beyond MNIST digits. | character recognition is a natural extension of digit recognition. | LOW |
| **food recognition/calorie estimation** | nothing found. | standard image classification. Food-101, Food-2K datasets exist. | LOW |
| **weather/climate prediction from sensor data** | nothing found. | time-series data naturally maps to temporal spike encoding. | MEDIUM |
| **music genre classification** | one undergrad thesis (mrahtz, 2016) on musical pattern recognition. no genre classification. | audio temporal patterns are a natural fit. | LOW-MEDIUM |
| **sports action recognition** | no SNN papers on standard sports datasets (UCF-101, HMDB-51). | temporal dynamics of actions suit SNNs. | MEDIUM |

### barely explored (1-3 papers)

| Domain | What exists | What's missing | Effort |
|--------|-----------|---------------|--------|
| **fraud/anomaly detection on tabular data** | one paper: Bayesian Optimization 1D-CSNN for BAF dataset (EPIA 2024). | no comparison with standard ML baselines (XGBoost, RF) on common fraud datasets. nothing on credit card fraud (Kaggle dataset). | LOW |
| **NLP/text classification** | ~3-4 papers: SNNLP (2024), Spikformer for text, sentence-level sentiment. | nothing on common benchmarks like AG News, IMDB Reviews, SST-2 with standard SNN frameworks. text encoding for SNNs is still basically unsolved. | MEDIUM |
| **emotion recognition from facial expressions** | one paper on SNN for facial expression + speech (2020). one lip-reading paper (CVPR 2024 workshop). | nothing on FER2013, AffectNet, or RAF-DB. | LOW-MEDIUM |
| **predictive maintenance / fault diagnosis** | ~3-4 papers including vibration-based bearing fault (2020-2025). | very few; no standard comparison across bearing fault benchmarks (CWRU, Paderborn). | LOW-MEDIUM |
| **financial time series** | ~3-5 papers including VMD-SNN (2025) and cross-market portfolio. | nobody's compared SNN vs LSTM/Transformer on standard stock datasets with proper backtesting. | MEDIUM |
| **network intrusion detection** | ~4-5 papers including convolutional SNN on UNSW-NB15 (2024). | nothing on newest CICIDS or TON_IoT. no snnTorch implementation. | LOW-MEDIUM |
| **3D point cloud processing** | two papers: Spiking PointNet (2023), SPCNNet (2026). | ModelNet40 and ShapeNet benchmarks with SNNs are still rare. | MEDIUM-HIGH |

---

## 2. datasets that haven't been benchmarked with SNNs

### neuromorphic datasets in Tonic that lack proper SNN benchmarks

Tonic is the PyTorch Vision equivalent for neuromorphic data. it provides these datasets but many have sparse or no published SNN results:

| Dataset | Type | Task | SNN benchmark status |
|---------|------|------|---------------------|
| **ASL-DVS** | Event vision | American Sign Language | very few SNN results. most work uses ANNs on the events. |
| **POKER-DVS** | Event vision | Card suit recognition | occasionally in Norse tutorials but rarely formally benchmarked. |
| **DVSLip** | Event vision | Lip reading | 1-2 papers (CVPR 2024 workshop). |
| **N-CALTECH101** | Event vision | Object recognition (101 classes) | some results but way fewer than N-MNIST or CIFAR10-DVS. |
| **NTIDIGITS** | Event audio | Spoken digits | rarely benchmarked with modern frameworks. |
| **DSEC** | Event vision | Depth estimation | no SNN-specific benchmarks. |
| **ThreeET_Eyetracking** | Event vision | Eye gaze tracking | extremely new, zero SNN results. |
| **EBSSA** | Event vision | Space situational awareness | zero SNN results. |

### standard ML datasets never tested with SNNs

| Dataset | Domain | Size | Why it'd work | Existing SNN work |
|---------|--------|------|--------------|-------------------|
| **Fashion-MNIST** | Image | 70K, 10 classes | direct drop-in for any MNIST SNN pipeline | some results but not a proper study |
| **EMNIST** | Character recognition | 814K, 62 classes | extension of MNIST to full alphabet | one student project (sofi12321) |
| **SVHN** | Street view house numbers | 600K+ | real-world digit recognition | almost nothing |
| **Food-101** | Food recognition | 101K, 101 classes | standard classification | zero |
| **Flowers-102** | Fine-grained | 8K, 102 classes | small dataset, easy to train | zero |
| **Stanford Cars** | Fine-grained | 16K, 196 classes | fine-grained challenge | zero |
| **UCF-101** | Video action recognition | 13K clips, 101 classes | temporal data suits SNNs | near-zero |
| **ESC-50** | Environmental sound | 2K, 50 classes | audio classification, natural for temporal SNNs | near-zero |
| **UrbanSound8K** | Urban sound | 8.7K, 10 classes | audio classification | zero |
| **GTZAN** | Music genre | 1K, 10 genres | audio temporal patterns | zero |
| **MIT-BIH Arrhythmia** | ECG | 48 recordings | time series, perfect for SNNs | 2-3 papers, not with snnTorch |
| **PTB-XL** | 12-lead ECG | 21K, multi-label | large ECG dataset | zero |
| **HAR (UCI)** | Human activity recognition | 10K, 6 classes | sensor time series | very few |
| **CWRU Bearing** | Vibration fault diagnosis | Variable | industrial time series | 2-3 papers |
| **AG News** | Text classification | 120K, 4 classes | NLP benchmark | zero (with snnTorch) |
| **IMDB Reviews** | Sentiment analysis | 50K, 2 classes | NLP benchmark | 1-2 papers, not standard frameworks |

### Heidelberg spiking datasets (SHD/SSC) -- gaps in framework coverage

SHD and SSC are the premier audio neuromorphic benchmarks. current SOTA on SHD is 96.41% (SpikCommander). but:

- nobody's compared snnTorch, SpikingJelly, Norse, and BindsNET on SHD with identical architectures
- no study on different spike encoding methods on SHD (rate vs temporal vs delta)
- SSC (the harder 35-class version) has way fewer results than SHD

---
