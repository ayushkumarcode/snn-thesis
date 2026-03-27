# SNN Research Gaps: Achievable Undergraduate Thesis Opportunities

**Research Date:** 2026-02-25
**Purpose:** Identify the lowest-effort paths to a genuine novel contribution in SNN research for a 3rd-year undergraduate thesis.
**Methodology:** Exhaustive web search across arXiv, IEEE Xplore, PMC, Nature, Springer, conference proceedings, GitHub, and community resources (Open Neuromorphic, snnTorch docs, Tonic library).

---

## Executive Summary

The SNN field is in a peculiar state: it is mature enough that good tools and datasets exist, but immature enough that enormous gaps remain in basic empirical coverage. Most SNN papers focus on image classification (MNIST, CIFAR-10, ImageNet) with surrogate gradient training. Entire application domains, datasets, and framework comparisons remain untouched or have only 1-2 papers. This creates a rich landscape for undergraduate contributions that are technically novel without requiring PhD-level ambition.

The single lowest-effort strategy for a genuine contribution is: **take an existing SNN architecture/method and apply it to a dataset or domain where nobody has tried it yet.** The second lowest-effort strategy is: **run the same experiment across multiple frameworks and report the differences.** Both of these are essentially "engineering" contributions -- running experiments and reporting results -- rather than "invention" contributions, but they are genuinely valuable to the community and count as novel work.

---

## Table of Contents

1. [Application Domains Where SNNs Have Not Been Tried](#1-untried-domains)
2. [Datasets Not Yet Benchmarked with SNNs](#2-unbenchmarked-datasets)
3. [Missing Framework/Method Comparison Studies](#3-missing-comparisons)
4. [Future Work Sections from Recent SNN Papers](#4-future-work-leads)
5. [Single-Paper Domains (Easy Second Data Point)](#5-single-paper-domains)
6. [Cross-Domain Application Opportunities](#6-cross-domain)
7. [Ranked Thesis Project Ideas by Effort/Novelty Ratio](#7-ranked-ideas)
8. [Sources](#8-sources)

---

<a name="1-untried-domains"></a>
## 1. Application Domains Where SNNs Have Not Been Tried (or Barely Tried)

### 1.1 Completely Untouched or Near-Untouched

| Domain | Status | Why SNNs Could Work | Effort |
|--------|--------|-------------------|--------|
| **Plant disease detection from leaf images** | Zero SNN papers found. Entire agricultural CV field uses CNNs/transformers. | Standard image classification; direct transfer of existing SNN architectures. | LOW |
| **Wildlife camera trap classification** | No SNN papers found. | Sparse, event-like data (animals appear briefly). SNNs could exploit temporal sparsity. | LOW-MEDIUM |
| **Satellite/remote sensing land cover** | One paper (SNN4Space, ESA) on EuroSAT and UC Merced. No follow-ups. | Standard image classification with large datasets. Energy efficiency argument strong for satellite edge computing. | LOW |
| **Document/OCR classification** | No SNN papers found beyond MNIST digits. | Character recognition is a natural extension of digit recognition. | LOW |
| **Food recognition/calorie estimation** | No SNN papers found. | Standard image classification. Food-101, Food-2K datasets available. | LOW |
| **Weather/climate prediction from sensor data** | No SNN papers found. | Time-series data naturally maps to temporal spike encoding. | MEDIUM |
| **Music genre classification** | One undergraduate thesis (mrahtz, 2016) on musical pattern recognition. No genre classification. | Audio temporal patterns are a natural fit for SNNs. | LOW-MEDIUM |
| **Sports action recognition** | No SNN papers on standard sports datasets (UCF-101, HMDB-51). | Temporal dynamics of actions suit SNNs. | MEDIUM |

### 1.2 Barely Explored (1-3 Papers Exist)

| Domain | Existing Work | Gap | Effort |
|--------|--------------|-----|--------|
| **Fraud/anomaly detection on tabular data** | One paper: Bayesian Optimization 1D-CSNN for BAF dataset (EPIA 2024). | No comparison with standard ML baselines (XGBoost, Random Forest) on common fraud datasets. No study on credit card fraud (Kaggle dataset). | LOW |
| **NLP/Text classification** | ~3-4 papers: SNNLP (2024), Spikformer for text, sentence-level sentiment. | No study on common benchmarks like AG News, IMDB Reviews, or SST-2 with standard SNN frameworks (snnTorch/SpikingJelly). Text encoding for SNNs remains largely unsolved. | MEDIUM |
| **Emotion recognition from facial expressions** | One paper on SNN for facial expression + speech (2020). One lip-reading paper (CVPR 2024 workshop). | No SNN study on FER2013, AffectNet, or RAF-DB facial expression datasets. | LOW-MEDIUM |
| **Predictive maintenance / fault diagnosis** | ~3-4 papers including vibration-based bearing fault (2020-2025). | Very few studies; no standard comparison across bearing fault benchmarks (CWRU, Paderborn). | LOW-MEDIUM |
| **Financial time series** | ~3-5 papers including VMD-SNN (2025) and cross-market portfolio. | No study comparing SNN vs LSTM/Transformer on standard stock datasets with proper backtesting. | MEDIUM |
| **Network intrusion detection** | ~4-5 papers including convolutional SNN on UNSW-NB15 (2024). | No study on newest CICIDS or TON_IoT datasets. No snnTorch implementation. | LOW-MEDIUM |
| **3D point cloud processing** | Two papers: Spiking PointNet (2023), SPCNNet (2026). | ModelNet40 and ShapeNet benchmarks with SNNs are still rare. | MEDIUM-HIGH |

---

<a name="2-unbenchmarked-datasets"></a>
## 2. Datasets Not Yet Benchmarked with SNNs

### 2.1 Neuromorphic Datasets in Tonic Library That Lack Comprehensive SNN Benchmarks

The Tonic library (the PyTorch Vision equivalent for neuromorphic data) provides these datasets, but many have sparse or no published SNN benchmark results:

| Dataset | Type | Task | SNN Benchmark Status |
|---------|------|------|---------------------|
| **ASL-DVS** | Event vision | American Sign Language | Very few SNN results published. Most work uses ANNs on the events. |
| **POKER-DVS** | Event vision | Card suit recognition | Occasionally used in Norse tutorials but rarely in formal benchmarks. |
| **DVSLip** | Event vision | Lip reading | Only 1-2 papers (CVPR 2024 workshop). |
| **N-CALTECH101** | Event vision | Object recognition (101 classes) | Some results exist but far fewer than N-MNIST or CIFAR10-DVS. |
| **NTIDIGITS** | Event audio | Spoken digits | Rarely benchmarked with modern SNN architectures (snnTorch, SpikingJelly). |
| **DSEC** | Event vision | Depth estimation | No SNN-specific benchmarks; only ANN-based event processing. |
| **ThreeET_Eyetracking** | Event vision | Eye gaze tracking | Extremely new; no SNN benchmark results. |
| **EBSSA** | Event vision | Space situational awareness | No SNN benchmark results. |

### 2.2 Standard ML Datasets Never Tested with SNNs

| Dataset | Domain | Size | Why It Would Work | Existing SNN Work |
|---------|--------|------|------------------|-------------------|
| **Fashion-MNIST** | Image classification | 70K, 10 classes | Direct drop-in for any MNIST SNN pipeline | Some results exist but not systematic |
| **EMNIST** | Character recognition | 814K, 62 classes | Extension of MNIST to full alphabet | One student project (sofi12321) |
| **SVHN** | Street view house numbers | 600K+ | Real-world digit recognition | Almost no SNN work |
| **Food-101** | Food recognition | 101K, 101 classes | Standard classification | Zero SNN papers |
| **Flowers-102** | Fine-grained classification | 8K, 102 classes | Small dataset, easy to train | Zero SNN papers |
| **Stanford Cars** | Fine-grained classification | 16K, 196 classes | Fine-grained recognition challenge | Zero SNN papers |
| **UCF-101** | Video action recognition | 13K clips, 101 classes | Temporal data suits SNNs | Near-zero SNN papers |
| **ESC-50** | Environmental sound | 2K, 50 classes | Audio classification, natural for temporal SNNs | Near-zero SNN papers |
| **UrbanSound8K** | Urban sound | 8.7K, 10 classes | Audio classification | Zero SNN papers |
| **GTZAN** | Music genre | 1K, 10 genres | Audio temporal patterns | Zero SNN papers |
| **MIT-BIH Arrhythmia** | ECG signals | 48 recordings | Time series, perfect for SNNs | 2-3 papers, not with snnTorch |
| **PTB-XL** | 12-lead ECG | 21K, multi-label | Large ECG dataset | Zero SNN papers |
| **HAR (UCI)** | Human activity recognition | 10K, 6 classes | Sensor time series | Very few SNN papers |
| **CWRU Bearing** | Vibration fault diagnosis | Variable | Industrial time series | 2-3 SNN papers |
| **AG News** | Text classification | 120K, 4 classes | NLP benchmark | Zero SNN papers (with snnTorch) |
| **IMDB Reviews** | Sentiment analysis | 50K, 2 classes | NLP benchmark | 1-2 papers, not with standard frameworks |

### 2.3 Heidelberg Spiking Datasets (SHD/SSC) -- Framework Coverage Gaps

The SHD (Spiking Heidelberg Digits) and SSC (Spiking Speech Commands) are the premier audio neuromorphic benchmarks. Current state of art on SHD is 96.41% (SpikCommander). However:
