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

- **Gap:** No systematic study comparing snnTorch, SpikingJelly, Norse, and BindsNET on SHD with identical architectures.
- **Gap:** No study on the effect of different spike encoding methods on SHD (rate vs temporal vs delta modulation).
- **Gap:** SSC (the harder 35-class version) has far fewer benchmark results than SHD.

---

<a name="3-missing-comparisons"></a>
## 3. Missing Framework/Method Comparison Studies

### 3.1 Framework vs Framework on Real Datasets

The Open Neuromorphic benchmark (Feb 2024) tested 11 SNN frameworks but ONLY on a synthetic single-layer fully-connected setup (not real datasets). The 2025 multimodal benchmark covered 5 frameworks but excluded snnTorch, Norse, and BindsNET. No study exists that does the following:

| Missing Comparison | What Would Be Needed | Effort | Impact |
|-------------------|---------------------|--------|--------|
| **snnTorch vs SpikingJelly vs Norse on SHD** | Same CSNN architecture, same hyperparameters, same hardware. Report accuracy, training time, memory, energy estimate. | LOW | HIGH -- directly useful to every SNN researcher choosing a framework. |
| **snnTorch vs SpikingJelly on DVS128 Gesture** | Same ConvSNN architecture. Both frameworks support DVS128 natively. | LOW-MEDIUM | HIGH |
| **snnTorch vs SpikingJelly on CIFAR10-DVS** | Same architecture. Both claim support. | LOW-MEDIUM | HIGH |
| **All 4 frameworks on Fashion-MNIST** | snnTorch, SpikingJelly, Norse, BindsNET with identical LIF-based architecture. | LOW | MEDIUM |
| **Framework comparison on N-CALTECH101** | No systematic comparison exists. | MEDIUM | MEDIUM |

### 3.2 Method vs Method Comparisons

| Missing Comparison | Details | Effort |
|-------------------|---------|--------|
| **Surrogate gradient vs ANN-to-SNN conversion on the same dataset/architecture** | Papers compare within their method but rarely against each other on identical setups. Especially missing for audio datasets (SHD, SSC). | MEDIUM |
| **Rate coding vs temporal coding vs delta modulation** | No systematic study comparing encoding methods across multiple datasets with the same architecture. | LOW-MEDIUM |
| **LIF vs Adaptive LIF vs Izhikevich neuron models** | Most papers use basic LIF. No systematic study on how neuron model choice affects accuracy/efficiency across datasets. | MEDIUM |
| **STDP vs surrogate gradient on the same task** | Very few direct comparisons. STDP papers compare to other STDP; gradient papers compare to other gradient methods. | MEDIUM |
| **Effect of number of timesteps** | How does varying T=4, 8, 16, 32, 64 affect accuracy/energy across datasets? Sparse data exists but no systematic study. | LOW |

### 3.3 SNN vs ANN Fair Comparisons

| Missing Comparison | Details | Effort |
|-------------------|---------|--------|
| **SNN vs ANN at equivalent parameter count on audio** | Most comparisons are on vision. Audio (SHD, ESC-50, UrbanSound8K) comparisons are nearly absent. | MEDIUM |
| **SNN vs ANN on time-series regression** | SNN regression is brand new (first paper: Royal Society Open Science, May 2024). No comparison study exists. | MEDIUM |
| **SNN vs ANN on tabular data** | Virtually unexplored. Can an SNN compete with XGBoost on tabular classification? | MEDIUM |
| **Energy estimation methodology comparison** | Papers use wildly different energy estimation methods. Some count MACs, some count spikes, some use synaptic operations. No paper standardizes and compares these methods. | LOW-MEDIUM |

---

<a name="4-future-work-leads"></a>
## 4. Future Work Suggestions from Recent SNN Papers

### 4.1 From Survey Papers (2024-2025)

**"The Promise of Spiking Neural Networks for Ubiquitous Computing" (arXiv, June 2025):**
- SNNs remain underexplored within the ubiquitous computing community.
- Suggested: Apply SNNs to wearable sensor data, smart home IoT, and mobile applications.
- Specific gap: No systematic evaluation of SNNs on standard HAR (Human Activity Recognition) benchmarks.

**"Spiking Neural Networks in Imaging: A Review and Case Study" (MDPI Sensors, 2025):**
- Current progress constrained by "reliance on small or custom datasets" and "narrow focus on classification tasks."
- Suggested: Move beyond classification to detection, segmentation, and regression.

**"Toward Large-scale Spiking Neural Networks" (arXiv, Sept 2024):**
- Suggested: Multi-task learning with SNNs, continual learning benchmarks, and real-world deployment studies.

**"SNN and Sound: A Comprehensive Review" (Biomedical Engineering Letters, 2024):**
- Speech enhancement using SNNs is "very recent" with very limited research.
- Music generation with SNNs is "significantly underexplored."
- Environmental sound classification with SNNs: near-zero papers.

**"Reconsidering the Energy Efficiency of Spiking Neural Networks" (arXiv, Sept 2024):**
- Prevailing energy evaluations "often oversimplify by focusing on computational aspects while neglecting data movement and memory access."
- Suggested: Honest energy comparisons that account for full system overhead. Under typical neuromorphic hardware conditions, SNNs need average spike rate below 6.4% to outperform quantized ANNs.

### 4.2 From Individual Research Papers

**"Spiking Neural Networks for Nonlinear Regression" (Royal Society Open Science, May 2024):**
- This is the first paper on SNN regression (non-classification). Authors explicitly suggest:
  - Applying the method to different regression tasks (temperature prediction, load forecasting, etc.)
  - Comparing spike encodings for regression tasks.
  - snnTorch now has regression tutorials (Parts I and II) as of late 2024.

**"Neuromorphic Data Augmentation for Training SNNs" (ECCV 2022):**
- Showed NDA improves CIFAR10-DVS accuracy by 10.1% and N-Caltech101 by 13.7%.
- Suggested: Test NDA on other neuromorphic datasets (DVS128 Gesture, ASL-DVS, SHD).

**"Spiking Diffusion Models" (arXiv, Aug 2024):**
- First SNN-based generative model achieving competitive image generation.
- Suggested: Apply to different generation tasks, smaller datasets, conditional generation.

**"MuSpike: A Benchmark for Symbolic Music Generation with SNNs" (arXiv, May 2025):**
- Explicitly states "standardized benchmarks and comprehensive evaluation methods are lacking" for SNN music generation.
- Provides 5 datasets and evaluates 5 SNN architectures. Easy to add a 6th architecture or new dataset.

### 4.3 From the Comprehensive Multimodal Framework Benchmark (July 2025)

This paper tested 5 frameworks (SpikingJelly, BrainCog, Sinabs, SNNGrow, Lava) but notably **excluded snnTorch, Norse, and BindsNET.** The authors explicitly state:
- Future work should include additional frameworks.
- Testing on more diverse datasets is needed.
- Energy consumption measurement methodology needs standardization.

---

<a name="5-single-paper-domains"></a>
## 5. Single-Paper Domains (Easy Second Data Point)

These are areas where only ONE substantial paper exists. Publishing a second study -- even a replication or extension -- is a genuine contribution.

| Topic | Single Existing Paper | What a 2nd Paper Could Do | Effort |
|-------|---------------------|--------------------------|--------|
| **SNN for satellite image classification** | SNN4Space (ESA, Kucik et al.) on EuroSAT/UC Merced | Use a different SNN architecture or framework. Add a 3rd satellite dataset. | LOW |
| **SNN for nonlinear regression** | Henkes, Eshraghian, Wessels (Royal Soc, 2024) | Apply to different regression benchmarks (Boston Housing, California Housing, energy prediction). Compare encoding methods. | LOW-MEDIUM |
| **SNN for underwater object detection** | SU-YOLO (2025) | Test on different underwater datasets or compare with Spiking-YOLO. | MEDIUM |
| **SNN for fraud detection** | Bayesian-Opt 1D-CSNN on BAF dataset (2024) | Test on Kaggle Credit Card Fraud dataset. Compare with non-spiking baselines. | LOW |
| **SNN for music pattern recognition** | mrahtz BEng thesis (2016, Brian2) | Redo with modern snnTorch. Use proper music datasets (GTZAN, MagnaTagATune). | LOW-MEDIUM |
| **SNN for driver distraction detection** | Spiking-DD (2024) | Test on different driving datasets or compare architectures. | MEDIUM |
| **SNN for lip reading** | SpikGRU2+ on DVSLip (CVPR 2024 workshop) | Different architecture on same dataset. Or same approach on new lip dataset. | MEDIUM |
| **SNN for glacier segmentation** | snn-glacier-segmentation (GitHub, 0 stars) | Any formal study would be the first peer-reviewed contribution. | LOW-MEDIUM |
| **SNN for sign language (event-based)** | DVS_Sign dataset with basic SNN | Apply modern architectures (CSNN, transformer-based). | MEDIUM |
| **SNN for 3D scene rendering (NeRF)** | SpiNeRF (Li et al., 2025) -- first SNN for NeRF | Any follow-up or comparison is novel. | HIGH |

---

<a name="6-cross-domain"></a>
## 6. Cross-Domain Application Opportunities

### 6.1 Vision Methods Applied to Audio

| Transfer | Specific Idea | Effort |
|----------|--------------|--------|
| CSNN architectures from image classification to audio spectrograms | Take a proven CSNN from CIFAR-10 work, apply to ESC-50 or UrbanSound8K spectrograms | LOW |
| Data augmentation from vision to audio events | NDA (Neuromorphic Data Augmentation) tested only on vision; apply to SHD/SSC | LOW-MEDIUM |
| Spiking ResNet from ImageNet to SHD | Transfer architecture, not weights | MEDIUM |

### 6.2 Classification Methods Applied to Regression

| Transfer | Specific Idea | Effort |
|----------|--------------|--------|
| snnTorch classification pipeline to regression | snnTorch has regression tutorials since 2024. Apply to energy forecasting, stock prediction, or sensor regression | LOW-MEDIUM |
| Surrogate gradient training for continuous output prediction | Most regression SNN work uses rate-coded output. Try membrane potential decoding for regression | MEDIUM |

### 6.3 Vision Methods Applied to Medical/Biomedical

| Transfer | Specific Idea | Effort |
|----------|--------------|--------|
| CSNN from CIFAR-10 to chest X-ray classification | CheXpert or ChestX-ray14 dataset. No SNN study exists. | LOW-MEDIUM |
| SNN from DVS128 Gesture to EMG-based gesture | Different sensor modality, similar temporal classification task | MEDIUM |

### 6.4 Audio Methods Applied to Vibration/Industrial

| Transfer | Specific Idea | Effort |
|----------|--------------|--------|
| SHD/SSC audio architectures to bearing fault diagnosis | Both are 1D temporal signals; architecture transfers directly | LOW-MEDIUM |
| Speech command SNN architecture to ECG classification | Both are short temporal signals with few classes | LOW-MEDIUM |

---

<a name="7-ranked-ideas"></a>
## 7. Ranked Thesis Project Ideas by Effort/Novelty Ratio

### TIER 1: Lowest Effort, Genuine Novelty (Recommended)

These projects require primarily "running experiments" rather than "inventing methods." Each fills a documented gap.

#### 1A. Framework Shootout: snnTorch vs SpikingJelly on SHD + DVS128 Gesture
- **What:** Same CSNN architecture, same hyperparameters, both frameworks. Report accuracy, training time, GPU memory, energy estimates.
- **Why novel:** No such comparison exists. The Open Neuromorphic benchmark (2024) only tested synthetic data. The multimodal benchmark (2025) excluded snnTorch.
- **Datasets:** SHD (audio), DVS128 Gesture (vision). Both available via Tonic.
- **Effort:** LOW. Both frameworks have tutorials for these exact datasets.
- **Deliverable:** Comparison tables, training curves, analysis of API differences.
- **Risk:** LOW. Both frameworks are well-documented.

#### 1B. SNN on ESC-50 or UrbanSound8K (Environmental Sound Classification)
- **What:** Apply a standard CSNN or recurrent SNN to ESC-50 or UrbanSound8K. Compare with CNN baseline.
- **Why novel:** Zero SNN papers exist on environmental sound classification. The "SNN and Sound" review (2024) explicitly calls this out as a gap.
- **Datasets:** ESC-50 (50 classes, 2000 clips) or UrbanSound8K (10 classes, 8732 clips).
- **Effort:** LOW. Convert audio to mel-spectrograms (standard), then rate-encode to spikes.
- **Deliverable:** First SNN results on these datasets. Energy efficiency comparison with CNN.
- **Risk:** LOW. Even if SNN underperforms CNN, the result is novel and publishable.

#### 1C. SNN for Plant Disease Classification
- **What:** Apply SNN to PlantVillage dataset (54K images, 38 classes of healthy/diseased leaves).
- **Why novel:** Zero SNN papers on agricultural image classification. Entire field uses CNNs.
- **Datasets:** PlantVillage (freely available, well-structured).
- **Effort:** LOW. Standard image classification pipeline. Use snnTorch CSNN.
- **Deliverable:** First SNN benchmark on plant disease. Energy argument strong for drone/edge deployment.
- **Risk:** LOW. Well-understood classification task.

#### 1D. SNN Regression Benchmark Study
- **What:** Since SNN regression is brand new (first paper May 2024), test snnTorch's regression capability on 3-4 standard regression datasets (Boston/California Housing, energy efficiency, etc.).
- **Why novel:** snnTorch added regression tutorials in late 2024 but no one has published a systematic benchmark.
- **Datasets:** Standard sklearn/UCI regression datasets.
- **Effort:** LOW. snnTorch tutorials provide starter code.
- **Deliverable:** First systematic SNN regression benchmark. Compare with MLP/Linear baselines.
- **Risk:** LOW-MEDIUM. Regression with SNNs is tricky; partial results are acceptable.

### TIER 2: Moderate Effort, Strong Novelty

#### 2A. Timestep Sensitivity Study Across Datasets
- **What:** Systematically vary the number of timesteps (T=2, 4, 8, 16, 32, 64) on 3+ datasets (MNIST, Fashion-MNIST, CIFAR-10, SHD) and measure accuracy, training time, and estimated energy.
- **Why novel:** No systematic study exists. Papers pick arbitrary T values and don't justify them.
- **Effort:** MEDIUM. Many training runs needed but each is straightforward.
- **Deliverable:** Guidelines for choosing T. Energy-accuracy tradeoff curves.

#### 2B. Encoding Method Comparison on Audio Data
- **What:** Compare rate coding, latency coding, delta modulation, and direct spike encoding on SHD and SSC datasets using the same SNN architecture.
- **Why novel:** No systematic comparison of encoding methods for audio neuromorphic data.
- **Effort:** MEDIUM. Need to implement multiple encoders.
- **Deliverable:** Definitive guide on which encoding works best for audio SNNs.

#### 2C. SNN for ECG Arrhythmia Classification
- **What:** Apply snnTorch CSNN to MIT-BIH or PTB-XL ECG dataset. Compare with CNN and LSTM baselines.
- **Why novel:** Only 2-3 SNN ECG papers exist, none using snnTorch. PTB-XL (21K recordings) has zero SNN results.
- **Effort:** MEDIUM. Need ECG preprocessing pipeline + spike encoding.
- **Deliverable:** SNN benchmark on clinical ECG data. Energy efficiency argument for wearables.

#### 2D. SNN for Network Intrusion Detection (CICIDS/UNSW-NB15)
- **What:** Apply SNN to network intrusion detection using CICIDS-2017 or UNSW-NB15 dataset. Compare with standard ML (Random Forest, XGBoost) and DNN baselines.
- **Why novel:** A few papers exist but none use snnTorch, and none provide reproducible comparisons with standard ML.
- **Effort:** MEDIUM. Tabular data needs spike encoding strategy.
- **Deliverable:** First snnTorch-based IDS. Energy efficiency argument for IoT edge deployment.

#### 2E. Neuromorphic Data Augmentation on DVS128 Gesture
- **What:** Apply NDA (Neuromorphic Data Augmentation) techniques to DVS128 Gesture dataset and measure accuracy improvement. NDA was only tested on CIFAR10-DVS and N-Caltech101.
- **Why novel:** NDA paper explicitly suggests this. DVS128 is the most popular gesture dataset.
- **Effort:** MEDIUM. NDA code is available; need to adapt for DVS128.
- **Deliverable:** Accuracy improvement on a widely-used benchmark.

### TIER 3: Higher Effort, Very Strong Novelty

#### 3A. SNN for Sentiment Analysis / Text Classification
- **What:** Apply SNN to IMDB Reviews or AG News. Key challenge: spike encoding for text (word embeddings to spikes).
- **Why novel:** NLP is explicitly called "underexplored in the neuromorphic setting" in multiple surveys. Text encoding for SNNs is an open problem.
- **Effort:** HIGH. Text-to-spike encoding is non-trivial.
- **Deliverable:** Potentially publishable even with modest accuracy, due to novelty.

#### 3B. SNN for Music Genre Classification (GTZAN)
- **What:** Apply SNN to GTZAN music genre dataset (10 genres, 1000 clips). Compare with CNN on mel-spectrograms.
- **Why novel:** The only SNN music paper is from 2016 (mrahtz, BEng thesis on pattern recognition, not genre classification). GTZAN has never been tested with SNNs.
- **Effort:** MEDIUM-HIGH. Audio processing pipeline + appropriate SNN architecture.
- **Deliverable:** First SNN music genre classification results.

#### 3C. Multi-Modal SNN: Vision + Audio Fusion
- **What:** Combine visual and audio modalities in an SNN for audiovisual classification (e.g., video scene classification).
- **Why novel:** Only 4-5 papers exist on multimodal SNNs, all very recent (2024-2025). The field is wide open. Text modality is completely absent.
- **Effort:** HIGH. Need to design fusion architecture.
- **Deliverable:** Contribution to rapidly growing subfield.

---

## 8. The Absolute Lowest-Effort Path to a Genuine Contribution

If the goal is to minimize effort while maximizing the "novelty" claim, here is the priority ranking:

### Option A: "First SNN Results on [Dataset X]"
Pick a dataset with zero SNN papers. Run a standard snnTorch CSNN. Report results.

**Best candidates (in order of ease):**
1. **ESC-50 or UrbanSound8K** -- Environmental sound. Convert to mel-spectrogram, rate-encode, classify.
2. **PlantVillage** -- Plant disease images. Standard image classification.
3. **GTZAN** -- Music genre. Mel-spectrograms + SNN.
4. **SVHN** -- Street View House Numbers. Slightly harder MNIST variant, zero SNN work.
5. **Food-101 or Flowers-102** -- Fine-grained image classification, zero SNN work.
6. **PTB-XL** -- ECG. Time series, zero SNN work.

### Option B: "Same Architecture, Different Frameworks"
Run an identical CSNN on snnTorch and SpikingJelly on the same dataset. Report accuracy, speed, memory, energy.

**Best candidates:**
1. **SHD** (audio) -- both frameworks support it but no head-to-head comparison.
2. **DVS128 Gesture** -- flagship neuromorphic dataset, no framework comparison.
3. **CIFAR10-DVS** -- popular but no framework comparison.
4. **Fashion-MNIST** -- simple but no systematic framework comparison.

### Option C: "Systematic Hyperparameter Study"
Vary one key SNN parameter across multiple settings and multiple datasets. Report results.

**Best candidates:**
1. **Number of timesteps (T)** -- varies wildly across papers (T=4 to T=100), no guidance exists.
2. **Membrane decay constant (beta)** -- critical parameter, no systematic study.
3. **Spike encoding method** -- rate vs temporal vs learnable, especially on audio data.

---

<a name="8-sources"></a>
## 8. Sources

### Survey Papers Identifying Gaps
- [The Promise of SNNs for Ubiquitous Computing (arXiv, June 2025)](https://arxiv.org/html/2506.01737v1)
- [SNN and Sound: A Comprehensive Review (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11362401/)
- [SNNs in Imaging: A Review and Case Study (MDPI Sensors, 2025)](https://www.mdpi.com/1424-8220/25/21/6747)
- [Toward Large-scale SNNs: A Comprehensive Survey (arXiv, Sept 2024)](https://arxiv.org/html/2409.02111v1)
- [SNN Architecture Search: A Survey (arXiv, Oct 2025)](https://arxiv.org/html/2510.14235v1)
- [Reconsidering the Energy Efficiency of SNNs (arXiv, Sept 2024)](https://arxiv.org/abs/2409.08290)
- [Exploring SNNs in Biomedical Applications (PMC, 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11362408/)
- [Direct Training High-Performance Deep SNNs: A Review (PMC, 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11322636/)
- [SNNs for Object Detection and Semantic Segmentation: A Review (2025)](https://www.oejournal.org/ioe/article/doi/10.29026/ioe.2025.250007)
- [Comprehensive Multimodal Benchmark of SNN Frameworks (ScienceDirect, July 2025)](https://www.sciencedirect.com/science/article/abs/pii/S0952197625015453)
- [SNN for Physiological and Speech Signals: A Review (PMC, 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11362433/)

### Framework Benchmarks
- [Open Neuromorphic SNN Library Benchmarks (Feb 2024)](https://open-neuromorphic.org/blog/spiking-neural-network-framework-benchmarking/)
- [SpikingJelly: Science Advances Paper (2023)](https://www.science.org/doi/10.1126/sciadv.adi1480)
- [A Practical Tutorial on SNNs: Comprehensive Review (MDPI, Nov 2025)](https://www.mdpi.com/2673-4117/6/11/304)

### Key Individual Papers with Future Work Suggestions
- [SNN for Nonlinear Regression (Royal Society Open Science, May 2024)](https://royalsocietypublishing.org/rsos/article/11/5/231606/92889/Spiking-neural-networks-for-nonlinear)
- [Neuromorphic Data Augmentation for Training SNNs (ECCV 2022)](https://arxiv.org/abs/2203.06145)
- [MuSpike: Benchmark for Symbolic Music Generation with SNNs (arXiv, May 2025)](https://arxiv.org/html/2508.19251)
- [SNNLP: Energy-Efficient NLP with SNNs (arXiv, Jan 2024)](https://arxiv.org/abs/2401.17911)
- [Spiking-DD: Driver Distraction Detection (2024)](https://www.researchgate.net/publication/382691405_Spiking-DD_Neuromorphic_Event_Camera_based_Driver_Distraction_Detection_with_Spiking_Neural_Network)
- [SNN for Low-Power Vibration-Based Predictive Maintenance (arXiv, June 2025)](https://arxiv.org/abs/2506.13416)
- [SpiNeRF: SNN for Neural Radiance Fields (PMC, 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12326478/)
- [Spiking Diffusion Models (arXiv, Aug 2024)](https://arxiv.org/abs/2408.16467)

### Dataset and Tool Resources
- [Tonic Neuromorphic Data Library](https://tonic.readthedocs.io/)
- [snnTorch Regression Tutorials](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_regression_1.html)
- [Spiking Heidelberg Digits (SHD) Dataset](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/)
- [Open Neuromorphic Community](https://open-neuromorphic.org/)

### Existing Student/Undergraduate SNN Projects (For Scope Reference)
- [SNN4Space (ESA) -- satellite classification](https://github.com/AndrzejKucik/SNN4Space)
- [Musical Pattern Recognition in SNNs (mrahtz, BEng)](https://github.com/mrahtz/musical-pattern-recognition-in-spiking-neural-networks)
- [SNN vs CNN Comparison (sofi12321)](https://github.com/sofi12321/SNN_image_classification)
- [Bayesian Optimization 1D-CSNN for Fraud Detection](https://github.com/dylanperdigao/Bayesian-Optimization-1D-CSNN)
- [SNN Gesture Classification DVS128 (DerrickL25)](https://github.com/DerrickL25/SNN_Gesture_Classification)

### SNN Robustness and Adversarial Research
- [Neuromorphic Computing Paradigms Enhance Robustness Through SNNs (Nature Communications, 2025)](https://www.nature.com/articles/s41467-025-65197-x)
- [Random Heterogeneous SNN for Adversarial Defense (PMC, 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12159496/)

### SNN for NLP / Text
- [Advancing SNNs for Sequential Tasks (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/2f55a8b7b1c2c6312eb86557bb9a2bd5-Paper-Conference.pdf)
- [Neuromorphic Sentiment Analysis Using SNNs (PMC, 2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10536645/)

### Financial / Time Series
- [Accurate and Efficient Stock Market Index Prediction: VMD-SNN (PMC, 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11873965/)
- [SNN for Cross-Market Portfolio (arXiv, 2025)](https://arxiv.org/pdf/2510.15921)

### Continual Learning and Catastrophic Forgetting
- [Brain-Inspired Algorithm That Mitigates Catastrophic Forgetting (Science Advances, 2024)](https://www.science.org/doi/10.1126/sciadv.adi2947)
- [Continual Multi-Label Learning with Evolving Spiking Networks (ScienceDirect, 2025)](https://www.sciencedirect.com/science/article/abs/pii/S0952197625018913)
- [Real-time Continual Learning on Intel Loihi 2 (arXiv, 2025)](https://arxiv.org/html/2511.01553v1)

### Generative Models with SNNs
- [Spiking-GAN: Time-To-First-Spike Coding (Semantic Scholar)](https://www.semanticscholar.org/paper/Spiking-GAN:-A-Spiking-Generative-Adversarial-Using-Kotariya-Ganguly/5d4aa57d0536c555a13c5be5ec30127866299f20)

### Explainability
- [Feature Attribution Explanations for SNNs (arXiv, 2023)](https://arxiv.org/abs/2311.02110)

---

## Confidence Assessment

| Finding | Confidence |
|---------|------------|
| Zero SNN papers on ESC-50, UrbanSound8K, Food-101, GTZAN, SVHN, PlantVillage, PTB-XL | HIGH -- searched multiple databases |
| No framework comparison on real neuromorphic datasets exists | HIGH -- confirmed via Open Neuromorphic benchmark paper |
| SNN regression is brand new (first paper May 2024) | HIGH -- confirmed via Royal Society paper |
| Environmental sound classification with SNNs is unexplored | HIGH -- confirmed via "SNN and Sound" review (2024) |
| NLP/text with SNNs is underexplored | HIGH -- confirmed via multiple surveys |
| Multimodal SNN (audio+vision+text) barely explored | HIGH -- confirmed via recent papers |
| Music generation/classification with SNNs has minimal papers | HIGH -- confirmed via MuSpike benchmark paper |
| Specific accuracy numbers for SoTA on SHD, CIFAR-10, ImageNet | HIGH -- from published papers |

---

*Report compiled: 2026-02-25*
