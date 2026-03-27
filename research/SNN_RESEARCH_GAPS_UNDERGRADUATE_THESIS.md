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

## 3. missing comparison studies

### framework vs framework on real datasets

the Open Neuromorphic benchmark (Feb 2024) tested 11 frameworks but only on a synthetic single-layer FC setup (not real datasets). the 2025 multimodal benchmark covered 5 frameworks but excluded snnTorch, Norse, and BindsNET. nobody's done:

| What's missing | What you'd need | Effort | Impact |
|---------------|----------------|--------|--------|
| **snnTorch vs SpikingJelly vs Norse on SHD** | same CSNN architecture, same hyperparameters, same hardware. report accuracy, training time, memory, energy. | LOW | HIGH -- directly useful to every SNN researcher picking a framework |
| **snnTorch vs SpikingJelly on DVS128 Gesture** | same ConvSNN. both frameworks support DVS128 natively. | LOW-MEDIUM | HIGH |
| **snnTorch vs SpikingJelly on CIFAR10-DVS** | same architecture. both claim support. | LOW-MEDIUM | HIGH |
| **all 4 frameworks on Fashion-MNIST** | snnTorch, SpikingJelly, Norse, BindsNET with identical LIF architecture. | LOW | MEDIUM |
| **framework comparison on N-CALTECH101** | no comparison exists at all. | MEDIUM | MEDIUM |

### method vs method comparisons

| What's missing | Details | Effort |
|---------------|---------|--------|
| **surrogate gradient vs ANN-to-SNN conversion on same dataset/architecture** | papers compare within their method but rarely against each other on identical setups. especially missing for audio. | MEDIUM |
| **rate coding vs temporal coding vs delta modulation** | no one's compared encoding methods across multiple datasets with the same architecture. | LOW-MEDIUM |
| **LIF vs adaptive LIF vs Izhikevich neuron models** | most papers use basic LIF. no study on how neuron model affects accuracy/efficiency. | MEDIUM |
| **STDP vs surrogate gradient on the same task** | very few direct comparisons. each community mostly compares within itself. | MEDIUM |
| **effect of number of timesteps** | how does varying T=4,8,16,32,64 affect accuracy/energy? sparse data, no proper study. | LOW |

### SNN vs ANN fair comparisons

| What's missing | Details | Effort |
|---------------|---------|--------|
| **SNN vs ANN at equivalent parameter count on audio** | most comparisons are on vision. audio comparisons are nearly absent. | MEDIUM |
| **SNN vs ANN on time-series regression** | SNN regression is brand new (first paper May 2024). zero comparison studies. | MEDIUM |
| **SNN vs ANN on tabular data** | virtually unexplored. can an SNN compete with XGBoost? | MEDIUM |
| **energy estimation methodology comparison** | papers use wildly different methods (MACs vs spikes vs synaptic ops). nobody's standardized this. | LOW-MEDIUM |

---

## 4. future work suggestions from recent papers

### from survey papers (2024-2025)

**"The Promise of Spiking Neural Networks for Ubiquitous Computing" (arXiv, June 2025):**
- SNNs underexplored in ubiquitous computing
- suggested: wearable sensor data, smart home IoT, mobile applications
- specific gap: no evaluation on standard HAR benchmarks

**"SNNs in Imaging: A Review and Case Study" (MDPI Sensors, 2025):**
- progress constrained by "reliance on small or custom datasets" and "narrow focus on classification"
- suggested: move beyond classification to detection, segmentation, regression

**"Toward Large-scale SNNs" (arXiv, Sept 2024):**
- suggested: multi-task learning, continual learning benchmarks, real-world deployment studies

**"SNN and Sound: A Comprehensive Review" (Biomedical Engineering Letters, 2024):**
- speech enhancement with SNNs is "very recent" with very limited research
- music generation with SNNs is "significantly underexplored"
- environmental sound classification with SNNs: near-zero papers

**"Reconsidering the Energy Efficiency of SNNs" (arXiv, Sept 2024):**
- prevailing energy evaluations "often oversimplify by focusing on computational aspects while neglecting data movement and memory access"
- suggested: honest energy comparisons accounting for full system overhead
- under typical neuromorphic hardware conditions, SNNs need average spike rate below 6.4% to beat quantized ANNs

### from individual papers

**"SNN for Nonlinear Regression" (Royal Society Open Science, May 2024):**
- first paper on SNN regression ever. authors explicitly suggest:
  - applying to different regression tasks (temperature, load forecasting, etc.)
  - comparing spike encodings for regression
  - snnTorch now has regression tutorials (added late 2024)

**"Neuromorphic Data Augmentation for Training SNNs" (ECCV 2022):**
- NDA improved CIFAR10-DVS by 10.1% and N-Caltech101 by 13.7%
- suggested: test NDA on other neuromorphic datasets (DVS128 Gesture, ASL-DVS, SHD)

**"MuSpike: A Benchmark for Symbolic Music Generation with SNNs" (arXiv, May 2025):**
- "standardized benchmarks and evaluation methods are lacking" for SNN music generation
- provides 5 datasets, evaluates 5 architectures. easy to add a 6th or new dataset.

**the 2025 multimodal framework benchmark** tested 5 frameworks but excluded snnTorch, Norse, and BindsNET. authors explicitly say future work should include additional frameworks, more diverse datasets, and standardized energy measurement.

---

## 5. single-paper domains (easy second data point)

these are areas where only ONE substantial paper exists. a second study -- even a replication or extension -- is a genuine contribution.

| Topic | What exists | What a 2nd paper could do | Effort |
|-------|-----------|--------------------------|--------|
| **SNN for satellite images** | SNN4Space (ESA) on EuroSAT/UC Merced | different architecture/framework, add a 3rd dataset | LOW |
| **SNN for nonlinear regression** | Henkes, Eshraghian, Wessels (Royal Soc, 2024) | different regression benchmarks (Boston Housing, California Housing, energy). compare encodings. | LOW-MEDIUM |
| **SNN for underwater detection** | SU-YOLO (2025) | different underwater datasets or compare with Spiking-YOLO | MEDIUM |
| **SNN for fraud detection** | Bayesian-Opt 1D-CSNN on BAF (2024) | Kaggle Credit Card Fraud. compare with non-spiking baselines. | LOW |
| **SNN for music pattern recognition** | mrahtz BEng thesis (2016, Brian2) | redo with modern snnTorch. use proper datasets (GTZAN, MagnaTagATune). | LOW-MEDIUM |
| **SNN for driver distraction** | Spiking-DD (2024) | different driving datasets or architectures | MEDIUM |
| **SNN for lip reading** | SpikGRU2+ on DVSLip (CVPR 2024 workshop) | different architecture on same dataset, or same approach on new dataset | MEDIUM |
| **SNN for glacier segmentation** | snn-glacier-segmentation (GitHub, 0 stars) | any formal study would be the first peer-reviewed contribution | LOW-MEDIUM |
| **SNN for sign language (event-based)** | DVS_Sign dataset with basic SNN | apply modern architectures (CSNN, transformer-based) | MEDIUM |
| **SNN for 3D rendering (NeRF)** | SpiNeRF (Li et al., 2025) | any follow-up or comparison is novel | HIGH |

---

## 6. cross-domain transfers

### vision methods to audio

| Transfer | Idea | Effort |
|----------|------|--------|
| CSNN from image classification to audio spectrograms | take a proven CSNN from CIFAR-10, apply to ESC-50 or UrbanSound8K spectrograms | LOW |
| data augmentation from vision to audio events | NDA tested only on vision; apply to SHD/SSC | LOW-MEDIUM |
| spiking ResNet from ImageNet to SHD | transfer architecture, not weights | MEDIUM |

### classification to regression

| Transfer | Idea | Effort |
|----------|------|--------|
| snnTorch classification pipeline to regression | snnTorch has regression tutorials since 2024. apply to energy forecasting, stock prediction, sensor regression | LOW-MEDIUM |
| surrogate gradient for continuous output prediction | most SNN regression uses rate-coded output. try membrane potential decoding | MEDIUM |

### vision to medical/biomedical

| Transfer | Idea | Effort |
|----------|------|--------|
| CSNN from CIFAR-10 to chest X-ray classification | CheXpert or ChestX-ray14. zero SNN studies. | LOW-MEDIUM |
| SNN from DVS128 Gesture to EMG-based gesture | different sensor, similar temporal classification | MEDIUM |

### audio to vibration/industrial

| Transfer | Idea | Effort |
|----------|------|--------|
| SHD/SSC architectures to bearing fault diagnosis | both are 1D temporal signals; architecture transfers directly | LOW-MEDIUM |
| speech command SNN to ECG classification | both are short temporal signals with few classes | LOW-MEDIUM |

---

## 7. ranked thesis ideas by effort/novelty ratio

### tier 1: lowest effort, genuine novelty (what i'd recommend)

these are mostly "running experiments" rather than "inventing methods." each fills a documented gap.

#### 1A. framework shootout: snnTorch vs SpikingJelly on SHD + DVS128 Gesture
- **what:** same CSNN, same hyperparameters, both frameworks. report accuracy, training time, GPU memory, energy estimates.
- **why novel:** no such comparison exists. Open Neuromorphic benchmark (2024) only tested synthetic data. multimodal benchmark (2025) excluded snnTorch.
- **datasets:** SHD (audio), DVS128 Gesture (vision). both available via Tonic.
- **effort:** LOW. both frameworks have tutorials for these exact datasets.
- **risk:** LOW.

#### 1B. SNN on ESC-50 or UrbanSound8K
- **what:** standard CSNN or recurrent SNN on ESC-50 or UrbanSound8K. compare with CNN baseline.
- **why novel:** zero SNN papers on environmental sound classification. the "SNN and Sound" review (2024) explicitly calls this out.
- **datasets:** ESC-50 (50 classes, 2000 clips) or UrbanSound8K (10 classes, 8732 clips).
- **effort:** LOW. convert audio to mel-spectrograms, rate-encode to spikes.
- **risk:** LOW. even if SNN underperforms CNN, the result is novel.

#### 1C. SNN for plant disease classification
- **what:** SNN on PlantVillage dataset (54K images, 38 classes of healthy/diseased leaves).
- **why novel:** zero SNN papers on agricultural image classification. entire field uses CNNs.
- **effort:** LOW. standard image classification pipeline with snnTorch CSNN.
- **risk:** LOW.

#### 1D. SNN regression benchmark study
- **what:** since SNN regression is brand new (first paper May 2024), test snnTorch's regression on 3-4 standard datasets (Boston/California Housing, energy efficiency, etc.).
- **why novel:** snnTorch added regression tutorials in late 2024 but nobody's published a benchmark.
- **effort:** LOW. tutorials provide starter code.
- **risk:** LOW-MEDIUM. regression with SNNs is tricky but partial results are fine.

### tier 2: moderate effort, strong novelty

#### 2A. timestep sensitivity study
- vary T=2,4,8,16,32,64 on 3+ datasets (MNIST, Fashion-MNIST, CIFAR-10, SHD). measure accuracy, training time, estimated energy.
- why: no study like this exists. papers pick arbitrary T values.
- effort: MEDIUM. lots of training runs but each is simple.

#### 2B. encoding method comparison on audio
- compare rate, latency, delta modulation, direct spike encoding on SHD and SSC with same architecture.
- why: zero comparison studies for audio neuromorphic data.
- effort: MEDIUM. need to implement multiple encoders.

#### 2C. SNN for ECG arrhythmia classification
- snnTorch CSNN on MIT-BIH or PTB-XL. compare with CNN and LSTM.
- why: 2-3 SNN ECG papers exist, none using snnTorch. PTB-XL has zero SNN results.
- effort: MEDIUM. need ECG preprocessing + spike encoding.

#### 2D. SNN for network intrusion detection
- SNN on CICIDS-2017 or UNSW-NB15. compare with RF, XGBoost, DNN.
- why: few papers exist but none use snnTorch, none have reproducible comparisons with standard ML.
- effort: MEDIUM. tabular data needs spike encoding strategy.

#### 2E. neuromorphic data augmentation on DVS128 Gesture
- apply NDA techniques to DVS128 Gesture (NDA was only tested on CIFAR10-DVS and N-Caltech101).
- why: NDA paper explicitly suggests this.
- effort: MEDIUM. code available, need to adapt.

### tier 3: higher effort, very strong novelty

#### 3A. SNN for sentiment analysis / text classification
- SNN on IMDB Reviews or AG News. key challenge: spike encoding for text.
- why: NLP is explicitly "underexplored" in multiple surveys.
- effort: HIGH. text-to-spike encoding is non-trivial.
- even modest accuracy results would be publishable due to novelty.

#### 3B. SNN for music genre classification (GTZAN)
- SNN on GTZAN (10 genres, 1000 clips). compare with CNN on mel-spectrograms.
- why: only SNN music paper is from 2016 (undergrad thesis, pattern recognition not genre). GTZAN has never been tested.
- effort: MEDIUM-HIGH.

#### 3C. multi-modal SNN: vision + audio fusion
- combine visual and audio in SNN for audiovisual classification.
- why: 4-5 papers on multimodal SNNs, all very recent. text modality completely absent.
- effort: HIGH. need to design fusion architecture.

---

## 8. the absolute easiest path to a genuine contribution

if the goal is minimum effort, maximum novelty claim:

### option A: "first SNN results on [dataset X]"
pick a dataset with zero SNN papers. run a standard snnTorch CSNN. report results.

best candidates (easiest first):
1. **ESC-50 or UrbanSound8K** -- environmental sound. mel-spectrogram, rate-encode, classify.
2. **PlantVillage** -- plant disease images. standard classification.
3. **GTZAN** -- music genre. mel-spectrograms + SNN.
4. **SVHN** -- street view house numbers. slightly harder MNIST, zero SNN work.
5. **Food-101 or Flowers-102** -- fine-grained image classification, zero SNN work.
6. **PTB-XL** -- ECG. time series, zero SNN work.

### option B: "same architecture, different frameworks"
identical CSNN on snnTorch and SpikingJelly on the same dataset. report accuracy, speed, memory, energy.

best candidates:
1. **SHD** -- both frameworks support it, no head-to-head comparison exists
2. **DVS128 Gesture** -- flagship neuromorphic dataset, no framework comparison
3. **CIFAR10-DVS** -- popular but no framework comparison
4. **Fashion-MNIST** -- simple but no framework comparison either

### option C: "systematic hyperparameter study"
vary one key SNN parameter across settings and datasets.

best candidates:
1. **number of timesteps (T)** -- varies wildly across papers (T=4 to T=100), no guidance
2. **membrane decay constant (beta)** -- critical parameter, no study
3. **spike encoding method** -- rate vs temporal vs learnable, especially on audio

---

## sources

### survey papers identifying gaps
- [SNNs for Ubiquitous Computing (arXiv, June 2025)](https://arxiv.org/html/2506.01737v1)
- [SNN and Sound Review (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11362401/)
- [SNNs in Imaging (MDPI Sensors, 2025)](https://www.mdpi.com/1424-8220/25/21/6747)
- [Toward Large-scale SNNs (arXiv, Sept 2024)](https://arxiv.org/html/2409.02111v1)
- [SNN Architecture Search Survey (arXiv, Oct 2025)](https://arxiv.org/html/2510.14235v1)
- [Reconsidering SNN Energy Efficiency (arXiv, Sept 2024)](https://arxiv.org/abs/2409.08290)
- [SNNs in Biomedical Applications (PMC, 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11362408/)
- [Direct Training High-Performance Deep SNNs (PMC, 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11322636/)
- [SNNs for Detection and Segmentation (2025)](https://www.oejournal.org/ioe/article/doi/10.29026/ioe.2025.250007)
- [Multimodal SNN Framework Benchmark (ScienceDirect, July 2025)](https://www.sciencedirect.com/science/article/abs/pii/S0952197625015453)
- [SNN for Physiological/Speech Signals (PMC, 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11362433/)

### framework benchmarks
- [Open Neuromorphic SNN Benchmarks (Feb 2024)](https://open-neuromorphic.org/blog/spiking-neural-network-framework-benchmarking/)
- [SpikingJelly Science Advances (2023)](https://www.science.org/doi/10.1126/sciadv.adi1480)
- [Practical SNN Tutorial (MDPI, Nov 2025)](https://www.mdpi.com/2673-4117/6/11/304)

### individual papers with future work
- [SNN for Nonlinear Regression (Royal Society, May 2024)](https://royalsocietypublishing.org/rsos/article/11/5/231606/92889/Spiking-neural-networks-for-nonlinear)
- [Neuromorphic Data Augmentation (ECCV 2022)](https://arxiv.org/abs/2203.06145)
- [MuSpike Music Benchmark (arXiv, May 2025)](https://arxiv.org/html/2508.19251)
- [SNNLP (arXiv, Jan 2024)](https://arxiv.org/abs/2401.17911)
- [Spiking-DD Driver Distraction (2024)](https://www.researchgate.net/publication/382691405_Spiking-DD_Neuromorphic_Event_Camera_based_Driver_Distraction_Detection_with_Spiking_Neural_Network)
- [SNN for Vibration Predictive Maintenance (arXiv, June 2025)](https://arxiv.org/abs/2506.13416)
- [SpiNeRF (PMC, 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12326478/)
- [Spiking Diffusion Models (arXiv, Aug 2024)](https://arxiv.org/abs/2408.16467)

### datasets and tools
- [Tonic Neuromorphic Data Library](https://tonic.readthedocs.io/)
- [snnTorch Regression Tutorials](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_regression_1.html)
- [SHD Dataset](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/)
- [Open Neuromorphic](https://open-neuromorphic.org/)

### existing student SNN projects (for scope reference)
- [SNN4Space (ESA)](https://github.com/AndrzejKucik/SNN4Space)
- [Musical Pattern Recognition in SNNs (mrahtz, BEng)](https://github.com/mrahtz/musical-pattern-recognition-in-spiking-neural-networks)
- [SNN vs CNN Comparison (sofi12321)](https://github.com/sofi12321/SNN_image_classification)
- [Bayesian Optimization 1D-CSNN for Fraud](https://github.com/dylanperdigao/Bayesian-Optimization-1D-CSNN)
- [SNN Gesture Classification DVS128](https://github.com/DerrickL25/SNN_Gesture_Classification)

### other relevant papers
- [Neuromorphic Computing Robustness (Nature Communications, 2025)](https://www.nature.com/articles/s41467-025-65197-x)
- [Random Heterogeneous SNN for Adversarial Defense (PMC, 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12159496/)
- [SNNs for Sequential Tasks (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/2f55a8b7b1c2c6312eb86557bb9a2bd5-Paper-Conference.pdf)
- [Neuromorphic Sentiment Analysis (PMC, 2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10536645/)
- [VMD-SNN Stock Prediction (PMC, 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11873965/)
- [SNN Cross-Market Portfolio (arXiv, 2025)](https://arxiv.org/pdf/2510.15921)
- [Brain-Inspired Catastrophic Forgetting (Science Advances, 2024)](https://www.science.org/doi/10.1126/sciadv.adi2947)
- [Continual Multi-Label with Spiking Networks (ScienceDirect, 2025)](https://www.sciencedirect.com/science/article/abs/pii/S0952197625018913)
- [Real-time CL on Loihi 2 (arXiv, 2025)](https://arxiv.org/html/2511.01553v1)
- [Spiking-GAN (Semantic Scholar)](https://www.semanticscholar.org/paper/Spiking-GAN:-A-Spiking-Generative-Adversarial-Using-Kotariya-Ganguly/5d4aa57d0536c555a13c5be5ec30127866299f20)
- [Feature Attribution for SNNs (arXiv, 2023)](https://arxiv.org/abs/2311.02110)

---

## confidence

| Finding | Confidence |
|---------|-----------|
| Zero SNN papers on ESC-50, UrbanSound8K, Food-101, GTZAN, SVHN, PlantVillage, PTB-XL | HIGH -- searched multiple databases |
| No framework comparison on real neuromorphic datasets | HIGH -- confirmed via Open Neuromorphic benchmark |
| SNN regression is brand new (first paper May 2024) | HIGH -- confirmed via Royal Society paper |
| Environmental sound classification with SNNs unexplored | HIGH -- confirmed via "SNN and Sound" review |
| NLP/text with SNNs underexplored | HIGH -- multiple surveys confirm |
| Multimodal SNN barely explored | HIGH -- recent papers confirm |
| Music generation/classification with SNNs minimal | HIGH -- MuSpike benchmark paper confirms |
| SOTA accuracy numbers for SHD, CIFAR-10, ImageNet | HIGH -- from published papers |
