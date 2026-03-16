# SNN-Based EEG Classification for Brain-Computer Interfaces
## Comprehensive Research Report
**Date:** 2026-02-25

---

## Executive Summary

Spiking Neural Networks (SNNs) for EEG-based Brain-Computer Interfaces (BCIs) represent an active and rapidly growing research area at the intersection of neuromorphic computing, neuroscience, and machine learning. The field has seen significant acceleration since 2023, with dozens of new architectures, benchmark results, and open-source implementations emerging. SNNs offer a biologically plausible alternative to conventional deep learning for EEG classification, with strong arguments around energy efficiency (up to 95% reduction vs. DNNs on neuromorphic hardware), temporal dynamics that naturally align with EEG signals, and suitability for real-time edge deployment.

However, SNNs currently trail state-of-the-art conventional deep learning (CNNs, Transformers) in raw classification accuracy by approximately 3-10 percentage points on most benchmark tasks, though the gap is narrowing rapidly. The primary EEG tasks addressed by SNN research include motor imagery classification, emotion recognition, seizure/epilepsy detection, stress detection, and more recently SSVEP classification. Multiple public datasets (BCI Competition IV-2a/2b, PhysioNet EEGMMIDB, DEAP, SEED) and open-source frameworks (snnTorch, SpikingJelly, Norse, combra-lab/snn-eeg) make this project feasible for an undergraduate student with strong Python/PyTorch skills but no neuroscience background. The topic would be considered genuinely novel at the undergraduate level, as SNN-based EEG classification remains a niche research frontier even in graduate-level work.

---

## 1. Current State of SNN-Based EEG Classification

### 1.1 Tasks Addressed by SNNs

SNN-based EEG classification research spans multiple BCI paradigms:

#### Motor Imagery (MI) Classification
The most extensively studied task. Subjects imagine moving body parts (left hand, right hand, feet, tongue) and EEG patterns over the motor cortex are classified.

**Key papers:**
- **SCNet** (2023): Combines CNN feature extraction with SNN biological interpretability using adaptive coding and surrogate gradient learning. Evaluated on PhysioNet, BCI IV-2a, and BCI IV-2b. Outperforms prior SNN methods.
- **HR-SNN** (2024): End-to-end SNN achieving 77.58% average accuracy on BCI IV-2a (4-class), surpassing all compared SNN models. On PhysioNet: 67.24% (global) and 74.95% (transfer learning).
- **NiSNN-A** (2024): Non-iterative SNN with attention mechanism for motor imagery. Combines accuracy gains with energy reduction.
- **LENet/RDSNN** (2024): Lightweight SNN achieving 73.65% on PhysioNet, 81.75% on BCI IV-2a, 84.56% on BCI IV-2b.
- **Lightweight SNN** (2025, ScienceDirect): Within-subject and cross-subject experiments on three public datasets demonstrate the superiority of the SNN model over classical CNN-based models.
- **combra-lab SNN-EEG** (2022, TMLR): Deployed on Intel Loihi, consuming 95% less energy than DNNs on NVIDIA Jetson TX2 with similar accuracy.

#### Emotion Recognition
The second-most studied SNN-EEG task. Subjects watch emotional stimuli while EEG is recorded; valence, arousal, and dominance are classified.

**Key papers:**
- **EESCN** (2024, Computer Methods and Programs in Biomedicine): Novel SNN achieving 94.56% (valence), 94.81% (arousal), 94.73% (dominance) on DEAP and 79.65% mean accuracy on SEED-IV. Faster running speed and less memory than prior SNN methods.
- **Fractal-SNN** (2023): Exploits multi-scale temporal-spectral-spatial information. Evaluated on DREAMER, DEAP, SEED-IV, and MPED.
- **NeuroSense** (Tan et al., 2021): Achieved 78.97%/67.76% (arousal/valence) on DEAP.
- **Bidirectional SNN** (Alzhrani et al., 2021): 94.83% accuracy on DREAMER.
- **BISNN** (2025): Bio-information-fused SNN for enhanced emotion recognition.

#### Epilepsy/Seizure Detection
A growing application area with strong clinical motivation.

**Key papers:**
- **EESNN** (2024): Recurrent spiking convolution structure achieving energy reduction by several orders of magnitude compared to ANNs.
- **Spiking Conformer** (2024): Trained on raw EEG, bypassing preprocessing. Requires 10x fewer operations than non-spiking equivalent.
- **SyNSense Xylo deployment** (2024): Real-time sub-milliwatt epilepsy detection on a neuromorphic edge inference processor.
- **Cross-patient SNN** (2024, Frontiers in Neuroscience): Efficient and generalizable cross-patient seizure detection.

#### Stress Detection
- **CSNN for Stress** (2025, Scientific Reports): Convolutional SNN achieving 98.75% accuracy with 10-fold cross-validation and F1 score of 98.60%.

#### SSVEP (Steady-State Visual Evoked Potentials)
- **Event-driven SNN for SSVEP** (2024, IEEE): Uses empirical mode decomposition and canonical correlation analysis with excitation-inhibition balanced SNN.

#### P300 Classification
- SNNs have been used for P300 signal reconstruction and data augmentation (2021, Frontiers), but dedicated P300 SNN classifiers remain rare compared to CNN-based approaches. This represents a potential research gap.

#### Sleep Staging
- **Hybrid SNN (HSNN)**: Demonstrated for automatic sleep staging from EEG signals.

#### Situational Awareness
- **SNN with SCTN neurons** (2024, Applied Sciences): Novel approach using spike continuous-time neurons for situational awareness from EEG.

### 1.2 Architecture Evolution

| Generation | Approach | Example |
|---|---|---|
| Early (pre-2020) | Reservoir/NeuCube + STDP | NeuCube framework |
| Mid (2020-2022) | Surrogate gradient SNNs | combra-lab SNN-EEG |
| Recent (2023-2024) | Hybrid CNN-SNN, attention | SCNet, HR-SNN, NiSNN-A |
| Emerging (2024-2025) | Spiking Transformers | Spikeformer, Spiking Conformer |
| Frontier (2025+) | Lightweight + edge deployment | LENet, Xylo-based systems |

---

## 2. Accuracy: SNNs vs. Conventional Approaches

### 2.1 Motor Imagery (BCI Competition IV Dataset 2a, 4-class)

| Method | Type | Accuracy (%) | Year |
|---|---|---|---|
| EEGEncoder (Transformer) | ANN | 86.46 | 2024 |
| CIACNet (Attention CNN) | ANN | 85.15 | 2024 |
| SNA-MHC (custom SNN+Attn) | SNN | 92.80* | 2024 |
| RDSNN (Lightweight SNN) | SNN | 81.75 | 2024 |
| HR-SNN (End-to-End SNN) | SNN | 77.58 | 2024 |
| CNN1D_MF | ANN | 69.20 | 2023 |
| DFBRTS | ANN | 78.16 | 2024 |

*Note: SNA-MHC's 92.80% is an outlier result that may use different evaluation protocols. Most SNN results cluster around 75-82% on this benchmark.

### 2.2 Motor Imagery (PhysioNet EEGMMIDB)

| Method | Type | Accuracy (%) | Year |
|---|---|---|---|
| Novel DL approach | ANN | 95.70 | 2025 |
| Optimized DL + DWT | ANN | 97.05 | 2024 |
| HR-SNN (transfer) | SNN | 74.95 | 2024 |
| HR-SNN (global) | SNN | 67.24 | 2024 |
| RDSNN | SNN | 73.65 | 2024 |
| combra-lab SNN-EEG | SNN | ~similar to DNN* | 2022 |

*combra-lab reports "similar classification performance" to DNNs with 95% less energy.

### 2.3 Emotion Recognition (DEAP Dataset)

| Method | Type | Valence (%) | Arousal (%) | Year |
|---|---|---|---|---|
| Graph CNN + Dual Attention | ANN | ~90+ | ~90+ | 2024 |
| EESCN | SNN | 94.56 | 94.81 | 2024 |
| NeuroSense | SNN | 67.76 | 78.97 | 2021 |
| Bidirectional SNN (DREAMER) | SNN | 94.83 (overall) | -- | 2021 |

### 2.4 Stress Detection

| Method | Type | Accuracy (%) | Year |
|---|---|---|---|
| CSNN | SNN | 98.75 | 2025 |
| Hybrid SNN | SNN | 94.00 | 2024 |

### 2.5 Key Takeaway on Accuracy

**The accuracy gap between SNNs and ANNs is task-dependent:**
- **Emotion recognition and stress detection:** SNNs can match or even exceed ANN performance (EESCN at 94.81% is competitive with state-of-the-art ANNs on DEAP).
- **Motor imagery (4-class):** SNNs still trail top ANNs by ~5-10 percentage points (77-82% vs. 85-87% for best ANNs), though the gap is narrowing rapidly.
- **Seizure detection:** SNNs achieve comparable accuracy with orders-of-magnitude lower energy consumption.

**The real argument for SNNs is not just accuracy but the accuracy-energy tradeoff.** An SNN achieving 80% accuracy at 1/20th the energy of a 85% accurate CNN may be the better choice for a wearable BCI.

---

## 3. Available Datasets

### 3.1 Motor Imagery Datasets

| Dataset | Subjects | Channels | Classes | Sampling Rate | Access | Notes |
|---|---|---|---|---|---|---|
| **BCI Competition IV-2a** | 9 | 22 (EEG) + 3 (EOG) | 4 (left hand, right hand, feet, tongue) | 250 Hz | [BBCI website](https://www.bbci.de/competition/iv/) / [Kaggle](https://www.kaggle.com/datasets/thngdngvn/bci-competition-iv-data-sets-2a) | Most popular MI benchmark (used in 31+ studies). 288 trials per subject across 2 sessions. |
| **BCI Competition IV-2b** | 9 | 3 (EEG) + 3 (EOG) | 2 (left hand, right hand) | 250 Hz | [BBCI website](https://www.bbci.de/competition/iv/) | Second most popular MI benchmark (14+ studies). |
| **PhysioNet EEGMMIDB** | 109 | 64 | 4 (open/close fists, imagine fists/feet) | 160 Hz | [PhysioNet](https://www.physionet.org/content/eegmmidb/1.0.0/) | Freely downloadable, no application needed. Largest freely available MI dataset. Third most popular (11+ studies). |
| **BNCI Horizon 2020** | Various | Various | Various | Various | [BNCI database](https://bnci-horizon-2020.eu/database/data-sets) | Collection of multiple BCI datasets. |

### 3.2 Emotion Recognition Datasets

| Dataset | Subjects | Channels | Classes | Stimuli | Access | Notes |
|---|---|---|---|---|---|---|
| **DEAP** | 32 | 32 (EEG) + 8 (peripheral) | Valence/Arousal/Dominance (continuous) | 40 music videos (1 min each) | [DEAP official](http://eecs.qmul.ac.uk/mmv/datasets/deap/) | Requires university email application (~1 month approval). 512 Hz sampling. Preprocessed files available (downsampled to 128 Hz). |
| **SEED** | 15 | 62 (EEG) | 3 (negative, neutral, positive) | Film clips | [BCMI Lab](https://bcmi.sjtu.edu.cn/home/seed/seed.html) | 200 Hz preprocessed. 3 sessions per subject (~1 week apart). |
| **SEED-IV** | 15 | 62 (EEG) | 4 (happy, sad, fear, neutral) | Film clips | [BCMI Lab](https://bcmi.sjtu.edu.cn/home/seed/index.html) | Extension of SEED with 4 emotion classes. |
| **SEED-VII** | -- | -- | 6 basic emotions + continuous | Multimodal | [BCMI Lab](https://bcmi.sjtu.edu.cn/home/seed/) | Newest variant with continuous labels. |
| **DREAMER** | 23 | 14 (EEG) | Valence/Arousal/Dominance | Film clips | Public | Lower channel count, useful for lightweight models. |

### 3.3 Seizure Detection Datasets

| Dataset | Notes |
|---|---|
| **CHB-MIT** | Scalp EEG from pediatric subjects with intractable seizures. PhysioNet. |
| **Bonn University** | 5 classes (healthy, epileptic zone, seizure). Classic benchmark. |
| **TUH EEG Corpus** | Large-scale clinical EEG dataset from Temple University Hospital. |

### 3.4 Recommendation for an Undergraduate Project

**Best starting point:** PhysioNet EEGMMIDB
- Freely available without application process
- Large (109 subjects)
- Well-documented
- Many published baselines for comparison
- Directly supported by combra-lab/snn-eeg codebase

**Best for emotion recognition:** DEAP
- Industry standard benchmark
- Supported by TorchEEG library
- Requires university email application (plan ahead)

---

## 4. The Argument for SNNs in BCI

### 4.1 Biological Plausibility

SNNs communicate through discrete spike events, mimicking the actual neural coding used by the brain. This creates a natural alignment between the computation model and the biological signals being decoded:

- EEG signals reflect aggregate neural spiking activity
- SNNs process information through spike timing and spike rates, the same coding schemes used by biological neurons
- Spike-Timing-Dependent Plasticity (STDP) in SNNs mirrors actual synaptic learning rules
- The temporal dynamics of spiking neurons (membrane potential, refractory periods) naturally capture the temporal structure in EEG

This biological plausibility provides not just a philosophical argument but practical benefits: SNN architectures can leverage neuroscientific knowledge about EEG generation to inform network design.

### 4.2 Energy Efficiency and Low Power

This is the strongest practical argument for SNNs in BCI:

- **combra-lab SNN on Intel Loihi:** 95% less energy per inference than DNNs on NVIDIA Jetson TX2
- **Spiking Conformer for seizure detection:** 10x fewer operations than non-spiking equivalent
- **SyNSense Xylo deployment:** Sub-milliwatt real-time epilepsy detection
- **Theoretical basis:** SNNs use additions instead of multiplications (no multiply-accumulate operations), and event-driven processing means energy is consumed only during spike transmission

**Why this matters for BCI:** Brain-computer interfaces, especially implantable or wearable ones, demand extreme energy efficiency. A BCI device running on a battery for days/weeks requires the kind of energy budgets only neuromorphic computing can deliver. The human brain operates complex neural networks on approximately 20W total -- SNNs on neuromorphic hardware approach this efficiency paradigm.

### 4.3 Real-Time Processing and Low Latency

- SNNs process temporal information natively -- they do not need to accumulate a window of data before processing
- Event-driven computation means processing happens asynchronously as spikes arrive
- Neuromorphic hardware (Loihi, SpiNNaker, Xylo, TrueNorth) enables parallel, event-driven operations with minimal latency
- This is critical for BCI applications where milliseconds matter (e.g., motor intent decoding for prosthetic control)

### 4.4 Temporal Dynamics

EEG signals are inherently temporal, and SNNs have native temporal processing capabilities:

- Spiking neurons maintain internal state (membrane potential) that naturally integrates temporal information
- No need for explicit temporal feature engineering or sliding windows
- Recurrent connections in SNNs can capture long-range temporal dependencies
- This contrasts with CNNs that treat EEG as quasi-static images and may miss temporal dynamics

### 4.5 Compact Models

- SNNs tend to have fewer parameters than equivalent ANNs
- EESCN showed faster running speed and less memory footprint than prior methods
- Lightweight SNN architectures (LENet) replace fully connected layers with classification convolution blocks, reducing parameter count
- Smaller models are easier to deploy on edge devices and require less training data

---

## 5. Open-Source Implementations on GitHub

### 5.1 SNN-EEG Specific Repositories

| Repository | Description | Language | Stars | Paper |
|---|---|---|---|---|
| [combra-lab/snn-eeg](https://github.com/combra-lab/snn-eeg) | PyTorch + Intel Loihi implementation for decoding EEG on neuromorphic hardware. Uses spatial conv, temporal conv, and recurrent layers. Trained on EEGMMIDB (motor imagery). | Python/PyTorch | Active | TMLR 2022 |
| [SuperBruceJia/EEG-DL](https://github.com/SuperBruceJia/EEG-DL) | Deep Learning library for EEG Tasks including SNN-related approaches. TensorFlow-based. | Python/TF | Popular | -- |
| [TheBrainLab/Awesome-Spiking-Neural-Networks](https://github.com/TheBrainLab/Awesome-Spiking-Neural-Networks) | Curated paper list with codes for SNN research including EEG applications. | Markdown | Large | -- |
| [SpikingChen/SNN-Daily-Arxiv](https://github.com/SpikingChen/SNN-Daily-Arxiv) | Daily updated arXiv papers about SNNs. Useful for tracking latest publications. | Markdown | Active | -- |

### 5.2 General SNN Frameworks (Applicable to EEG)

| Framework | Repository | Key Features | Documentation |
|---|---|---|---|
| **snnTorch** | [jeshraghian/snntorch](https://github.com/jeshraghian/snntorch) | PyTorch-based, excellent tutorials (7 part series), surrogate gradient support, GPU-accelerated. Best for beginners. | [snntorch.readthedocs.io](https://snntorch.readthedocs.io/) |
| **SpikingJelly** | [fangwei123456/spikingjelly](https://github.com/fangwei123456/spikingjelly) | Full-stack SNN toolkit: preprocessing, building, training, neuromorphic chip deployment. Published in Science Advances. | [spikingjelly.readthedocs.io](https://spikingjelly.readthedocs.io/) |
| **Norse** | [norse/norse](https://github.com/norse/norse) | PyTorch primitives for bio-inspired neural components. Research-oriented. | [norse.github.io/norse](https://norse.github.io/norse/) |

### 5.3 EEG Processing Libraries

| Library | Repository | Key Features |
|---|---|---|
| **TorchEEG** | [torcheeg/torcheeg](https://github.com/torcheeg/torcheeg) | PyTorch-based EEG analysis. Plug-and-play API for preprocessing, built-in DEAP/SEED dataset loaders, multiple DL models. |
| **MNE-Python** | [mne-tools/mne-python](https://github.com/mne-tools/mne-python) | Industry standard for EEG preprocessing, filtering, artifact removal. Essential tool. |
| **MOABB** | [NeuroTechX/moabb](https://github.com/NeuroTechX/moabb) | Mother of All BCI Benchmarks. Standardized evaluation framework for BCI algorithms. |

### 5.4 Recommended Stack for Undergraduate Project

```
EEG Data Loading:    TorchEEG (for DEAP/SEED) or MNE-Python (for PhysioNet/BCI IV)
EEG Preprocessing:   MNE-Python (filtering, artifact removal, epoching)
SNN Framework:       snnTorch (best tutorials) or SpikingJelly (most features)
Base Framework:      PyTorch
Spike Encoding:      snnTorch built-in encoders (rate coding, latency coding)
Evaluation:          scikit-learn metrics, MOABB benchmarks
```

---

## 6. Feasibility for an Undergraduate Without Neuroscience Background

### 6.1 Assessment: FEASIBLE with Appropriate Scoping

**Reasons it IS feasible:**

1. **No wet-lab work required:** All datasets are publicly available and pre-recorded. No need to recruit subjects, apply for ethics approval, or operate EEG equipment.

2. **Strong tooling ecosystem:** Libraries like TorchEEG, MNE-Python, and snnTorch abstract away much of the complexity. TorchEEG can load DEAP/SEED datasets with a few lines of code.

3. **PyTorch foundation:** If you know PyTorch, the jump to snnTorch is manageable. snnTorch extends PyTorch and the tutorials are written for people with deep learning (not neuroscience) backgrounds.

4. **Excellent tutorials available:**
   - snnTorch has a 7-part tutorial series starting from spike encoding fundamentals
   - combra-lab/snn-eeg has a complete working pipeline you can study and modify
   - TorchEEG has workflow examples for DEAP and SEED

5. **Published reference implementations:** Unlike cutting-edge research in some fields, there are multiple working codebases you can reference, reproduce, and extend.

6. **The neuroscience you need is learnable:** You need to understand:
   - What EEG measures (aggregate electrical activity from the scalp)
   - What motor imagery is (imagining movement)
   - Basic frequency bands (alpha, beta, mu rhythms)
   - What Event-Related Desynchronization/Synchronization (ERD/ERS) means
   - These concepts can be learned in 1-2 weeks from review papers and tutorials

### 6.2 Challenges and Mitigations

| Challenge | Difficulty | Mitigation |
|---|---|---|
| Understanding EEG preprocessing (filtering, artifact removal) | Medium | Use MNE-Python tutorials; many datasets come preprocessed |
| Spike encoding (converting EEG to spikes) | Medium | snnTorch has built-in encoders; follow tutorial 1 |
| Surrogate gradient training | Medium-Hard | snnTorch/SpikingJelly handle this automatically |
| Cross-subject variability | Hard | Start with within-subject classification, then attempt cross-subject |
| Hyperparameter tuning | Medium | Follow published papers' configurations as starting points |
| Understanding neuroscience terminology | Medium | Keep a glossary; focus on the 20 terms you actually need |
| Debugging temporal SNN dynamics | Medium | Use snnTorch visualization tools for membrane potentials |

### 6.3 Recommended Scope for Undergraduate Thesis

**Minimum Viable Project (safe scope):**
- Reproduce combra-lab/snn-eeg results on PhysioNet EEGMMIDB
- Compare SNN vs. EEGNet (standard CNN baseline) on the same dataset
- Report accuracy, parameter count, and estimated energy consumption
- Timeline: ~3-4 months

**Ambitious but Achievable Project:**
- Implement an SNN for motor imagery classification using snnTorch on BCI Competition IV-2a
- Compare multiple spike encoding methods (rate coding vs. temporal coding)
- Benchmark against 2-3 CNN baselines (EEGNet, ShallowConvNet, DeepConvNet)
- Analyze energy efficiency using theoretical multiply-accumulate operation counts
- Timeline: ~4-6 months

**Stretch Goals (if time permits):**
- Cross-subject transfer learning with SNNs
- Hybrid CNN-SNN architecture
- Deployment on neuromorphic hardware simulator
- Real-time inference demonstration

### 6.4 Essential Background Reading (1-2 Week Crash Course)

1. **EEG Basics:** "An Introduction to EEG" -- any introductory neuroscience textbook chapter
2. **BCI Fundamentals:** BCI Competition IV website documentation
3. **SNN Fundamentals:** snnTorch Tutorial Series (Tutorials 1-5)
4. **SNN for EEG:** "Spiking neural networks for EEG signal analysis: From theory to practice" (ScienceDirect, 2025)
5. **Motor Imagery BCI:** "An in-depth survey on Deep Learning-based Motor Imagery EEG classification" (ScienceDirect, 2023)

---

## 7. Novelty Assessment at the Undergraduate Level

### 7.1 Verdict: YES, This Would Be Considered Novel

**Strong indicators of novelty for an undergraduate thesis:**

1. **Niche research area:** SNN-based EEG classification is primarily published in top-tier journals (Nature Scientific Reports, Frontiers in Neuroscience, IEEE TMLR, Neurocomputing) by PhD students and postdocs. An undergraduate tackling this topic demonstrates ambition and capability beyond the typical undergraduate scope.

2. **Few undergraduate implementations exist:** While CNN-based EEG classification is becoming common in undergraduate projects, SNN-based approaches remain rare at this level. Most undergraduate BCI projects use traditional machine learning (SVM, Random Forest) or standard deep learning (EEGNet).

3. **Active research frontier:** New papers are being published monthly (2024-2025). The field has not yet converged on best practices, meaning even a systematic comparison study would contribute useful knowledge.

4. **Multiple novelty angles available:**

| Novelty Approach | Description | Risk Level |
|---|---|---|
| Systematic SNN benchmark | Compare 3+ SNN architectures on same dataset with unified evaluation | Low risk, high value |
| Novel spike encoding for EEG | Test wavelet-based or adaptive encoding vs. standard rate/temporal coding | Medium risk |
| Hybrid CNN-SNN architecture | Combine CNN feature extraction with SNN temporal processing | Medium risk |
| Cross-dataset SNN evaluation | Train on one MI dataset, test on another | Low risk, useful contribution |
| SNN for under-explored task | Apply SNN to P300 or SSVEP (few existing SNN papers) | Medium-High risk, high novelty |
| Energy-accuracy Pareto analysis | Systematic study of accuracy vs. energy tradeoff across architectures | Low risk, high value |
| Lightweight SNN for edge | Design minimal SNN for real-time BCI on resource-constrained hardware | Medium risk |

### 7.2 What Would NOT Be Novel

- Simply reproducing combra-lab/snn-eeg without any extension or comparison
- Using a standard CNN (EEGNet) on a standard dataset without any SNN component
- Only reviewing the literature without implementation

### 7.3 Recommended Novelty Strategy

For maximum novelty with manageable risk:

**"A Systematic Evaluation of Spiking Neural Network Architectures for Motor Imagery EEG Classification: Accuracy, Energy Efficiency, and Practical Deployment Considerations"**

This framing allows you to:
- Implement and compare 2-3 SNN approaches (e.g., basic LIF-based SNN, SCNet-style hybrid, attention-based NiSNN)
- Benchmark against established CNN baselines (EEGNet)
- Evaluate on a standard dataset (BCI IV-2a or PhysioNet)
- Analyze the accuracy-energy tradeoff (novel contribution)
- Discuss practical implications for wearable BCI deployment
- Even negative results (SNNs underperforming) are publishable and useful

---

## 8. Research Gaps and Opportunities

### 8.1 Identified Gaps

1. **P300 classification with SNNs:** Very few dedicated papers. Most P300 BCI work uses CNNs. An SNN-based P300 speller would be genuinely novel.

2. **Standardized SNN-EEG benchmarks:** No equivalent of MOABB exists for SNN-based BCI. Different papers use different preprocessing, splits, and metrics, making comparison difficult.

3. **Cross-subject SNN generalization:** Most SNN-EEG papers report within-subject results. Cross-subject transfer learning with SNNs is under-explored.

4. **Real-time SNN-BCI demonstration:** Few papers demonstrate actual real-time decoding with SNNs. Most are offline analyses.

5. **Spike encoding optimization for EEG:** The optimal method to convert EEG signals into spike trains is not settled. Rate coding, temporal coding, delta modulation, and learned encodings all exist but systematic comparisons are rare.

6. **Hybrid SNN-Transformer for EEG:** While Spiking Transformers exist for vision, their application to EEG is just beginning.

### 8.2 Low-Hanging Fruit for Undergraduate Contribution

- Systematic comparison of spike encoding methods on the same EEG dataset
- Reproducing EESCN or HR-SNN results and extending to a new dataset
- Energy analysis of SNN vs. CNN for EEG (many papers claim efficiency but few quantify it rigorously)
- Applying an existing SNN framework (snnTorch) to EEG for the first time with proper documentation

---

## 9. Network and Research Community Map

### 9.1 Key Research Groups

| Group | Affiliation | Focus | Notable Work |
|---|---|---|---|
| COMBRA Lab | Various | SNN on neuromorphic hardware for EEG | snn-eeg (Loihi deployment) |
| BCMI Lab (Wei-Long Zheng) | Shanghai Jiao Tong University | SEED dataset, EEG emotion recognition | SEED, SEED-IV, SEED-VII |
| Nikola Kasabov | AUT, New Zealand | NeuCube framework | Pioneering SNN-EEG work |
| Jason Eshraghian | UC Santa Cruz | snnTorch framework | Training SNNs tutorial paper (IEEE Proceedings) |
| SpikingJelly team | Peking University | SpikingJelly framework | Science Advances publication |
| SyNSense | Zurich | Xylo neuromorphic chip | Real-time seizure detection |

### 9.2 Key Conferences and Venues

- **IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)**
- **Frontiers in Neuroscience**
- **Neurocomputing (Elsevier)**
- **IEEE Transactions on Neural Systems and Rehabilitation Engineering**
- **NeurIPS (Neuromorphic Computing Workshop)**
- **Nature Scientific Reports**
- **Transactions on Machine Learning Research (TMLR)**

---

## 10. Confidence Assessment

| Finding | Confidence | Reasoning |
|---|---|---|
| SNNs trail ANNs in MI accuracy by 5-10% | HIGH | Multiple independent benchmarks confirm this |
| SNNs achieve 95% energy reduction on Loihi | HIGH | Published in TMLR with reproducible code |
| EESCN achieves 94.81% on DEAP | MEDIUM-HIGH | Single paper, not independently reproduced |
| The field is growing rapidly (2024-2025) | HIGH | Multiple new publications per month |
| Undergraduate feasibility | HIGH | Based on available tools, tutorials, and datasets |
| Novelty at undergraduate level | HIGH | Based on analysis of typical undergraduate projects |
| P300 SNN is a gap | HIGH | Systematic search found minimal dedicated work |
| Accuracy gap will close by 2026 | MEDIUM | Trend-based extrapolation |

---

## 11. Recommended Follow-Up Actions

1. **Download and explore PhysioNet EEGMMIDB dataset** -- no application needed, immediate access
2. **Complete snnTorch tutorials 1-5** (~1 week) to understand spike encoding and SNN training
3. **Clone and run combra-lab/snn-eeg** to reproduce baseline results
4. **Install TorchEEG** and load a sample dataset to understand EEG data format
5. **Read the EESCN paper** (Computer Methods and Programs in Biomedicine, 2024) for a complete SNN-EEG pipeline
6. **Apply for DEAP dataset access** now (takes ~1 month for approval)
7. **Draft thesis proposal** around a systematic SNN vs. CNN comparison for motor imagery EEG classification
8. **Identify supervisor expertise** -- this project sits at the intersection of neuromorphic computing and signal processing; ensure your supervisor is comfortable with the scope

---

## 12. Sources and References

### Key Papers
- [Spiking neural networks for EEG signal analysis using wavelet transform](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1652274/full) (Frontiers, 2025)
- [A lightweight spiking neural network for EEG-based motor imagery classification](https://www.sciencedirect.com/science/article/abs/pii/S0893608025006215) (Neural Networks, 2025)
- [Decoding EEG With Spiking Neural Networks on Neuromorphic Hardware](https://openreview.net/forum?id=ZPBJPGX3Bz) (TMLR, 2022)
- [EESCN: A novel spiking neural network method for EEG-based emotion recognition](https://www.sciencedirect.com/science/article/abs/pii/S016926072300593X) (CMPB, 2024)
- [A convolutional spiking neural network with adaptive coding for motor imagery classification](https://www.sciencedirect.com/science/article/abs/pii/S0925231223005933) (Neurocomputing, 2023)
- [HR-SNN: An End-to-End Spiking Neural Network for Four-class Classification Motor Imagery Brain-Computer Interface](https://ieeexplore.ieee.org/document/10511071/) (IEEE, 2024)
- [NiSNN-A: Non-iterative Spiking Neural Networks with Attention](https://arxiv.org/html/2312.05643) (arXiv, 2024)
- [Constructing lightweight and efficient spiking neural networks for EEG-based motor imagery classification](https://www.sciencedirect.com/science/article/abs/pii/S1746809424010589) (BSPC, 2024)
- [Advancing EEG based stress detection using spiking neural networks](https://www.nature.com/articles/s41598-025-10270-0) (Scientific Reports, 2025)
- [Efficient and generalizable cross-patient epileptic seizure detection through a spiking neural network](https://pmc.ncbi.nlm.nih.gov/articles/PMC10805904/) (Frontiers, 2024)
- [Real-time Sub-milliwatt Epilepsy Detection Implemented on a Spiking Neural Network Edge Inference Processor](https://arxiv.org/html/2410.16613v1) (Computers in Biology and Medicine, 2024)
- [Fractal Spiking Neural Network Scheme for EEG-Based Emotion Recognition](https://pmc.ncbi.nlm.nih.gov/articles/PMC10712674/) (2023)
- [Unleashing the potential of spiking neural networks for epileptic seizure detection](https://www.sciencedirect.com/science/article/abs/pii/S0925231224007057) (Neurocomputing, 2024)
- [Spiking neural networks for EEG signal analysis: From theory to practice](https://www.sciencedirect.com/science/article/abs/pii/S089360802501007X) (Neural Networks, 2025)
- [Review of deep learning models with Spiking Neural Networks for multimodal neuroimaging data](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1623497/full) (Frontiers, 2025)
- [An in-depth survey on Deep Learning-based Motor Imagery EEG classification](https://www.sciencedirect.com/science/article/pii/S093336572300252X) (Neural Computing and Applications, 2023)
- [Brain-computer interfaces in 2023-2024](https://onlinelibrary.wiley.com/doi/full/10.1002/brx2.70024) (Brain-X, 2025)

### Datasets
- [BCI Competition IV](https://www.bbci.de/competition/iv/)
- [PhysioNet EEGMMIDB](https://www.physionet.org/content/eegmmidb/1.0.0/)
- [DEAP Dataset](http://eecs.qmul.ac.uk/mmv/datasets/deap/)
- [SEED Dataset](https://bcmi.sjtu.edu.cn/home/seed/seed.html)

### GitHub Repositories
- [combra-lab/snn-eeg](https://github.com/combra-lab/snn-eeg)
- [jeshraghian/snntorch](https://github.com/jeshraghian/snntorch)
- [fangwei123456/spikingjelly](https://github.com/fangwei123456/spikingjelly)
- [norse/norse](https://github.com/norse/norse)
- [torcheeg/torcheeg](https://github.com/torcheeg/torcheeg)
- [TheBrainLab/Awesome-Spiking-Neural-Networks](https://github.com/TheBrainLab/Awesome-Spiking-Neural-Networks)
- [SpikingChen/SNN-Daily-Arxiv](https://github.com/SpikingChen/SNN-Daily-Arxiv)
- [SuperBruceJia/EEG-DL](https://github.com/SuperBruceJia/EEG-DL)

### Frameworks and Tools
- [snnTorch Documentation](https://snntorch.readthedocs.io/)
- [SpikingJelly Documentation](https://spikingjelly.readthedocs.io/)
- [TorchEEG Documentation](https://torcheeg.readthedocs.io/)
- [Open Neuromorphic](https://open-neuromorphic.org/)
- [SNN Framework Benchmarks](https://open-neuromorphic.org/blog/spiking-neural-network-framework-benchmarking/)
