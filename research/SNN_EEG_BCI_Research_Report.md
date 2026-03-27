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

