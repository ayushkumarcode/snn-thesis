# SNN-based EEG classification for brain-computer interfaces

i looked into whether SNNs could be used for EEG-based BCIs, and honestly it's a pretty active area. the field has picked up a lot since 2023, with new architectures, benchmarks, and open-source code coming out regularly. the basic pitch is that SNNs are biologically plausible (the brain itself uses spikes), energy efficient (up to 95% less energy than DNNs on neuromorphic hardware), and naturally suited to temporal signals like EEG.

that said, SNNs currently trail state-of-the-art CNNs and transformers by about 3-10 percentage points on most EEG benchmarks, though the gap is closing. the main tasks people are tackling with SNN-EEG include motor imagery classification, emotion recognition, seizure detection, stress detection, and SSVEP classification. there's plenty of public datasets (BCI Competition IV-2a/2b, PhysioNet EEGMMIDB, DEAP, SEED) and frameworks (snnTorch, SpikingJelly, Norse, combra-lab/snn-eeg) to make this doable for an undergrad who knows Python/PyTorch but not neuroscience.

---

## 1. what people are doing with SNN-based EEG classification

### motor imagery (MI) classification

this is the most studied task by far. subjects imagine moving body parts (left hand, right hand, feet, tongue) and you classify the EEG patterns over the motor cortex.

key papers:
- **SCNet** (2023): CNN feature extraction + SNN biological interpretability with adaptive coding and surrogate gradient learning. Tested on PhysioNet, BCI IV-2a, and BCI IV-2b. Beats prior SNN methods.
- **HR-SNN** (2024): End-to-end SNN getting 77.58% average on BCI IV-2a (4-class), beating all compared SNN models. On PhysioNet: 67.24% (global) and 74.95% (transfer learning).
- **NiSNN-A** (2024): Non-iterative SNN with attention for motor imagery. Combines accuracy gains with energy reduction.
- **LENet/RDSNN** (2024): Lightweight SNN getting 73.65% on PhysioNet, 81.75% on BCI IV-2a, 84.56% on BCI IV-2b.
- **Lightweight SNN** (2025, ScienceDirect): Within-subject and cross-subject experiments on three public datasets show the SNN beats classical CNN-based models.
- **combra-lab SNN-EEG** (2022, TMLR): Deployed on Intel Loihi, consuming 95% less energy than DNNs on NVIDIA Jetson TX2 with similar accuracy.

### emotion recognition

second-most studied task. subjects watch emotional stimuli while EEG is recorded, and you classify valence, arousal, dominance.

key papers:
- **EESCN** (2024, Computer Methods and Programs in Biomedicine): Gets 94.56% (valence), 94.81% (arousal), 94.73% (dominance) on DEAP and 79.65% on SEED-IV. Faster and uses less memory than prior SNN methods.
- **Fractal-SNN** (2023): Uses multi-scale temporal-spectral-spatial information. Tested on DREAMER, DEAP, SEED-IV, and MPED.
- **NeuroSense** (Tan et al., 2021): 78.97%/67.76% (arousal/valence) on DEAP.
- **Bidirectional SNN** (Alzhrani et al., 2021): 94.83% accuracy on DREAMER.
- **BISNN** (2025): Bio-information-fused SNN for emotion recognition.

### epilepsy/seizure detection

growing application area with obvious clinical motivation.

key papers:
- **EESNN** (2024): Recurrent spiking convolution structure, energy reduction by several orders of magnitude vs ANNs.
- **Spiking Conformer** (2024): Trained on raw EEG, no preprocessing needed. 10x fewer operations than non-spiking equivalent.
- **SyNSense Xylo deployment** (2024): Real-time sub-milliwatt epilepsy detection on a neuromorphic edge processor.
- **Cross-patient SNN** (2024, Frontiers in Neuroscience): Efficient cross-patient seizure detection.

### other tasks

- **Stress detection:** CSNN (2025, Scientific Reports) gets 98.75% with 10-fold CV and F1 of 98.60%.
- **SSVEP:** Event-driven SNN (2024, IEEE) using empirical mode decomposition and CCA with excitation-inhibition balanced SNN.
- **P300:** SNNs have been used for P300 signal reconstruction and data augmentation (2021, Frontiers), but dedicated P300 SNN classifiers are rare. this is a potential gap.
- **Sleep staging:** Hybrid SNN (HSNN) demonstrated for automatic sleep staging.
- **Situational awareness:** SNN with SCTN neurons (2024, Applied Sciences).

### how architectures have evolved

| Generation | Approach | Example |
|---|---|---|
| Early (pre-2020) | Reservoir/NeuCube + STDP | NeuCube framework |
| Mid (2020-2022) | Surrogate gradient SNNs | combra-lab SNN-EEG |
| Recent (2023-2024) | Hybrid CNN-SNN, attention | SCNet, HR-SNN, NiSNN-A |
| Emerging (2024-2025) | Spiking Transformers | Spikeformer, Spiking Conformer |
| Frontier (2025+) | Lightweight + edge deployment | LENet, Xylo-based systems |

---

## 2. accuracy comparison: SNNs vs conventional approaches

### motor imagery (BCI Competition IV Dataset 2a, 4-class)

| Method | Type | Accuracy (%) | Year |
|---|---|---|---|
| EEGEncoder (Transformer) | ANN | 86.46 | 2024 |
| CIACNet (Attention CNN) | ANN | 85.15 | 2024 |
| SNA-MHC (custom SNN+Attn) | SNN | 92.80* | 2024 |
| RDSNN (Lightweight SNN) | SNN | 81.75 | 2024 |
| HR-SNN (End-to-End SNN) | SNN | 77.58 | 2024 |
| CNN1D_MF | ANN | 69.20 | 2023 |
| DFBRTS | ANN | 78.16 | 2024 |

*SNA-MHC's 92.80% is an outlier -- might use different evaluation protocols. most SNN results cluster around 75-82% on this benchmark.

### motor imagery (PhysioNet EEGMMIDB)

| Method | Type | Accuracy (%) | Year |
|---|---|---|---|
| Novel DL approach | ANN | 95.70 | 2025 |
| Optimized DL + DWT | ANN | 97.05 | 2024 |
| HR-SNN (transfer) | SNN | 74.95 | 2024 |
| HR-SNN (global) | SNN | 67.24 | 2024 |
| RDSNN | SNN | 73.65 | 2024 |
| combra-lab SNN-EEG | SNN | ~similar to DNN* | 2022 |

*combra-lab reports "similar classification performance" to DNNs with 95% less energy.

### emotion recognition (DEAP dataset)

| Method | Type | Valence (%) | Arousal (%) | Year |
|---|---|---|---|---|
| Graph CNN + Dual Attention | ANN | ~90+ | ~90+ | 2024 |
| EESCN | SNN | 94.56 | 94.81 | 2024 |
| NeuroSense | SNN | 67.76 | 78.97 | 2021 |
| Bidirectional SNN (DREAMER) | SNN | 94.83 (overall) | -- | 2021 |

### stress detection

| Method | Type | Accuracy (%) | Year |
|---|---|---|---|
| CSNN | SNN | 98.75 | 2025 |
| Hybrid SNN | SNN | 94.00 | 2024 |

### the takeaway on accuracy

the gap between SNNs and ANNs depends a lot on the task:
- **emotion recognition and stress detection:** SNNs can match or beat ANNs (EESCN at 94.81% is competitive with SOTA on DEAP)
- **motor imagery (4-class):** SNNs still trail top ANNs by ~5-10pp (77-82% vs 85-87% for best ANNs), though the gap is narrowing
- **seizure detection:** SNNs get comparable accuracy with orders-of-magnitude lower energy

the real argument for SNNs isn't just accuracy -- it's the accuracy-energy tradeoff. an SNN at 80% accuracy that uses 1/20th the energy of an 85% CNN might be the better choice for a wearable BCI.

---

## 3. available datasets

### motor imagery

| Dataset | Subjects | Channels | Classes | Sampling Rate | Access | Notes |
|---|---|---|---|---|---|---|
| **BCI Competition IV-2a** | 9 | 22 EEG + 3 EOG | 4 (left hand, right hand, feet, tongue) | 250 Hz | [BBCI website](https://www.bbci.de/competition/iv/) / [Kaggle](https://www.kaggle.com/datasets/thngdngvn/bci-competition-iv-data-sets-2a) | Most popular MI benchmark (31+ studies). 288 trials/subject across 2 sessions. |
| **BCI Competition IV-2b** | 9 | 3 EEG + 3 EOG | 2 (left hand, right hand) | 250 Hz | [BBCI website](https://www.bbci.de/competition/iv/) | Second most popular (14+ studies). |
| **PhysioNet EEGMMIDB** | 109 | 64 | 4 (open/close fists, imagine fists/feet) | 160 Hz | [PhysioNet](https://www.physionet.org/content/eegmmidb/1.0.0/) | Freely downloadable, no application needed. Largest free MI dataset. |
| **BNCI Horizon 2020** | Various | Various | Various | Various | [BNCI database](https://bnci-horizon-2020.eu/database/data-sets) | Collection of multiple BCI datasets. |

### emotion recognition

| Dataset | Subjects | Channels | Classes | Stimuli | Access | Notes |
|---|---|---|---|---|---|---|
| **DEAP** | 32 | 32 EEG + 8 peripheral | Valence/Arousal/Dominance (continuous) | 40 music videos (1 min each) | [DEAP official](http://eecs.qmul.ac.uk/mmv/datasets/deap/) | Needs university email application (~1 month wait). 512 Hz, preprocessed files at 128 Hz available. |
| **SEED** | 15 | 62 EEG | 3 (negative, neutral, positive) | Film clips | [BCMI Lab](https://bcmi.sjtu.edu.cn/home/seed/seed.html) | 200 Hz preprocessed. 3 sessions/subject (~1 week apart). |
| **SEED-IV** | 15 | 62 EEG | 4 (happy, sad, fear, neutral) | Film clips | [BCMI Lab](https://bcmi.sjtu.edu.cn/home/seed/index.html) | Extension of SEED with 4 emotions. |
| **SEED-VII** | -- | -- | 6 basic emotions + continuous | Multimodal | [BCMI Lab](https://bcmi.sjtu.edu.cn/home/seed/) | Newest, continuous labels. |
| **DREAMER** | 23 | 14 EEG | Valence/Arousal/Dominance | Film clips | Public | Lower channel count, good for lightweight models. |

