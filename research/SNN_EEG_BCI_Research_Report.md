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
