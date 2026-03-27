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
