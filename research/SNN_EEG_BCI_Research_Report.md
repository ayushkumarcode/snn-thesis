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
