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
