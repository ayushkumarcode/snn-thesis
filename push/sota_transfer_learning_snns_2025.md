# State-of-the-Art in Transfer Learning for Spiking Neural Networks (2024-2026)

**Research Report for COMP30040 Thesis**
**Date: 5 March 2026**
**Context: PANNs+SNN head achieves 92.50% on ESC-50; scratch SNN: 47.15%; gap collapse from 16.7pp to 0.95pp with equal features**

---

## Executive Summary

This report surveys the state-of-the-art in transfer learning for Spiking Neural Networks (SNNs), covering ANN-to-SNN conversion, knowledge distillation, and hybrid ANN-SNN architectures with a focus on 2024-2026 publications. The field has undergone a dramatic acceleration: ANN-to-SNN conversion has matured to the point where single-timestep lossless conversion is now possible for vision models (ICML 2025, CVPR 2025), knowledge distillation from ANN teachers to SNN students has produced at least 8 distinct methods in 2024-2025 alone, and hybrid ANN-SNN architectures with pretrained ANN feature extractors and SNN classifier heads have emerged as a practical deployment paradigm for neuromorphic hardware.

Critically, **the thesis finding that the SNN-ANN accuracy gap collapses from 16.7pp to 0.95pp when both receive equal-quality features is consistent with -- but extends -- the broader literature**. The closest parallel is Spiking Vocos (2025), which achieves ANN-comparable audio quality at 14.7% energy using self-architectural distillation. The SAFE paper (ICLR 2025 submission) uses a CNN feature extractor + SNN classifier for audio fidelity and finds comparable performance with fewer parameters. However, **no prior work has demonstrated this gap-collapse phenomenon specifically for environmental sound classification on ESC-50**, making the thesis finding genuinely novel.

The report identifies zero prior works combining PANNs (or any AudioSet-pretrained model) with an SNN classifier head for ESC-50. This confirms the thesis approach occupies a clear gap in the literature.

---

## 1. ANN-to-SNN Conversion: Recent Advances (2024-2026)

### 1.1 Training-Free Conversion Methods

The dominant trend in 2024-2025 is training-free conversion of pretrained ANNs to SNNs, eliminating the need for SNN-specific training entirely.

