# SNN Accuracy Improvement Techniques: Exhaustive Research Report

**Date:** 2026-03-25
**Context:** ESC-50 environmental sound classification, Conv SNN (622K params), 47.15% SNN vs 63.85% ANN
**Objective:** Techniques to close or significantly narrow the 16.7pp SNN-ANN accuracy gap

---

## Executive Summary

This report catalogues every promising technique found across ~40 recent papers (2022-2026) that could improve our SNN's accuracy on ESC-50. The techniques fall into three tiers of expected impact:

**Tier 1 -- High Impact, Moderate Effort (likely +5-15pp):**
1. ANN-to-SNN Conversion (preserve ~60-63% of ANN's 63.85%)
2. Knowledge Distillation from ANN teacher (typically +3-11pp)
3. Hybrid Training: ANN init + SNN fine-tune (typically +5-10pp)
4. TET Loss Function (reported +10pp on DVS-CIFAR10, consistently +2-5pp elsewhere)
5. Learnable Neuron Parameters (PLIF/GLIF) (+1-3pp)

**Tier 2 -- Moderate Impact, Moderate Effort (likely +2-5pp):**
6. Temporal Batch Normalization (TEBN/tdBN/BNTT)
7. Ternary Spikes (negative spikes)
8. Residual/Skip Connections (SEW-ResNet style)
9. Channel/Temporal Attention (TCJA-SNN)
10. Advanced Surrogate Gradients (AdaLi, learnable)

**Tier 3 -- Lower/Uncertain Impact, Higher Effort:**
11. NAS for SNNs
