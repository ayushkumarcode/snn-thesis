# Spiking Neural Networks for Audio Processing: Keyword Spotting & Speech Command Recognition

## Comprehensive Research Report -- February 2025

---

## Executive Summary

Spiking Neural Networks (SNNs) for audio keyword spotting and speech command recognition have matured significantly in 2024-2025, reaching a point where they are a viable and compelling undergraduate thesis topic. The accuracy gap between SNNs and conventional ANNs has narrowed dramatically: state-of-the-art SNNs now achieve 96.9% on Google Speech Commands V2 (35-class), approaching the ANN ceiling of ~97-98%. Multiple open-source frameworks (snnTorch, SpikingJelly, sparch) provide well-documented starting points, and several complete implementations exist on GitHub with 300-600 lines of core Python code. The energy efficiency argument is substantiated by hardware benchmarks showing 10-200x lower energy per inference on neuromorphic hardware (Intel Loihi) versus conventional processors. This is a well-scoped, feasible thesis project with clear benchmarks, available code, and a strong research narrative around energy-efficient edge AI.

---

## 1. SNN vs ANN Accuracy on Google Speech Commands Dataset

### 1.1 Current State of the Art (as of early 2025)

| Model | Type | Dataset (Task) | Accuracy | Parameters | Year | Code Available |
|-------|------|----------------|----------|------------|------|----------------|
| **SpikCommander** | SNN (Spiking Transformer) | GSC V2 (35-class) | **96.92%** | 2.13M | 2025 | Yes |
| **SpikCommander** | SNN (Spiking Transformer) | GSC V2 (35-class) | **97.08%** (T=200) | 2.13M | 2025 | Yes |
