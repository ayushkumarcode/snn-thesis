# Multimodal Spiking Neural Networks: Comprehensive Research Report

**Date**: 2026-02-25
**Research Focus**: Combining different data types (vision, audio, event camera, IMU) in a single spiking neural network
**Purpose**: Assess feasibility for an undergraduate thesis project

---

## Executive Summary

Multimodal SNNs -- combining different sensory data types within a single spiking neural network -- represent an **active and rapidly growing research area** that has seen significant acceleration since 2023. The field is no longer purely theoretical: multiple working implementations exist for audio-visual classification, event camera + RGB fusion, and sensor fusion for robotics. Critically, a paper published in August 2024 demonstrates *exactly* the simplified version proposed (MNIST digits + audio digits fusion), achieving 98.43% accuracy. This means a multimodal SNN thesis project is **achievable at the undergraduate level**, provided the scope is carefully bounded. The area has enough existing work to build upon but enough open problems to contribute meaningfully.

**Verdict: This is feasible as an undergraduate project. It sits at the boundary between "well-explored for PhDs" and "emerging for undergrads" -- an ideal sweet spot for a thesis that can demonstrate both competence and novelty.**

---

## 1. Has Anyone Combined Vision + Audio in an SNN?

**Yes -- this is now an established sub-field with at least 6 major papers from 2023-2025.**

### Key Papers and Systems

| Paper/System | Year | Datasets | Accuracy | Key Innovation |
|---|---|---|---|---|
| **SMMT** (Spiking Multi-Modal Transformer) | 2023 | CREMA-D, UrbanSound8K-AV | ~66% (CREMA-D) | Spiking Cross-Attention (SCA) mechanism for audio-visual fusion |
| **MISNet** (Multimodal Interaction Spiking Network) | 2024 | 5 audio-visual datasets | Competitive | MLIF neuron that synchronizes audiovisual spikes in a single neuron |
