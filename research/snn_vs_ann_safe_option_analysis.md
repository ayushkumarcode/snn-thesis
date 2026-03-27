# SNN vs ANN Image Classification Comparison: "Safe Option" Thesis Analysis

**Date**: 2026-02-25
**Purpose**: Evaluate feasibility, risks, and differentiation strategies for the "safe, guaranteed-to-work" undergraduate thesis option
**Verdict**: Viable but requires deliberate scoping to avoid being trivial

---

## 1. Standard Benchmarks and Current Accuracy Landscape

### The Three Standard Datasets

| Dataset | Classes | Image Size | Complexity | Role in SNN Research |
|---------|---------|------------|------------|---------------------|
| MNIST | 10 digits | 28x28 grayscale | Trivial | Baseline sanity check; considered "solved" |
| Fashion-MNIST | 10 clothing types | 28x28 grayscale | Low-Moderate | Drop-in MNIST replacement; slightly more realistic |
| CIFAR-10 | 10 object classes | 32x32 RGB | Moderate | Real test of SNN capability; where gaps emerge |

Additional relevant datasets for stronger projects:
- **N-MNIST**: Neuromorphic version of MNIST (event-camera recorded) -- but note: research shows N-MNIST can be classified without temporal information, so it does not truly test SNN temporal advantages ([Iyer et al., 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8027306/))
- **CIFAR10-DVS**: True event-stream version of CIFAR-10 -- this is where SNNs genuinely shine
- **DVS128 Gesture**: Temporal gesture recognition -- plays to SNN strengths

### Current ANN/CNN State-of-the-Art (for comparison baseline)

| Dataset | Simple CNN | Best CNN/ViT | Notes |
|---------|-----------|-------------|-------|
| MNIST | ~99.5% | 99.84% | Effectively saturated |
| Fashion-MNIST | ~93-95% | 96.7% (best CNN) | ViT approaches exceed 96% |
| CIFAR-10 | ~93-94% | 99.5%+ (ViT/AutoML) | Massive architecture-dependent range |

Sources: [Papers With Code MNIST](https://paperswithcode.com/sota/image-classification-on-mnist), [Papers With Code CIFAR-10](https://paperswithcode.com/sota/image-classification-on-cifar-10), [State-of-the-Art Fashion-MNIST](https://www.mdpi.com/2227-7390/12/20/3174)

---

## 2. The Accuracy Gap: SNN vs ANN on Each Benchmark

This is the core question. Here is a consolidated view of where things stand as of early 2025.

### MNIST

| Method | Accuracy | Gap vs ANN | Notes |
|--------|----------|-----------|-------|
| ANN baseline (same arch) | 98.23% | -- | Simple FC network |
