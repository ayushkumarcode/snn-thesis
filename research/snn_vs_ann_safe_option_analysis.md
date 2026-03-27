# SNN vs ANN Image Classification: The "Safe Option" Thesis

looked into whether doing a straight SNN vs ANN comparison on image classification would work as a thesis. the verdict: it's viable, but you have to be deliberate about scoping or it'll be too trivial.

---

## 1. The Standard Benchmarks

### Three Standard Datasets

| Dataset | Classes | Image Size | Complexity | Role in SNN Research |
|---------|---------|------------|------------|---------------------|
| MNIST | 10 digits | 28x28 grayscale | Trivial | Baseline sanity check; considered "solved" |
| Fashion-MNIST | 10 clothing types | 28x28 grayscale | Low-Moderate | Drop-in MNIST replacement; slightly more realistic |
| CIFAR-10 | 10 object classes | 32x32 RGB | Moderate | Real test of SNN capability; where gaps show up |

other datasets worth considering for a stronger project:
- **N-MNIST**: neuromorphic MNIST (event-camera) -- but research shows N-MNIST can be classified without temporal info, so it doesn't really test SNN temporal advantages ([Iyer et al., 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8027306/))
- **CIFAR10-DVS**: true event-stream CIFAR-10 -- this is where SNNs genuinely shine
- **DVS128 Gesture**: temporal gesture recognition -- plays to SNN strengths

### Current ANN/CNN SOTA (for comparison)

| Dataset | Simple CNN | Best CNN/ViT | Notes |
|---------|-----------|-------------|-------|
| MNIST | ~99.5% | 99.84% | Effectively saturated |
| Fashion-MNIST | ~93-95% | 96.7% (best CNN) | ViT approaches exceed 96% |
| CIFAR-10 | ~93-94% | 99.5%+ (ViT/AutoML) | Massive architecture-dependent range |
