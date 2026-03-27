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

Sources: [Papers With Code MNIST](https://paperswithcode.com/sota/image-classification-on-mnist), [Papers With Code CIFAR-10](https://paperswithcode.com/sota/image-classification-on-cifar-10), [Fashion-MNIST SOTA](https://www.mdpi.com/2227-7390/12/20/3174)

---

## 2. The Accuracy Gap

this is the core question. here's where things stand.

### MNIST

| Method | Accuracy | Gap vs ANN | Notes |
|--------|----------|-----------|-------|
| ANN baseline (same arch) | 98.23% | -- | Simple FC network |
| SNN (surrogate gradient, LIF) | 98.1-98.7% | **0.0-0.5%** | Nearly closed |
| SNN (STDP unsupervised) | ~95-97% | 1-3% | Bio-plausible but weaker |
| SNN (Forward-Forward) | 98.69% | **~0%** | Very recent (2025) |

gap is effectively closed on MNIST. surrogate-gradient SNNs match ANNs. solved problem -- fine for completeness but proves nothing on its own.

### Fashion-MNIST

| Method | Accuracy | Gap vs ANN | Notes |
|--------|----------|-----------|-------|
| CNN baseline | ~93-95% | -- | Standard CNN |
| SNN (Sa-SNN, attention) | 94.13% | **~0-1%** | Best SNN result |
| SNN (surrogate gradient) | ~90-92% | 2-4% | Typical implementation |
| SNN (STDP-based) | ~87-89% | 5-8% | Unsupervised methods |
| SNN (Forward-Forward) | 90.27% | ~3-5% | Recent but limited |

meaningful gap exists (2-5% for typical implementations). narrows with fancy architectures like attention-based SNNs. more informative than MNIST.

### CIFAR-10

| Method | SNN Accuracy | ANN Equiv. | Gap | Time Steps |
|--------|-------------|-----------|-----|-----------|
| VGG16 (ANN-SNN conversion) | 95.91% | ~96.5% | **~0.6%** | T=many |
| ResNet20 (conversion) | 96.64% | ~97% | **~0.4%** | T=varies |
| STAA-SNN (direct, CVPR 2025) | **97.14%** | ~97.5% | **~0.4%** | T=4 |
| ResNet19 (surrogate gradient) | 95.44% | ~96% | **~0.6%** | T~3 |
| VGG (direct, few steps) | 83-93% | ~93-94% | **1-10%** | T=1-4 |
| Simple SNN (snnTorch tutorial-level) | ~85-90% | ~93% | **3-8%** | T=varies |

this is where the comparison gets genuinely interesting. gap ranges from nearly zero (SOTA methods, big architectures) to 3-10% (simpler stuff an undergrad would actually build). heavily dependent on architecture choice, time steps, training method, and encoding.

### Summary

| Dataset | Typical Undergrad SNN Gap | Best Known SNN Gap | Status |
|---------|-------------------|-------------------|--------|
| MNIST | 0-1% | ~0% | Solved -- not interesting alone |
| Fashion-MNIST | 2-5% | ~0-1% | Moderate interest |
| CIFAR-10 | 3-8% | ~0.4% | Most interesting |

---

## 3. What Would Make This More Than Running Tutorials
