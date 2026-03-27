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
| SNN (surrogate gradient, LIF) | 98.1-98.7% | **0.0-0.5%** | Nearly closed |
| SNN (STDP unsupervised) | ~95-97% | 1-3% | Bio-plausible but weaker |
| SNN (Forward-Forward) | 98.69% | **~0%** | Very recent (2025) |

**Verdict**: The gap on MNIST is effectively closed. Surrogate-gradient-trained SNNs match ANNs. This is a solved problem -- including it is fine for completeness but it alone proves nothing.

Sources: [Forward-Forward SNN](https://arxiv.org/html/2502.20411v1), [Sigma-delta neuron benchmarks](https://arxiv.org/pdf/2501.15547)

### Fashion-MNIST

| Method | Accuracy | Gap vs ANN | Notes |
|--------|----------|-----------|-------|
| CNN baseline | ~93-95% | -- | Standard CNN |
| SNN (Sa-SNN, attention) | 94.13% | **~0-1%** | Best SNN result |
| SNN (surrogate gradient) | ~90-92% | 2-4% | Typical implementation |
| SNN (STDP-based) | ~87-89% | 5-8% | Unsupervised methods |
| SNN (Forward-Forward) | 90.27% | ~3-5% | Recent but limited |

**Verdict**: A meaningful gap exists (2-5% for typical implementations). The gap narrows with sophisticated architectures like attention-based SNNs. This dataset is more informative than MNIST for a comparison study.

Sources: [Sa-SNN](https://peerj.com/articles/cs-2549.pdf), [GLSNN](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2020.576841/full)

### CIFAR-10

| Method | SNN Accuracy | ANN Equiv. | Gap | Time Steps |
|--------|-------------|-----------|-----|-----------|
| VGG16 (ANN-SNN conversion) | 95.91% | ~96.5% | **~0.6%** | T=many |
| ResNet20 (conversion) | 96.64% | ~97% | **~0.4%** | T=varies |
| STAA-SNN (direct, CVPR 2025) | **97.14%** | ~97.5% | **~0.4%** | T=4 |
| ResNet19 (surrogate gradient) | 95.44% | ~96% | **~0.6%** | T~3 |
| VGG (direct, few steps) | 83-93% | ~93-94% | **1-10%** | T=1-4 |
| Simple SNN (snnTorch tutorial-level) | ~85-90% | ~93% | **3-8%** | T=varies |

**Verdict**: This is where the comparison becomes genuinely interesting. The gap ranges from nearly zero (with state-of-the-art methods on large architectures) to 3-10% (with simpler implementations an undergraduate would actually build). The accuracy gap is heavily dependent on:
1. Architecture choice (VGG vs ResNet vs simple CNN)
2. Number of time steps
3. Training method (conversion vs direct training)
4. Encoding scheme

Sources: [STAA-SNN CVPR 2025](https://arxiv.org/pdf/2503.02689), [ANN-SNN Conversion](https://proceedings.mlr.press/v202/jiang23a/jiang23a.pdf), [Training by Differentiation on Spike](https://openaccess.thecvf.com/content/CVPR2022/papers/Meng_Training_High-Performance_Low-Latency_Spiking_Neural_Networks_by_Differentiation_on_Spike_CVPR_2022_paper.pdf)

### Summary Accuracy Gap Table

| Dataset | Typical UG SNN Gap | Best Known SNN Gap | Status |
|---------|-------------------|-------------------|--------|
| MNIST | 0-1% | ~0% | Solved -- not interesting alone |
| Fashion-MNIST | 2-5% | ~0-1% | Moderate interest |
| CIFAR-10 | 3-8% | ~0.4% | Most interesting for study |

---

## 3. What Would Make This More Than "Running snnTorch Tutorials"

This is the critical question. Here is an honest assessment.

### What the snnTorch tutorials already cover (your baseline risk)

The snnTorch documentation provides 8+ tutorials that already demonstrate:
- Spike encoding (rate, latency, delta) -- [Tutorial 1](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html)
- Training a fully-connected SNN on MNIST -- [Tutorial 5](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html)
- Training a convolutional SNN on MNIST -- [Tutorial 6](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html)
- Neuromorphic datasets with Tonic -- [Tutorial 7](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_7.html)
- Rate vs latency coded loss functions

**If your thesis is "I ran tutorials 5 and 6, then also trained a CNN on the same data, and compared accuracies" -- that is NOT a thesis. It is a lab exercise.** A high school student has already published this exact comparison in the [National High School Journal of Science](https://nhsjs.com/2024/advancements-in-image-classification-comparing-spiking-convolutional-and-artificial-neural-networks/), comparing SNNs, CNNs, and ANNs on MNIST, CIFAR-10, and N-MNIST.

### What elevates it beyond tutorials

To cross the threshold from "lab exercise" to "thesis," you need at least ONE of:

1. **A systematic study with controlled variables** -- not just "does it work" but "how do specific factors affect the accuracy-efficiency tradeoff across conditions"
2. **An original experimental design** -- testing a hypothesis that is not already answered in existing literature
3. **A novel combination** -- applying known techniques to a new context, or combining techniques not previously combined
4. **A quantitative analysis dimension** that tutorials do not cover (energy estimation, robustness, Pareto analysis)

---

## 4. Specific Angles That Could Add Value

### Angle A: Systematic Encoding Scheme Comparison (Moderate Value, High Feasibility)

**What it is**: Compare rate coding, latency (TTFS) coding, delta coding, phase coding, and burst coding on the same architectures and datasets, with controlled variables.

**Why it adds value**: The [Frontiers paper by Park et al.](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.638474/full) did this for a limited set of encodings. You could extend it with:
