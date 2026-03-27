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

this is the critical question. being honest here.

### What snnTorch tutorials already cover (the baseline risk)

snnTorch provides 8+ tutorials demonstrating:
- spike encoding (rate, latency, delta) -- [Tutorial 1](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html)
- training FC SNN on MNIST -- [Tutorial 5](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html)
- training conv SNN on MNIST -- [Tutorial 6](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html)
- neuromorphic datasets with Tonic -- [Tutorial 7](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_7.html)
- rate vs latency coded loss functions

**if the thesis is just "i ran tutorials 5 and 6, trained a CNN on the same data, compared accuracies" -- that's NOT a thesis, it's a lab exercise.** a high school student already published this exact comparison in the [National High School Journal of Science](https://nhsjs.com/2024/advancements-in-image-classification-comparing-spiking-convolutional-and-artificial-neural-networks/), comparing SNNs, CNNs, and ANNs on MNIST, CIFAR-10, and N-MNIST.

### What Elevates It

to cross from "lab exercise" to "thesis," you need at least ONE of:

1. **Systematic study with controlled variables** -- not just "does it work" but "how do specific factors affect the accuracy-efficiency tradeoff"
2. **Original experimental design** -- testing a hypothesis not already answered
3. **Novel combination** -- known techniques in new context, or techniques not previously combined
4. **Quantitative analysis dimension** that tutorials don't cover (energy estimation, robustness, Pareto analysis)

---

## 4. Angles That Could Add Value

### Angle A: Encoding Scheme Comparison (moderate value, high feasibility)

compare rate, latency, delta, phase, burst coding on same architectures and datasets with controlled variables. the [Frontiers paper by Park et al.](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.638474/full) did this for limited encodings. you could extend with more encodings, additional datasets, snnTorch-specific implementations (reproducibility), multi-dimensional comparison (accuracy, spike count, convergence, noise robustness).

risk: partially done. contribution would be breadth and reproducibility, not individual novelty.

estimated time: 4-6 weeks after setup.

### Angle B: Hyperparameter Sensitivity (moderate value, high feasibility)

systematic search over SNN-specific hyperparams: membrane decay (tau), firing threshold, time steps, surrogate gradient function (arctan, sigmoid, fast-sigmoid), and interactions.

this matters because SNN training is really sensitive to these. a firing threshold of 1.0 vs 0.25 can swing accuracy from 96% to 41% ([Bojkovic et al., 2024](https://proceedings.mlr.press/v238/bojkovic24a/bojkovic24a.pdf)). no undergrad-accessible guide to tuning these across datasets exists.

thesis-worthy if you: plot Pareto frontiers of accuracy vs spike count, identify which hyperparams matter most, provide practical guidelines.

estimated time: 3-5 weeks.

### Angle C: Energy/Efficiency via NeuroBench (high value, moderate feasibility)

measure computational cost using proxy metrics: SynOps, spike counts, MAC vs AC operations, memory accesses, using [NeuroBench](https://neurobench.readthedocs.io/en/latest/).

the paper ["Are SNNs Really More Energy-Efficient Than ANNs?"](https://cea.hal.science/cea-03852141/file/Are_SNNs_Really_More_Energy_Efficient_Than_ANNs__An_In_Depth_Hardware_Aware_Study_versionacceptee.pdf) showed SNN energy advantage is conditional and often overstated. replicating this at undergrad level using NeuroBench would be genuinely valuable. key claim to test: SNNs need >93% spike sparsity with VGG16 at T=6 to be more efficient than ANNs.

this goes beyond accuracy to address the *actual claimed advantage* of SNNs. could produce Pareto curves. challenges a common claim with empirical data.

estimated time: 4-6 weeks.

### Angle D: Adversarial Robustness (high value, moderate feasibility)

compare SNN and ANN vulnerability to FGSM, PGD, and natural noise/corruption. SNNs show inherent robustness advantages ([ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740392.pdf)) -- 2-4.6% improvement in adversarial accuracy over equivalent ANNs on CIFAR-10 with VGG/ResNet. somewhat under-explored at undergrad level. practical relevance for safety-critical applications.

estimated time: 3-5 weeks.

### Angle E: Architecture-Controlled Fair Comparison (moderate value, high feasibility)

build identical architectures (same layers, same params) for ANN and SNN. follows [Deng et al., "Rethinking the performance comparison between SNNs and ANNs"](https://web.ece.ucsb.edu/~lip/publications/SNN-vs-ANN-NeuralNetworks2020.pdf). most naive comparisons are unfair -- different architectures, training regimes, hyperparameter budgets. a rigorous controlled comparison is more scientifically valuable.

key finding from literature: "On ANN-oriented workloads, SNNs fail to beat their ANN counterparts; while on SNN-oriented workloads, SNNs can fully perform better." testing this claim would be worthwhile.

estimated time: 3-4 weeks.

### Angle F: Time Steps Pareto Analysis (high value, moderate feasibility)

systematically vary T=1 to T=32+, plot three-way tradeoff between accuracy, inference latency, and spike sparsity. references framework from ["Exploring Tradeoffs in SNNs"](https://direct.mit.edu/neco/article/35/10/1627/117019/Exploring-Trade-Offs-in-Spiking-Neural-Networks). directly answers: "how many time steps do i actually need?" answer varies by dataset, architecture, encoding.

estimated time: 3-4 weeks.

---

## 5. How Many Projects Have Already Done This?

### Direct matches found

1. **High school student paper (2024)**: published in [NHSJS](https://nhsjs.com/2024/advancements-in-image-classification-comparing-spiking-convolutional-and-artificial-neural-networks/) -- compared SNN, CNN, ANN on MNIST, CIFAR-10, N-MNIST. found SNN matched accuracy but consumed 142% more power and 128% more memory on commercial hardware.

2. **Virginia Tech class project (2020)**: [GitHub](https://github.com/oshears/adv-ml-2020-snn-project) -- compared 784-100 ANN to 784-100 SNN on MNIST. grad course project.

3. **UNSW Bachelor Honours (2022)**: biologically-inspired ANNs with spiking neurons, benchmarking against traditional networks.

4. **KCL BSc (2018)**: [GitHub](https://github.com/LucaMozzo/SpikingNeuralNetwork) -- SNN for MNIST in C++ from scratch. more implementation than comparison focused.

5. **Multiple Kaggle/GitHub repos**: numerous basic SNN MNIST implementations.

### How Saturated Is This?

**basic "SNN vs ANN accuracy on MNIST"**: HIGHLY saturated. done by high school students, undergrads, grad students, academic papers.

**"SNN vs ANN on CIFAR-10 with systematic analysis"**: less saturated but still significant academic coverage.

**risk of being too generic**: HIGH if you only compare accuracy. MODERATE if you add one analytical dimension (energy, robustness). LOW if you add two or more dimensions with controlled methodology.

---

## 6. Strong vs Weak Version

### WEAK (risky for grade)

- run snnTorch tutorials on MNIST
- train CNN on MNIST
- compare accuracy numbers
- maybe Fashion-MNIST
- report: "SNNs got 98%, ANNs got 99%, so ANNs are slightly better"
- no energy analysis, no controlled variables, no encoding comparison, default hyperparams

this is weak because a high school student already published it. no experimental design beyond "run and report." easily dismissed as tutorial replication.

red flags: only MNIST, one encoding, default hyperparams, no statistics, no energy analysis.

### MODERATE (solid 2:1)

- all three datasets
- two+ architectures (FC and conv)
- fair comparison (matched params)
- at least two encoding schemes
- some hyperparameter sensitivity
- spike count / SynOps reported
- multiple trials with error bars
- structured analysis of accuracy gap

### STRONG (first-class potential, publishable)

- title like: "A Systematic Multi-Dimensional Comparison of Spiking and Artificial Neural Networks: Accuracy, Efficiency, and Robustness Trade-offs"
- three datasets plus neuromorphic dataset (CIFAR10-DVS or DVS128 Gesture)
- three+ architectures with matched params
- systematic encoding comparison
- hyperparameter sensitivity with Pareto frontiers
- NeuroBench energy estimation
- adversarial robustness comparison (FGSM, PGD at multiple epsilon)
- time steps vs accuracy vs efficiency curves
- statistical rigor (multiple seeds, CIs, significance tests)
- clear conclusions with practical recommendations
- reproducible codebase

strong because: multi-dimensional analysis, tests actual claimed SNN advantages, controlled methodology, addresses open questions, far exceeds tutorials, potentially submittable to a workshop.

---

## 7. Timeline

### 12-Week Plan for Strong Version

| Week | Activity | Deliverable |
|------|----------|-------------|
| 1 | Setup, snnTorch + NeuroBench, run tutorials | Working environment |
| 2 | ANN baselines: FC + CNN on all 3 datasets | Baseline numbers with error bars |
| 3 | SNN equivalents with matched architectures | SNN numbers, initial comparison |
| 4 | Encoding comparison: rate, latency, delta, direct | Encoding comparison tables/plots |
| 5 | Hyperparameter sensitivity: grid search over tau, threshold, T, surrogate function | Sensitivity plots, key params identified |
| 6 | NeuroBench: SynOps, spike counts, sparsity | Energy proxy metrics |
| 7 | Time steps sweep T=1 to T=32: Pareto curves | Three-way tradeoff plots |
| 8 | Adversarial robustness: FGSM, PGD at multiple epsilon | Robustness comparison |
| 9 | Neuromorphic dataset (if time) | Extended results |
| 10 | Analysis, statistical testing, plots | Complete results section |
| 11-12 | Writing, polishing, documentation | Final report |

### Is This Realistic?

**yes, with caveats:**

in favor:
