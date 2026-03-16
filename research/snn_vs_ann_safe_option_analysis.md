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
- More encoding schemes
- Additional datasets
- snnTorch-specific implementations (making results reproducible in a popular framework)
- Multi-dimensional comparison: accuracy, spike count, convergence speed, robustness to noise

**Risk**: This has been partially done in academic papers. Your contribution would be the breadth and reproducibility angle, not novelty of the individual encodings.

**Estimated time**: 4-6 weeks of development after setup

### Angle B: Hyperparameter Sensitivity Study (Moderate Value, High Feasibility)

**What it is**: Systematic grid/random search over SNN-specific hyperparameters: membrane decay (tau), firing threshold, time steps, surrogate gradient function (arctan, sigmoid, fast-sigmoid), and their interactions.

**Why it adds value**: Research confirms SNN training is highly sensitive to these parameters. A firing threshold of 1.0 vs 0.25 can cause accuracy to swing from 96% to 41% ([Bojkovic et al., 2024](https://proceedings.mlr.press/v238/bojkovic24a/bojkovic24a.pdf)). Yet there is no comprehensive undergraduate-accessible guide to tuning these parameters across datasets.

**What makes it thesis-worthy**:
- Plot Pareto frontiers of accuracy vs spike count at different parameter combinations
- Identify which hyperparameters matter most (sensitivity analysis)
- Provide practical guidelines for practitioners

**Estimated time**: 3-5 weeks of development

### Angle C: Energy/Efficiency Analysis via Proxy Metrics (High Value, Moderate Feasibility)

**What it is**: Measure and compare computational cost using proxy metrics like synaptic operations (SynOps), spike counts, MAC vs AC operations, memory accesses -- using the [NeuroBench framework](https://neurobench.readthedocs.io/en/latest/).

**Why it adds value**: The paper ["Are SNNs Really More Energy-Efficient Than ANNs?"](https://cea.hal.science/cea-03852141/file/Are_SNNs_Really_More_Energy_Efficient_Than_ANNs__An_In_Depth_Hardware_Aware_Study_versionacceptee.pdf) showed that SNN energy advantage is conditional and often overstated. Replicating this analysis at an undergraduate level using NeuroBench would be genuinely valuable. Key finding to test: SNNs need >93% spike sparsity with VGG16 at T=6 to be more efficient than ANNs.

**What makes it thesis-worthy**:
- Goes beyond accuracy to address the *actual claimed advantage* of SNNs
- Uses a real benchmarking framework (NeuroBench) with standardized metrics
- Can produce Pareto curves of accuracy vs energy proxy
- Challenges a common claim in the field with empirical evidence

**Estimated time**: 4-6 weeks (NeuroBench has good documentation and tutorials)

### Angle D: Adversarial Robustness Comparison (High Value, Moderate Feasibility)

**What it is**: Compare SNN and ANN vulnerability to adversarial attacks (FGSM, PGD) and natural noise/corruption.

**Why it adds value**: SNNs exhibit inherent robustness advantages ([ECCV 2020 paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740392.pdf)). SNN-BP shows 2-4.6% improvement in adversarial accuracy over equivalent ANNs on CIFAR-10 with VGG and ResNet. This is a genuinely useful and somewhat under-explored angle at the undergraduate level.

**What makes it thesis-worthy**:
- Tests a non-obvious SNN advantage (not just accuracy)
- Practical relevance for safety-critical applications
- Combines two interesting research areas
- Can use standard adversarial attack libraries (Foolbox, ART)

**Estimated time**: 3-5 weeks

### Angle E: Architecture-Controlled Fair Comparison (Moderate Value, High Feasibility)

**What it is**: Build identical architectures for ANN and SNN (same layers, same parameter count), following the methodology from [Deng et al., "Rethinking the performance comparison between SNNs and ANNs"](https://web.ece.ucsb.edu/~lip/publications/SNN-vs-ANN-NeuralNetworks2020.pdf) and [Event-based Optical Flow comparison](https://arxiv.org/html/2407.20421v1).

**Why it adds value**: Most naive comparisons are unfair -- different architectures, different training regimes, different hyperparameter budgets. A rigorously controlled comparison is more scientifically valuable than a casual one.

**Key insight from literature**: "On ANN-oriented workloads, SNNs fail to beat their ANN counterparts; while on SNN-oriented workloads, SNNs can fully perform better." Testing this claim with controlled experiments would be valuable.

**Estimated time**: 3-4 weeks

### Angle F: Time Steps vs Accuracy vs Sparsity Pareto Analysis (High Value, Moderate Feasibility)

**What it is**: Systematically vary time steps from T=1 to T=32+ and plot the three-way tradeoff between accuracy, inference latency, and spike sparsity, referencing the framework from ["Exploring Tradeoffs in Spiking Neural Networks"](https://direct.mit.edu/neco/article/35/10/1627/117019/Exploring-Trade-Offs-in-Spiking-Neural-Networks).

**Why it adds value**: This directly addresses the key practical question: "How many time steps do I actually need?" The answer varies by dataset, architecture, and encoding scheme.

**Estimated time**: 3-4 weeks

---

## 5. How Many Existing Projects Have Done Exactly This?

### Direct Matches Found (SNN vs ANN comparison on standard benchmarks)

1. **High School Student Paper (2024)**: Published in [NHSJS](https://nhsjs.com/2024/advancements-in-image-classification-comparing-spiking-convolutional-and-artificial-neural-networks/) -- compared SNN, CNN, and ANN on MNIST, CIFAR-10, N-MNIST. Found SNN matched accuracy but consumed 142% more power and 128% more memory on commercial hardware.

2. **Virginia Tech Class Project (2020)**: [GitHub repo](https://github.com/oshears/adv-ml-2020-snn-project) -- compared 784-100 ANN to 784-100 SNN on MNIST. Graduate course project.

3. **UNSW Bachelor Honours Thesis (2022)**: Investigated biologically-inspired ANNs with spiking neurons, benchmarking against traditional networks on classification tasks.

4. **King's College London BSc (2018)**: [GitHub repo](https://github.com/LucaMozzo/SpikingNeuralNetwork) -- SNN for MNIST digit recognition in C++ from scratch. More implementation-focused than comparison-focused.

5. **Multiple Kaggle/GitHub repos**: Numerous repositories with basic SNN MNIST implementations using snnTorch, SpikingJelly, or Norse.

### Assessment of Saturation

**The basic "SNN vs ANN accuracy on MNIST" comparison is HIGHLY saturated.** It has been done by:
- High school students (published)
- Undergraduate course projects (multiple)
- Graduate course projects (multiple)
- Academic papers (dozens)

**The "SNN vs ANN on CIFAR-10 with systematic analysis" is LESS saturated** but still has significant academic coverage.

**Risk of being too generic**: HIGH if you only compare accuracy. MODERATE if you add one analytical dimension (energy, robustness, encoding comparison). LOW if you add two or more analytical dimensions with controlled experimental methodology.

---

## 6. Strong Version vs Weak Version of This Project

### WEAK Version (Grade: Pass/Low 2:1 -- risky)

**What it looks like**:
- Run snnTorch tutorials 5 and 6 on MNIST
- Train a simple CNN on MNIST
- Compare accuracy numbers
- Maybe also run on Fashion-MNIST
- Report: "SNNs achieved 98% and ANNs achieved 99%, therefore ANNs are slightly better"
- No energy analysis, no controlled variables, no encoding comparison
- Uses default hyperparameters throughout

**Why it is weak**:
- A high school student already published this
- No experimental design beyond "run and report"
- No contribution to knowledge
- No analysis of *why* results differ
- Easily dismissed as "tutorial replication"

**Red flags**:
- Only MNIST results
- Only one encoding scheme
- Only default hyperparameters
- No statistical significance testing
- No energy/efficiency analysis

### MODERATE Version (Grade: Solid 2:1)

**What it looks like**:
- All three datasets (MNIST, Fashion-MNIST, CIFAR-10)
- Two or more architectures (FC and convolutional)
- Fair comparison methodology (same parameter counts)
- At least two encoding schemes compared
- Some hyperparameter sensitivity analysis
- Spike count / synaptic operations reported
- Multiple trials with error bars
- Structured analysis of accuracy gap across conditions

### STRONG Version (Grade: First -- publishable potential)

**What it looks like**:
- Title: "A Systematic Multi-Dimensional Comparison of Spiking and Artificial Neural Networks: Accuracy, Efficiency, and Robustness Trade-offs"
- Three datasets (MNIST, Fashion-MNIST, CIFAR-10) plus one neuromorphic dataset (CIFAR10-DVS or DVS128 Gesture)
- Three+ architectures with matched parameter counts
- Systematic encoding comparison (rate, latency, delta, direct input)
- Hyperparameter sensitivity analysis with Pareto frontiers
- Energy estimation via NeuroBench (SynOps, spike sparsity, Eff_MACs vs Eff_ACs)
- Adversarial robustness comparison (FGSM, PGD at multiple epsilon values)
- Time steps vs accuracy vs efficiency three-way tradeoff curves
- Statistical rigor (multiple seeds, confidence intervals, significance tests)
- Clear conclusions with practical recommendations
- Reproducible codebase with documentation

**Why it is strong**:
- Multi-dimensional analysis (not just accuracy)
- Tests the actual claimed advantages of SNNs (efficiency, robustness)
- Controlled methodology following best practices from published papers
- Addresses open questions with practical implications
- Significantly exceeds what any tutorial provides
- Could be submitted to a workshop paper (e.g., NICE, NeuroAI workshop)

---

## 7. Timeline Feasibility: Can This Be Done in 2-3 Months?

### Assumption: 2-3 months of actual development time (roughly 8-12 weeks)

### Week-by-Week Breakdown for the STRONG Version

| Week | Activity | Deliverable |
|------|----------|-------------|
| 1 | Environment setup, snnTorch + NeuroBench installation, run tutorials end-to-end | Working dev environment, familiarity with framework |
| 2 | Implement ANN baselines: FC and CNN on all 3 datasets | Baseline accuracy numbers with error bars |
| 3 | Implement SNN equivalents with matched architectures | SNN accuracy numbers, initial comparison |
| 4 | Encoding scheme comparison: implement rate, latency, delta, direct input | Encoding comparison tables and plots |
| 5 | Hyperparameter sensitivity: grid search over tau, threshold, time steps, surrogate function | Sensitivity analysis plots, identify key parameters |
| 6 | NeuroBench integration: measure SynOps, spike counts, sparsity | Energy proxy metrics for all models |
| 7 | Time steps sweep (T=1 to T=32): accuracy/efficiency Pareto curves | Three-way tradeoff plots |
| 8 | Adversarial robustness experiments: FGSM and PGD at multiple epsilon | Robustness comparison tables |
| 9 | Neuromorphic dataset experiments (CIFAR10-DVS if time permits) | Extended results |
| 10 | Analysis, statistical testing, plot generation | Complete results section |
| 11-12 | Writing, polishing, documentation | Final report |

### Feasibility Assessment

**Is this realistic in 2-3 months?** YES, with caveats:

**Factors in favor**:
- snnTorch is well-documented with extensive tutorials and Colab notebooks
- MNIST/Fashion-MNIST train quickly (minutes on GPU)
- CIFAR-10 is manageable on a single GPU (hours, not days)
- NeuroBench is built on top of snnTorch -- integration is straightforward
- No custom hardware needed
- No novel algorithm development required
- All tools are open-source and free

**Factors that could slow you down**:
- CIFAR-10 SNN training can be slow (sequential time steps multiply training time)
- Hyperparameter sweeps multiply experiment count significantly
- Debugging SNN training issues (gradient problems, spike vanishing) can be time-consuming
- The transition from "it works on MNIST" to "it works on CIFAR-10" is non-trivial
- Writing up results comprehensively takes longer than expected

**Critical dependency**: GPU access. SNN training on CIFAR-10 with multiple time steps and multiple seeds will require significant GPU time. Google Colab free tier may be insufficient. Colab Pro or university GPU cluster is recommended.

### Scoped-Down "Guaranteed Completion" Version (8 weeks)

If you want to ensure completion with buffer:
1. Drop adversarial robustness (save for future work)
2. Drop neuromorphic dataset (save for future work)
3. Focus on: 3 datasets x 2 architectures x 3 encodings x hyperparameter sweep + NeuroBench energy metrics
4. This still produces a strong thesis but with less breadth

---

## Key Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| Thesis perceived as too generic | HIGH if no differentiator | HIGH | Add at least 2 of the angles from Section 4 |
| SNN training fails to converge on CIFAR-10 | MODERATE | MODERATE | Use proven architectures from papers; start with conversion, then try direct training |
| Compute budget insufficient | MODERATE | HIGH | Request university GPU; budget Colab Pro; reduce hyperparameter grid |
| Results replicate known findings without new insight | MODERATE | MODERATE | Frame as a *systematic reproducibility study* with added dimensions |
| Takes longer than expected | HIGH | MODERATE | Have a scoped-down version ready; prioritize experiments by impact |

---

## Recommended Thesis Framing

Do NOT frame this as: "I compared SNNs and ANNs on image classification"
(This sounds like a tutorial exercise)

DO frame this as one of:

**Option 1 -- Efficiency Focus**: "Evaluating the accuracy-efficiency tradeoff of spiking neural networks: A controlled multi-dataset study using NeuroBench metrics"

**Option 2 -- Multi-dimensional Comparison**: "Beyond accuracy: A systematic comparison of spiking and artificial neural networks across accuracy, energy, and robustness dimensions"

**Option 3 -- Encoding Study**: "Impact of spike encoding schemes on SNN classification performance: A comprehensive empirical study across datasets and architectures"

**Option 4 -- Hyperparameter Study**: "Sensitivity analysis of spiking neuron parameters for image classification: Practical guidelines for SNN practitioners"

Each of these frames the work as answering a specific question rather than just "comparing things."

---

## Confidence Assessment

| Finding | Confidence | Basis |
|---------|-----------|-------|
| MNIST gap is closed (~0%) | VERY HIGH | Multiple papers, reproducible |
| Fashion-MNIST gap is 2-5% for typical implementations | HIGH | Multiple sources |
| CIFAR-10 gap is 0.4-8% depending on method | HIGH | Extensive literature |
| Basic comparison has been done many times | VERY HIGH | Found multiple student projects doing this |
| Energy analysis angle adds significant value | HIGH | NeuroBench is well-documented, papers confirm conditional efficiency |
| Robustness angle adds significant value | HIGH | Published evidence of SNN robustness advantage |
| 2-3 month timeline is feasible for strong version | MODERATE-HIGH | Depends on GPU access and CIFAR-10 training speed |
| Weak version risks being dismissed as trivial | HIGH | High school student already published equivalent |

---

## References and Key Papers

1. Deng & Gu (2020). ["Rethinking the performance comparison between SNNs and ANNs"](https://web.ece.ucsb.edu/~lip/publications/SNN-vs-ANN-NeuralNetworks2020.pdf) -- Essential reading for fair comparison methodology
2. Lemaire et al. (2022). ["Are SNNs Really More Energy-Efficient Than ANNs?"](https://cea.hal.science/cea-03852141/file/Are_SNNs_Really_More_Energy_Efficient_Than_ANNs__An_In_Depth_Hardware_Aware_Study_versionacceptee.pdf) -- Critical for energy analysis claims
3. Park et al. (2021). ["Neural Coding in SNNs: A Comparative Study"](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.638474/full) -- Encoding scheme comparison methodology
4. Sharmin et al. (2020). ["Inherent Adversarial Robustness of Deep SNNs"](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740392.pdf) -- Robustness analysis
5. Patino-Saucedo et al. (2023). ["Exploring Trade-Offs in SNNs"](https://direct.mit.edu/neco/article/35/10/1627/117019/Exploring-Trade-Offs-in-Spiking-Neural-Networks) -- Pareto analysis framework
6. NeuroBench Collaboration (2025). ["The NeuroBench Framework"](https://www.nature.com/articles/s41467-025-56739-4) -- Standardized benchmarking
7. STAA-SNN (CVPR 2025). ["Spatial-Temporal Attention Aggregator for SNNs"](https://arxiv.org/pdf/2503.02689) -- Current state-of-the-art SNN on CIFAR-10 (97.14%)
8. Luo (2024). ["Advancements in Image Classification: Comparing Spiking, Convolutional, and Artificial Neural Networks"](https://nhsjs.com/2024/advancements-in-image-classification-comparing-spiking-convolutional-and-artificial-neural-networks/) -- The high school paper to differentiate from

---

## Bottom Line

This IS a safe option. It WILL work -- you will get numbers, plots, and a complete thesis. The risk is not failure; the risk is mediocrity. The difference between a weak thesis and a strong thesis with this topic is NOT the topic itself -- it is the depth of analysis.

**Minimum viable differentiator**: Include NeuroBench energy metrics + at least one encoding scheme comparison + multiple datasets. This lifts you above "tutorial replication" into "systematic empirical study."

**Recommended approach**: Combine Angles B (hyperparameter sensitivity), C (energy analysis), and F (time steps Pareto analysis) for maximum impact with reasonable effort. Add Angle D (robustness) if time permits.
