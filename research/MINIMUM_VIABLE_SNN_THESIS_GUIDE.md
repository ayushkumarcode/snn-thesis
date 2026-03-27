# Minimum Viable Undergraduate SNN Thesis: The Efficient Path to a Good Grade

> Research compiled: 2026-02-25
> Context: University of Manchester COMP30040 Third Year Project (40 credits)
> Assessment: Report 55%, Achievements 30%, Screencast 15%

---

## EXECUTIVE SUMMARY

The fastest path to a completed, high-quality SNN thesis at Manchester is a **comparative benchmarking study** using the **snnTorch** framework. The project compares SNN performance against a conventional ANN baseline on 1-2 standard datasets (MNIST + one other), measuring accuracy, spike count, and estimated energy efficiency. This approach works because: (a) snnTorch tutorials provide 80% of the code you need, (b) the research question is clear and naturally produces tables/graphs for the report, (c) you can scale ambition up or down based on available time, and (d) it directly addresses a genuine open question in the field (when/why SNNs outperform ANNs).

**Estimated total coding hours: 30-50 hours** (not including report writing).
**Minimum datasets: 1 (MNIST), recommended: 2 (MNIST + Fashion-MNIST or N-MNIST).**
**Minimum experiments: 3-4 distinct experimental configurations producing 2-3 results tables.**
**Literature review scope: 20-30 references (15-20 core papers + 5-10 supporting).**

---

## 1. THE SIMPLEST PROJECT THAT WOULD STILL EARN A GOOD GRADE

### Recommended Project: "Comparative Analysis of Spiking vs. Conventional Neural Networks for Image Classification"

**Research question:** "How does SNN performance compare to equivalent ANN architectures on standard image classification benchmarks, and under what conditions do SNNs offer advantages?"

This is the optimal "minimum viable thesis" because:

1. **It is a genuine research question.** The SNN vs ANN performance comparison is an active area of research. A 2020 paper from UCSB ("Rethinking the performance comparison between SNNs and ANNs") explicitly noted that on ANN-oriented workloads SNNs fail to beat ANNs, while on SNN-oriented workloads SNNs can outperform -- this tension is worth investigating.

2. **The code is largely pre-written.** snnTorch Tutorial 5 (fully-connected SNN on MNIST) and Tutorial 6 (convolutional SNN on MNIST) provide the SNN side. A standard PyTorch CNN provides the ANN baseline. You are adapting and extending existing code, not building from scratch.

3. **It produces natural report structure.** Background (SNN theory) -> Methodology (architectures, datasets, metrics) -> Results (comparison tables) -> Discussion (why gaps exist, when SNNs win) -> Conclusion.

4. **It scales.** Minimum version: 1 dataset, 2 architectures. Ambitious version: 3 datasets, 4 architectures, hyperparameter sweeps, encoding comparison.

### What makes this a 2:1 vs. First Class:

| Grade | What you need |
|-------|--------------|
| **2:1 (60-69%)** | ANN vs SNN on MNIST with accuracy comparison. Clear report with adequate literature review. Basic analysis of results. |
| **First (70%+)** | Multiple datasets (MNIST + Fashion-MNIST or N-MNIST). Multiple architectures (FC + CNN). Additional metrics beyond accuracy (spike count, training time, estimated energy). Thoughtful discussion of WHY results differ. Encoding scheme comparison (rate vs. latency coding). |

### Alternative "minimum viable" projects (ranked by efficiency):

| Project | Implementation Effort | Report Writing Effort | Risk Level | Grade Ceiling |
|---------|----------------------|----------------------|------------|---------------|
| **SNN vs ANN comparison (recommended)** | Low (adapt tutorials) | Medium | Low | First |
| **Spike encoding scheme comparison** | Low-Medium | Medium | Low | First |
| **ANN-to-SNN conversion study** | Low (use sinabs or SNN Toolbox) | Medium | Low | 2:1/First |
| **STDP unsupervised learning on MNIST** | Medium (BindsNET) | Medium | Medium | First |
| **Neuron model comparison (LIF vs Izhikevich)** | Medium | Higher (more theory) | Low | First |
| **DVS128 gesture recognition** | Higher | Medium | Higher | First |

---

## 2. HOW MANY EXPERIMENTS/RESULTS ARE TYPICALLY NEEDED

Based on analysis of successful undergraduate SNN projects and Manchester's assessment criteria ("the report has to be a complete account... how you evaluated it and with what results"):

### Minimum for a 2:1: 3-4 experiments producing 2 results tables

**Experiment Set A (Baseline):**
1. Train ANN (standard CNN) on MNIST -- record accuracy, training time
2. Train SNN (equivalent architecture using snnTorch) on MNIST -- record accuracy, training time, spike count

**Experiment Set B (Variation):**
3. Vary a key SNN parameter (e.g., number of time steps: 10, 25, 50, 100) and record accuracy vs. time steps
4. Test on a second dataset (Fashion-MNIST) to show generalisability

**This produces:**
- Table 1: ANN vs SNN accuracy comparison
- Table 2: Effect of time steps on SNN accuracy
- Figure 1: Training loss curves (ANN vs SNN)
- Figure 2: Accuracy vs. time steps plot

### Recommended for a First: 6-8 experiments producing 3-4 results tables + 4-6 figures

Add to the above:
5. Compare fully-connected vs convolutional SNN architectures
6. Compare rate coding vs latency coding input encoding
7. Measure and compare spike counts / synaptic operations (energy proxy)
8. Test on N-MNIST (neuromorphic dataset) -- shows SNN advantage on temporal data

**Additional outputs:**
- Table 3: Architecture comparison (FC-SNN vs CSNN vs FC-ANN vs CNN)
- Table 4: Encoding scheme comparison
- Figure 3: Spike raster plots (visualise spiking activity)
- Figure 4: Confusion matrices
- Figure 5: Energy estimation comparison (synaptic operations)

### What examiners actually care about (from Manchester assessment docs):

The Manchester marking criteria state the report needs:
- "An elucidation of the problem and the objectives"
- "An in-depth investigation of the context and literature"
- "A critical appraisal... indicating the rationale for any design/implementation decisions"
- "Evaluation (with hindsight) of the project outcome"

**The key insight: examiners care MORE about your analysis of results than the quantity of results.** Three well-analysed experiments with insightful discussion will outscore eight experiments with superficial commentary. For each result, you should explain: what you expected, what happened, and WHY.

---

## 3. MINIMUM NUMBER OF DATASETS

### Absolute minimum: 1 (MNIST)

MNIST is universally accepted as the baseline benchmark for SNN research. The seminal Diehl & Cook (2015) paper used only MNIST and has 2000+ citations. Every SNN framework tutorial uses MNIST.

### Recommended: 2 datasets

| Dataset combination | Why this works | Difficulty |
|--------------------|---------------|------------|
| **MNIST + Fashion-MNIST** (recommended) | Same format (28x28 grayscale), harder task. Shows your SNN generalises. Both load identically in PyTorch. | Easy -- literally change one line of code |
| **MNIST + N-MNIST** | Static vs neuromorphic. Shows SNN advantage on temporal data. Uses Tonic library with snnTorch (Tutorial 7). | Medium -- different data loading pipeline |
| **MNIST + CIFAR-10** | Tests on colour images. Good ANN comparison point. | Medium -- requires architecture changes |

### For a First Class mark: 2-3 datasets

Adding N-MNIST or DVS128 Gesture as a third dataset demonstrates understanding of neuromorphic data and shows SNNs in their natural domain. This is where the SNN advantage over ANNs actually appears.

### Datasets to avoid:
- Custom/obscure datasets (examiners cannot verify your claims)
- Datasets requiring significant preprocessing work (time sink)
