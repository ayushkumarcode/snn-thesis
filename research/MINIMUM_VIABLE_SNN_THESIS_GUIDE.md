# Minimum viable SNN thesis -- the efficient path to a decent grade

So i worked out the fastest path to a completed, high-quality SNN thesis at Manchester. It's basically a **comparative benchmarking study** using **snnTorch**. Compare SNN performance against a conventional ANN baseline on 1-2 standard datasets (MNIST + one other), measuring accuracy, spike count, and estimated energy efficiency. This works because: (a) snnTorch tutorials give you like 80% of the code already, (b) the research question is clear and naturally produces tables/graphs for the report, (c) you can scale ambition up or down depending on time, and (d) it addresses a real open question (when/why do SNNs outperform ANNs).

**Estimated coding time: 30-50 hours** (not including report writing).
**Minimum datasets: 1 (MNIST), recommended: 2 (MNIST + Fashion-MNIST or N-MNIST).**
**Minimum experiments: 3-4 distinct configurations producing 2-3 results tables.**
**Lit review scope: 20-30 references (15-20 core papers + 5-10 supporting).**

---

## 1. The simplest project that could still get a good grade

### Recommended: "Comparative Analysis of Spiking vs. Conventional Neural Networks for Image Classification"

**Research question:** "How does SNN performance compare to equivalent ANN architectures on standard image classification benchmarks, and under what conditions do SNNs offer advantages?"

Why this is the optimal minimum viable thesis:

1. **It's a genuine research question.** The SNN vs ANN comparison is active research. A 2020 paper from UCSB ("Rethinking the performance comparison between SNNs and ANNs") noted that on ANN-oriented workloads SNNs fail to beat ANNs, while on SNN-oriented workloads SNNs can outperform -- this tension is worth investigating.

2. **The code is mostly pre-written.** snnTorch Tutorial 5 (FC SNN on MNIST) and Tutorial 6 (conv SNN on MNIST) give you the SNN side. A standard PyTorch CNN is the ANN baseline. You're adapting existing code, not building from scratch.

3. **It produces natural report structure.** Background (SNN theory) -> Methodology (architectures, datasets, metrics) -> Results (comparison tables) -> Discussion (why gaps exist, when SNNs win) -> Conclusion.

4. **It scales.** Minimum version: 1 dataset, 2 architectures. Ambitious version: 3 datasets, 4 architectures, hyperparameter sweeps, encoding comparison.

### What makes this a 2:1 vs first:

| Grade | What you need |
|-------|--------------|
| **2:1 (60-69%)** | ANN vs SNN on MNIST with accuracy comparison. Clear report with adequate lit review. Basic analysis of results. |
| **First (70%+)** | Multiple datasets (MNIST + Fashion-MNIST or N-MNIST). Multiple architectures (FC + CNN). Additional metrics beyond accuracy (spike count, training time, estimated energy). Thoughtful discussion of WHY results differ. Encoding scheme comparison (rate vs. latency coding). |

### Alternative minimum viable projects (ranked by efficiency):

| Project | Implementation Effort | Report Writing Effort | Risk Level | Grade Ceiling |
|---------|----------------------|----------------------|------------|---------------|
| **SNN vs ANN comparison (recommended)** | Low (adapt tutorials) | Medium | Low | First |
| **Spike encoding scheme comparison** | Low-Medium | Medium | Low | First |
| **ANN-to-SNN conversion study** | Low (use sinabs or SNN Toolbox) | Medium | Low | 2:1/First |
| **STDP unsupervised learning on MNIST** | Medium (BindsNET) | Medium | Medium | First |
| **Neuron model comparison (LIF vs Izhikevich)** | Medium | Higher (more theory) | Low | First |
| **DVS128 gesture recognition** | Higher | Medium | Higher | First |

---

## 2. How many experiments/results are needed

Based on successful undergrad SNN projects and Manchester's assessment criteria ("the report has to be a complete account... how you evaluated it and with what results"):

### Minimum for a 2:1: 3-4 experiments producing 2 results tables

**Experiment Set A (Baseline):**
1. Train ANN (standard CNN) on MNIST -- record accuracy, training time
2. Train SNN (equivalent architecture using snnTorch) on MNIST -- record accuracy, training time, spike count
