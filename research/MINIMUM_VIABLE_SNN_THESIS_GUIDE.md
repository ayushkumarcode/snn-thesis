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
