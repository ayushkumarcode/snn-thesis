# Spiking Neural Networks for Natural Language Processing / Text Tasks

> **Deep Research Report**
> **Date:** 2026-02-25
> **Scope:** SpikingBERT, SpikeGPT, SpikeLLM, SpikeLM, text encoding as spikes, sentiment analysis, feasibility assessment for undergraduate thesis

---

## Executive Summary

Spiking Neural Networks for NLP is a **genuinely novel and rapidly emerging research direction** that has only become viable since 2023. The field has produced several landmark papers (SpikingBERT at AAAI 2024, SpikeGPT at UCSC, SpikeLM at ICML 2024, SpikeLLM at ICLR 2025), but performance still lags behind conventional models by 3-15% on standard benchmarks. The primary value proposition is **energy efficiency** (10-60x reduction), not accuracy improvement.

For an undergraduate thesis, this direction offers **exceptionally high novelty** but comes with **significant technical risk**. The most feasible approach would be a **focused binary sentiment classification task** using the ANN-to-SNN conversion pipeline with pre-trained word embeddings encoded as Poisson spike trains. Open-source code exists but requires substantial adaptation. A realistic project would compare SNN vs. ANN performance on 2-3 text datasets, quantify the accuracy-energy tradeoff, and contribute to an almost-empty undergraduate research space.

**Bottom line:** High risk, high reward. Very few undergraduates have attempted this worldwide. It would be a standout thesis if scoped correctly, but needs careful management to avoid scope creep.

---

## Part 1: The Key Models -- What Are They and Do They Work?

### 1.1 SpikeGPT (UC Santa Cruz, Feb 2023)

**What it is:** The first generative pre-trained language model built with spiking neural networks, inspired by the RWKV architecture (not standard transformer). Created by Jason Eshraghian's lab at UCSC.

**Architecture:** Replaces multi-head self-attention with a linear-complexity spiking mechanism. Uses binary, event-driven spiking activation units. Two variants: 45M and 216M parameters.

**Does it work?** Partially. Results are mixed:

| Benchmark | SpikeGPT 216M (pretrained) | BERT | GPT-2 Small |
|-----------|---------------------------|------|-------------|
| SST-2 (sentiment) | 88.76% | 91.73% | -- |
| SST-5 (5-class sentiment) | 51.27% | 53.21% | -- |
| MR (movie reviews) | 85.63% | 86.72% | -- |
| Subj (subjectivity) | 95.30% | N/A | -- |
| WikiText-2 (perplexity) | 18.01 PPL | -- | 37.50 PPL |
| WikiText-103 (perplexity) | 39.75 PPL | -- | 29.41 PPL |
