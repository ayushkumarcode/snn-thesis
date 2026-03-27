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

**Energy:** Claims 32.2x fewer operations on neuromorphic hardware. Theoretical energy reduced from 3.29x10^10 pJ to 1.02x10^9 pJ.

**Verdict:** Works competitively on simple classification tasks (within 1-3% of BERT). Language generation is weaker. The 216M pretrained model is required -- the 45M model underperforms significantly.

**Code:** https://github.com/ridgerchu/SpikeGPT (public, Python/PyTorch)

**Paper:** https://arxiv.org/abs/2302.13939

---

### 1.2 SpikingBERT (Penn State, AAAI 2024)

**What it is:** A spiking language model created by distilling knowledge from a pre-trained BERT model into a spiking architecture using a novel implicit differentiation technique. This overcomes the non-differentiability problem of SNNs without surrogate gradients.

**Architecture:** Uses Average Spiking Rate (ASR) convergence at equilibrium to develop a spiking attention mechanism. Employs a 3-stage training pipeline: general knowledge distillation, task-based internal layer KD, and prediction layer distillation.

**Does it work?** Yes, on GLUE benchmark tasks (SST-2, MNLI, QQP, QNLI, RTE, MRPC, STS-B). It is the first spiking LM evaluated on multiple GLUE tasks. Without distillation, there is a 4-5% performance loss. With distillation, performance is competitive but still below BERT-base.

**Key details:**
- Convergence time steps (t_conv): 125
- Threshold voltage (vth): 1.0
- Max sequence length: 128
- Requires multi-GPU training (DataParallel)

**Verdict:** Demonstrates that BERT-like capabilities can be approximated with spiking neurons, but requires a complex 3-stage distillation pipeline. Not trivial to reproduce.

**Code:** https://github.com/NeuroCompLab-psu/SpikingBERT (public, Python/PyTorch)

**Paper:** https://arxiv.org/abs/2308.10873

---

### 1.3 SpikeLM (ICML 2024)

**What it is:** The first *fully spiking* mechanism for general language tasks (both discriminative and generative). Introduces "elastic bi-spiking" -- spikes have bi-directional amplitude and frequency encoding, while still maintaining the additive nature of SNNs.

**Does it work?** This is currently the best-performing spiking language model:

| Task | BERT-base | SpikeBERT | SpikeLM | Gap from BERT |
|------|-----------|-----------|---------|---------------|
| SST-2 | 92.3% | 85.4% | 87.0% | -5.3% |
| MNLI-m/mm | 83.8/83.4 | 71.4/71.0 | 77.1/77.2 | -6.7/-6.2% |
| QQP (F1) | 90.5 | 68.2 | 83.9 | -6.6% |
| QNLI | 90.7 | 66.4 | 85.3 | -5.4% |
| CoLA | 60.0 | 16.9 | 38.8 | -21.2% |
| STS-B | 89.4 | 18.7 | 84.9 | -4.5% |
| MRPC (F1) | 89.8 | 82.0 | 85.7 | -4.1% |
| RTE | 69.3 | 57.5 | 69.0 | -0.3% |
| **Average gap** | -- | -- | -- | **~6.7%** |

**Key achievements:**
- Reduces performance gap from 28.3% (LIF-BERT) to 6.7% vs BERT-base
- Outperforms SpikeBERT by 16.8% without requiring distillation
- 12.9x energy savings with T=1 timestep, 3.7x with T=4

