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

**Verdict:** Best spiking NLP model as of 2024. The 6.7% gap is notable but still significant for practical use.

**Code:** https://github.com/Xingrun-Xing/SpikeLM (public, Python/PyTorch/CUDA)

**Paper:** https://arxiv.org/abs/2406.03287

---

### 1.4 SpikeLLM (ICLR 2025)

**What it is:** Scales spiking neural networks to *large* language models (7B-70B parameters) using saliency-based spiking. The first attempt to make billion-parameter LLMs spike-driven.

**Architecture:** Uses Generalized Integrate-and-Fire (GIF) neurons with saliency detection to allocate more spiking steps to important channels. Employs first-order gradients for activation saliency and second-order Hessian metrics for weight saliency.

**Does it work?** Results on LLaMA-2 models:

| Model | Config | WikiText2 PPL | Zero-shot Avg Accuracy |
|-------|--------|---------------|----------------------|
| LLaMA-2-7B | 4W4A | -- | 50.65% (baseline 47.58%) |
| LLaMA-2-7B | 2W16A | 14.16 (baseline 38.05) | -- |
| LLaMA-2-13B | 2W8A | 13.56 (baseline 53.87) | 52.49% |
| LLaMA-2-70B | 2W16A | 6.35 (baseline 10.04) | 59.93% |

**Verdict:** Impressive that spiking mechanisms can work at 70B scale. But this is more of a quantization/compression technique than a from-scratch spiking model. The "spiking" here is about efficient representation, not biological plausibility.

**Code:** https://github.com/Xingrun-Xing2/SpikeLLM (public but sparse documentation)

**Paper:** https://arxiv.org/abs/2407.04752

---

### 1.5 SpikeZIP-TF (ICML 2024)

**What it is:** A lossless ANN-to-SNN conversion method for transformer architectures. Key innovation: ANN and SNN are *exactly equivalent*, incurring zero accuracy degradation.

**Results:**
- **SST-2: 93.79% accuracy** (highest SNN result for this benchmark)
- Outperforms SpikeGPT and SpikeBERT on English and Chinese text tasks
- 3.65% improvement on MR, 5.24% on SST-5 vs prior SNN methods

**Verdict:** If your goal is to demonstrate SNN equivalence with no accuracy loss, this is the strongest approach. But the energy savings may be smaller since exact conversion requires more time steps.

**Paper:** https://arxiv.org/abs/2406.03470

---

### 1.6 SpikingMiniLM (2024)

**What it is:** An energy-efficient spiking transformer for natural language understanding. Introduces a multistep encoding method to convert text embeddings into spike trains. Targets the MiniLM architecture (smaller than BERT).

**Verdict:** Achieves similar performance to BERT-MINI with fewer parameters and much lower energy consumption. Potentially more feasible for an undergraduate project due to smaller model size.

**Paper:** https://link.springer.com/article/10.1007/s11432-024-4101-6

---

## Part 2: SNNs for Sentiment Analysis and Text Classification

### 2.1 Existing Work

Several groups have successfully applied SNNs to sentiment analysis:

#### SSA-SpiNNaker (PMC, 2023)
- **Task:** Binary sentiment analysis on IMDB (50,000 movie reviews)
- **Method:** Train ANN, convert to SNN using Integrate-and-Fire neurons, deploy on SpiNNaker hardware
- **Accuracy:** Claims 100% on test samples (vs 90% for the original ANN -- suspicious claim, likely on a subset)
- **Energy:** 3,970 Joules for ~10,000 words
- **Hardware:** SpiNNaker neuromorphic platform (University of Manchester)
- **Paper:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10536645/

#### Energy-Efficient Sentiment Classification (ICANN 2023)
- **Task:** Sentiment classification on IMDB
- **Energy result:** SNN energy consumption reduced to 1.36% of a Transformer model (64.93x improvement)
- **Paper:** https://link.springer.com/chapter/10.1007/978-3-031-44204-9_43

#### Spiking CNN for Text Classification (ICLR 2023)
- **Task:** 6 text classification benchmarks (MR, SST-2, SST-5, Subj, ChnSenti, Waimai)
- **Method:** Conversion + fine-tuning of TextCNN, Poisson spike trains from word embeddings
- **Results:**

| Dataset | Original TextCNN | Spiking CNN | Accuracy Drop |
|---------|-----------------|-------------|---------------|
| MR (movie reviews) | 77.41% | 75.45% | -1.96% |
| SST-2 (binary sentiment) | 83.25% | 80.91% | -2.34% |
| Subj (subjectivity) | 94.00% | 90.60% | -3.40% |
| SST-5 (5-class) | 45.48% | 41.63% | -3.85% |
| ChnSenti (Chinese) | 86.74% | 85.02% | -1.72% |
| Waimai (Chinese food) | 88.49% | 86.66% | -1.83% |

- **Average accuracy drop:** 2.51% across all datasets
- **Energy:** >10x reduction compared to TextCNN
- **Adversarial robustness:** +13.55% robust accuracy under adversarial attacks
- **Timesteps:** 50 (fine-tuned SNNs at 50 steps outperform converted SNNs at 80 steps)
- **Code:** https://github.com/Lvchangze/snn (public, Python)
- **Paper:** https://arxiv.org/abs/2406.19230

#### SNNLP (Jan 2024)
- **Task:** Sentiment analysis with novel text-to-spike encoding
- **Key result:** New encoding method outperforms Poisson rate-coding by ~13%
- **Energy:** 32x more efficient during inference, 60x during training vs DNNs
- **Paper:** https://arxiv.org/abs/2401.17911

#### SNN Topic Modeling (2024)
- **Task:** Document topic modeling using STDP learning
- **Method:** Transform text into spike sequences, each neuron represents a topic, STDP modifies synaptic weights
- **Paper:** https://www.sciencedirect.com/science/article/pii/S0893608024004180

---

## Part 3: How to Encode Text as Spikes

This is the central technical challenge for SNN-NLP. Four main approaches exist:
