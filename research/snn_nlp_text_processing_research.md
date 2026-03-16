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

### 3.1 Poisson Rate Coding (Most Common)

**How it works:**
1. Take a pre-trained word embedding (e.g., Word2Vec, GloVe) for each word
2. Normalize and shift all embedding components to [0, 1] range
3. Each component generates a Poisson spike train with firing rate proportional to its value
4. Higher value = more frequent spikes; lower value = fewer spikes

**Process (from Spiking CNN paper):**
- Calculate mean (mu) and standard deviation (sigma) of embedding values
- Clip within [mu - 3*sigma, mu + 3*sigma]
- Normalize by subtracting mu and dividing by 6*sigma
- Scale to [0, 1]
- Generate Poisson spike trains proportional to each scaled value

**Pros:** Simple, well-understood, preserves semantic information from pre-trained embeddings
**Cons:** Requires many timesteps (50-100), introduces stochastic noise, slow inference

### 3.2 Temporal/Latency Coding

**How it works:** Information is encoded in the *timing* of spikes rather than their rate. Higher values produce earlier spikes, lower values produce later spikes.

**Pros:** More information per spike, faster inference, lower energy
**Cons:** More complex to implement, less robust to noise, less explored for text

### 3.3 Direct Embedding Encoding (SpikeLM approach)

**How it works:** Uses elastic bi-spiking with bi-directional amplitude and frequency encoding. Spikes carry direction and amplitude information in a single timestep, with a controlled firing rate strategy.

**Pros:** Works with fewer timesteps (T=1 possible), preserves more information
**Cons:** Requires custom neuron models, harder to implement

### 3.4 Word2Spike (2025)

**How it works:** A novel rate coding mechanism that creates a one-to-one mapping from multi-dimensional word vectors to spike-based attractor states using Poisson processes.

**Key finding:** Spike-based embeddings preserve essential semantic and relational structure despite realistic noise.

**Paper:** https://arxiv.org/html/2509.07361

### 3.5 Integer Word Index Encoding (Simplest)

**How it works:** Words are converted to integer indices, then each index generates a corresponding spike pattern. Used in the SSA-SpiNNaker paper.

**Pros:** Very simple to implement
**Cons:** Loses semantic information, limited vocabulary handling

### Recommended Approach for Undergraduate Project

**Use Poisson Rate Coding with pre-trained embeddings (approach 3.1).** This is:
- Best documented with working code
- Used in the highest-quality reproducible paper (Spiking CNN for Text Classification)
- Compatible with snnTorch/SpikingJelly frameworks
- Straightforward to explain and implement

---

## Part 4: Results Compared to Conventional NLP Models

### Summary Table: SNN vs. ANN Performance on NLP Tasks

| Model | Task | SNN Accuracy | ANN Baseline | Gap | Energy Savings |
|-------|------|-------------|-------------|-----|---------------|
| SpikeGPT 216M | SST-2 | 88.76% | 91.73% (BERT) | -2.97% | 32x fewer ops |
| SpikeLM | SST-2 | 87.0% | 92.3% (BERT-base) | -5.3% | 12.9x (T=1) |
| SpikeZIP-TF | SST-2 | 93.79% | ~93% (BERT) | ~0% | TBD |
| Spiking CNN | SST-2 | 80.91% | 83.25% (TextCNN) | -2.34% | >10x |
| Spiking CNN | MR | 75.45% | 77.41% (TextCNN) | -1.96% | >10x |
| SpikeLM | MNLI | 77.1% | 83.8% (BERT-base) | -6.7% | 12.9x (T=1) |
| SpikeLM | QNLI | 85.3% | 90.7% (BERT-base) | -5.4% | 12.9x (T=1) |
| SSA-SpiNNaker | IMDB | 100%* | 90% (ANN) | +10%* | 3970J |
| SNNLP | Sentiment | -- | -- | -- | 32x inference |

*The 100% claim from SSA-SpiNNaker is likely on a subset and should be treated with skepticism.

### Key Findings:
1. **Simple binary tasks (SST-2, MR, IMDB):** SNNs achieve within 1-5% of ANNs
2. **Complex multi-class/inference tasks (MNLI, CoLA):** Gap widens to 6-21%
3. **Energy efficiency:** Consistently 10-60x better than ANNs
4. **The conversion approach (train ANN, convert to SNN) generally outperforms direct SNN training**
5. **More timesteps = better accuracy but more energy; the sweet spot is 50-100 timesteps**

---

## Part 5: Is This Too Ambitious for an Undergraduate?

### Honest Assessment

**Yes, full-scale SNN-NLP (like SpikingBERT or SpikeLM) is too ambitious.** But a focused, well-scoped subset is achievable and would be exceptional.

### Why Full-Scale Is Too Hard:
1. **Multi-stage training pipelines** (SpikingBERT requires 3 distillation stages, 200+ epochs)
2. **Custom CUDA kernels** (SpikeLM is 7.6% CUDA code)
3. **Multi-GPU requirements** (SpikingBERT uses DataParallel on 8 A800 GPUs for pretraining)
4. **Novel neuron models** that are not in standard frameworks
5. **Sparse documentation** -- most repos have minimal README and no tutorials
6. **Debugging SNN dynamics** is fundamentally harder than debugging ANNs

### Why a Focused Subset IS Feasible:

1. **The Spiking CNN text classification pipeline is well-documented** with public code (https://github.com/Lvchangze/snn). This is the most reproducible entry point.
2. **Binary sentiment analysis is a simple, well-understood task** with clean datasets (IMDB, SST-2)
3. **The conversion + fine-tuning approach** means you train a standard ANN first (familiar territory), then convert
4. **snnTorch is well-documented** with 14+ tutorials, even if none are NLP-specific
5. **The SpiNNaker sentiment analysis paper** shows the task is achievable with simple models
6. **The novelty is in the APPLICATION, not the architecture** -- you don't need to invent new neuron models

### Comparison with Other Undergraduate SNN Projects Found:

| Project | Scope | Result |
|---------|-------|--------|
| Shape Detector SNN (Manchester BSc) | Single-layer SNN for shapes | Clean project, 107 commits |
| Musical Pattern SNN (BEng) | Audio pattern recognition | "Only small portion achieved" |
| SNN for Digit Recognition (KCL) | MNIST classification | Successful |
| **Proposed: SNN for Sentiment** | **Binary text classification** | **Higher novelty, similar complexity** |

**An SNN sentiment analysis project would be MORE novel than any existing undergraduate SNN project found in our research.**

---

## Part 6: Simpler NLP Tasks Where SNNs Could Work

### Tier 1: Most Feasible (Recommended)

#### Binary Sentiment Analysis (IMDB or SST-2)
- **Why:** Clear positive/negative labels, large datasets, proven SNN results
- **Expected accuracy:** ~80-88% (vs ~90-93% for ANNs)
- **Datasets:** IMDB (50K reviews), SST-2 (67K sentences)
- **Approach:** TextCNN -> convert to Spiking CNN -> fine-tune
- **Code base:** https://github.com/Lvchangze/snn

#### Spam Detection (Binary Classification)
- **Why:** Even simpler than sentiment -- highly separable classes
- **Datasets:** SMS Spam Collection (5,574 messages), Enron spam dataset
- **Expected accuracy:** Should exceed 90% (spam is easier than sentiment)
- **Approach:** Same conversion pipeline, simpler features
- **Note:** No existing SNN spam detection paper found -- this would be novel

### Tier 2: Moderately Feasible

#### Subjectivity Classification
- **Why:** Binary task (subjective vs objective), proven SNN results (90.60% on Subj dataset)
- **Dataset:** Subj dataset
- **Approach:** Same as above

#### News Topic Classification (Binary Subset)
- **Why:** Take AG News or 20 Newsgroups, reduce to 2 categories
- **Dataset:** AG News (subset), 20 Newsgroups (subset)
- **Approach:** Conversion pipeline with simple features

### Tier 3: Stretch Goals

#### Multi-class Sentiment (SST-5)
- **Why:** 5-class fine-grained sentiment is harder
- **Expected accuracy:** ~41-51% (vs ~45-53% for ANNs)
- **Risk:** Performance drop is more noticeable with more classes

#### Aspect-Based Sentiment
- **Why:** Recent paper (Jan 2026) on "Efficient Aspect Term Extraction using SNN"
- **Risk:** More complex architecture, less available code

---

## Part 7: Open-Source Code Available

### Directly Relevant Repositories

| Repository | What | Stars | Quality | Usability |
|-----------|------|-------|---------|-----------|
| [Lvchangze/snn](https://github.com/Lvchangze/snn) | Spiking CNN text classification | ~Low | HIGH | Best starting point |
| [ridgerchu/SpikeGPT](https://github.com/ridgerchu/SpikeGPT) | SpikeGPT implementation | ~High | MEDIUM | Complex to run |
| [NeuroCompLab-psu/SpikingBERT](https://github.com/NeuroCompLab-psu/SpikingBERT) | SpikingBERT distillation | ~Low | MEDIUM | Needs multi-GPU |
| [Xingrun-Xing/SpikeLM](https://github.com/Xingrun-Xing/SpikeLM) | SpikeLM ICML 2024 | ~Low | MEDIUM | Needs 8xA800 GPUs |
| [Xingrun-Xing2/SpikeLLM](https://github.com/Xingrun-Xing2/SpikeLLM) | SpikeLLM ICLR 2025 | ~43 | LOW | Sparse docs |

### SNN Frameworks (General Purpose)

| Framework | Docs | NLP Support | Best For |
|-----------|------|-------------|----------|
| [snnTorch](https://github.com/jeshraghian/snntorch) | Excellent (14+ tutorials) | No NLP tutorials | Building custom SNN models |
| [SpikingJelly](https://github.com/fangwei123456/spikingjelly) | Good | Limited | PyTorch-native SNN ops |
| [Norse](https://github.com/norse/norse) | Good | None | Research-oriented |
| [BindsNET](https://github.com/BindsNET/bindsnet) | Fair | None | Biologically plausible learning |

### Curated Paper Lists

| Resource | URL |
|----------|-----|
| Awesome Spiking Neural Networks (TheBrainLab) | https://github.com/TheBrainLab/Awesome-Spiking-Neural-Networks |
| Awesome SNNs (yfguo91) | https://github.com/yfguo91/Awesome-Spiking-Neural-Networks |
| SNN Daily ArXiv | https://github.com/SpikingChen/SNN-Daily-Arxiv |
| SpikingJelly publications | https://github.com/fangwei123456/spikingjelly/blob/master/publications.md |

---

## Part 8: Feasibility vs. Novelty Tradeoff Assessment

### Novelty Score: 9/10

- Almost no undergraduate has attempted SNN for NLP anywhere in the world
- The field itself is only 2-3 years old (first papers: 2023)
- Even PhD-level work in this area is rare and published at top venues (AAAI, ICML, ICLR)
- An undergraduate demonstrating SNN text classification with energy analysis would be **publishable**
- The SpiNNaker connection at Manchester makes this particularly compelling

### Feasibility Score: 5/10 (Full-scale) / 7/10 (Scoped correctly)

**Risk factors:**
- Text-to-spike encoding is non-trivial and under-documented
- SNN training dynamics are harder to debug than ANNs
- No NLP-specific tutorials in any SNN framework
- Performance will be worse than conventional models (this must be framed as an energy-efficiency study)
- GPU requirements can be managed if using conversion approach (not pre-training from scratch)

**Mitigating factors:**
- The Spiking CNN code (Lvchangze/snn) provides a working pipeline
- Binary sentiment is a well-understood task
- snnTorch is well-documented for the SNN fundamentals
- The conversion approach means you start with a working ANN
- Energy analysis is straightforward to add

### Recommended Scope for Undergraduate Thesis

**Title suggestion:** "Energy-Efficient Sentiment Analysis Using Spiking Neural Networks: A Conversion-Based Approach"

**Minimum Viable Project:**
1. Train a TextCNN on IMDB (binary sentiment) -- standard ANN baseline
2. Convert to Spiking CNN using the conversion pipeline from Lvchangze/snn
3. Fine-tune with surrogate gradients
4. Compare accuracy vs. ANN baseline
5. Measure/estimate energy consumption
6. Report on 2 datasets (IMDB + SST-2)

**Stretch goals (if time permits):**
- Test different encoding schemes (Poisson vs. latency coding)
- Try spam detection as a second task
- Vary number of timesteps and analyze accuracy-energy tradeoff
- Test adversarial robustness (SNNs are reportedly more robust)
- Deploy on SpiNNaker (if Manchester access is available)

**Expected deliverables:**
- Jupyter notebooks with reproducible experiments
- Accuracy comparison table (SNN vs. ANN)
- Energy efficiency analysis
- 40-60 page thesis report

---

## Part 9: Confidence Assessment

| Finding | Confidence | Source Quality |
|---------|-----------|---------------|
| SpikeGPT achieves ~88% on SST-2 | HIGH | Published paper with code |
| SpikeLM closes gap to 6.7% from BERT | HIGH | ICML 2024 paper with detailed tables |
| Spiking CNN drops ~2.5% accuracy on text tasks | HIGH | ICLR 2023 poster with code and 6 datasets |
| Energy savings of 10-60x are real | MEDIUM | Multiple papers claim this, but measured differently |
| SSA-SpiNNaker achieves 100% on IMDB | LOW | Likely tested on subset, claims seem inflated |
| Undergraduate can complete binary sentiment SNN | MEDIUM-HIGH | Based on scope comparison with existing UG projects |
| SNN NLP will remain niche for 2-3 more years | MEDIUM | Based on publication trajectory |

---

## Part 10: Research Gaps and Follow-Up Recommendations

### What Could Not Be Determined:
1. Exact SpikingBERT accuracy numbers on individual GLUE tasks (paper PDF behind some access barriers)
2. Whether any UK undergraduate has attempted SNN-NLP (likely not, but cannot confirm)
3. Actual wall-clock training times for the conversion pipeline on consumer GPUs
4. Whether snnTorch has any undocumented NLP examples in development

### Recommended Follow-Up Actions:
1. **Download and run** the Lvchangze/snn code on a sample dataset to verify it works
2. **Contact Jason Eshraghian** (SpikeGPT author, very active on Twitter/GitHub) -- he is known to be supportive of students
3. **Check if Manchester SpiNNaker access is available** for deployment experiments
4. **Read the full SpikeLM paper** (https://arxiv.org/abs/2406.03287) for architecture details
5. **Start with snnTorch tutorials 1-5** to build SNN fundamentals before tackling text

---

## Key Sources

### Papers
- [SpikeGPT: Generative Pre-trained Language Model with SNNs](https://arxiv.org/abs/2302.13939) - UC Santa Cruz, 2023
- [SpikingBERT: Distilling BERT to Train Spiking Language Models](https://arxiv.org/abs/2308.10873) - Penn State, AAAI 2024
- [SpikeLM: Towards General Spike-Driven Language Modeling](https://arxiv.org/abs/2406.03287) - ICML 2024
- [SpikeLLM: Scaling up SNN to Large Language Models](https://arxiv.org/abs/2407.04752) - ICLR 2025
- [Spiking CNNs for Text Classification](https://arxiv.org/abs/2406.19230) - ICLR 2023
- [Neuromorphic Sentiment Analysis Using SNNs](https://pmc.ncbi.nlm.nih.gov/articles/PMC10536645/) - PMC 2023
- [SNNLP: Energy-Efficient NLP Using SNNs](https://arxiv.org/abs/2401.17911) - 2024
- [SpikeZIP-TF: Conversion for Transformer-based SNN](https://arxiv.org/abs/2406.03470) - ICML 2024
- [SpikingMiniLM: Energy-efficient Spiking Transformer for NLU](https://link.springer.com/article/10.1007/s11432-024-4101-6) - 2024
- [Leveraging SNNs for Topic Modeling](https://www.sciencedirect.com/science/article/pii/S0893608024004180) - 2024

### Code Repositories
- [SpikeGPT](https://github.com/ridgerchu/SpikeGPT)
- [SpikingBERT](https://github.com/NeuroCompLab-psu/SpikingBERT)
- [SpikeLM](https://github.com/Xingrun-Xing/SpikeLM)
- [SpikeLLM](https://github.com/Xingrun-Xing2/SpikeLLM)
- [Spiking CNN for Text Classification](https://github.com/Lvchangze/snn)
- [snnTorch](https://github.com/jeshraghian/snntorch)
- [SpikingJelly](https://github.com/fangwei123456/spikingjelly)
- [Awesome SNNs (TheBrainLab)](https://github.com/TheBrainLab/Awesome-Spiking-Neural-Networks)

### Tutorials and Documentation
- [snnTorch Tutorials](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html)
- [SpikeGPT UCSC News](https://news.ucsc.edu/2023/03/eshraghian-spikegpt/)
- [SNN Encoding Techniques Survey](https://link.springer.com/article/10.1007/s11063-021-10562-2)
