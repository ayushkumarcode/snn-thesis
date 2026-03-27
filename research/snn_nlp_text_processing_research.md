# spiking neural networks for NLP / text tasks

i looked into whether SNNs can do NLP stuff -- SpikingBERT, SpikeGPT, SpikeLLM, text encoding as spikes, sentiment analysis. this is a genuinely novel and rapidly emerging direction that's only really been viable since 2023. there are some landmark papers (SpikingBERT at AAAI 2024, SpikeGPT from UCSC, SpikeLM at ICML 2024, SpikeLLM at ICLR 2025), but performance still lags behind conventional models by 3-15% on standard benchmarks. the main argument is energy efficiency (10-60x reduction), not accuracy.

for an undergrad thesis: exceptionally high novelty but significant technical risk. the most feasible approach would be a focused binary sentiment classification task using ANN-to-SNN conversion with pre-trained word embeddings encoded as Poisson spike trains. open-source code exists but needs adaptation. a realistic project would compare SNN vs ANN on 2-3 text datasets and quantify the accuracy-energy tradeoff.

high risk, high reward. very few undergrads have tried this worldwide. it'd be a standout thesis if scoped right, but needs careful management to avoid scope creep.

---

## the key models -- what are they and do they work?

### SpikeGPT (UC Santa Cruz, Feb 2023)

first generative pre-trained language model built with SNNs. uses an RWKV-inspired architecture (not standard transformer). replaces self-attention with linear-complexity spiking mechanism. 45M and 216M parameter variants.

does it work? partially:

| Benchmark | SpikeGPT 216M | BERT |
|-----------|---------------|------|
| SST-2 (sentiment) | 88.76% | 91.73% |
| SST-5 (5-class) | 51.27% | 53.21% |
| MR (movie reviews) | 85.63% | 86.72% |
| Subj (subjectivity) | 95.30% | N/A |
| WikiText-2 (perplexity) | 18.01 PPL | -- |

claims 32.2x fewer operations on neuromorphic hardware. works competitively on simple classification (within 1-3% of BERT). generation is weaker. the 216M pretrained model is needed -- 45M underperforms badly.

code: https://github.com/ridgerchu/SpikeGPT

---

### SpikingBERT (Penn State, AAAI 2024)

spiking language model created by distilling BERT into a spiking architecture using implicit differentiation. novel spiking attention mechanism based on Average Spiking Rate convergence at equilibrium. 3-stage training pipeline: general KD, task-based internal layer KD, prediction layer KD.

works on GLUE tasks (SST-2, MNLI, QQP, QNLI, RTE, MRPC, STS-B). first spiking LM evaluated on multiple GLUE tasks. without distillation, 4-5% performance loss. with distillation, competitive but still below BERT-base.

not trivial to reproduce -- 125 convergence timesteps, multi-GPU DataParallel needed.

code: https://github.com/NeuroCompLab-psu/SpikingBERT

---

### SpikeLM (ICML 2024)

first *fully spiking* mechanism for general language tasks (discriminative + generative). introduces "elastic bi-spiking" -- spikes have bi-directional amplitude and frequency encoding while staying additive.

this is currently the best-performing spiking language model:

| Task | BERT-base | SpikeLM | Gap |
|------|-----------|---------|-----|
| SST-2 | 92.3% | 87.0% | -5.3% |
| MNLI-m/mm | 83.8/83.4 | 77.1/77.2 | -6.7/-6.2% |
| QQP (F1) | 90.5 | 83.9 | -6.6% |
| QNLI | 90.7 | 85.3 | -5.4% |
| CoLA | 60.0 | 38.8 | -21.2% |
| STS-B | 89.4 | 84.9 | -4.5% |
| RTE | 69.3 | 69.0 | -0.3% |
| **Avg gap** | -- | -- | **~6.7%** |

reduces the gap from 28.3% (LIF-BERT) to 6.7% vs BERT-base. 12.9x energy savings at T=1, 3.7x at T=4.

code: https://github.com/Xingrun-Xing/SpikeLM

---

### SpikeLLM (ICLR 2025)

scales SNNs to large language models (7B-70B parameters) using saliency-based spiking. first attempt at billion-parameter spiking LLMs. uses Generalized Integrate-and-Fire neurons with saliency detection.

results on LLaMA-2:

| Model | Config | WikiText2 PPL | Zero-shot Avg |
|-------|--------|---------------|---------------|
| LLaMA-2-7B | 4W4A | -- | 50.65% |
| LLaMA-2-7B | 2W16A | 14.16 | -- |
| LLaMA-2-70B | 2W16A | 6.35 | 59.93% |

impressive that spiking works at 70B. but honestly this is more of a quantization/compression technique than a from-scratch spiking model.

code: https://github.com/Xingrun-Xing2/SpikeLLM (sparse docs)

---

### SpikeZIP-TF (ICML 2024)

lossless ANN-to-SNN conversion for transformers. ANN and SNN are *exactly equivalent* -- zero accuracy degradation.

gets **93.79% on SST-2** (highest SNN result for this benchmark). outperforms SpikeGPT and SpikingBERT.

if you want to show SNN equivalence with no accuracy loss, this is the strongest approach. but energy savings may be smaller since exact conversion needs more timesteps.

paper: https://arxiv.org/abs/2406.03470

---

### SpikingMiniLM (2024)

energy-efficient spiking transformer for NLU. targets MiniLM (smaller than BERT). multistep encoding for text embeddings to spike trains. might be more feasible for an undergrad project due to smaller size.

paper: https://link.springer.com/article/10.1007/s11432-024-4101-6

---

## SNNs for sentiment analysis and text classification

several groups have done this successfully:

### spiking CNN for text classification (ICLR 2023)
the most relevant paper for an undergrad project. conversion + fine-tuning of TextCNN, Poisson spike trains from word embeddings.

| Dataset | TextCNN | Spiking CNN | Drop |
|---------|---------|-------------|------|
| MR | 77.41% | 75.45% | -1.96% |
| SST-2 | 83.25% | 80.91% | -2.34% |
| Subj | 94.00% | 90.60% | -3.40% |
| SST-5 | 45.48% | 41.63% | -3.85% |
| ChnSenti | 86.74% | 85.02% | -1.72% |
| Waimai | 88.49% | 86.66% | -1.83% |

average drop: 2.51%. energy: >10x reduction. also +13.55% robust accuracy under adversarial attacks. 50 timesteps.

code: https://github.com/Lvchangze/snn -- **best starting point for a project**

### other work

- **SSA-SpiNNaker (PMC, 2023):** binary sentiment on IMDB, ANN-to-SNN conversion, deployed on SpiNNaker. claims 100% accuracy (likely on a subset, seems inflated).
- **Energy-Efficient Sentiment (ICANN 2023):** SNN energy reduced to 1.36% of a transformer (64.93x improvement)
- **SNNLP (Jan 2024):** novel text-to-spike encoding outperforms Poisson rate-coding by ~13%. 32x more efficient at inference, 60x during training.
- **SNN Topic Modeling (2024):** STDP-based, each neuron represents a topic.

---

## how to encode text as spikes

this is the central technical challenge. four main approaches:

### 1. Poisson rate coding (most common, recommended)

take pre-trained word embeddings (Word2Vec, GloVe), normalize to [0,1], each component generates a Poisson spike train with firing rate proportional to its value. the Spiking CNN paper details the exact process: calculate mean/std, clip within 3 sigma, normalize, scale, generate Poisson trains.

pros: simple, well-understood, preserves semantic info from pre-trained embeddings, working code exists
cons: needs many timesteps (50-100), stochastic noise, slow inference

### 2. temporal/latency coding

information encoded in spike *timing* -- higher values produce earlier spikes.

pros: more info per spike, faster inference, lower energy
cons: more complex, less robust to noise, less explored for text

### 3. direct embedding (SpikeLM approach)

elastic bi-spiking with bi-directional amplitude and frequency encoding. works with T=1.

pros: fewer timesteps, preserves more info
cons: needs custom neuron models, harder to implement

### 4. Word2Spike (2025)

rate coding with one-to-one mapping from word vectors to spike-based attractor states. preserves semantic structure despite noise.

### 5. integer word index encoding (simplest)

words converted to integer indices, each generates a spike pattern. used in SSA-SpiNNaker.

pros: dead simple
cons: loses semantic information

### what i'd recommend for an undergrad project

**Poisson rate coding with pre-trained embeddings.** best documented, working code exists (Lvchangze/snn), compatible with snnTorch/SpikingJelly, straightforward to explain.

---

## SNN vs ANN performance summary

| Model | Task | SNN Acc | ANN Baseline | Gap | Energy Savings |
|-------|------|---------|-------------|-----|---------------|
| SpikeGPT 216M | SST-2 | 88.76% | 91.73% (BERT) | -2.97% | 32x fewer ops |
| SpikeLM | SST-2 | 87.0% | 92.3% (BERT-base) | -5.3% | 12.9x (T=1) |
| SpikeZIP-TF | SST-2 | 93.79% | ~93% (BERT) | ~0% | TBD |
| Spiking CNN | SST-2 | 80.91% | 83.25% (TextCNN) | -2.34% | >10x |
| Spiking CNN | MR | 75.45% | 77.41% (TextCNN) | -1.96% | >10x |
| SpikeLM | MNLI | 77.1% | 83.8% (BERT-base) | -6.7% | 12.9x |
| SpikeLM | QNLI | 85.3% | 90.7% (BERT-base) | -5.4% | 12.9x |
| SNNLP | Sentiment | -- | -- | -- | 32x inference |

key findings:
1. simple binary tasks (SST-2, MR, IMDB): SNNs within 1-5% of ANNs
2. complex tasks (MNLI, CoLA): gap widens to 6-21%
3. energy efficiency: consistently 10-60x better
4. conversion approach (train ANN, convert to SNN) generally outperforms direct SNN training
5. more timesteps = better accuracy but more energy; sweet spot is 50-100 timesteps

---

## is this too ambitious for an undergrad?

### honest answer

**full-scale SNN-NLP (SpikingBERT, SpikeLM) is too ambitious.** but a focused subset is achievable and would be outstanding.

### why full-scale is too hard:
1. multi-stage training pipelines (SpikingBERT needs 3 distillation stages, 200+ epochs)
2. custom CUDA kernels (SpikeLM is 7.6% CUDA code)
3. multi-GPU requirements (SpikingBERT uses DataParallel on 8 A800 GPUs)
4. novel neuron models not in standard frameworks
5. sparse documentation
6. debugging SNN dynamics is fundamentally harder than ANNs

### why a focused version IS feasible:
1. the Spiking CNN text classification pipeline is well-documented with public code
2. binary sentiment is simple and well-understood with clean datasets (IMDB, SST-2)
3. conversion + fine-tuning means you train a standard ANN first (familiar territory)
4. snnTorch has 14+ tutorials even if none are NLP-specific
5. novelty is in the APPLICATION, not the architecture -- don't need to invent new neuron models

### comparison with other undergrad SNN projects:

| Project | Scope | Result |
|---------|-------|--------|
| Shape Detector SNN (Manchester BSc) | Single-layer SNN for shapes | Clean, 107 commits |
| Musical Pattern SNN (BEng) | Audio pattern recognition | "Only small portion achieved" |
| SNN for Digit Recognition (KCL) | MNIST classification | Successful |
| **SNN for Sentiment** | **Binary text classification** | **Higher novelty, similar complexity** |

---

## simpler NLP tasks that could work

### tier 1: most feasible (recommended)

**binary sentiment analysis (IMDB or SST-2)**
- clear pos/neg labels, large datasets, proven SNN results
- expected accuracy: ~80-88% (vs ~90-93% for ANNs)
- code: https://github.com/Lvchangze/snn

**spam detection**
- even simpler -- highly separable classes
- datasets: SMS Spam Collection (5,574 messages), Enron
- expected accuracy: should exceed 90%
- no existing SNN spam detection paper found -- this would be novel

### tier 2: moderately feasible

**subjectivity classification** -- binary task, proven results (90.60% on Subj dataset)

**news topic classification (binary subset)** -- take AG News or 20 Newsgroups, reduce to 2 categories

### tier 3: stretch goals

**multi-class sentiment (SST-5)** -- 5 classes is harder. expected ~41-51%.

**aspect-based sentiment** -- recent paper (Jan 2026) but more complex, less code available.

---

## open-source code

### directly relevant

| Repo | What | Usability |
|------|------|-----------|
| [Lvchangze/snn](https://github.com/Lvchangze/snn) | Spiking CNN text classification | **Best starting point** |
| [ridgerchu/SpikeGPT](https://github.com/ridgerchu/SpikeGPT) | SpikeGPT | Complex to run |
| [NeuroCompLab-psu/SpikingBERT](https://github.com/NeuroCompLab-psu/SpikingBERT) | SpikingBERT | Needs multi-GPU |
| [Xingrun-Xing/SpikeLM](https://github.com/Xingrun-Xing/SpikeLM) | SpikeLM | Needs 8xA800 |
| [Xingrun-Xing2/SpikeLLM](https://github.com/Xingrun-Xing2/SpikeLLM) | SpikeLLM | Sparse docs |

### SNN frameworks

| Framework | NLP support | Best for |
|-----------|------------|---------|
| snnTorch | No NLP tutorials | Building custom SNN models |
| SpikingJelly | Limited | PyTorch-native SNN ops |
| Norse | None | Research-oriented |
| BindsNET | None | Bio-plausible learning |

---

## novelty vs feasibility

**novelty: 9/10** -- almost no undergrad has attempted this. the field is 2-3 years old. even PhD work here is rare and published at AAAI/ICML/ICLR. an undergrad demonstrating SNN text classification with energy analysis could be publishable.

**feasibility: 5/10 (full-scale) / 7/10 (properly scoped)**

### recommended scope

**minimum viable project:**
1. train TextCNN on IMDB (binary sentiment) -- standard ANN baseline
2. convert to Spiking CNN using Lvchangze/snn pipeline
3. fine-tune with surrogate gradients
4. compare accuracy vs ANN
5. estimate energy consumption
6. report on 2 datasets (IMDB + SST-2)

**stretch goals:**
- test different encodings (Poisson vs latency)
- try spam detection as second task
- vary timesteps, analyze accuracy-energy tradeoff
- test adversarial robustness
- deploy on SpiNNaker

---

## confidence notes
