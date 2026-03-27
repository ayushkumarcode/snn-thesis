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

