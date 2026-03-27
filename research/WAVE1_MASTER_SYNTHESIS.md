# Synthesis of 21 past Manchester theses + course guidance + SNN research

i went through 21 Manchester BSc thesis papers, 5 course guidance documents, and all the existing SNN research i'd already gathered to figure out what actually works for a thesis at this level.

---

## 1. Paper analysis summary

| # | Author | Topic | SNN Relevance | Novelty Level | Word Count | Key Pattern |
|---|--------|-------|---------------|---------------|------------|-------------|
| 1 | Andrei Hutu | Privacy-preserving billing (crypto) | 1/5 | (c)/(d) Novel protocol | ~12,827 | Conference-grade work; gap analysis in intro |
| 2 | Thomas Gill | QBF solver optimisation | 1/5 | (c)/(d) Novel pre-resolution | ~14,712 | Supervisor-suggested novel technique |
| 3 | **Tyler Gaffey** | **SNN music genre classification** | **5/5** | **(b) Apply to new domain** | **~21,588** | **Directly SNN; Oliver Rhodes supervisor** |
| 4 | Shay Boual | Cellular automata GOE | 2/5 | (c) Extension + optimisation | ~14,984 | Theory + implementation + evaluation triad |
| 5 | Yi Wu | EEG classification (SVM) | 2/5 | (b) Application | ~10,896 | Descoped original goal; 66.7% accuracy sufficient |
| 6 | Vishal K. Sekar | Alzheimer's prediction (ML) | 2/5 | (b) Comparison | ~14,982 | 749 ensemble configs; breadth over depth |
| 7 | Asma Ali | MANET/FANET simulation | 1/5 | (b) Existing methods, new condition | ~12,000 | Comparative simulation study |
| 8 | Salman Ashraf | Fake news detection (BERT) | 2/5 | (b)/(c) Novel combination | ~13,967 | 9 model configs; in-domain vs out-of-domain |
| 9 | Alexander Havlin | ZK-SNARKs + CNN smart contracts | 2/5 | (b)/(c) Integration novelty | N/A | Two complex domains integrated |
| 10 | Robert Chiru | Motion diffusion (UNet->Transformer) | 2/5 | (c) Modification + novel augmentation | ~70 pages | Testable hypotheses; ablation studies |
| 11 | Jakub Rozanski | Low-light CV pedestrian guidance | 2/5 | (b)/(c) Application + modification | ~14,168 | Multi-axis evaluation; custom dataset |
| 12 | Rose Halsall | 3D hand modelling + texture | 1.5/5 | (b)/(c) Integration novelty | ~14,472 | Pipeline approach; success criteria upfront |
| 13 | Hanin Muhammad Amri | LLM cultural alignment (evo opt) | 1/5 | (c) Novel combination | ~14,234 | Clear research questions; quantitative results |
| 14 | Alexandru Buburuzan | Diffusion inpainting (multimodal) | 1/5 | **(d) Novel method** | ~14,074 | **Outlier -- conference-grade, the ceiling** |
| 15 | Patrick Gransbury | Mathematics of Transformers | 2/5 | (c) Modification + extension | ~15,000 | Math derivation + implementation + novel "Enough Attention" |
| 16 | **Brian Ezinwoke** | **SNN for HFT price spikes** | **5/5** | **(c) Extension of existing** | **~14,695** | **Directly SNN; Oliver Rhodes supervisor; STDP + Bayesian Opt** |
| 17 | Shubham Aggarwal | Drone landing (Decision Transformer) | 2/5 | (b) Application | ~12,345 | Delivered 1/3 objectives; rest as "future work" |
| 18 | Maximilian Bolt | Adversarial attacks on LLMs | 2/5 | (c)/(d) Novel metric | ~11,525 | Novel "cost" metric; human survey validation |
| 19 | Benjamin Hatton | Slimmable NNs on NVDLA hardware | 2/5 | (c) Integration novelty | ~13,807 | Hardware-software integration project |
| 20 | Patrick Devine | ECG analysis web tool (Django) | 2/5 | (b) Application | ~10,940 | NHS collaboration; proof-of-concept sufficient |
| 21 | Nathan Oldfield | Ethics of neuromorphic computing | 4/5 | (b) Qualitative research | N/A | **No code at all; purely qualitative; still passed** |

---

## 2. Patterns i noticed

### 2.1 Novelty distribution
- **Level (b) -- apply existing methods to new domain:** 8/21 papers (38%)
- **Level (c) -- modify/extend existing methods:** 11/21 papers (52%)
- **Level (d) -- novel method:** 1/21 papers (5%) -- the outlier (Buburuzan)
- **Level (a) -- pure replication:** 0/21 papers (0%)

The sweet spot is (b) or (c). Nobody does pure replication, but genuinely novel methods are rare and not expected. Most successful theses take existing techniques and either apply them to a new context or extend them a bit.

### 2.2 Word count distribution
- **Range:** 10,896 -- 21,588 words
- **Median:** ~14,000 words
- **Most common range:** 12,000 -- 15,000 words

### 2.3 Structural patterns (basically universal)
Every strong paper followed this intro structure:
1. **Motivation** (real-world context, statistics, why this matters)
2. **Problem statement** (specific gap or question)
3. **Aims & objectives** (numbered, measurable, with success criteria)
4. **Evaluation strategy** (metrics defined before experiments)
5. **Report structure** (chapter-by-chapter roadmap)

### 2.4 What makes a paper stand out
- **Quantitative results in the abstract** (specific numbers, not vague claims)
- **Explicit success criteria** defined before experiments
- **Multi-axis evaluation** (not just accuracy -- also efficiency, robustness, failure analysis)
- **Honest limitation acknowledgment** (valued, not penalised)
- **Comparison against at least one baseline**
- **Negative results reported and analysed** (several papers actually gained credibility from honest failures)

### 2.5 Common patterns that are apparently fine
- Descoping from original ambitious goals (Yi Wu, Aggarwal)
- Delivering fewer objectives than planned with rest as "future work" (Aggarwal)
- Supervisor-suggested novel contributions (Gill)
- Modest accuracy/results with thorough analysis (Yi Wu's 66.7% was sufficient)
- Proof-of-concept framing without full deployment (Devine)
- No code at all for a research project (Oldfield)

---

## 3. The two directly SNN papers (most relevant precedents)

### Tyler Gaffey (2024) -- SNN for music genre classification
- **Supervisor:** Oliver Rhodes (SpiNNaker group)
- **Framework:** snnTorch + PyTorch + librosa
- **Approach:** Compared 5 spike encoding methods on audio spectrograms
- **Key result:** Poisson encoding matched ANN performance
- **Novelty:** Application to underexplored domain + thorough comparison
- **Lessons:** The "can SNNs compete with ANNs on X?" framing works. Negative results (autoencoders failed, CNNs underperformed) were valued. Honest about compute constraints.

### Brian Ezinwoke (2025) -- SNN for HFT price spike prediction
- **Supervisor:** Oliver Rhodes (SpiNNaker group)
- **Framework:** Custom implementation
- **Approach:** Extended existing STDP architecture (Gao et al.) + Bayesian Optimisation with novel PSA metric
- **Key result:** 17.44% return, Sharpe Ratio 19.71 outperforming supervised baseline
- **Novelty:** Extension of existing architecture + novel evaluation metric
- **Lessons:** "Extend, don't invent" works. Same supervisor sets consistent expectations. Quantitative evaluation with concrete metrics is essential.

---

## 4. Marking criteria (COMP30040)

| Component | Weight | What Matters |
|-----------|--------|-------------|
| **Report** | **55%** | Writing quality, critical analysis, evaluation depth |
| **Achievements** | **30%** | Working output, demonstrated via weekly supervisor meetings |
| **Screencast** | **15%** | 8-min video explaining project + results |
| **Q&A** | 0% (informs Achievements) | 25 min with second marker; validates understanding |

The report is worth almost DOUBLE the code/achievements. A brilliant analysis of modest results beats impressive results with shallow analysis.

What the documents never say: "novel", "original", "groundbreaking", "publishable". They care about: clear objectives, proper methodology, thorough evaluation, honest reflection.

Biggest differentiator between 2:1 and first: the critical appraisal section. A first-class student explains WHY results are limited, WHAT they learned, HOW their approach compares to alternatives, and WHAT they'd do differently.

---

## 5. Novelty expectations (UK undergrad level)

- **QAA Level 6 (Bachelor's):** Requires "self-direction and originality in tackling problems" -- NOT original research
- **Even Cambridge:** A genuine contribution to the field is "not a requirement" for highest marks
- **Edinburgh (for Master's!):** "Not expected that the dissertation will report notable or original contributions to knowledge"
- **A first (70-79) requires zero novelty** if you demonstrate thorough understanding + rigorous evaluation
- **80+ requires:** Independent thought, ambitious scope, thorough evaluation with statistical rigour

---

## 6. What research i'd already done

### Top application domains ranked:
1. **Audio SHD/SSC** -- Easiest path, pre-encoded spikes, SNNs beat ANNs
2. **ECG/Heartbeat** -- Nearly as easy, great clinical narrative, snnTorch delta encoding
3. **Audio Keyword Spotting (GSC)** -- Extends SHD to raw audio
4. **Network Intrusion Detection** -- Tabular data, good narrative
5. **EEG/BCI** -- Feasible but more preprocessing complexity
6. **Time-Series Forecasting** -- High novelty but finicky training
7. **NLP/Text** -- Highest novelty but highest risk

### Technical stack decisions:
- **Training:** Surrogate gradient (recommended) > ANN-to-SNN conversion > STDP
- **Framework:** snnTorch (learning/general) or SpikingJelly (DVS128/performance)
- **Neuron model:** LIF (standard) or PLIF (1-2% better, learnable decay)
- **Encoding:** Rate coding (baseline) > Direct coding (best accuracy) > TTFS (best energy)
- **Energy measurement:** NeuroBench or manual SynOps counting (no hardware needed)

### Previously identified project directions:
1. Framework Shootout (snnTorch vs SpikingJelly on SHD + DVS128)
2. SNN on ESC-50/UrbanSound8K (zero prior papers)
3. SNN for Music Generation (MIDI)
4. SNN for Plant Disease (PlantVillage)
5. Multi-dimensional SNN vs ANN comparison

---

## 7. How to frame the project

Based on all 21 papers, the formula that works at Manchester BSc level:

```
1. Pick a well-defined task/domain
2. Apply existing SNN methods to it (or extend modestly)
3. Compare against a baseline (ANN or alternative SNN config)
4. Evaluate along multiple axes (accuracy + efficiency + at least one more)
5. Analyse honestly (including failures and limitations)
6. Frame as "comparative study" or "investigation of X" rather than "I invented Y"
```

### Three viable framings:

**Framing A: "SNN for Domain X" (Novel Application)**
> "Can Spiking Neural Networks achieve competitive performance on [domain] classification while offering energy efficiency advantages?"
- Pick a domain with zero or few SNN papers
- The novelty is automatic (first SNN results on this data)
- Examples: Environmental sound, plant disease, ECG on PTB-XL

**Framing B: "SNN Comparison Study" (Framework/Method Study)**
> "A comparison of [X] across [Y conditions] for spiking neural network classification"
- Pick 2-3 things to vary (frameworks, encoding methods, neuron models, architectures)
- The novelty is in the breadth and rigour of comparison
- Examples: snnTorch vs SpikingJelly on SHD + DVS128, encoding method comparison on audio

**Framing C: "SNN Extension" (Modify Existing Approach)**
> "Extending [existing SNN method] with [modification] for [task]"
- Take a published SNN architecture and add something (new encoding, new regularisation, new evaluation metric)
- The novelty is in the specific modification
- Examples: STDP + Bayesian Optimisation (like Ezinwoke), adding NeuroBench energy metrics to existing benchmarks

All three framings have worked at Manchester. Choose based on personal interest and available time.

---

## 8. Decision matrix

| Direction | Effort | Novelty | Risk | Grade Ceiling | Natural SNN Fit |
|-----------|--------|---------|------|---------------|-----------------|
| SHD Audio Classification | LOW | Moderate | LOW | First (75+) | HIGH (temporal data) |
| ECG Heartbeat Detection | LOW-MED | High | LOW | First (78+) | HIGH (spike-like QRS) |
| Environmental Sound (ESC-50) | LOW | VERY HIGH | LOW | First (80+) | HIGH (temporal) |
| DVS128 Gesture Recognition | MEDIUM | Low-Mod | LOW | First (75+) | VERY HIGH (native events) |
| Framework Comparison | LOW-MED | Genuine gap | LOW | First (75+) | N/A (meta-study) |
| Plant Disease (PlantVillage) | LOW | VERY HIGH | LOW-MED | First (78+) | LOW (static images) |
| Music Generation (MIDI) | MED | VERY HIGH | MEDIUM | First (80+) | HIGH (event-like) |
| Multi-dim SNN vs ANN | MEDIUM | Moderate | LOW | First (72-78) | N/A (meta-study) |
| NLP Sentiment Analysis | HIGH | EXTREME | HIGH | First (82+) if done | LOW (text) |
| Time-Series Forecasting | HIGH | High | MEDIUM-HIGH | First (80+) if done | HIGH (temporal) |

---

## 9. The pragmatic recommendation

Given a "just get it done" mindset, here's what makes sense:

### Pick ONE from the top 3, commit, start coding:

**Option 1: SNN on Environmental Sound (ESC-50)** -- best novelty-to-effort ratio
- Zero prior SNN papers = automatic novelty
- Mel-spectrograms -> rate encoding -> snnTorch convolutional SNN
- ANN baseline (same architecture with ReLU) for comparison
- Energy comparison via NeuroBench
- ~200-400 lines of code; 4-6 weeks to working results

**Option 2: SNN on SHD Audio** -- lowest effort, strongest tooling
- Data is pre-encoded as spikes (no encoding pipeline needed)
- snnTorch or SpikingJelly have built-in loaders
- SNNs actually beat ANNs here (rare!) -- great narrative
- Add value with: encoding comparison, neuron model comparison, or timestep analysis
- ~200 lines for baseline; 2-4 weeks to working results

**Option 3: SNN ECG Classification** -- best real-world narrative
- snnTorch has built-in delta encoding for ECG
- MIT-BIH is small and clean
- PTB-XL 12-lead is virtually untouched = strong novelty
- Clinical relevance makes a compelling motivation section
- ~300-500 lines of code; 4-6 weeks to working results

### Report strategy (worth 55% of the mark):
1. Strong motivation section grounded in real-world context
2. 3-4 numbered objectives with explicit success criteria
3. Thorough background chapter (~20% of word count)
4. Multi-axis evaluation (accuracy + energy + at least one more dimension)
5. Honest critical appraisal with lessons learned
6. Future work section showing awareness of what's next

---

## 10. Web research updates (Feb 2026)

### 10.1 ESC-50 / Environmental Sound -- confirmed zero SNN papers
A March 2025 peer-reviewed paper (arxiv 2503.11206) explicitly states: *"No state-of-the-art solution has yet encoded environmental sound datasets using spike-based methods and performed classification with a spiking neural network."*
- That paper only benchmarked encoding methods on ESC-10 (10-class subset), getting ~69% accuracy with a basic 4-layer LIF network
- No published SNN paper reports results on the full ESC-50 dataset (50 classes)
- One paper (SATRN) achieves 95.5% on UrbanSound8K but not ESC-50
- This is the single strongest novelty opportunity i found

### 10.2 DVS128 Gesture -- confirmed saturated
- SOTA: 99.59% (TENNs-PLEIADES, not even an SNN)
- SNN-only SOTA: ~99.3% (SG-SNN)
