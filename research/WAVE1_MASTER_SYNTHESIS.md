# Master Synthesis: 21 Undergraduate Paper Analyses + Course Guidance + Existing Research

**Generated:** 2026-02-25
**Purpose:** Collated findings from Wave 1 analysis of 21 Manchester BSc thesis papers, 5 course guidance documents, and all existing SNN research in this repository.

---

## 1. Paper Analysis Summary Table

| # | Author | Topic | SNN Relevance | Novelty Level | Word Count | Key Pattern |
|---|--------|-------|---------------|---------------|------------|-------------|
| 1 | Andrei Hutu | Privacy-preserving billing (crypto) | 1/5 | (c)/(d) Novel protocol | ~12,827 | Conference-grade work; gap analysis in intro |
| 2 | Thomas Gill | QBF solver optimisation | 1/5 | (c)/(d) Novel pre-resolution | ~14,712 | Supervisor-suggested novel technique |
| 3 | **Tyler Gaffey** | **SNN music genre classification** | **5/5** | **(b) Apply to new domain** | **~21,588** | **Directly SNN; Oliver Rhodes supervisor** |
| 4 | Shay Boual | Cellular automata GOE | 2/5 | (c) Extension + optimisation | ~14,984 | Theory + implementation + evaluation triad |
| 5 | Yi Wu | EEG classification (SVM) | 2/5 | (b) Application | ~10,896 | Descoped original goal; 66.7% accuracy sufficient |
| 6 | Vishal K. Sekar | Alzheimer's prediction (ML) | 2/5 | (b) Systematic comparison | ~14,982 | 749 ensemble configs; breadth over depth |
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

## 2. Key Patterns Extracted

### 2.1 Novelty Distribution
- **Level (b) -- Apply existing methods to new domain:** 8/21 papers (38%)
- **Level (c) -- Modify/extend existing methods:** 11/21 papers (52%)
- **Level (d) -- Novel method:** 1/21 papers (5%) -- the outlier (Buburuzan)
- **Level (a) -- Pure replication:** 0/21 papers (0%)

**Takeaway:** The sweet spot is (b) or (c). Nobody does pure replication, but genuinely novel methods are rare and not expected. Most successful theses take existing techniques and either apply them to a new context or extend them modestly.

### 2.2 Word Count Distribution
- **Range:** 10,896 -- 21,588 words
- **Median:** ~14,000 words
- **Most common range:** 12,000 -- 15,000 words

### 2.3 Structural Patterns (Universal)
Every high-quality paper followed this introduction structure:
1. **Motivation** (real-world context, statistics, why this matters)
2. **Problem Statement** (specific gap or question)
3. **Aims & Objectives** (numbered, measurable, with success criteria)
4. **Evaluation Strategy** (metrics defined before experiments)
5. **Report Structure** (chapter-by-chapter roadmap)

### 2.4 What Differentiates Strong Papers
- **Quantitative results in the abstract** (specific numbers, not vague claims)
- **Explicit success criteria** defined before experiments
- **Multi-axis evaluation** (not just accuracy -- also efficiency, robustness, failure analysis)
- **Honest limitation acknowledgment** (valued, not penalised)
- **Comparison against at least one baseline**
- **Negative results reported and analysed** (several papers gained credibility from honest failures)

### 2.5 Common Acceptable Patterns
- Descoping from original ambitious goals (Yi Wu, Aggarwal)
- Delivering fewer objectives than planned with rest as "future work" (Aggarwal)
- Supervisor-suggested novel contributions (Gill)
- Modest accuracy/results with thorough analysis (Yi Wu's 66.7% was sufficient)
- Proof-of-concept framing without full deployment (Devine)
- No code at all for a research project (Oldfield)

---

## 3. The Two Directly SNN Papers (Most Relevant Precedents)

### Tyler Gaffey (2024) -- SNN for Music Genre Classification
- **Supervisor:** Oliver Rhodes (SpiNNaker group)
- **Framework:** snnTorch + PyTorch + librosa
- **Approach:** Systematic comparison of 5 spike encoding methods on audio spectrograms
- **Key result:** Poisson encoding matched ANN performance
- **Novelty:** Application to underexplored domain + thorough comparison
- **Lessons:** The "can SNNs compete with ANNs on X?" framing works; negative results (autoencoders failed, CNNs underperformed) were valued; honest about compute constraints

### Brian Ezinwoke (2025) -- SNN for HFT Price Spike Prediction
- **Supervisor:** Oliver Rhodes (SpiNNaker group)
- **Framework:** Custom implementation
- **Approach:** Extended existing STDP architecture (Gao et al.) + Bayesian Optimisation with novel PSA metric
- **Key result:** 17.44% return, Sharpe Ratio 19.71 outperforming supervised baseline
- **Novelty:** Extension of existing architecture + novel evaluation metric
- **Lessons:** "Extend, don't invent" works; same supervisor sets consistent expectations; quantitative evaluation with concrete metrics is essential

---

## 4. Marking Criteria (COMP30040)

| Component | Weight | What Matters |
|-----------|--------|-------------|
| **Report** | **55%** | Writing quality, critical analysis, evaluation depth |
| **Achievements** | **30%** | Working output, demonstrated via weekly supervisor meetings |
| **Screencast** | **15%** | 8-min video explaining project + results |
| **Q&A** | 0% (informs Achievements) | 25 min with second marker; validates understanding |

**Critical insight:** The report is worth almost DOUBLE the code/achievements. A brilliant analysis of modest results beats impressive results with shallow analysis.

**What the documents never say:** "novel", "original", "groundbreaking", "publishable". They care about: clear objectives, proper methodology, thorough evaluation, honest reflection.

**Biggest differentiator (2:1 vs First):** The critical appraisal section. A First student explains WHY results are limited, WHAT they learned, HOW their approach compares to alternatives, and WHAT they'd do differently.

---

## 5. Novelty Expectations (UK Undergraduate Level)

- **QAA Level 6 (Bachelor's):** Requires "self-direction and originality in tackling problems" -- NOT original research
- **Even Cambridge:** A genuine contribution to the field is "not a requirement" for highest marks
- **Edinburgh (for Master's!):** "Not expected that the dissertation will report notable or original contributions to knowledge"
- **A First (70-79) requires zero novelty** if you demonstrate thorough understanding + rigorous evaluation
- **80+ requires:** Independent thought, ambitious scope, thorough evaluation with statistical rigour

---

## 6. Existing Research Summary (What's Already Been Done in This Repo)

### Top Application Domains Ranked (from domain research):
1. **Audio SHD/SSC** -- Easiest path, pre-encoded spikes, SNNs beat ANNs
2. **ECG/Heartbeat** -- Nearly as easy, great clinical narrative, snnTorch delta encoding
3. **Audio Keyword Spotting (GSC)** -- Extends SHD to raw audio
4. **Network Intrusion Detection** -- Tabular data, good narrative
5. **EEG/BCI** -- Feasible but more preprocessing complexity
6. **Time-Series Forecasting** -- High novelty but finicky training
7. **NLP/Text** -- Highest novelty (9/10) but highest risk

### Technical Stack Decisions (from technical research):
- **Training:** Surrogate gradient (recommended) > ANN-to-SNN conversion > STDP
- **Framework:** snnTorch (learning/general) or SpikingJelly (DVS128/performance)
- **Neuron model:** LIF (standard) or PLIF (1-2% better, learnable decay)
- **Encoding:** Rate coding (baseline) > Direct coding (best accuracy) > TTFS (best energy)
- **Energy measurement:** NeuroBench or manual SynOps counting (no hardware needed)

### Previously Identified Project Directions:
1. Framework Shootout (snnTorch vs SpikingJelly on SHD + DVS128)
2. SNN on ESC-50/UrbanSound8K (zero prior papers)
3. SNN for Music Generation (MIDI)
4. SNN for Plant Disease (PlantVillage)
5. Multi-dimensional SNN vs ANN comparison

---

## 7. Synthesis: What the Papers Tell Us About How to Frame Our Project

### The Proven Thesis Formula
Based on all 21 papers, the formula that works at Manchester BSc level:

```
1. Pick a well-defined task/domain
