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
2. Apply existing SNN methods to it (or extend modestly)
3. Compare against a baseline (ANN or alternative SNN config)
4. Evaluate along multiple axes (accuracy + efficiency + at least one more)
5. Analyse honestly (including failures and limitations)
6. Frame as "comparative study" or "investigation of X" rather than "I invented Y"
```

### The Three Viable Framings for Our SNN Project

**Framing A: "SNN for Domain X" (Novel Application)**
> "Can Spiking Neural Networks achieve competitive performance on [domain] classification while offering energy efficiency advantages?"
- Pick a domain with zero or few SNN papers
- The novelty is automatic (first SNN results on this data)
- Examples: Environmental sound, plant disease, ECG on PTB-XL

**Framing B: "Systematic SNN Comparison" (Framework/Method Study)**
> "A systematic comparison of [X] across [Y conditions] for spiking neural network classification"
- Pick 2-3 things to vary (frameworks, encoding methods, neuron models, architectures)
- The novelty is in the breadth and rigour of comparison
- Examples: snnTorch vs SpikingJelly on SHD + DVS128, encoding method comparison on audio

**Framing C: "SNN Extension" (Modify Existing Approach)**
> "Extending [existing SNN method] with [modification] for [task]"
- Take a published SNN architecture and add something (new encoding, new regularisation, new evaluation metric)
- The novelty is in the specific modification
- Examples: STDP + Bayesian Optimisation (like Ezinwoke), adding NeuroBench energy metrics to existing benchmarks

**All three framings have worked at Manchester. Choose based on personal interest and available time.**

---

## 8. Decision Matrix

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

## 9. The Pragmatic Recommendation

Given the "just get it over with" mindset, here is the optimal strategy:

### Pick ONE from the top 3, commit today, start coding tomorrow:

**Option 1: SNN on Environmental Sound (ESC-50)** -- Best novelty-to-effort ratio
- Zero prior SNN papers = automatic novelty
- Mel-spectrograms -> rate encoding -> snnTorch convolutional SNN
- ANN baseline (same architecture with ReLU) for comparison
- Energy comparison via NeuroBench
- ~200-400 lines of code; 4-6 weeks to working results

**Option 2: SNN on SHD Audio** -- Lowest effort, strongest tooling
- Data is pre-encoded as spikes (no encoding pipeline needed)
- snnTorch or SpikingJelly have built-in loaders
- SNNs actually beat ANNs here (rare!) -- great narrative
- Add value with: encoding comparison, neuron model comparison, or timestep analysis
- ~200 lines for baseline; 2-4 weeks to working results

**Option 3: SNN ECG Classification** -- Best real-world narrative
- snnTorch has built-in delta encoding for ECG
- MIT-BIH is small and clean
- PTB-XL 12-lead is virtually untouched = strong novelty
- Clinical relevance makes a compelling motivation section
- ~300-500 lines of code; 4-6 weeks to working results

### The Report Strategy (worth 55% of the mark):
1. Strong motivation section grounded in real-world context
2. 3-4 numbered objectives with explicit success criteria
3. Thorough background chapter (~20% of word count)
4. Multi-axis evaluation (accuracy + energy + at least one more dimension)
5. Honest critical appraisal with lessons learned
6. Future work section showing awareness of what's next

---

## 10. Wave 2 Research Updates (Web Research, Feb 2026)

### 10.1 ESC-50 / Environmental Sound -- CONFIRMED ZERO SNN PAPERS
A March 2025 peer-reviewed paper (arxiv 2503.11206) explicitly states: *"No state-of-the-art solution has yet encoded environmental sound datasets using spike-based methods and performed classification with a spiking neural network."*
- The paper only benchmarked **encoding methods** on ESC-10 (10-class subset), achieving ~69% accuracy with a basic 4-layer LIF network
- **No published SNN paper reports results on the full ESC-50 dataset** (50 classes)
- One paper (SATRN) achieves 95.5% on UrbanSound8K but not ESC-50
- **This is the single strongest novelty opportunity identified**

### 10.2 DVS128 Gesture -- CONFIRMED SATURATED
- SOTA: 99.59% (TENNs-PLEIADES, not even an SNN)
- SNN-only SOTA: ~99.3% (SG-SNN)
- 14+ papers with code on leaderboard
- DVS128 is now the "MNIST of event-based vision" -- useful for validation but NOT a thesis contribution
- Better alternatives: DailyDVS-200 (ECCV 2024, 200 classes) or EgoEvGesture (March 2025, first-person, SOTA only 62.7%)

### 10.3 Novel Application Gaps -- UPDATED STATUS (Feb 2026)
| Domain | SNN Papers | Status | Verdict |
|--------|-----------|--------|---------|
| **Wildlife Camera Trap** | **0** | Zero papers, excellent datasets (Snapshot Serengeti 3.2M images) | **TOP PICK** |
| **Food Recognition (Food-101)** | **0** | Zero papers, clean benchmark | **STRONG** |
| Plant Disease (PlantVillage) | 1-3 | Only 1 hybrid paper (2021), pure SNN untried | **STRONG** |
| Music Generation (MIDI) | 4-10 | **GAP CLOSED** -- MuSpike benchmark (Aug 2025) tested 5 architectures | AVOID |
| SVHN | 4-10 | Already a standard SNN benchmark | AVOID |
| Wearable HAR | 10+ | Saturated, 2025 comprehensive survey exists | AVOID |

### 10.35 Framework Comparison -- CONFIRMED GENUINE GAP
- No three-way snnTorch vs SpikingJelly vs Norse comparison on real neuromorphic data exists
- The closest work (2025 multimodal benchmark) deliberately EXCLUDES snnTorch and Norse
- Open Neuromorphic benchmark: synthetic data only, appears dormant
- Spyx paper: speed only, explicitly ignores accuracy
- The question "same architecture, same hyperparams, three frameworks -- same accuracy?" has NEVER been answered
- This is a tractable, useful, publication-worthy gap

### 10.4 SHD Audio -- APPROACHING SATURATION
- SOTA: 96.41% (SpikCommander, 2025)
- SHD is still useful as a **secondary validation benchmark** but not as a primary thesis contribution
- SSC (35 classes, SOTA ~82%) has more room for improvement

### 10.4 Revised Top Recommendation

**The single strongest thesis option as of February 2026:**

> **"Spiking Neural Networks for Environmental Sound Classification"**
> - Apply SNNs to ESC-50 (first to do so -- documented, citable gap)
> - Compare spike encoding methods (rate, delta, latency, direct)
> - Benchmark against ANN baselines on same dataset
> - Report energy efficiency via NeuroBench
> - Use SHD/SSC as secondary validation to show method generalises
> - Even 70-85% accuracy on ESC-50 is publishable given zero prior work

**Strong alternative: SNN ECG Classification on PTB-XL**
- Only ~20-30 SNN-ECG papers exist total (small but credible field)
- PTB-XL (21,799 ECGs, 12-lead) has NO comprehensive SNN benchmark
- Best SNN on MIT-BIH: 98.3% (SparrowSNN) vs 99.7% DNN
- No snnTorch/SpikingJelly ECG tutorial exists (contributing one = impact)
- Compelling energy narrative: SNNs 30-1000x less energy for wearable cardiac monitors
- Risk: PTB-XL is 12-lead (more complex than single-lead MIT-BIH)

**Surprise contender: Wildlife Camera Trap Classification**
- ZERO SNN papers exist (confirmed Feb 2026)
- Excellent datasets: Snapshot Serengeti (3.2M images, 48 species), Caltech Camera Traps, iWildCam
- Killer real-world narrative: camera traps are battery-powered in remote locations = SNNs' low-power advantage is directly relevant
- CNN baselines: 90-96% depending on dataset
- Risk: RGB images (not event-based), so less natural SNN fit than audio

**REVISED FINAL TOP 5 (February 2026, web-research verified):**

| Rank | Direction | Novelty | Effort | Risk | Why |
|------|-----------|---------|--------|------|-----|
| **1** | **SNN on ESC-50 (Environmental Sound)** | VERY HIGH (zero papers, peer-review confirmed) | LOW | LOW | Best novelty-to-effort ratio of all options |
| **2** | **SNN ECG on PTB-XL** | HIGH (no comprehensive benchmark) | LOW-MED | LOW | Compelling clinical narrative; 30-1000x energy savings |
| **3** | **SNN Wildlife Camera Trap** | VERY HIGH (zero papers) | LOW-MED | LOW-MED | Outstanding real-world motivation; large datasets |
| **4** | **Framework Shootout (snnTorch vs SpikingJelly vs Norse)** | GENUINE (confirmed gap) | LOW-MED | LOW | Immediately useful to SNN community |
| **5** | **SNN Food Recognition (Food-101)** | VERY HIGH (zero papers) | LOW | LOW | Clean benchmark but weaker real-world narrative |

**Directions REMOVED from previous recommendations:**
- ~~Music Generation (MIDI)~~ -- MuSpike benchmark (Aug 2025) closed this gap
- ~~DVS128 Gesture~~ -- Saturated at 99.59%; the "MNIST of event-based vision"
- ~~SHD as primary~~ -- Approaching saturation at 96.4%; use as secondary validation only
- ~~SVHN~~ -- Already a standard SNN benchmark
- ~~Wearable HAR~~ -- Saturated with 10+ papers and a 2025 survey

**Why ESC-50 is the #1 pick:**
1. Zero prior work = automatic novelty (no risk of being scooped by something that already exists)
2. The gap is peer-review confirmed and CITABLE in your literature review
3. ESC-50 is a well-known, respected benchmark in the sound community
4. Implementation effort is LOW (mel-spectrograms -> spike encoding -> convolutional SNN)
5. Natural SNN advantage narrative (temporal patterns in environmental sounds)
6. Multiple encoding methods to compare = rich evaluation section
7. ANN baselines readily available for comparison (~97% accuracy)
8. A March 2025 paper only got ~69% on ESC-10 with basic encoding -- massive room for improvement

---

## 11. Wave 3 Research Updates: Alternative Directions (Feb 2026)

### 11.1 SNN for Math Proof Verification -- DEAD END
- **Zero papers** combining SNNs with formal proof verification
- Fundamental mismatch: theorem proving = sequential symbolic generation (LLMs), not temporal pattern recognition (SNNs)
- Current SOTA uses 671B-parameter transformers (DeepSeek-Prover-V2)
- No SNN framework interfaces with Lean/Coq/Isabelle
- **Verdict: Do not pursue**

### 11.2 SNN for Satellite Image Classification -- VIABLE BUT MEDIOCRE
- ~15-25 papers exist already (not a virgin domain)
- SNN4Space codebase (ESA-funded) achieves 95.07% on EuroSAT
- Real neuromorphic hardware in orbit: Loihi on TechEdSat-13, BrainChip Akida on SpaceX
- Datasets: EuroSAT (27K images, 10 classes), UC Merced (2.1K, 21 classes)
- Strong narrative ("edge AI in space") but LOW natural SNN fit (standard RGB images, no temporal dynamics)
- Weak Manchester/SpiNNaker connection
- **Verdict: Functional but ESC-50 beats it on novelty, ECG beats it on narrative**

### 11.3 SNN for Robot Reflexes (Simulation-Only) -- VIABLE AND IMPRESSIVE
- ~5-10 papers in SNN + legged locomotion (most from 2024-2025)
- SpikeGym framework exists (SNN + RL for Isaac Gym, ~7 min training)
- VERY HIGH natural SNN fit: biological reflexes literally use spiking neurons; 3.5ms latency
- Strong Manchester/SpiNNaker connection: SpOmnibot, PushBot projects
- ICRA 2026 workshop on neuromorphic field robotics; commercial neuromorphic robots shipping 2026
- **Risk: RL training instability + SNN debugging = double complexity; significant setup overhead**
- **Verdict: Most impressive option if pulled off, but highest effort and risk. Only pursue if genuinely excited about robotics + RL**

### 11.4 FINAL TOP 7 (All Waves Combined, February 2026)

| Rank | Direction | Novelty | Effort | Risk | SNN Fit | Narrative | Pragmatic? |
|------|-----------|---------|--------|------|---------|-----------|-----------|
| **1** | **SNN on ESC-50 (Environmental Sound)** | VERY HIGH (0 papers) | LOW | LOW | HIGH | Strong | YES |
| **2** | **SNN ECG on PTB-XL** | HIGH (no benchmark) | LOW-MED | LOW | HIGH | Excellent | YES |
| **3** | **SNN Wildlife Camera Trap** | VERY HIGH (0 papers) | LOW-MED | LOW-MED | LOW | Excellent | YES |
| **4** | **SNN Robot Reflexes (Simulation)** | HIGH (~5-10 papers) | MED-HIGH | MEDIUM | VERY HIGH | Excellent | Only if excited |
| **5** | **Framework Shootout** | GENUINE (confirmed gap) | LOW-MED | LOW | N/A | Good | YES |
| **6** | **SNN Satellite Classification** | MODERATE (~15-25) | LOW-MED | LOW | LOW | Strong | Mediocre |
| **7** | **SNN Food Recognition (Food-101)** | VERY HIGH (0 papers) | LOW | LOW | LOW | Weak | YES but boring |

**Eliminated:** Math proof verification (fundamental mismatch), Music Generation (gap closed), DVS128 (saturated), SHD as primary (approaching saturation), SVHN (standard benchmark), Wearable HAR (saturated)

---

*This document synthesises outputs from 21 paper analysis agents, 6 context analysis agents, 5 web research agents, 3 Wave 3 research agents, and cross-references with existing research in this repository.*
