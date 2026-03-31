# Project Audit Report — COMP30040 SNN-ESC50

Date: 31 March 2026

## Executive Summary

This audit scores the project against all 12 rubric criteria across Report (55%), Screencast (15%), and Achievements (30%). The project has exceptional experimental breadth and genuine novelty but the current thesis drafts have critical gaps that must be addressed before submission.

**Overall assessment: the experimental work is Outstanding (90-95). The thesis drafts are currently at Excellent (80-85) level but need: (1) formal academic writing, (2) incorporation of missing results, (3) figures/tables, and (4) LaTeX formatting to reach Outstanding.**

---

## Rubric Scoring: Report (55% of total)

### 3.1 Abstract and Introduction (15% of Report)

**Current band: Above Expectations / Excellent (75-82)**

**Evidence:**
- Introduction covers motivation, RQs, contributions, scope — structurally complete
- Four clear RQs, six specific contributions
- Scope/limitations section is honest (PGD caveat, partial deployment)
- Evaluation strategy mentioned but not derived from literature

**Gaps to Outstanding (90-100):**
- Writing is informal ("ok so", "way too much", "isn't just inconvenient") — needs formal academic register throughout
- No abstract exists yet (only an introduction)
- Rhythm-SNN (61.10%) and dendritic (61.65%) results — the BEST SNN results — are not mentioned at all
- GitHub link is "[GitHub — TBD]"
- TODO marker on line 113
- Evaluation strategy should be "derived from peer-reviewed literature" per rubric
- Needs quantified energy context (specific mJ numbers for edge devices)

**Action items:**
1. Write a precise 250-word abstract covering all key numbers
2. Rewrite entire introduction in formal academic prose
3. Add rhythm-SNN and full-deploy results to contributions
4. Derive evaluation strategy from NeuroBench (Yik et al. 2025) and ESC-50 CV protocol (Piczak 2015)
5. Remove all TODOs and placeholder links

---

### 3.2 Background and Theory (25% of Report)

**Current band: Expected / Above Expectations (65-75)**

**Evidence:**
- Covers ESC-50, LIF, encodings, SpiNNaker, energy, adversarial, PANNs, CL
- Prior SNN audio work table with 3 entries
- Mathematical formulations for LIF neuron
- Some peer-reviewed references

**Gaps to Outstanding (90-100):**
- Only ~1,800 words — far too thin for 25% of report (should be ~3,000-3,600)
- Literature review is breadth-only, no critical comparison or synthesis
- Only 3 prior SNN audio papers cited — need wider coverage of SNN image classification to contextualize gaps
- No discussion of SpiNNaker 2, Loihi 2, BrainScaleS (hardware landscape)
- No discussion of other audio datasets (UrbanSound8K, AudioSet, FSD50K, SHD)
- Missing formal mathematical presentation (Definition/Theorem style as in Gransbury sample)
- No critical comparison of surrogate gradient methods before ablation
- ESC-50 SOTA is listed as 98.25% but current SOTA is 99.1% (OmniVec2, CVPR 2024)
- One TODO marker remaining
- Informal writing tone

**Action items:**
1. Expand to ~3,500 words with critical literature synthesis
2. Add formal LIF equations with numbered definitions
3. Expand prior work coverage: SNN image classification (CIFAR-10 gaps as context), SNN speech/audio
4. Add hardware landscape section (Loihi 2, BrainScaleS, SpiNNaker 2)
5. Critical comparison of surrogate gradient approaches
6. Update ESC-50 SOTA to 99.1%

---

### 3.3 Technical Quality, Methodology and Evaluation (35% of Report)

**Current band: Excellent (82-88)**

**Evidence:**
- Controlled comparison principle well-articulated
- All 7 encoding methods with mathematical formulations
- 5-fold CV using predefined folds
- Comprehensive SpiNNaker deployment methodology
- NeuroBench energy framework
- Statistical tests (paired t-test, Wilcoxon)
- Negative results honestly documented with root causes

**Gaps to Outstanding (90-100):**
- Missing results from the thesis drafts:
  * Rhythm-SNN (61.10% ± 1.99%) — best SNN result, not in any chapter
  * Full SpiNNaker deployment (57.35% T=3, 55.05% T=1) — not incorporated
  * 50 pruned SpiNNaker deployments — not incorporated
  * Energy reduction (T=3: 142 nJ, T=1: 47 nJ) — not incorporated
  * Noise robustness 5-fold — not in any chapter
  * Encoding transfer matrix — not mentioned
  * Pruning resilience — not mentioned
  * Neuron ablation — not mentioned
  * Stochastic resonance — not mentioned
  * Saliency maps — not mentioned
  * Spike drop robustness — not mentioned
  * Temporal ablation — not mentioned
- Energy inconsistency: pruned sweep shows 4705 nJ at T=3/0%, but neurobench_t3_t1 shows 142 nJ at T=3 — must be resolved
- No figures or tables in current drafts (critical for Outstanding)
- No effect sizes (Cohen's d) alongside p-values
- No confidence intervals on per-fold results
- Loss function inconsistency across scripts (per-timestep CE vs summed membrane CE in energy_sweep.py)
- No random seed management in training scripts (reproducibility gap)

**Action items:**
1. Incorporate ALL missing experimental results into report
2. Resolve energy measurement inconsistency (4705 vs 142 nJ at T=3)
3. Create 15+ figures: architecture diagram, encoding comparison bar chart, Pareto frontier, training curves, confusion matrices, t-SNE plots, raster plots
4. Create 10+ tables: encoding comparison, SpiNNaker 5-fold, pruning sweep, adversarial, energy
5. Add effect sizes and confidence intervals
6. Document the loss function difference as a methodology note

---

### 3.4 Summary and Conclusions (15% of Report)

**Current band: Above Expectations / Excellent (75-82)**

**Evidence:**
- Six contributions restated concisely
- Direct answers to all four RQs
- Three limitations acknowledged
- Ten future work directions
- Final synthesis statement

**Gaps to Outstanding (90-100):**
- No reflection on negative results as contributions
- Rhythm-SNN and full-deploy results not mentioned
- "What I would do differently" reflection missing
- Future work could be more specific and tied to literature
- Impact statement missing

**Action items:**
1. Add critical reflection on negative results
2. Incorporate latest results
3. Add impact statement
4. Make future work more ambitious yet realistic

---

### 3.5 Presentation, Structure and Language (10% of Report)

**Current band: Below Expectations (50-55) — HIGHEST PRIORITY GAP**

**Evidence:**
- Current drafts are Markdown, not LaTeX
- No figures, no tables, no formatted equations
- Informal writing throughout (contractions, casual phrases)
- No front matter (declaration, copyright, abbreviations, lists)
- No bibliography
- No page numbers, headers, or professional formatting

**Gaps to Outstanding (90-100):**
- Everything. This criterion requires a complete LaTeX document with professional formatting.

**Action items:**
1. Create complete LaTeX document using MUThesis template
2. Professional front matter (title, abstract, declaration, acknowledgements, abbreviations, ToC, LoF, LoT)
3. All figures at high resolution with descriptive captions
4. All tables properly formatted with booktabs
5. All equations numbered and referenced
6. BibTeX bibliography with 40+ references
7. Hyperref for clickable cross-references
8. No contractions, formal academic prose throughout

---

## Rubric Scoring: Screencast (15% of total)

### 4.1-4.4 Screencast (all criteria)

**Current band: N/A (not yet produced)**

**Assessment:** Need to write comprehensive screencast plan. The rubric heavily rewards creativity and professionalism. A podcast/interview format using NotebookLM scored very highly for a previous student.

---

## Rubric Scoring: Achievements (30% of total)

### 5.1 Complexity (33% of Achievements)

**Current band: Outstanding (92-95)**

**Evidence:**
- Surrogate gradient SNN training (not taught in any UoM course)
- SpiNNaker neuromorphic hardware deployment (supervisor: "no student has ever done this")
- Ring buffer quantization debugging (hardware-level understanding)
- 7 spike encoding implementations with mathematical formulations
- NeuroBench energy benchmarking framework integration
- Pruning with mask enforcement (discovered and fixed fake pruning bug)
- Rhythm-SNN with learnable oscillatory threshold modulation
- Dendritic branching with learnable delays
- Current injection batched deployment protocol for SpiNNaker
- Full pipeline: audio → spectrogram → encoding → SNN → SpiNNaker → classification

**Rubric match:** "solves a long-standing hard problem or combines ideas from several areas in an original way" — combines audio signal processing, spiking neural networks, surrogate gradient optimization, neuromorphic hardware deployment, energy benchmarking, adversarial robustness, and continual learning. Draws from neuroscience, electrical engineering, and ML.

---

### 5.2 Scale (33% of Achievements)

**Current band: Outstanding (93-96)**

**Evidence:**
- 50+ unique experiments across CSF3 A100 GPUs and SpiNNaker
- 7 encoding comparisons × 5 folds = 35 training runs
- 34 Rhythm/advanced SNN training runs on CSF3
- 76 energy optimization experiments
- 50 SpiNNaker pruning deployments (10 levels × 5 folds)
- 10 unpruned SpiNNaker deployments (T=3 and T=1 × 5 folds)
- 147-paper energy optimization literature survey
- ~400+ hours of project work
- Adversarial robustness 5-fold, noise robustness 5-fold, continual learning 5-fold
- Full experiment log (1050+ lines)
- 46 decision records

**Rubric match:** "ownership of problems... much greater could be delivered given more time" — clearly demonstrated through the iterative debugging of SpiNNaker (runs 1-6, 9 Poisson configurations, Option A/B/C), the comprehensive encoding comparison, and the energy optimization research survey.

---

### 5.3 Achievement (34% of Achievements)

**Current band: Outstanding (90-94)**

**Evidence:**
- FIRST SNN on full ESC-50 (verified: zero prior peer-reviewed work)
- Rhythm-SNN: 61.10% accuracy (+14pp over baseline, only 2.75pp below ANN)
- Full SpiNNaker deployment: 57.35% with only 2.15pp hardware gap
- Pruning Pareto: 85% pruning → maintained accuracy with ~27 nJ energy
- SpiNNaker BEATS snnTorch at 60% and 80% pruning (negative hardware gaps)
- PANNs+SNN: 92.5% (gap collapses from 17pp to 1pp — key finding)
- Adversarial: SNN 6x more robust than ANN
- Continual learning: SNN forgets 4.7pp less
- Information preservation framework for encoding hierarchy
- ICONS 2026 paper in progress (publication-quality work)
- Well-documented SpiNNaker 1 limitation (ring buffer quantization)

**Rubric match:** "the provided solution is hard to improve upon" — for a 3rd year undergraduate thesis, this breadth and depth of results is exceptional. Multiple findings are independently publication-worthy.

---

## Critical Issues Found

### Issue 1: Energy Measurement Inconsistency (HIGH PRIORITY)
The pruned sweep (MASTER_RESULTS.json) reports T=3 unpruned energy as 4,706 nJ, while neurobench_t3_t1_fold4.json reports T=3 energy as 142 nJ (AC-only). This is a 33x discrepancy that MUST be resolved before the report.

**Hypothesis:** The pruned sweep likely uses a different NeuroBench counting methodology (including dense conv MACs) while the T3/T1 file uses AC-only counting. Need to verify by reading the actual measurement scripts.

