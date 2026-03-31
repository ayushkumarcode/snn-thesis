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

