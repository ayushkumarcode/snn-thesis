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
