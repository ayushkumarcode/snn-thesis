# Self-Assessment — COMP30040 SNN-ESC50
## Updated: 31 March 2026 (post-polish)

## Report (55% of total)

### 3.1 Abstract and Introduction (15%) — Score: 90 (Outstanding)
- Abstract: precise 200-word overview covering all key numbers in proper context
- Introduction: energy motivation, biological blueprint, 4 RQs, 6 contributions
- Evaluation strategy derived from NeuroBench and ESC-50 CV protocol
- All cross-references resolve, no TODOs remaining

### 3.2 Background and Theory (25%) — Score: 88 (Excellent)
- Formal LIF definition (Definition 2.1) with numbered equations (2.1-2.15)
- All 7 encoding methods with mathematical formulations and citations
- Hardware landscape: SpiNNaker, Loihi 2, BrainScaleS-2, TrueNorth
- Energy threshold analysis (Dampfhoffer), NeuroBench framework
- Prior SNN audio work table identifying the research gap
- 35+ peer-reviewed references throughout

### 3.3 Technical Quality, Methodology, Evaluation (35%) — Score: 92 (Outstanding)
- Controlled comparison principle clearly articulated
- 14 numbered equations in methodology, 7 encoding formulations
- 3 figures: architecture diagram, SpiNNaker pipeline, encoding bar chart
- 20+ tables with booktabs formatting and descriptive captions
- Complete SpiNNaker deployment pipeline with parameter table
- Negative results documented with root causes (burst, delta, KD, augmentation)
- Information preservation principle (novel framework)
- Statistical tests (paired t-test p=0.001, Wilcoxon p=0.0625)
- All 50 pruned deployments, 5-fold T=3 and T=1 results

### 3.4 Summary and Conclusions (15%) — Score: 88 (Excellent)
- C1-C6 restated with specific numbers
- RQ1-RQ4 explicitly answered with evidence
- 3 honest limitations
- 7 future work directions tied to literature
- Critical reflection and impact statement

### 3.5 Presentation, Structure, Language (10%) — Score: 88 (Excellent)
- Professional LaTeX: 82 pages, 416 KB PDF
- Complete front matter: title, abstract, declaration, copyright, acknowledgements
- ToC, List of Figures (3), List of Tables (20+), all hyperlinked
- ALL citations resolved to numbered references (zero [?] markers)
- ALL figure references resolved (zero Figure ?? markers)
- Formal academic prose, no contractions, no casual language
- Word count: 14,525 (within 10,000-15,000 range)
- Booktabs tables, numbered equations, clean typography

### Report weighted score: ~90 (Outstanding)
Breakdown: 0.15*90 + 0.25*88 + 0.35*92 + 0.15*88 + 0.10*88 = 89.9

---

## Screencast (15% of total) — Score: N/A (plan written, not produced)
- Comprehensive 8-minute documentary-style script in SCREENCAST_PLAN.md
- Minute-by-minute breakdown with 9 segments and 5 demo moments
- 3 format options evaluated, production schedule included

---

## Achievements (30% of total)

### 5.1 Complexity (33%) — Score: 93 (Outstanding)
- Surrogate gradient SNN training (not taught at UoM)
- SpiNNaker hardware deployment (supervisor: "no student has ever done this")
- 7 encoding implementations with mathematical formulations
- Rhythm-SNN with learnable oscillatory modulation
- NeuroBench energy framework integration
- Combines audio ML + neuroscience + hardware engineering

### 5.2 Scale (33%) — Score: 94 (Outstanding)
- 84 SpiNNaker JSON result files (50 pruned + 10 unpruned + 24 diagnostic)
- 7 encoding comparisons x 5 folds = 35 training runs
- 34 advanced technique experiments on CSF3 A100
- 76 energy optimisation experiments
- 147-paper literature survey
- ~400+ hours of work

### 5.3 Achievement (34%) — Score: 92 (Outstanding)
- First SNN on full ESC-50 (verified: zero prior work)
- Rhythm-SNN: 61.10% (+14pp, only 2.75pp below ANN)
- SpiNNaker: 57.35% with 2.15pp gap
- PANNs gap collapse: 16.70pp to 0.95pp
- 85% pruning: ~27 nJ (16.6x cheaper than ANN)
- SpiNNaker BEATS snnTorch at 60% and 80% pruning
- 6x adversarial robustness, 4.7pp less forgetting

### Achievements weighted score: ~93 (Outstanding)

---

## Overall projected grade: ~91 (Outstanding)
Report (55%): 90 x 0.55 = 49.5
Achievements (30%): 93 x 0.30 = 27.9
Screencast (15%): TBD (plan at 85+ level)

Total (excl. screencast): 77.4 / 85% = ~91
