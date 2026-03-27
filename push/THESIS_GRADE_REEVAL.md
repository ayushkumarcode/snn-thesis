# Re-evaluation Against Actual Thesis Marking Criteria
*16 March 2026*

## Manchester COMP30040 Breakdown
- **Report (55%)** — written quality, structure, literature, evaluation
- **Achievements (30%)** — software output, experiments, results
- **Screencast (15%)** — 8-minute video presentation

## Our Project vs Marking Criteria

### Report Quality (targeting 55% component)

| Dimension | Our Status | Grade Band |
|-----------|-----------|------------|
| Literature review | Exhaustive: Larroza, Dominguez-Morales, Sharmin, Kong, Dampfhoffer, 21+ references, confirmed zero prior ESC-50 SNN work | **80+** |
| Problem framing | Clear RQs (4), contributions (6), gap identified and validated | **80+** |
| Methodology description | 7 encodings, architecture, training, SpiNNaker — all documented | **75-80** |
| Evaluation quality | 5-fold CV, p-values for all claims, negative results documented, 3 findings corrected by own validation | **80+** |
| Critical analysis | Root-cause analysis of SpiNNaker failure, honest energy framing, gradient obfuscation acknowledged | **80+** |
| Writing quality | 10 thesis chapters in markdown, ICONS paper in LaTeX, needs final polish | **70-75** (not yet in LaTeX thesis) |
| Honest limitations | Documented: SNN uses more energy in software, SpiNNaker FC1 fails, pruning/SR claims corrected | **80+** |

### Achievements (targeting 30% component)

| Dimension | Our Status | Grade Band |
|-----------|-----------|------------|
| Working implementation | SNN + ANN trained, 7 encodings, all 5 folds | **75-80** |
| Hardware deployment | SpiNNaker FC2-only (33.1% 5-fold), FC1+FC2 pipeline proven (15%) | **80+** (rare for UG) |
| Scope of experiments | 14+ experiment scripts, adversarial, PANNs, continual learning, etc. | **80+** |
| Novelty | First SNN on ESC-50 (confirmed), 7 encoding comparison, gap collapse finding | **80+** |
| Conference paper | ICONS 2026 submission drafted — publishable-quality work | **85+** |
| Code quality | Modular src/, systematic experiments/, documented decisions | **70-75** |

### Key Differentiators (from marking criteria research)

| Factor | 2:1 (60-69) | First (70-79) | High First (80+) | Our Project |
|--------|-------------|---------------|-------------------|-------------|
| **Evaluation** | Basic testing | Systematic, appropriate metrics | Statistical confidence, baselines | **80+**: p-values, 5-fold, Cohen's d, multiple baselines |
| **Critical Analysis** | Acknowledges limitations | Honest discussion | Identifies WHY things failed | **80+**: root-cause analysis of SpiNNaker, weight distribution, corrected own claims |
| **Independence** | Significant supervisor guidance | Self-directed | Student drives the project | **75-80**: autonomous experiment design, CSF3/SpiNNaker deployment |
| **Originality** | Follows existing approaches | Some personal creativity | Novel contribution | **80+**: first SNN on ESC-50, gap collapse insight, encoding transfer matrix |

## Previous Grade Assessment Was Too Harsh

The methodology reviewer graded findings on PEER REVIEW standards (would a reviewer at ICONS accept this?). But undergraduate thesis marking is different:

- A reviewer demands n=5 folds for every claim → **Grade C for single-fold**
- A thesis examiner rewards the PROCESS of running 14 experiments, finding 3 were wrong, and correcting them → **Grade A for self-correction**

### Revised Assessment for THESIS (not paper)

| Finding | Paper Grade | Thesis Grade | Why Different |
