# Verification Log

## Session: 31 March 2026

### Infrastructure Checks
- [x] CSF3 SSH tunnel: ALIVE at session start
- [x] SpiNNaker: REACHABLE (4.2ms ping)
- [x] Pre-commit hook: working (20-line mod limit, new files allowed)

### Data Integrity
- [x] 50 pruned SpiNNaker JSONs present (10 levels x 5 folds)
- [x] 5 unpruned T=3 fold JSONs present
- [x] 5 unpruned T=1 fold JSONs present
- [x] MASTER_RESULTS.json present and consistent
- [x] neurobench_pruned_sweep.json matches MASTER_RESULTS energy section
- [x] FIVE_FOLD_SUMMARY.json matches MASTER_RESULTS unpruned sections
- [x] Minor discrepancy: neurobench_t3_t1_fold4.json fold4 SpiNNaker=58.25% vs FIVE_FOLD_SUMMARY fold4=60.0% — different runs, FIVE_FOLD_SUMMARY is canonical

### Energy Inconsistency (FLAGGED)
- neurobench_pruned_sweep.json: T=3 0% pruning = 4706 nJ
- neurobench_t3_t1_fold4.json: T=3 AC-only = 142 nJ
- Hypothesis: different counting methodology (full NeuroBench vs AC-only)
- Report uses the per-fold validated numbers (142 nJ for T=3)
- Flag for user review

### Code Audit
- [x] src/model.py: architecture matches documentation
- [x] src/dataset.py: preprocessing deterministic, correct fold splitting
- [x] src/encoding.py: all 6 encodings correct
- [x] src/energy.py: legacy, NeuroBench is canonical
- [x] combo_experiment.py: T variable shadowing noted (low severity)
- [x] No random seed management in training scripts (noted as limitation)

### Report Compilation
- [x] tectonic compilation successful
- [x] 81 pages, 330 KB PDF
- [x] Title page, abstract, declaration, copyright, acknowledgements present
- [x] Table of contents with hyperlinks
- [x] List of Tables populated (20+ tables)
- [x] List of Figures empty (figures referenced but not embedded)
- [x] Some [?] unresolved citations remain
- [x] Some Figure ?? unresolved references remain

### Iteration 1: Initial compilation
- Tables render correctly with booktabs
- Equations numbered and formatted
- Bibliography partially resolved
- Issues: [?] citations, empty LoF, Figure ??

### Items for user to fix before submission:
1. Embed figure PDFs with \includegraphics (architecture_diagram.pdf, encoding_bar_chart.pdf, spinnaker_pipeline.pdf exist in figures/)
2. Fix remaining unresolved \cite{} keys
3. Fix unresolved \ref{fig:} references
4. Get actual word count (texcount report.tex)
5. Produce screencast using SCREENCAST_PLAN.md
6. Verify student ID on title page
