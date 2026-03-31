# Devil's Advocate Review

## CRITICAL ISSUES (now fixed)
1. ANN per-fold values: Chapter 4 had [66.25,64.00,68.75,60.00,60.25] — WRONG. Fixed to [63.25,59.50,65.25,68.75,62.50] matching source JSON.
2. Wilcoxon p-value: Chapter 5 said "Wilcoxon p=0.001" — IMPOSSIBLE with n=5. Fixed to clarify both tests (paired t-test p=0.001, Wilcoxon p=0.0625).

## Remaining items for user review
- Model identity: report mixes baseline SNN (47.15%) and Rhythm-SNN (57.35%) — mostly clear from context but could be more explicit
- Energy estimates ~27nJ and ~14nJ are extrapolations, not NeuroBench-validated — stated with tildes
- 86nJ PANNs-SpiNNaker figure is FC2-only energy, omits CNN14 — discussed in text
- No confusion matrices in main body (data exists in appendix)
- No learning curves shown
- Background missing ANN-to-SNN conversion discussion
