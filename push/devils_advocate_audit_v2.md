# Devil's Advocate Audit v2 — Every Remaining Gap
*Generated: 15 March 2026*

## Critical Findings

### 1. Noise robustness claim is NOT statistically significant
p=0.07-0.94 at all SNR levels. Cannot claim "SNN degrades less" without qualification.
**Action:** Soften language to "trend towards" or "directionally more robust."

### 2. T=20 > T=25 is noise, not real
p=0.45. The temporal ablation "peak" is not significant.
**Action:** Report as "accuracy plateaus around T=15-20" not "peaks at T=20."

### 3. ICONS LaTeX has inconsistent adversarial numbers
Line 70: "9.4x" but results section says "6.0x". Must be 6.0x (5-fold).
**Fixed in this commit.**

### 4. Noise robustness clean accuracy discrepancy
Noise robustness shows 54.25% clean SNN but canonical is 47.15%.
**Root cause:** Noise robustness evaluates fold-by-fold best models which may differ from the training-time reported accuracy. The 47.15% is the mean of per-fold best-epoch accuracies.

### 5. Continual learning is single-fold (thesis contribution C6)
Cannot defend a thesis contribution with n=1.
**Action:** Run on CSF3 (submitted in 5fold_upgrades job).

### 6. No figures included in LaTeX \includegraphics
**Fixed in this commit.**

### 7. Zero unit tests, stale README, incomplete requirements.txt
Lower priority but needed before submission.

## Statistical Tests Now Complete
- Core SNN vs ANN: p=0.0028 (significant)
- All encoding comparisons: p<0.002 (all significant except rate≈phase p=0.93)
- Adversarial at eps=0.1: p=0.0073 (significant)
- Noise robustness: NOT significant
- T=20 vs T=25: NOT significant
