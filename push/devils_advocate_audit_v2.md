# Devil's Advocate Audit v2 -- Every Remaining Gap

being harsh with myself about what still needs fixing.

## Critical Findings

### 1. Noise robustness claim is NOT statistically significant
p=0.07-0.94 at all SNR levels. can't claim "SNN degrades less" without qualification.
**action:** soften language to "trend towards" or "directionally more robust."

### 2. T=20 > T=25 is noise, not real
p=0.45. the temporal ablation "peak" is not significant.
**action:** report as "accuracy plateaus around T=15-20" not "peaks at T=20."

### 3. ICONS LaTeX has inconsistent adversarial numbers
line 70: "9.4x" but results section says "6.0x". must be 6.0x (5-fold).
**fixed.**

### 4. Noise robustness clean accuracy discrepancy
noise robustness shows 54.25% clean SNN but canonical is 47.15%.
**root cause:** noise robustness evaluates fold-by-fold best models which may differ from training-time reported accuracy. the 47.15% is the mean of per-fold best-epoch accuracies.

### 5. Continual learning is single-fold (thesis contribution C6)
can't defend a thesis contribution with n=1.
**action:** run on CSF3 (submitted in 5fold_upgrades job).

### 6. No figures included in LaTeX
**fixed.**
