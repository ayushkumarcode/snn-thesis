# FINAL VALIDATED FINDINGS (15 March 2026)
*After 5-fold validation — correcting single-fold outliers*

## Findings that HOLD (5-fold validated)

### 1. Encoding Hierarchy (Grade B, p<0.002)
direct (47.15%) >> rate (24.00%) ≈ phase (24.15%) > population (19.15%) > latency (16.30%) >> delta (7.25%) ≈ burst (6.50%)
- All pairwise comparisons significant (p<0.002) except rate≈phase (p=0.93)
- Rate and phase are statistically tied despite 7x spike count difference
- Cohen's d ranges from 3.75 to 8.13 — large effects

### 2. PANNs Gap Collapse (Grade C+, p=0.034)
SNN 92.50% vs ANN 93.45% (gap 0.95pp, down from 16.7pp)
- Statistically significant (p=0.034) meaning SNN IS worse, but only by <1pp
- PANNs+Linear (93.80%) beats both — SNN head adds no accuracy benefit
- **The scientific insight holds:** gap is feature-learning, not spiking

### 3. Adversarial Robustness (Grade C, p=0.007)
FGSM eps=0.1: SNN 16.55%±5.49% vs ANN 2.75%±0.61% (6.0x more robust)
PGD eps=0.05: SNN 9.75% vs ANN 0.05% (195x)
- 5-fold validated, all epsilons significant
- Caveat: uses standard PGD not SA-PGD

### 4. SpiNNaker Deployment (Grade C-, p=0.0016)
FC2-only hybrid: 33.1%±6.9% (5-fold, 2000 inferences)
FC1+FC2 exc-only: 15.0% (20 samples, pipeline proven)
- Hardware gap 12.8±4.1pp is significant

### 5. Temporal Ablation (Grade C+, 5-fold)
92% of full accuracy at T=10 (60% energy saving)
T=20 vs T=25: NOT significant (p=0.45) — plateau, not peak

### 6. Encoding Transfer Matrix (Grade B-, 5-fold)
Transfer ratio = 0.255±0.006 (remarkably consistent across folds)
Encodings are highly specific — 75% accuracy loss when swapped
Novel finding, validated across all 5 folds

### 7. Noise Robustness (Grade C, 5-fold)
SNN degrades directionally less but NOT statistically significant at any SNR
At 0dB: SNN=7.05%, ANN=6.95% — effectively equal

### 8. Statistical Tests (all computed)
Core SNN vs ANN: p=0.0028, d=-2.93
Every encoding pairwise: all p<0.002
Adversarial each epsilon: all p<0.05

## Findings CORRECTED by 5-fold validation

### Pruning Resilience — CORRECTED
- Single-fold: "SNN retains 93.2%, ANN collapses to 36.8%"
- **5-fold: SNN=33.0% (61% ret), ANN=35.0% (57% ret) — similar degradation**
- SNN slightly more resilient at 90% but both are severely degraded

### Neuron Ablation — CORRECTED
- Single-fold: "SNN beats ANN at 10-30% ablation"
- **5-fold: ANN maintains higher absolute accuracy at all rates**
- On relative basis SNN retains slightly more but both collapse rapidly

### Stochastic Resonance — CORRECTED
- Single-fold: "SR detected at sigma=0.02 (+0.25pp)"
- **5-fold: +0.10pp at sigma=0.2, detected in only 3/5 folds — NOISE**
- Cannot be claimed as a finding

## Clean Accuracy Discrepancy — EXPLAINED
- Training-time canonical: 47.15% (CSF3 CUDA, used in encoding table)
- Re-evaluation on GPU: 54.25% (same models, used in adversarial/noise tables)
- Cause: BatchNorm running statistics differ across hardware
- Robustness RATIOS are unaffected (both numerator and denominator use same eval)

## Remaining Gaps
1. UrbanSound8K cross-dataset: script exists, never run
2. Continual learning: single-fold only (thesis contribution C6)
3. NeuroBench energy: single-fold only
4. Surrogate ablation: single-fold, single-seed
5. SpiNNaker latency/energy: no actual hardware measurement
