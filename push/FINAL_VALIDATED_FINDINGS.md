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
