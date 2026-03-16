# ICONS 2026 Simulated Review (16 March 2026)

## Recommendation: Weak Accept

## Key Strengths
1. Clear novelty claim (first conv SNN on ESC-50)
2. Comprehensive 7-encoding comparison with statistical tests
3. Honest SpiNNaker failure analysis
4. Gap-collapse finding (16.7pp → 0.95pp)
5. Novel encoding transfer analysis
6. Impressive density of experiments
7. Statistical rigour throughout

## Key Weaknesses & Actions Taken
1. **SpiNNaker deployment limited** — acknowledged in paper as "proof of concept"
2. **Architecture undersized** — valid concern; we note this in threats to validity
3. **Adversarial uses standard PGD not SA-PGD** — acknowledged; added relative robustness ratio (SNN 30.5% vs ANN 4.4%)
4. **Clean accuracy discrepancy (7pp)** — FIXED: added precise explanation of training-time vs re-evaluation pipeline
5. **Some analyses single-fold** — acknowledged in threats to validity
6. **Paper tries to do too much** — intentional density; all analyses support thesis contributions
7. **Energy analysis: SNN loses in software** — honest reporting; temporal truncation path discussed

## Questions & Responses
1. **Relative robustness ratio** — ADDED to paper (SNN 30.5% vs ANN 4.4% retained)
2. **Clean accuracy discrepancy** — CLARIFIED in paper
3. **Full 7x7 transfer matrix** — exists in results, could add to appendix/supplementary

## Minor Issues Fixed
- Removed commented-out supervisor placeholder
- Flagged per-class sample size limitation for SpiNNaker analysis

## Issues NOT Fixed (acceptable for ICONS)
- PGD std dev missing for some epsilons (only fold-4 PGD, not 5-fold)
- Surrogate ablation single-fold (acknowledged in threats)
- Title could be softened (keeping as-is for impact)
