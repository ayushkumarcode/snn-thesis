# ICONS 2026 -- Simulated Review Notes

did a mock review of our paper to anticipate what reviewers will say.

## Overall: Weak Accept

## Strengths
1. clear novelty claim (first conv SNN on ESC-50)
2. 7-encoding comparison with statistical tests
3. honest SpiNNaker failure analysis
4. gap-collapse finding (16.7pp -> 0.95pp)
5. novel encoding transfer analysis
6. impressive density of experiments
7. statistical rigour throughout

## Weaknesses & What We Did About Them
1. **SpiNNaker deployment limited** -- acknowledged as "proof of concept" in paper
2. **Architecture undersized** -- noted in threats to validity
3. **Adversarial uses standard PGD not SA-PGD** -- acknowledged; added relative robustness ratio (SNN 30.5% vs ANN 4.4%)
4. **Clean accuracy discrepancy (7pp)** -- FIXED: added precise explanation of training-time vs re-evaluation pipeline
5. **Some analyses single-fold** -- acknowledged in threats to validity
6. **Paper tries to do too much** -- intentional density; all analyses support thesis contributions
7. **Energy analysis: SNN loses in software** -- honest reporting; temporal truncation path discussed

## Questions & Responses
1. **relative robustness ratio** -- ADDED to paper (SNN 30.5% vs ANN 4.4% retained)
2. **clean accuracy discrepancy** -- CLARIFIED in paper
3. **full 7x7 transfer matrix** -- exists in results, could add to appendix
