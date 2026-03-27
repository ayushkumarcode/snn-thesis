# Methodology Review -- Honest Re-evaluation of All Findings

going through every finding and grading it honestly. being harsh on purpose.

## Summary Grades

| # | Finding | Folds | Grade | Key Weakness |
|---|---------|-------|-------|-------------|
| 1 | SNN vs ANN 7 encodings | 5 | **B** | Single architecture, no hyperparameter sweep |
| 2 | PANNs gap collapse | 5 | **C+** | PANNs+Linear > SNN; p=0.034 means SNN is actually *worse* |
| 3 | Adversarial robustness | 5 | **C** | Gradient obfuscation not addressed |
| 4 | SpiNNaker deployment | 5 | **C-** | Only 2.1% of model on hardware |
| 5 | NeuroBench energy | 1 | **C** | ANN wins in software; energy constants may be wrong |
| 6 | Temporal ablation | 5 | **C+** | T=20>T=25 is suspicious |
| 7 | Encoding transfer | 1 | **D+** | Trivially expected (input distribution shift) |
| 8 | Pruning resilience | 1 | **D** | Both models broken at 90%; relative metric misleading |
| 9 | Neuron ablation | 1 | **D** | Relative metric; "beats ANN" claim is suspect |
| 10 | Stochastic resonance | 1 | **F** | 0.25pp on 400 samples = noise, not resonance |
| 11 | Noise robustness | 5 | **C** | 1.3pp relative difference is trivially weak |
| 12 | Few-shot curves | 1 | **D** | Single seed, single fold |
| 13 | Spike Pareto | 1 | **D** | Single seed, single fold |
| 14 | Saliency maps | 1 | **F** | 10 samples |
| 15 | Weight distributions | 1 | **C-** | Descriptive, not causal |
| 16 | Spike drop | 1 | **D+** | Circular reasoning |

## Top 5 Strongest
1. 7-encoding comparison (B) -- core novelty
2. PANNs gap collapse (C+) -- genuine insight
