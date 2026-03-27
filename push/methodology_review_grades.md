# Methodology Review — Brutal Re-evaluation of All Findings
*Generated: 15 March 2026 by critical review agent*

## Summary Grades

| # | Finding | Folds | Grade | Key Weakness |
|---|---------|-------|-------|-------------|
| 1 | SNN vs ANN 7 encodings | 5 | **B** | Single architecture, no hyperparameter sweep |
| 2 | PANNs gap collapse | 5 | **C+** | PANNs+Linear > SNN; p=0.034 means SNN is *worse* |
| 3 | Adversarial robustness | 5 | **C** | Gradient obfuscation not addressed |
| 4 | SpiNNaker deployment | 5 | **C-** | Only 2.1% of model on hardware |
| 5 | NeuroBench energy | 1 | **C** | ANN wins in software; wrong energy constants |
| 6 | Temporal ablation | 5 | **C+** | T=20>T=25 suspicious |
| 7 | Encoding transfer | 1 | **D+** | Trivially expected (input distribution shift) |
| 8 | Pruning resilience | 1 | **D** | Both models broken at 90%; relative metric misleading |
| 9 | Neuron ablation | 1 | **D** | Relative metric; "beats ANN" claim suspect |
| 10 | Stochastic resonance | 1 | **F** | 0.25pp on 400 samples = noise, not resonance |
