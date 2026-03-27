# Completeness Audit (Updated 16 March 2026)

## 5-Fold Validated (DONE -- 23 experiments)
all core training (7 encodings + ANN + 2 augmented), adversarial, noise,
temporal ablation, pruning, neuron ablation, stochastic resonance,
encoding transfer, PANNs, SpiNNaker FC2 5-fold, statistical tests,
NeuroBench energy, continual learning, few-shot learning, spike Pareto.

## Previously Single-Fold -- NOW ALL DONE

| Experiment | Status |
|-----------|--------|
| Continual learning | **DONE** -- SNN 69.9%+/-4.3%, ANN 74.7%+/-2.4% |
| NeuroBench energy | **DONE** -- SNN 968+/-37 nJ, ANN 454+/-11 nJ |
| Few-shot learning | **DONE** -- 5 folds, SNN-ANN gap widens at low data |
| Spike Pareto | **DONE** -- 5 folds, SNN 44% acc at 0.4% spike rate |
| Surrogate ablation | single fold, single seed (acceptable -- design guidance, not contribution) |

## SpiNNaker Full Deployment Status
- ROOT CAUSE fixed: `initialize(v=0.0)` + `set_number_of_neurons_per_core`
- step 3a: 231/256 hidden neurons fired (VERIFIED)
- step 4 exc-only: 5% (saturated hidden layer, 229/256 fired)
- step 4 top-k=100: 0% (too sparse, 0-11/256 fired)
- step 4 top-k=500: failed (sPyNNaker NotImplementedError: times differ)
- step 4 top-k=200: RUNNING
- FC2-only remains primary result: 33.1%+/-6.9% (5-fold validated)

## ICONS Paper Status
