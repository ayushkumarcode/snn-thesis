# Adversarial Robustness + Continual Learning -- Key Findings

---

## Adversarial Robustness

### the problem with our evaluation (Wang et al., arXiv:2512.22522)

standard PGD and FGSM attacks are broken for SNNs. the attacks work by following gradients, but SNN gradients during backprop come from the surrogate function (an approximation), not the real spike function. so the attack follows the wrong gradient -- it thinks it's fooling the model but it's actually confused by the approximation. this is gradient masking.

their fix is SA-PGD -- adjusts step sizes and dynamically changes the surrogate shape during the attack to get real gradients. results: standard PGD overestimates SNN robustness by 5-13 percentage points.

### our results in context
- our ratio (14.9x: SNN 26% vs ANN 1.75%) is likely inflated
- literature shows 2-8x ratios on vision benchmarks with confirmed methods
- two reasons our ratio is so high:
  1. ANN collapses to 1.75% -- near random chance for 50 classes -- partly due to gradient masking inflating apparent ANN weakness
  2. gradient masking makes SNN look more robust than it is
- the **direction** is real (SNN is genuinely more robust). the **magnitude** is overstated.

### what holds up
- we are the first to evaluate SNN adversarial robustness on any audio task -- zero prior work
- absolute numbers (SNN=26%, ANN=1.75%) are more defensible than the ratio
- correct framing: "results are upper bounds; SA-PGD evaluation recommended for definitive assessment"
- cite: Wang et al. arXiv:2512.22522

### best SNN adversarial numbers from literature (vision, with dedicated defenses)
- ~45% PGD-7 on CIFAR-10 (RandHet-SNN + RAT)
