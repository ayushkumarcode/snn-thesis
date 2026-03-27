# Adversarial Robustness + Continual Learning — Research Findings
*From deep-research agent, March 2026*

---

## Adversarial Robustness

### What Wang et al. (arXiv:2512.22522) actually say
Standard PGD and FGSM attacks are broken for SNNs. The reason: attacks work by following gradients, but SNN gradients during backprop come from the surrogate function (an approximation), not the real spike function. So the attack follows the wrong gradient — it thinks it's fooling the model but it's actually just confused by the approximation. This is called gradient masking.

Their fix is SA-PGD (Stable Adaptive PGD) — adjusts step sizes and dynamically changes the surrogate shape during the attack to get real gradients. Results: standard PGD overestimates SNN robustness by 5-13 percentage points in attack success rate.

### Our results in context
- Our ratio (14.9x: SNN 26% vs ANN 1.75%) is likely inflated
- Literature shows 2-8x ratios on vision benchmarks with confirmed methods
- Two reasons our ratio is so high:
  1. ANN collapses to 1.75% — near random chance for 50 classes — partly due to gradient masking inflating the apparent ANN weakness
  2. Gradient masking makes SNN look more robust than it is
- The **direction** is real (SNN is genuinely more robust). The **magnitude** is overstated.

### What holds up
- We are the first to evaluate SNN adversarial robustness on any audio task — zero prior work
- Absolute numbers (SNN=26%, ANN=1.75%) are more defensible than the ratio
- Correct framing: "results are upper bounds; SA-PGD evaluation recommended for definitive assessment"
- Cite: Wang et al. arXiv:2512.22522

### Best SNN adversarial numbers in literature (vision, with dedicated defences)
- ~45% PGD-7 on CIFAR-10 (RandHet-SNN + RAT)
- ~57% FGSM on CIFAR-10 (Robust Stable SNN)
- These use dedicated robustness methods; our result is vanilla SNN with no defence

---

## Continual Learning

