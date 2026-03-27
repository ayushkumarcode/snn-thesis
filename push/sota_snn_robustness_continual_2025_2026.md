# SNN Adversarial Robustness and Continual Learning -- Literature (2024-2026)

our key results for context:
- adversarial: SNN 26% vs ANN 1.75% at eps=0.1 FGSM (14.9x ratio); SNN 19.25% vs ANN 0% at eps=0.05 PGD
- continual learning: SNN forgetting 74.4% vs ANN forgetting 81.3% (SNN forgets 6.9pp less)

---

## PART 1: ADVERSARIAL ROBUSTNESS

### Wang et al. (arXiv:2512.22522) -- the overestimation problem

this is the paper that makes me nervous about our adversarial results. they found that standard PGD and FGSM attacks are broken for SNNs. the reason: attacks follow gradients, but SNN gradients during backprop come from the surrogate function (an approximation), not the real spike function. so the attack follows the wrong gradient. this is gradient masking / gradient obfuscation specific to SNNs.

their fix is SA-PGD (Stable Adaptive PGD) -- adjusts step sizes and dynamically changes surrogate shape during the attack. results: standard PGD overestimates SNN robustness by 5-13 percentage points in attack success rate.

| Dataset | Architecture | Standard STBP ASR | SA-PGD ASR | Improvement |
|---------|-------------|-------------------|------------|-------------|
| CIFAR-10 (AT) | SEWResNet19 | 75.38% | 88.44% | +13.06 pp |
| CIFAR-100 (AT) | SEWResNet19 | 88.22% | 93.19% | +4.97 pp |
| CIFAR10-DVS | VGG9 | 36.10% | 49.10% | +13.00 pp |

ASR = attack success rate (higher = attack works better = model is actually less robust).

**implications for us:** our SNN robustness numbers (26% FGSM, 19.25% PGD) may be **inflated** due to gradient masking. the true gap between our SNN and ANN may be smaller than 14.9x. this is a legitimate threat to validity.

**recommendation:** acknowledge this as a threat to validity in the thesis. note that SA-PGD on our audio SNN would be important future work.

