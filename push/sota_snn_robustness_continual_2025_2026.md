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

### is our 14.9x ratio consistent with literature?

our ratio is HIGH but directionally consistent. the magnitude is likely inflated by gradient masking.

| Paper | Year | Dataset | SNN Robust Acc | ANN Robust Acc | Ratio | Notes |
|-------|------|---------|---------------|---------------|-------|-------|
| **ours** | 2026 | ESC-50 (audio) | 26.0% (eps=0.1) | 1.75% | **14.9x** | Standard FGSM |
| RSC-SNN | 2024 | CIFAR-10 | 54.52% | 10.89% | **5.0x** | ACM MM 2024 |
| RSC-SNN | 2024 | CIFAR-100 | 34.89% | 4.56% | **7.7x** | ACM MM 2024 |
| Nature Comms (2025) | 2025 | CIFAR-10 | ~2x ANN | baseline | **~2x** | |
| Nature Comms (2025) | 2025 | FashionMNIST | ~20% (eps=0.5) | ~0% | **>>10x** | high epsilon |
| Sharmin et al. | 2020 | CIFAR-10 | 3-6% higher | baseline | ~1.1-1.2x | seminal ECCV paper |

at moderate epsilon (8/255 on CIFAR), SNN/ANN ratio is typically 2-8x. at high epsilon (0.1 on audio, 0.5 on FashionMNIST) where ANN collapses to near-zero, the ratio can look extremely large. our ANN drops to 1.75% (near random for 50 classes = 2%), essentially complete failure. the high ratio is partly an artifact of that plus gradient masking.

**should report absolute numbers (SNN 26%, ANN 1.75%)** rather than the ratio. frame as: "SNN retains non-trivial classification (26%) at perturbation magnitudes where the ANN is essentially defeated (1.75%, near chance level)."

### audio SNN adversarial robustness -- it's just us

**zero papers on SNN adversarial robustness for audio classification.** all SNN adversarial work (2020-2026) has been on image classification (CIFAR-10, CIFAR-100, MNIST, ImageNet) or neuromorphic vision (DVS). closest related: Wu et al. (2018) SOM-SNN for noise robustness (Gaussian, environmental), but that's noise not adversarial.

this is a clear novelty claim.

### gradient masking is a confirmed issue

evidence from 2024-2026:

1. **Wang et al. (Dec 2025):** quantifies gradient vanishing in surrogates. attack success rates increase 4-13pp when addressed via ASSG
