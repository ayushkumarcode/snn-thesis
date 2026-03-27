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
2. **Lin & Sengupta (Apr 2025):** gradient-based attacks are ineffective against SNNs with local learning. hybrid transferability-based attack is much stronger. apparent robustness largely disappears under their attack
3. **Gradient Sparsity Trail (Sep 2025):** identifies two types of gradient sparsity in SNNs that impair white-box attacks, creating false robustness signals
4. **HART Attack (ICLR 2024):** attacks combining rate and temporal information are significantly stronger than either alone
5. **RSC-SNN (ACM MM 2024):** addresses gradient obfuscation via EOT (Expectation Over Transformation)

**consensus:** standard FGSM/PGD with fixed surrogate will overestimate SNN robustness. degree of overestimation is 5-13+ pp. but the relative ordering (SNN > ANN) is likely correct -- every paper confirms some inherent advantage, the debate is about magnitude.

### defense and attack landscape (2024-2026)

**defense methods:**

| Paper | Venue | Method | CIFAR-10 PGD-7 | CIFAR-10 FGSM | Clean |
|-------|-------|--------|----------------|---------------|-------|
| SNN-RAT | NeurIPS 2022 | Regularized AT | 45.23% | ~52% | ~89% |
| FEEL-SNN | NeurIPS 2024 | Frequency Encoding + Evolutionary Leak | Improved over RAT | Improved | ~89% |
| Robust Stable SNN | arXiv 2024 | DLIF + MPPD + AT+Reg | **40.30%** | **56.71%** | **88.91%** |
| RSC-SNN | ACM MM 2024 | Randomized Smoothing Coding | 39.98% | 54.52% | 82.03% |
| RandHet-SNN | iScience 2025 | Random heterogeneous time constants | **44.86%** (PGD10) | 53.53% | 90.25% |
| TGO | ICLR 2026 | Threshold Guarding Optimization | 6.14% (vanilla) | 51.40% | 88.79% |

**mechanistic understanding:**
- RSC-SNN: Poisson coding is conceptually equivalent to randomized smoothing
- TGO: threshold-neighboring neurons are the weak point; reducing them by 40% improves robustness
- natural spike-induced gradient sparsity creates inherent (but limited) robustness
- temporal encoding + early-exit decoding = key to SNN robustness advantage (Nature Comms 2025)

---

## PART 2: CONTINUAL LEARNING

### best SNN CL results (2024-2026)

**task-incremental learning (TIL) -- Split CIFAR-100:**

| Paper | Venue | Method | Accuracy | Steps | Notes |
|-------|-------|--------|----------|-------|-------|
| DSD-SNN | IJCAI 2023 | Dynamic growth + pruning | 81.17% | 20-step | 37.48% parameter usage |
| HLOP-SNN | ICLR 2024 | Hebbian orthogonal projection | ~85%+ | 10-step | near-zero forgetting |
| SCA-SNN | Neural Networks 2024 | Context-aware similarity reuse | **86.45%** | 20-step | beats DNN methods |
| PS-SNN | Scientific Reports 2026 | Pattern separation + expandable | N/A | 10-step | surpasses DNN methods |
| LT-Gate | arXiv 2025 | Local timescale gates | retained ~95% of Task A | 2 tasks | minimal forgetting |

**class-incremental learning (CIL) -- much harder:**

| Paper | Method | CIFAR-100 CIL | Steps |
|-------|--------|---------------|-------|
| DSD-SNN | Dynamic structure | ~50-55% | 10-step |
| SCA-SNN | Context-aware | **57.06%** | 10-step |
| PS-SNN | Pattern separation | **76.42%** | 10-step |

### typical forgetting rates

| Method | Forgetting | Notes |
|--------|-----------|-------|
| Naive sequential (no CL method) | 70-90%+ | catastrophic forgetting baseline |
| HLOP-SNN | Near-zero | orthogonal projection |
