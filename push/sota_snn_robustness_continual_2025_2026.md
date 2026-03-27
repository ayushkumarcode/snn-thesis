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
| LT-Gate | ~2.8pp drop from peak | vs 5.8pp for HLOP, 7.1pp for DSD-SNN |
| DSD-SNN | ~5-10% | structure growth compensates |
| NACA | "markedly mitigated" | ~2% improvement + 98% less energy |
| **ours (SNN, no CL method)** | **74.4%** | naive sequential baseline |
| **ours (ANN, no CL method)** | **81.3%** | naive sequential baseline |

our 74.4%/81.3% forgetting is the naive sequential baseline -- expected to be high. the interesting finding is the 6.9pp gap between SNN and ANN.

### is the 6.9pp gap consistent with literature?

yes, directionally consistent but quantitative comparisons are scarce. the literature broadly supports mild inherent SNN advantages for CL:

theoretical arguments: spike sparsity = implicit regularization, temporal dynamics encode task-specific info, LIF leak prevents over-commitment to specific weight configs.

our 6.9pp result is valuable because it isolates the SNN vs ANN effect without any CL method confound, uses identical architectures, and demonstrates an inherent advantage. the magnitude (~8.5% relative reduction) is modest but meaningful.

### SNN CL on audio tasks -- very limited

1. **Spiking Compressed CL (Dequino et al.):** CL on Spiking Heidelberg Digits. 92.46% sample-incremental. only 2.2% loss on old classes in progressive learning. **most directly comparable** to us.
2. **NACA (Science Advances 2023):** TIDigits (speech). neuromodulation-assisted credit assignment.
3. **AGMP (Frontiers 2025):** SHD (audio-derived). effective CL without replay.
4. **ours:** ESC-50 environmental sound -- novel domain for SNN CL.

environmental sound is a distinctly different domain from speech/digits. our evaluation on ESC-50 super-categories appears novel -- nobody has done SNN CL on environmental sound before.

---

## PART 3: WHAT THIS MEANS FOR THE THESIS

### adversarial robustness narrative
1. SNNs have genuine inherent adversarial robustness -- confirmed across 20+ papers
2. the magnitude is debated: 2-8x typical on vision at standard epsilon
3. our 14.9x is consistent but likely inflated by ANN near-complete failure + gradient masking
4. standard PGD evaluation is now known to be unreliable (Wang et al. 2025) -- threat to validity
5. we're the FIRST to evaluate on audio
6. mechanisms: spike thresholding (noise filtering), temporal integration, gradient sparsity, input discretization

### continual learning narrative
1. our 6.9pp gap is directionally consistent with theory
2. SOTA SNN CL methods get near-zero forgetting -- our naive baseline is expected to be bad
3. the interesting finding is the SNN-ANN gap itself, not absolute forgetting
4. SNN CL on environmental sound is novel
5. STDP and surrogate gradient both active; combining them is the trend

### citations needed

**must-cite:**
1. Wang et al. (2512.22522) SA-PGD -- threat to validity
2. Sharmin et al. (ECCV 2020) -- foundational robustness paper
3. FEEL-SNN (NeurIPS 2024) -- SOTA defense
4. RSC-SNN (ACM MM 2024) -- Poisson = randomized smoothing
5. RandHet-SNN (iScience 2025) -- heterogeneity mechanism
6. TGO (ICLR 2026) -- threshold-neighboring neurons
7. Nature Comms SNN robustness (2025)
8. HLOP-SNN (ICLR 2024) -- SOTA CL
9. DSD-SNN (IJCAI 2023) -- architecture expansion CL
10. SCA-SNN (Neural Networks 2024) -- context-aware CL
11. PS-SNN (Scientific Reports 2026) -- best SNN CIL
12. AGMP (Frontiers 2025) -- astrocyte-gated CL on SHD

### research gaps we fill
1. no SNN adversarial robustness on audio -- we fill this
2. no SNN CL on environmental sound -- we fill this
3. very few direct SNN-vs-ANN forgetting comparisons without CL methods -- we provide this

### confidence

| Finding | Confidence | Basis |
|---------|------------|-------|
| SNNs have inherent adversarial advantage | HIGH | 20+ papers, 2020-2026 |
| Standard FGSM/PGD overestimates SNN robustness | HIGH | Wang 2025, Lin & Sengupta 2025 |
| Our 14.9x is directionally correct but inflated | HIGH | consistent with pattern at high epsilon |
| SNNs have mild inherent CL advantage | MEDIUM | theoretical support strong, empirical scarce |
| Our 6.9pp gap is meaningful | MEDIUM | consistent with theory but single experiment |
| Audio SNN adversarial robustness is novel | HIGH | found zero prior work |
| Audio SNN CL on ESC-50 is novel | HIGH | no prior work |

---

## Sources

### Adversarial
- [Wang et al. SA-PGD](https://arxiv.org/abs/2512.22522)
- [TGO ICLR 2026](https://arxiv.org/abs/2602.20548)
- [Lin & Sengupta local learning](https://arxiv.org/abs/2504.08897)
- [FEEL-SNN NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/a73474c359ed523e6cd3174ed29a4d56-Paper-Conference.pdf)
- [RSC-SNN ACM MM 2024](https://arxiv.org/abs/2407.20099)
- [RandHet-SNN iScience 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12159496/)
- [Nature Comms SNN robustness 2025](https://www.nature.com/articles/s41467-025-65197-x)
- [Gradient Sparsity Trail](https://arxiv.org/abs/2509.23762)
- [Sharmin ECCV 2020](https://arxiv.org/abs/2003.10399)
- [HART ICLR 2024](https://openreview.net/forum?id=xv8iGxENyI)

### Continual Learning
- [HLOP-SNN ICLR 2024](https://arxiv.org/abs/2402.11984)
- [DSD-SNN IJCAI 2023](https://arxiv.org/abs/2308.04749)
- [SCA-SNN Neural Networks 2024](https://arxiv.org/abs/2411.05802)
- [PS-SNN Scientific Reports 2026](https://www.nature.com/articles/s41598-026-42970-6)
- [AGMP Frontiers 2025](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1768235/full)
- [LT-Gate](https://arxiv.org/abs/2510.12843)
- [NACA Science Advances 2023](https://www.science.org/doi/10.1126/sciadv.adi2947)
