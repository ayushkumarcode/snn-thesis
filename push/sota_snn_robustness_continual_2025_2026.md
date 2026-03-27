# State-of-the-Art: SNN Adversarial Robustness and Continual Learning (2024-2026)

**Research Date:** 10 March 2026
**Context:** UoM COMP30040 Thesis -- SNNs for ESC-50 Audio Classification
**Our Key Results:**
- Adversarial: SNN 26% vs ANN 1.75% at eps=0.1 FGSM (14.9x ratio); SNN 19.25% vs ANN 0% at eps=0.05 PGD
- Continual Learning: SNN forgetting 74.4% vs ANN forgetting 81.3% (SNN forgets 6.9pp less)

---

## PART 1: ADVERSARIAL ROBUSTNESS OF SNNs

### 1.1 Wang et al. (2512.22522) -- SA-PGD and the Overestimation Problem

**Full Citation:** Jihang Wang, Dongcheng Zhao, Ruolin Chen, Qian Zhang, Yi Zeng. "Towards Reliable Evaluation of Adversarial Robustness for Spiking Neural Networks." arXiv:2512.22522, December 2025.

**Core Problem Identified:**
The binary, discontinuous nature of spike activations causes **vanishing gradients** when using standard surrogate gradient methods for generating adversarial attacks. This means standard PGD (and FGSM) attacks may be **weaker than they should be** when applied to SNNs, making SNNs appear more robust than they truly are. This is fundamentally a **gradient masking / gradient obfuscation** problem specific to SNNs.

**Two Key Contributions:**

1. **ASSG (Adaptive Sharpness Surrogate Gradient):** Dynamically adjusts the surrogate function's shape based on the input distribution during attack iterations. Rather than using a fixed surrogate gradient (e.g., fast_sigmoid or atan), ASSG evolves the sharpness parameter to maintain gradient accuracy while mitigating vanishing gradients. The gradient-vanishing degree is measured as G(x) = integral from -x to x of g(t)dt, and ASSG concentrates this around the theoretical upper bound (~0.87).

2. **SA-PGD (Stable Adaptive Projected Gradient Descent):** An adversarial attack with adaptive step size under L-infinity constraint. Key mechanical differences from standard PGD:
   - Uses L1-normalized momentum and gradient oscillation degree computation
   - Per-step L-infinity-norm clipping prevents excessive single-dimension updates
   - Adaptive step-size: clip(m_k / sqrt(v_k + xi) * eta_k, -eta_k, eta_k)
   - Maintains stability across 1000+ iterations vs. standard PGD's early convergence

**Experimental Results (eps = 8/255):**

| Dataset | Architecture | Standard STBP ASR | SA-PGD ASR | Improvement |
|---------|-------------|-------------------|------------|-------------|
| CIFAR-10 (AT) | SEWResNet19 | 75.38% | 88.44% | +13.06 pp |
| CIFAR-100 (AT) | SEWResNet19 | 88.22% | 93.19% | +4.97 pp |
| CIFAR10-DVS | VGG9 | 36.10% | 49.10% | +13.00 pp |

ASR = Attack Success Rate (higher = attack is more effective = model is less robust).

**Critical Finding:** Previous works using only 10 PGD iterations significantly underestimated attack effectiveness. The PSN neuron model showed 98%+ ASR with ASSG. Tested across LIF, LIF-2, IF, and PSN neuron models.

**Implications for Our Work:**
Our SNN adversarial evaluation uses standard FGSM and PGD with surrogate gradients (fast_sigmoid). Wang et al. would argue our SNN robustness numbers (26% FGSM, 19.25% PGD) may be **inflated** due to gradient masking. The true robustness gap between our SNN and ANN may be smaller than 14.9x. This is a legitimate threat to validity that we must acknowledge in the thesis discussion.

**Recommendation for thesis:** Acknowledge this as a threat to validity. State that standard gradient-based attacks may underestimate true vulnerability of SNNs due to surrogate gradient mismatch, per Wang et al. (2025). Note that applying SA-PGD to our audio SNN would be an important future work direction.

---

### 1.2 Is Our 14.9x SNN Robustness Ratio Consistent with Literature?

**Short answer: Our ratio is HIGH but directionally consistent. The magnitude is likely inflated by gradient masking.**

**Literature Comparison Table (FGSM, eps = 8/255 unless noted):**

| Paper | Year | Dataset | SNN Robust Acc | ANN Robust Acc | SNN/ANN Ratio | Notes |
|-------|------|---------|---------------|---------------|---------------|-------|
| **Ours** | 2026 | ESC-50 (audio) | 26.0% (eps=0.1) | 1.75% | **14.9x** | Standard FGSM |
| RSC-SNN (Wu et al.) | 2024 | CIFAR-10 | 54.52% | 10.89% | **5.0x** | ACM MM 2024 |
| RSC-SNN | 2024 | CIFAR-100 | 34.89% | 4.56% | **7.7x** | ACM MM 2024 |
| Nature Comms (2025) | 2025 | CIFAR-10 | ~2x ANN | baseline | **~2x** | "Twice the robustness" claim |
| Nature Comms (2025) | 2025 | FashionMNIST | ~20% (eps=0.5) | ~0% | **>>10x** | At high epsilon |
| RandHet-SNN | 2025 | CIFAR-10 | 53.53% | N/A | N/A | vs standard SNN baseline |
| Sharmin et al. | 2020 | CIFAR-10 | 3-6% higher than ANN | baseline | **~1.1-1.2x** | Seminal ECCV paper |

**Key Observations:**
- At **moderate epsilon** (e.g., 8/255 on CIFAR), the SNN/ANN robustness ratio is typically 2-8x
- At **high epsilon** (e.g., 0.1 on audio, 0.5 on FashionMNIST), where ANN accuracy collapses to near-zero, the ratio can appear extremely large (10x+)
- Our 14.9x ratio is measured at eps=0.1, which is a relatively aggressive perturbation for audio spectrograms
- The ANN drops to 1.75% (near random for 50 classes = 2%), essentially complete failure
- **The high ratio is partly an artifact of ANN near-total failure** rather than exceptional SNN robustness
- **Gradient masking likely further inflates this** per Wang et al.

**Recommendation for thesis:** Report the absolute numbers (SNN 26%, ANN 1.75%) rather than emphasizing the ratio. Frame it as: "The SNN retains non-trivial classification ability (26%) at perturbation magnitudes where the ANN is essentially defeated (1.75%, near chance level of 2%)." Acknowledge gradient masking caveat.

---

### 1.3 SNN Adversarial Robustness in Audio -- Literature Gap

**Finding: There are ZERO papers on SNN adversarial robustness specifically for audio classification.**

All SNN adversarial robustness work (2020-2026) has been conducted on:
- Image classification: CIFAR-10, CIFAR-100, MNIST, SVHN, Tiny-ImageNet, ImageNet
- Neuromorphic vision: CIFAR10-DVS, DVS-CIFAR10, N-Caltech101
- Event-based data: DVS128 Gesture

The closest related work:
- **Wu et al. (2018), Frontiers in Neuroscience:** SOM-SNN framework for robust sound classification, but focuses on noise robustness (Gaussian, environmental), NOT adversarial robustness
- **General audio adversarial robustness** (non-SNN): Active area for ASR systems, keyword spotting, but exclusively with ANNs (transformers, CNNs)

**This represents a significant novelty claim for the thesis:** Our work appears to be the first to evaluate SNN adversarial robustness on any audio classification task.

---

### 1.4 Gradient Masking -- Confirmed Issue for SNNs

**Gradient masking is now a confirmed and well-documented issue for SNN adversarial evaluation.**

Key evidence from 2024-2026 literature:

1. **Wang et al. (2512.22522, Dec 2025):** Directly quantifies gradient vanishing in surrogate functions. Shows attack success rates increase 4-13 pp when gradient masking is addressed via ASSG.

2. **Lin & Sengupta (2504.08897, Apr 2025):** Shows gradient-based attacks are **ineffective** against SNNs trained with local learning rules. Proposes hybrid transferability-based attack that is much stronger. The key finding: "local learning methods demonstrate more robustness than global methods" under standard attacks, but this apparent robustness largely disappears under their hybrid attack.

3. **Gradient Sparsity Trail (2509.23762, Sep 2025):** Identifies two types of gradient sparsity in SNNs: (a) architectural sparsity from design choices, (b) natural sparsity from spike signal nature. Both impair white-box attack effectiveness, creating false robustness signals.

4. **HART Attack (ICLR 2024):** "Threaten Spiking Neural Networks through Combining Rate and Temporal Information." Shows that attacks combining rate and temporal information are significantly stronger than rate-only or temporal-only attacks against SNNs.

5. **RSC-SNN (ACM MM 2024):** Addresses gradient obfuscation by applying the EOT (Expectation Over Transformation) method to obtain more accurate gradient estimates when evaluating SNN robustness.

**Consensus:** Standard FGSM/PGD with a fixed surrogate gradient function will **overestimate** SNN robustness. The degree of overestimation varies by architecture and neuron model but can be substantial (5-13+ pp in attack success rate).

**Implication for our thesis:** Our standard FGSM/PGD evaluation likely overestimates our SNN's true adversarial robustness. However, the **relative ordering** (SNN > ANN) is likely correct -- every paper in the literature confirms SNNs have some inherent robustness advantage, the debate is about how large that advantage truly is.

---

### 1.5 Comprehensive Landscape of SNN Adversarial Robustness (2024-2026)

#### Major Papers and Best Numbers Reported

**A. Defense Methods:**

| Paper | Venue | Method | CIFAR-10 PGD-7 | CIFAR-10 FGSM | CIFAR-10 Clean |
|-------|-------|--------|----------------|---------------|----------------|
| SNN-RAT (Ding et al.) | NeurIPS 2022 | Regularized AT | 45.23% | ~52% | ~89% |
| FEEL-SNN (2024) | NeurIPS 2024 | Frequency Encoding + Evolutionary Leak | Improved over RAT | Improved | ~89% |
| Robust Stable SNN (2024) | arXiv 2405.20694 | DLIF + MPPD + AT+Reg | **40.30%** (VGG11) | **56.71%** | **88.91%** |
| RSC-SNN (Wu et al.) | ACM MM 2024 | Randomized Smoothing Coding | 39.98% | 54.52% | 82.03% |
| RandHet-SNN (Wang et al.) | iScience 2025 | Random heterogeneous time constants | **44.86%** (PGD10) | 53.53% | 90.25% |
| TGO (Wang et al.) | ICLR 2026 | Threshold Guarding Optimization | 6.14% (vanilla), better w/ AT | 51.40% | 88.79% |
| RTE (Wang et al.) | arXiv 2508.11279 | Robust Temporal Self-Ensemble | 36.38% (APGD) | N/A | 81.90% |
| Sparse Conversion (Schmolli et al.) | CPAL 2025 | ANN-to-SNN conversion + sparsity | 40.0% | N/A | 83.2% |

**B. Attack Methods (making evaluation more reliable):**

| Paper | Venue | Attack Method | Key Finding |
|-------|-------|---------------|-------------|
| SA-PGD (Wang et al.) | arXiv Dec 2025 | Adaptive surrogate + adaptive step | Robustness overestimated by 5-13 pp |
| HART (Bu et al.) | ICLR 2024 | Combined rate + temporal attack | Stronger than rate-only attacks |
| Hybrid Attack (Lin & Sengupta) | arXiv Apr 2025 | Transferability-based | Local-learning robustness largely disappears |

**C. Mechanistic Understanding:**

| Paper | Venue | Key Insight |
|-------|-------|-------------|
| RSC-SNN | ACM MM 2024 | Poisson coding is conceptually equivalent to randomized smoothing |
| TGO (ICLR 2026) | ICLR 2026 | Threshold-neighboring neurons are the weak point; reducing them by 40% improves robustness |
| Gradient Sparsity (2025) | arXiv | Natural spike-induced gradient sparsity creates inherent (but limited) robustness |
| Nature Comms (2025) | Nature Comms | Temporal encoding + early-exit decoding = key to SNN robustness advantage |

---

### 1.6 Best SNN Adversarial Robustness on Any Audio Task

**No SNN adversarial robustness results exist for any audio task in the literature (as of March 2026).**

Our results (SNN 26% FGSM eps=0.1; SNN 19.25% PGD eps=0.05 on ESC-50) are, to our knowledge, the **first reported SNN adversarial robustness numbers on any audio/sound classification benchmark**.

For context, the best adversarial robustness numbers on **image** tasks are:
- CIFAR-10 (eps=8/255): ~45% PGD-7 robust accuracy (RandHet-SNN + RAT)
- CIFAR-100 (eps=8/255): ~26% PGD robust accuracy (RSC-SNN)
- ImageNet (eps=8/255): ~9% PGD robust accuracy (RSC-SNN)

These are not directly comparable to our audio numbers due to different epsilon scales and data domains, but they establish that SNN adversarial robustness is a genuinely active research area where robust accuracy in the 20-50% range under strong perturbations is typical.

---

## PART 2: CONTINUAL LEARNING WITH SNNs

### 2.1 Best SNN Continual Learning Results (2024-2026)

The field has seen rapid progress. Here is a summary of the best results:

**Task-Incremental Learning (TIL) -- Split CIFAR-100:**

| Paper | Venue | Method | CIFAR-100 TIL Acc | Steps | Notes |
|-------|-------|--------|-------------------|-------|-------|
| DSD-SNN (Chen et al.) | IJCAI 2023 | Dynamic structure growth + pruning | 81.17% | 20-step | 37.48% parameter usage |
| HLOP-SNN (Xiao et al.) | ICLR 2024 | Hebbian orthogonal projection | ~85%+ | 10-step | Near-zero forgetting |
| SCA-SNN (2024) | Neural Networks (ScienceDirect) | Context-aware similarity reuse | **86.45%** | 20-step | Best SNN TIL |
| PS-SNN (2026) | Scientific Reports | Pattern separation + expandable | N/A | 10-step | Surpasses DNN methods |
| LT-Gate (2025) | arXiv 2510.12843 | Local timescale gates | Retained ~95% of Task A perf | 2 tasks | Minimal forgetting under timescale shift |

**Class-Incremental Learning (CIL) -- Split CIFAR-100:**

| Paper | Venue | Method | CIFAR-100 CIL Acc | Steps |
|-------|-------|--------|-------------------|-------|
| DSD-SNN | IJCAI 2023 | Dynamic structure | ~50-55% | 10-step |
| SCA-SNN | Neural Networks 2024 | Context-aware | **57.06%** | 10-step |
| PS-SNN | Scientific Reports 2026 | Pattern separation | **76.42%** (B0, 10-step) | 10-step |

**Key Observations:**
- TIL accuracy on Split CIFAR-100 ranges from ~81-86% for the best SNN methods
- CIL is much harder; best SNN accuracy is 57-76%
- PS-SNN (2026) represents a significant jump in CIL performance
- SNNs are now approaching DNN-level performance on these benchmarks
