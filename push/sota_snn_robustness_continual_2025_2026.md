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
