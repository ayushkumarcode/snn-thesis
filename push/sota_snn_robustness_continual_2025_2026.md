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
