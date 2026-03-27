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
