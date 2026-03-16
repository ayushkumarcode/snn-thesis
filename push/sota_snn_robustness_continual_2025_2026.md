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

### 2.2 Typical Forgetting Rates for SNNs

**Forgetting rates vary enormously by method:**

| Method | Forgetting | Scenario | Notes |
|--------|-----------|----------|-------|
| Naive sequential (no CL method) | 70-90%+ | Any | Catastrophic forgetting baseline |
| HLOP-SNN | **Near-zero** | TIL | Orthogonal projection eliminates forgetting |
| LT-Gate | **~2.8pp drop** from peak | Domain-IL | vs 5.8pp for HLOP, 7.1pp for DSD-SNN |
| DSD-SNN | ~5-10% | TIL | Structure growth compensates |
| SCA-SNN | Not explicitly reported | TIL/CIL | Competitive with DNN methods |
| NACA (Science Advances) | "Markedly mitigated" | 5-class sequential | ~2% accuracy improvement + 98% less energy |
| **Ours (no CL method)** | **74.4% (SNN)** | TIL, 5 super-cats | No CL method applied, raw sequential |
| **Ours (no CL method)** | **81.3% (ANN)** | TIL, 5 super-cats | No CL method applied, raw sequential |

**Our results in context:** Our 74.4% SNN forgetting and 81.3% ANN forgetting are measured **without any continual learning method** -- this is the naive sequential baseline. This is the expected regime: without any CL technique, both SNNs and ANNs exhibit catastrophic forgetting. The interesting finding is the **6.9pp gap** between SNN and ANN forgetting.

### 2.3 Is 6.9pp Less Forgetting for SNN vs ANN Consistent with Literature?

**Yes, this is directionally consistent but quantitative comparisons are scarce.**

The literature broadly supports that SNNs have mild inherent advantages over ANNs for continual learning, but very few papers provide direct SNN vs ANN forgetting comparisons under identical conditions:

1. **Nature Communications (2025):** The neuromorphic computing paper states SNNs "exploit temporal processing capabilities" that can help with stability, but does not provide direct forgetting comparisons.

2. **NACA (Science Advances, 2023):** Tested on both ANNs and SNNs. NACA mitigated forgetting in both, but the relative advantage of SNN vs ANN was not the focus.

3. **DSD-SNN (IJCAI 2023):** Compares with DNN methods (EWC, GEM, RCL) and shows SNN methods can match or exceed them, but does not isolate the SNN-vs-ANN effect without a CL method.

4. **Theoretical argument:** SNNs have inherent properties that should help with continual learning:
   - Spike sparsity = implicit regularization (fewer parameters are heavily activated per task)
   - Temporal dynamics provide additional state that can encode task-specific information
   - LIF leak acts as a natural forgetting mechanism that prevents over-commitment to specific weight configurations

**Our 6.9pp result** is valuable because:
- It isolates the SNN vs ANN effect without any CL method confound
- It uses identical architectures (same conv+FC structure, same parameter count)
- It demonstrates an inherent advantage, not one conferred by a CL algorithm
- The magnitude (6.9pp, ~8.5% relative reduction in forgetting) is modest but meaningful

**Recommendation:** Frame this as evidence for an inherent mild SNN advantage in continual learning, consistent with the theoretical arguments in the literature. Note that dedicated CL methods (HLOP, DSD-SNN, etc.) can reduce forgetting much further.

### 2.4 Task-Incremental Learning with SNNs

**The dominant paradigm for SNN continual learning is task-incremental learning (TIL).**

Major approaches (2023-2026):

**A. Architecture-Based (Expansion/Isolation):**
- **DSD-SNN (IJCAI 2023):** Dynamically grows neurons for new tasks, prunes redundant ones. 81.17% on 20-step CIFAR-100 TIL. Uses 37.48% of parameters.
- **PS-SNN (Scientific Reports 2026):** Predefined orthogonal class centers + neurogenesis-inspired expansion. 76.42% CIL. State-of-the-art SNN CIL.
- **SCA-SNN (Neural Networks 2024):** Context-aware neuron reuse. 86.45% TIL on 20-step CIFAR-100. Surpasses DNN methods (DER++, HAT, iCaRL).

**B. Regularization/Projection-Based:**
- **HLOP-SNN (ICLR 2024):** Hebbian learning-based orthogonal projection. Near-zero forgetting. Compatible with multiple error propagation methods (BPTT, e-prop, etc.). Can combine with memory replay.
- **LT-Gate (arXiv 2025):** Local timescale gates with dual time constants. 95% retention of Task A performance. Better than HLOP under timescale shift.

**C. Replay-Based:**
- **Spiking Compressed CL (Dequino et al.):** Latent replay with lossy time compression on SHD dataset. 92.46% sample-incremental, 92.05% class-incremental. Only 2.2% accuracy loss on old tasks in progressive learning.

**D. Bio-Inspired:**
- **NACA (Science Advances 2023):** Neuromodulation-assisted credit assignment. Tested on MNIST (10 classes sequential) and TIDigits (speech).
- **AGMP (Frontiers 2025):** Astrocyte-gated multi-timescale plasticity. Online continual learning without replay buffers. Matches offline BPTT accuracy on DVS128 Gesture and SHD.

### 2.5 STDP vs Surrogate Gradient for Continual Learning

**Both approaches are active, with different strengths:**

| Aspect | STDP / Local Rules | Surrogate Gradient (BPTT) |
|--------|-------------------|--------------------------|
| **Biological plausibility** | High | Low |
| **Scalability** | Limited (shallow networks) | Good (deep networks, ResNets) |
| **Best CL accuracy** | ~95% on MNIST variants | 86%+ on CIFAR-100 |
| **Forgetting** | Naturally lower (local updates) | Requires explicit CL methods |
| **Hardware compatibility** | Excellent (on-chip learning) | Poor (requires backprop infrastructure) |
| **Key papers** | NACA (2023), AGMP (2025) | HLOP (ICLR 2024), DSD-SNN (IJCAI 2023), SCA-SNN (2024) |

**Key insight from Lin & Sengupta (2025):** Local learning methods show apparent adversarial robustness advantages but also have limitations for continual learning at scale. The hybrid approach (combining local and global learning) may be optimal.

**Emerging trend:** Combining STDP with supervised signals:
- AGMP (2025) couples fast eligibility traces (STDP-like) with slow astrocytic gating and broadcast error signals
- NACA (2023) uses neuromodulator-modulated STDP
- Three-factor learning rules (2025) generalize STDP with a third modulatory signal

**Our approach (surrogate gradient, no CL method)** is the simplest baseline. The literature shows that adding any CL method would dramatically reduce our 74.4% forgetting rate.

### 2.6 SNN Continual Learning on Audio Tasks

**Very limited work exists, but the gap is narrowing:**

1. **Spiking Compressed CL (Dequino et al.):** Continual learning on **Spiking Heidelberg Digits (SHD)** -- the closest to audio CL with SNNs. Used latent replay. Achieved 92.46% sample-incremental accuracy. Progressive learning: 78.4% final accuracy adding German digits to English digits, with only 2.2% loss on older classes. This is the **most directly comparable** work to ours.

2. **NACA (Science Advances 2023):** Tested on **TIDigits** (spoken digit recognition), a speech dataset. Demonstrated mitigated forgetting with neuromodulation-assisted credit assignment.

3. **AGMP (Frontiers 2025):** Evaluated on **SHD (Spiking Heidelberg Digits)**, which is derived from audio. Demonstrated effective continual learning without replay buffers.

4. **Our work (ESC-50):** 5 super-categories of environmental sound, task-incremental. SNN forgetting = 74.4%, ANN forgetting = 81.3%. No CL method applied.

**Assessment:** Environmental sound classification (ESC-50) is a distinctly different audio domain from speech/digit recognition. Our continual learning evaluation on ESC-50 super-categories appears to be **novel** -- no prior work has evaluated SNN continual learning on environmental sound classification.

---

## PART 3: SYNTHESIS AND THESIS IMPLICATIONS

### 3.1 Key Narrative Points for Thesis

**Adversarial Robustness:**
1. SNNs demonstrate genuine inherent adversarial robustness compared to ANNs -- this is confirmed across 20+ papers (2020-2026)
2. The magnitude of the advantage is debated: 2-8x on vision benchmarks at standard epsilon
3. Our 14.9x ratio at eps=0.1 FGSM is consistent with the pattern but likely inflated by (a) ANN near-complete failure at this epsilon and (b) gradient masking
4. Standard PGD evaluation of SNNs is now known to be unreliable (Wang et al. 2025) -- this is a threat to validity
5. Our work is the **first** to evaluate SNN adversarial robustness on audio classification
6. The mechanisms behind SNN robustness include: spike thresholding (noise filtering), temporal integration, gradient sparsity, and input discretization

**Continual Learning:**
1. Our 6.9pp less forgetting for SNN vs ANN is directionally consistent with the theoretical argument that SNNs have inherent CL advantages (sparsity, temporal dynamics, leak)
2. State-of-the-art SNN CL methods achieve near-zero forgetting (HLOP-SNN) or 85%+ accuracy on CIFAR-100 TIL -- our naive baseline is expected to show high forgetting
3. The **interesting finding** is the SNN-ANN gap itself, not the absolute forgetting rate
4. SNN continual learning on environmental sound is **novel**
5. STDP-based and surrogate gradient-based CL methods both show promise; combining them is the emerging trend

### 3.2 Recommended Citations for Thesis

**Must-cite (directly relevant to our findings):**

| # | Paper | Relevance |
|---|-------|-----------|
| 1 | Wang et al. (2512.22522) SA-PGD | Threat to validity for our adversarial evaluation |
| 2 | Sharmin et al. (ECCV 2020) | Foundational "inherent robustness" paper |
| 3 | FEEL-SNN (NeurIPS 2024) | Current SOTA defense method |
| 4 | RSC-SNN (ACM MM 2024) | Poisson coding = randomized smoothing insight |
| 5 | RandHet-SNN (iScience 2025) | Heterogeneity mechanism for robustness |
| 6 | TGO (ICLR 2026) | Threshold-neighboring neurons as weakness |
| 7 | Nature Comms SNN robustness (2025) | High-profile SNN robustness validation |
| 8 | HLOP-SNN (ICLR 2024) | SOTA SNN continual learning |
| 9 | DSD-SNN (IJCAI 2023) | SNN CL architecture expansion |
| 10 | SCA-SNN (Neural Networks 2024) | Context-aware SNN CL |
| 11 | PS-SNN (Scientific Reports 2026) | Best SNN CIL result |
| 12 | AGMP (Frontiers 2025) | Astrocyte-gated CL on SHD (audio) |

**Should-cite (supporting context):**

| # | Paper | Relevance |
|---|-------|-----------|
| 13 | SNN-RAT (NeurIPS 2022) | Baseline adversarial training method |
| 14 | HIRE-SNN (ICCV 2021) | Crafted noise training for robustness |
| 15 | Gradient Sparsity Trail (2509.23762) | Gradient sparsity mechanism |
| 16 | HART Attack (ICLR 2024) | Rate+temporal combined attack |
| 17 | Lin & Sengupta (2504.08897) | Local learning robustness |
| 18 | Robust Stable SNN (2405.20694) | DLIF neuron + MPPD minimization |
| 19 | RTE (2508.11279) | Temporal self-ensemble |
| 20 | NACA (Science Advances 2023) | Bio-inspired CL |
| 21 | Dequino et al. | SHD continual learning |
| 22 | LT-Gate (2510.12843) | Timescale-robust continual learning |
| 23 | Sparse Conversion (CPAL 2025) | Sparse + robust ANN-to-SNN |

### 3.3 Quantitative Summary Table for Discussion Chapter

**SNN Adversarial Robustness -- Literature Comparison:**

| Work | Domain | SNN Robust | ANN Robust | Attack | Epsilon |
|------|--------|-----------|-----------|--------|---------|
| **Ours** | **Audio (ESC-50)** | **26.00%** | **1.75%** | **FGSM** | **0.1** |
| **Ours** | **Audio (ESC-50)** | **19.25%** | **0.00%** | **PGD** | **0.05** |
| RSC-SNN | Vision (CIFAR-10) | 54.52% | 10.89% | FGSM | 8/255 |
| RSC-SNN | Vision (CIFAR-100) | 34.89% | 4.56% | FGSM | 8/255 |
| RandHet-SNN | Vision (CIFAR-10) | 53.53% | ~52% (standard SNN) | FGSM | 8/255 |
| Robust Stable SNN | Vision (CIFAR-10) | 56.71% | N/A | FGSM | 8/255 |
| RTE | Vision (CIFAR-10) | 36.38% | N/A | APGD | 8/255 |

**SNN Continual Learning -- Literature Comparison:**

| Work | Domain | Forgetting | Scenario | CL Method |
|------|--------|-----------|----------|-----------|
| **Ours (SNN)** | **Audio (ESC-50)** | **74.4%** | **TIL, 5 tasks** | **None (baseline)** |
| **Ours (ANN)** | **Audio (ESC-50)** | **81.3%** | **TIL, 5 tasks** | **None (baseline)** |
| HLOP-SNN | Vision (CIFAR-100) | ~0% | TIL | Orthogonal projection |
| DSD-SNN | Vision (CIFAR-100) | ~5-10% | TIL | Structure growth |
| LT-Gate | Vision | ~2.8pp drop | Domain-IL | Timescale gates |
| Dequino et al. | Audio (SHD) | ~2.2% loss | Progressive CL | Latent replay |

---

## PART 4: RESEARCH GAPS AND CONFIDENCE ASSESSMENT

### 4.1 Research Gaps Identified

1. **No SNN adversarial robustness work on audio** -- our work fills this gap
2. **No SNN continual learning on environmental sound** -- our work fills this gap
3. **Very few direct SNN-vs-ANN forgetting comparisons** without CL methods -- our work provides this
4. **SA-PGD has not been applied to audio SNNs** -- future work
5. **No large-scale SNN adversarial evaluation** (ImageNet-scale) with reliable attacks
6. **Limited work on combining adversarial robustness and continual learning for SNNs**

### 4.2 Confidence Assessment

| Finding | Confidence | Basis |
|---------|------------|-------|
| SNNs have inherent adversarial robustness advantage over ANNs | **HIGH** | 20+ papers, multiple groups, 2020-2026 |
| Standard FGSM/PGD overestimates SNN robustness | **HIGH** | Wang et al. 2025, Lin & Sengupta 2025, gradient sparsity work |
| Our 14.9x ratio is directionally correct but magnitude inflated | **HIGH** | Consistent with literature pattern at high epsilon |
| SNNs have mild inherent CL advantage | **MEDIUM** | Theoretical support strong, empirical comparisons scarce |
| Our 6.9pp gap is meaningful | **MEDIUM** | Consistent with theory, but single experiment (fold 4, 5 tasks) |
| Audio SNN adversarial robustness is novel | **HIGH** | Exhaustive literature search found zero prior work |
| Audio SNN continual learning on ESC-50 is novel | **HIGH** | No prior work on environmental sound CL with SNNs |

### 4.3 Recommended Follow-ups (Future Work)

1. Apply SA-PGD or ASSG-enhanced attacks to our audio SNN to get more reliable robustness estimates
2. Apply a dedicated CL method (HLOP or DSD-SNN) to our ESC-50 task structure
3. Test adversarial robustness with AutoAttack (gradient-free component removes masking concern)
4. Evaluate on additional audio datasets (UrbanSound8K, SHD)
5. Investigate whether encoding method affects adversarial robustness (our work uses direct encoding; RSC-SNN shows randomized smoothing coding helps)
6. Study the interaction between adversarial robustness and continual learning -- does adversarial training help or hurt CL?

---

## SOURCES

### Adversarial Robustness

- [Wang et al. (2025) -- SA-PGD, ASSG](https://arxiv.org/abs/2512.22522)
- [Wang et al. (2025) -- TGO, ICLR 2026](https://arxiv.org/abs/2602.20548)
- [Lin & Sengupta (2025) -- Local Learning Robustness](https://arxiv.org/abs/2504.08897)
- [FEEL-SNN -- NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/a73474c359ed523e6cd3174ed29a4d56-Paper-Conference.pdf)
- [RSC-SNN -- ACM MM 2024](https://arxiv.org/abs/2407.20099)
- [RandHet-SNN -- iScience 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12159496/)
- [Neuromorphic Computing Robustness -- Nature Communications 2025](https://www.nature.com/articles/s41467-025-65197-x)
- [Gradient Sparsity Trail -- arXiv 2025](https://arxiv.org/abs/2509.23762)
- [Robust Stable SNN -- arXiv 2024](https://arxiv.org/abs/2405.20694)
- [RTE -- arXiv 2025](https://arxiv.org/abs/2508.11279)
- [HART Attack -- ICLR 2024](https://openreview.net/forum?id=xv8iGxENyI)
- [Sharmin et al. (2020) -- ECCV Inherent Robustness](https://arxiv.org/abs/2003.10399)
- [HIRE-SNN -- ICCV 2021](https://arxiv.org/abs/2110.11417)
- [SNN-RAT -- NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9cf904c86cc5f9ac95646c07d2cfa241-Abstract-Conference.html)
- [Sparse Conversion -- CPAL 2025](https://arxiv.org/abs/2505.15833)
- [Robust Conversion -- TMLR 2024](https://arxiv.org/abs/2311.09266)
- [Input Filtering Defense -- ScienceDirect 2024](https://www.sciencedirect.com/science/article/abs/pii/S1383762124001462)

### Continual Learning

- [HLOP-SNN -- ICLR 2024](https://arxiv.org/abs/2402.11984)
- [DSD-SNN -- IJCAI 2023](https://arxiv.org/abs/2308.04749)
- [SCA-SNN -- Neural Networks 2024](https://arxiv.org/abs/2411.05802)
- [PS-SNN -- Scientific Reports 2026](https://www.nature.com/articles/s41598-026-42970-6)
- [AGMP -- Frontiers in Neuroscience 2025](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1768235/full)
- [LT-Gate -- arXiv 2025](https://arxiv.org/abs/2510.12843)
- [NACA -- Science Advances 2023](https://www.science.org/doi/10.1126/sciadv.adi2947)
- [Spiking Compressed CL on SHD](https://github.com/dequino/spiking-compressed-continual-learning)
- [NCL Survey -- arXiv 2024](https://arxiv.org/abs/2410.09218)
- [STDP CL Survey -- ResearchGate 2024](https://www.researchgate.net/publication/382527302)
