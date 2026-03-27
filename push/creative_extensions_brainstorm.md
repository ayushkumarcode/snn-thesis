# Creative Extensions Brainstorm for SNN-ESC50

ranked by impact vs effort. ideas for new experiments that could strengthen the thesis and/or the ICONS paper.

## Tier 1: Do These First (low effort, high payoff)

#### 1. Encoding Transfer Matrix (1-2 days, zero risk)
train with encoding X, test with encoding Y -> 7x7 transfer matrix. **nobody has published this.** probes whether SNNs learn encoding-specific circuits or general audio features. guaranteed novel figure.

#### 2. Temporal Ablation -- How Many Timesteps Needed? (1 day, zero risk)
evaluate trained SNN using only first T timesteps (T=1,2,5,10,15,20,25). no retraining. if SNN reaches 90% accuracy by timestep 10/25, that's 60% energy savings. directly actionable for deployment.

#### 3. Noise Robustness Profiling (2-3 days, very low risk)
SNN vs ANN accuracy across SNR levels (clean, 20dB, 10dB, 0dB, -5dB) with Gaussian, babble, pink noise. bridges our adversarial finding to real-world. probably the single most reviewer-friendly addition for ICONS.

#### 4. Few-Shot Learning Curves (2-3 days, low risk)
train with 100%, 50%, 25%, 10%, 5% of data. does the SNN-ANN gap shrink or widen? **directly tests the central thesis narrative** that SNNs need more data. either outcome is scientifically valuable.

#### 5. Neuron Ablation / Fault Tolerance (1 day, very low risk)
randomly silence 10-50% of neurons at inference. compare SNN vs ANN graceful degradation. mimics hardware faults on neuromorphic chips. "SNN maintains X% accuracy with 30% neuron death" is a headline finding.

---

## Tier 2: High Value, Moderate Effort

#### 6. Spike Efficiency Frontier / Pareto (2-3 days, low risk)
add L1 spike regularization at varying strengths -> map full accuracy-vs-spike-count Pareto curve. converts energy analysis from single point to design space. hardware designers would love this.

