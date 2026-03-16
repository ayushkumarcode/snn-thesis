# Creative Extensions Brainstorm for SNN-ESC50
*Generated: 15 March 2026*

## Top 18 Ideas — Ranked by Impact vs Effort

### Tier 1: DO THESE FIRST (low effort, high payoff, ICONS-ready)

#### 1. Encoding Transfer Matrix (1-2 days, ZERO risk)
Train with encoding X, test with encoding Y → 7×7 transfer matrix. **Nobody has published this.** Probes whether SNNs learn encoding-specific circuits or general audio features. Guaranteed novel figure.

#### 2. Temporal Ablation — How Many Timesteps Needed? (1 day, ZERO risk)
Evaluate trained SNN using only first T timesteps (T=1,2,5,10,15,20,25). No retraining. If SNN reaches 90% accuracy by timestep 10/25, that's 60% energy savings. Directly actionable deployment finding.

#### 3. Noise Robustness Profiling (2-3 days, very low risk)
SNN vs ANN accuracy across SNR levels (clean, 20dB, 10dB, 0dB, -5dB) with Gaussian, babble, pink noise. Bridges adversarial finding to real-world. **The single most reviewer-friendly addition for ICONS.**

#### 4. Few-Shot Learning Curves (2-3 days, low risk)
Train with 100%, 50%, 25%, 10%, 5% of data. Does the SNN-ANN gap shrink or widen? **Directly tests the central thesis narrative** ("SNNs need more data"). Either outcome is scientifically valuable.

#### 5. Neuron Ablation / Fault Tolerance (1 day, very low risk)
Randomly silence 10-50% of neurons at inference. Compare SNN vs ANN graceful degradation. Mimics hardware faults on neuromorphic chips. "SNN maintains X% accuracy with 30% neuron death" is a headline finding.

---

### Tier 2: HIGH VALUE, MODERATE EFFORT

#### 6. Spike Efficiency Frontier / Pareto (2-3 days, low risk)
Add L1 spike regularization at varying strengths → map full accuracy-vs-spike-count Pareto curve. Converts energy analysis from single point to design space. Hardware designers love this.

#### 7. SNN Saliency Maps / Spike Grad-CAM (3-4 days, medium risk)
Adapt Grad-CAM for surrogate gradients → spectrogram heatmaps showing what the SNN "looks at." Compare SNN vs ANN saliency. If SNN highlights transients while ANN highlights sustained textures → proof of different computational strategies.

#### 8. Pruning Resilience (2 days, low risk)
Magnitude pruning at 30-90% sparsity, compare SNN vs ANN. Key question: do weight sparsity + activation sparsity compound? If SNN tolerates 70% weight pruning → "92% total sparsity" is a powerful hardware number.

#### 9. Stochastic Resonance (1-2 days, medium-high risk but HUGE if positive)
Add controlled noise to membrane potentials at inference. Does noise IMPROVE classification? Stochastic resonance is well-known in biology but barely tested in trained SNNs. A positive result would be the most biologically interesting finding in the thesis.

---

### Tier 3: INTERESTING BUT LESS ICONS-CRITICAL

#### 10. Biological Firing Pattern Analysis (2-3 days)
Do hidden neurons develop temporal specialization (onset detectors vs sustained-pattern detectors)? Cluster neurons by firing profiles. Connection to auditory neuroscience.

#### 11. SpiNNaker Latency Benchmarking (1-2 days)
Actual wall-clock inference time on SpiNNaker vs GPU vs CPU. Real measurements > theoretical estimates. ICONS reviewers love hardware numbers.

#### 12. Weight Distribution Analysis (1 day)
Compare SNN vs ANN weight distributions (kurtosis, sparsity, spectral properties). Quick post-hoc analysis.

#### 13. Membrane Potential Trajectories (2 days)
PCA/UMAP of membrane potential dynamics → "neural trajectory" plots. Gorgeous figures. Do different sound classes create distinct dynamical attractors?

#### 14. Ensemble of Encodings (1-2 days)
Combine top 3-4 encodings via voting. More interesting: error complementarity analysis (do different encodings make different mistakes?).

#### 15. LIF Beta/Threshold Landscape (3-4 days)
2D sweep of LIF parameters → accuracy heatmap. Is the SNN robust or fragile to biophysical parameters?

#### 16. Sound Event Detection (2-3 days)
Use SNN temporal dynamics for frame-level event detection, not just clip-level classification.

#### 17. Cross-Domain Transfer to Speech Commands (3-4 days)
Test SNN on Google Speech Commands v2 (35 classes). Do SNN audio features transfer?

#### 18. Real-Time Microphone Demo (2-3 days)
Live audio → mel → SNN → classification with spike visualization. Great for thesis defense, not for paper.

---

## Recommended 1-Week Sprint

| Day | Task | Expected Output |
|-----|------|----------------|
| 1 | Encoding Transfer Matrix (#1) | 7×7 heatmap figure |
| 1 | Temporal Ablation (#2) | Accuracy-vs-timesteps curve |
| 2-3 | Noise Robustness (#3) | SNR degradation curves (SNN vs ANN) |
| 2-3 | Few-Shot Learning Curves (#4) on CSF3 | Data efficiency curves |
| 4 | Neuron Ablation (#5) | Fault tolerance comparison |
| 5 | Analysis, figures, writing | Publication-ready results |

## Recommended Week 2 Add-ons
- Spike Efficiency Frontier (#6)
- Stochastic Resonance (#9) — high risk, extraordinary payoff
- SNN Saliency Maps (#7)
