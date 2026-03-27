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

#### 7. SNN Saliency Maps / Spike Grad-CAM (3-4 days, medium risk)
adapt Grad-CAM for surrogate gradients -> spectrogram heatmaps showing what the SNN "looks at." compare SNN vs ANN saliency. if SNN highlights transients while ANN highlights sustained textures -> proof of different computational strategies.

#### 8. Pruning Resilience (2 days, low risk)
magnitude pruning at 30-90% sparsity, compare SNN vs ANN. do weight sparsity + activation sparsity compound? if SNN tolerates 70% weight pruning -> "92% total sparsity" is a powerful hardware number.

#### 9. Stochastic Resonance (1-2 days, medium-high risk but huge if positive)
add controlled noise to membrane potentials at inference. does noise IMPROVE classification? stochastic resonance is well-known in biology but barely tested in trained SNNs. a positive result would be the most biologically interesting finding in the thesis.

---

## Tier 3: Interesting But Less Critical

#### 10. Biological Firing Pattern Analysis (2-3 days)
do hidden neurons develop temporal specialization (onset detectors vs sustained-pattern detectors)? cluster neurons by firing profiles. connection to auditory neuroscience.

#### 11. SpiNNaker Latency Benchmarking (1-2 days)
actual wall-clock inference time on SpiNNaker vs GPU vs CPU. real measurements beat theoretical estimates. ICONS reviewers love hardware numbers.

#### 12. Weight Distribution Analysis (1 day)
compare SNN vs ANN weight distributions (kurtosis, sparsity, spectral properties). quick post-hoc analysis.

#### 13. Membrane Potential Trajectories (2 days)
PCA/UMAP of membrane potential dynamics -> "neural trajectory" plots. do different sound classes create distinct dynamical attractors? could make gorgeous figures.

#### 14. Ensemble of Encodings (1-2 days)
combine top 3-4 encodings via voting. more interesting: error complementarity analysis (do different encodings make different mistakes?).

#### 15. LIF Beta/Threshold Landscape (3-4 days)
2D sweep of LIF parameters -> accuracy heatmap. is the SNN robust or fragile to biophysical parameters?

#### 16. Sound Event Detection (2-3 days)
use SNN temporal dynamics for frame-level event detection, not just clip-level classification.

#### 17. Cross-Domain Transfer to Speech Commands (3-4 days)
test SNN on Google Speech Commands v2 (35 classes). do SNN audio features transfer?

#### 18. Real-Time Microphone Demo (2-3 days)
live audio -> mel -> SNN -> classification with spike visualization. great for thesis defense, not for paper.

---

## Recommended Sprint Plan

| Day | Task | Expected Output |
|-----|------|----------------|
| 1 | Encoding Transfer Matrix (#1) | 7x7 heatmap figure |
| 1 | Temporal Ablation (#2) | accuracy-vs-timesteps curve |
| 2-3 | Noise Robustness (#3) | SNR degradation curves |
| 2-3 | Few-Shot Learning Curves (#4) on CSF3 | data efficiency curves |
| 4 | Neuron Ablation (#5) | fault tolerance comparison |
| 5 | analysis, figures, writing | publication-ready results |

week 2 add-ons: spike efficiency frontier (#6), stochastic resonance (#9), SNN saliency maps (#7).
