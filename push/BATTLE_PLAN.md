# Plan: ICONS 2026 + Thesis Extensions

---

## where we are

- **ICONS 2026 deadline:** April 1 (17 days from when this was written)
- **acceptance rate:** ~59% (13/22 in 2023). peer-reviewed, rebuttal allowed.
- **even if rejected:** still presented as poster. zero downside to submitting.
- **our position:** i think we're in the top 20-30% of ICONS submissions based on content. zero audio papers at ICONS 2025 -- wide open.
- **all 7 novelty claims confirmed** across literature search.

---

## PHASE 1: ICONS PAPER (March 15-31)

### the story
"First SNN on ESC-50: 7 encodings, SpiNNaker deployment, and the gap-collapse finding"

### 4 contributions (cut from 6)
1. first convolutional SNN on ESC-50 + 7-encoding comparison
2. SpiNNaker deployment with root-cause analysis
3. PANNs+SNN: gap collapses from 17pp to 1pp
4. SNN adversarial robustness on audio

### cut from paper (thesis only)
surrogate ablation, continual learning, augmentation, t-SNE, temporal analysis, per-class analysis

### new experiments to add

**must do (zero risk, 1-2 days each):**

| Experiment | Time | What it produces | Why it helps |
|-----------|------|------------------|-------------|
| SpiNNaker latency measurement | 0.5 day | ms per inference on hardware | ICONS reviewers expect real numbers |
| SpiNNaker energy from provenance | 1-2 days | mJ per inference (real, not theoretical) | fills biggest gap |
| Temporal ablation (truncate timesteps) | 0.5 day | accuracy-vs-timesteps curve | "X% accuracy in Y ms" = headline result |
| Encoding transfer matrix | 1 day | 7x7 heatmap | novel figure nobody has published |

**should do (if time permits):**

| Experiment | Time | What it produces | Why it helps |
|-----------|------|------------------|-------------|
| Noise robustness profiling | 2-3 days | SNN vs ANN degradation curves | bridges adversarial to real-world |
| 1-fold UrbanSound8K | 2 days | cross-dataset validation | kills "single dataset" objection |
| Neuron ablation / fault tolerance | 1 day | graceful degradation comparison | hardware reliability finding |

**high-risk high-reward (only if ahead of schedule):**

| Experiment | Time | What it produces | Why it helps |
|-----------|------|------------------|-------------|
| Full SpiNNaker deploy via IF_cond_exp | 2-3 days | FC1+FC2 on hardware | game-changer if it works |
| Few-shot learning curves | 2-3 days | data efficiency comparison | tests central thesis narrative |

### paper production (parallel with experiments)

| Task | Time |
|------|------|
| Set up Overleaf with ACM template | 0.5 day |
| Convert ICONS draft to LaTeX | 1-2 days |
| Create Figure 1: architecture diagram | 1 day |
| Create Figure 2: SpiNNaker pipeline | 1 day |
| Create Figure 3: encoding bar chart | 0.5 day |
| Rewrite abstract (150-200 words) | 0.5 day |
| Final polish + supervisor review | 2-3 days |
| Submit on EasyChair | 0.5 day |

### title (recommended)
> **Spiking Neural Networks for Environmental Sound Classification: From Seven Encodings to SpiNNaker Deployment**

---

## PHASE 2: THESIS EXTENSIONS (run on CSF3 in parallel)

these go in the thesis, not the ICONS paper.

### tier 1: zero-risk, guaranteed novel

| # | Experiment | Days | Value |
|---|-----------|------|-------|
| 1 | Encoding transfer matrix (7x7) | 1 | HIGH -- novel figure |
| 2 | Temporal ablation (timestep truncation) | 0.5 | HIGH -- deployment finding |
| 3 | Neuron ablation / fault tolerance | 1 | MED-HIGH |
| 4 | Weight distribution analysis | 0.5 | MEDIUM |
| 5 | Pruning resilience (30-90% sparsity) | 2 | MED-HIGH |

### tier 2: high value, moderate effort

| # | Experiment | Days | Value |
|---|-----------|------|-------|
| 6 | Noise robustness profiling | 2-3 | HIGH |
| 7 | Few-shot learning curves | 2-3 | HIGH |
| 8 | Spike efficiency Pareto (L1 reg) | 2-3 | HIGH |
| 9 | SNN saliency maps (spike Grad-CAM) | 3-4 | HIGH |
| 10 | Stochastic resonance | 1-2 | HIGH if positive |

### tier 3: SpiNNaker-specific

| # | Experiment | Days | Value |
|---|-----------|------|-------|
| 11 | Full deploy via IF_cond_exp + MaxPool | 2-3 | VERY HIGH if works |
| 12 | Spike drop robustness | 1-2 | HIGH -- explains gap |
| 13 | WTA lateral inhibition | 1-2 | MEDIUM |
| 14 | SpiNNaker energy from provenance | 1-2 | HIGH |
| 15 | SpiNNaker 2 readiness (NIR export) | 2-3 | MEDIUM |

### tier 4: ambitious (only if everything else done)

| # | Experiment | Days | Value |
|---|-----------|------|-------|
| 16 | On-chip STDP learning for FC2 | 5-7 | VERY HIGH |
| 17 | LSM reservoir on SpiNNaker | 5-7 | VERY HIGH |
| 18 | Izhikevich resonator neurons | 3-5 | HIGH |
| 19 | Cross-domain transfer (Speech Commands) | 3-4 | MED-HIGH |
| 20 | Real-time microphone demo | 2-3 | LOW (science), HIGH (demo) |

---

## PHASE 3: AFTER ICONS SUBMISSION

| Date | Event | Action |
|------|-------|--------|
| April 1 | ICONS submitted | shift to thesis writing |
| May 18 | Reviews back | prepare rebuttals (pre-drafted responses ready) |
| May 25 | Rebuttal due | submit |
| June 5 | Decision | celebrate or plan poster |
| ~July | DCASE 2026 deadline | second paper opportunity (perfect topic match) |
| ~Sep | ICASSP 2027 deadline | third paper opportunity |

---

## prepared reviewer responses

1. **"47% is low"** -- baseline datum. PANNs+SNN proves 92.5% when features are good. gap identifies the bottleneck.
2. **"SNN uses more energy"** -- honest: yes in software. path: reduce spike rate from 25.8% to <6.4%. on neuromorphic hardware, AC costs 5.1x less than MAC.
3. **"SpiNNaker 33% with high variance"** -- first quantified hardware gap. 50-class task is 6.25x harder than prior work (8 pure tones). root cause documented.
4. **"Only ESC-50"** -- standard benchmark with predefined 5-fold. UrbanSound8K as future work (or add 1-fold result).
5. **"PANNs isn't neuromorphic"** -- hybrid edge paradigm: CNN14 in cloud, SNN on edge. precedent: Seekings et al. ICONS 2024.
