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
