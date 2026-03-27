# BATTLE PLAN: ICONS 2026 + Thesis Extensions
*Generated: 15 March 2026 — Synthesized from 8 research agents*

---

## THE SITUATION

- **ICONS 2026 deadline:** April 1 (17 days)
- **Acceptance rate:** ~59% (13/22 in 2023). Peer-reviewed, rebuttal allowed.
- **Even if rejected:** still presented as poster. Zero downside to submitting.
- **Our position:** Top 20-30% of ICONS submissions in content. Zero audio papers at ICONS 2025 — wide open.
- **All 7 novelty claims confirmed** across exhaustive literature search.

---

## PHASE 1: ICONS PAPER (March 15-31)

### The Story
"First SNN on ESC-50: 7 encodings, SpiNNaker deployment, and the gap-collapse finding"

### 4 Contributions (cut from 6)
1. First convolutional SNN on ESC-50 + 7-encoding comparison
2. SpiNNaker deployment with root-cause analysis
3. PANNs+SNN: gap collapses from 17pp to 1pp
4. SNN adversarial robustness on audio

### CUT from paper (thesis only)
Surrogate ablation, continual learning, augmentation, t-SNE, temporal analysis, per-class analysis

### New experiments to ADD (strengthen paper)

**MUST DO (zero-risk, 1-2 days each):**

| Experiment | Time | What it produces | Why it helps |
|-----------|------|------------------|-------------|
| SpiNNaker latency measurement | 0.5 day | ms per inference on hardware | ICONS reviewers expect real numbers |
| SpiNNaker energy from provenance | 1-2 days | mJ per inference (real, not theoretical) | Fills biggest gap in paper |
| Temporal ablation (truncate timesteps) | 0.5 day | Accuracy-vs-timesteps curve | "X% accuracy in Y ms" = headline result |
| Encoding transfer matrix | 1 day | 7×7 heatmap | Novel figure nobody has published |

**SHOULD DO (if time permits, 2-3 days each):**

| Experiment | Time | What it produces | Why it helps |
|-----------|------|------------------|-------------|
| Noise robustness profiling (SNR sweep) | 2-3 days | SNN vs ANN degradation curves | Bridges adversarial to real-world |
