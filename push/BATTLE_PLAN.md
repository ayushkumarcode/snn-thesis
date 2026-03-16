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
| 1-fold UrbanSound8K | 2 days | Cross-dataset validation | Kills "single dataset" objection |
| Neuron ablation / fault tolerance | 1 day | Graceful degradation comparison | Hardware reliability finding |

**HIGH-RISK HIGH-REWARD (attempt only if ahead of schedule):**

| Experiment | Time | What it produces | Why it helps |
|-----------|------|------------------|-------------|
| Full SpiNNaker deploy via IF_cond_exp | 2-3 days | FC1+FC2 on hardware | Game-changer if it works |
| Few-shot learning curves | 2-3 days | Data efficiency comparison | Tests central thesis narrative |

### Paper production (parallel with experiments)

| Task | Time | Owner |
|------|------|-------|
| Set up Overleaf with ACM template | 0.5 day | You |
| Convert ICONS2026_draft.md to LaTeX | 1-2 days | Claude |
| Create Figure 1: Architecture diagram | 1 day | Claude + you |
| Create Figure 2: SpiNNaker pipeline | 1 day | Claude + you |
| Create Figure 3: Encoding bar chart | 0.5 day | Claude |
| Rewrite abstract (150-200 words) | 0.5 day | Claude + you |
| Final polish + supervisor review | 2-3 days | You + supervisor |
| Submit on EasyChair | 0.5 day | You |

### Title (recommended)
> **Spiking Neural Networks for Environmental Sound Classification: From Seven Encodings to SpiNNaker Deployment**

---

## PHASE 2: THESIS EXTENSIONS (March 15 - code deadline)

These go in the thesis, not the ICONS paper. Run on CSF3 in parallel with paper production.

### Tier 1: Zero-risk, guaranteed novel results

| # | Experiment | Days | Scientific value |
|---|-----------|------|-----------------|
| 1 | Encoding transfer matrix (7×7) | 1 | HIGH — novel figure, encoding coupling |
| 2 | Temporal ablation (timestep truncation) | 0.5 | HIGH — deployment finding |
| 3 | Neuron ablation / fault tolerance | 1 | MED-HIGH — hardware reliability |
| 4 | Weight distribution analysis (SNN vs ANN) | 0.5 | MEDIUM — post-hoc, quick |
| 5 | Pruning resilience (30-90% sparsity) | 2 | MED-HIGH — compound sparsity |

### Tier 2: High value, moderate effort

| # | Experiment | Days | Scientific value |
|---|-----------|------|-----------------|
| 6 | Noise robustness profiling (SNR sweep) | 2-3 | HIGH — real-world relevance |
| 7 | Few-shot learning curves | 2-3 | HIGH — tests thesis narrative |
| 8 | Spike efficiency Pareto (L1 regularization) | 2-3 | HIGH — energy design space |
| 9 | SNN saliency maps (spike Grad-CAM) | 3-4 | HIGH — interpretability |
| 10 | Stochastic resonance | 1-2 | HIGH if positive — biology connection |

### Tier 3: SpiNNaker-specific

| # | Experiment | Days | Scientific value |
|---|-----------|------|-----------------|
| 11 | Full deploy via IF_cond_exp + MaxPool | 2-3 | VERY HIGH if it works |
| 12 | Spike drop robustness analysis | 1-2 | HIGH — explains hardware gap |
| 13 | WTA lateral inhibition on output | 1-2 | MEDIUM — hardware motif |
| 14 | SpiNNaker energy from provenance | 1-2 | HIGH — real measurements |
| 15 | SpiNNaker 2 readiness (NIR export) | 2-3 | MEDIUM — forward-looking |

### Tier 4: Ambitious (only if everything else done)

| # | Experiment | Days | Scientific value |
|---|-----------|------|-----------------|
| 16 | On-chip STDP learning for FC2 | 5-7 | VERY HIGH — headline result |
| 17 | LSM reservoir on SpiNNaker | 5-7 | VERY HIGH — novel architecture |
| 18 | Izhikevich resonator neurons | 3-5 | HIGH — filterbank on hardware |
| 19 | Cross-domain transfer (Speech Commands) | 3-4 | MED-HIGH |
| 20 | Real-time microphone demo | 2-3 | LOW (science), HIGH (demo) |

---

## PHASE 3: AFTER ICONS SUBMISSION

| Date | Event | Action |
|------|-------|--------|
| April 1 | ICONS submitted | Shift to thesis writing |
| May 18 | Reviews back | Prepare rebuttals (pre-drafted responses ready) |
| May 25 | Rebuttal due | Submit |
| June 5 | Decision | Celebrate or plan poster |
| ~July | DCASE 2026 deadline | Second paper opportunity (perfect topic match) |
| ~Sep | ICASSP 2027 deadline | Third paper opportunity |

---

## KEY PREPARED RESPONSES (for reviewers)

1. **"47% is low"** → Baseline datum. PANNs+SNN proves 92.5% when features are good. Gap identifies the bottleneck.
2. **"SNN uses more energy"** → Honest: yes in software. Path: reduce spike rate from 25.8% to <6.4%. On neuromorphic hardware, AC costs 5.1x less than MAC.
3. **"SpiNNaker 33% with high variance"** → First quantified hardware gap. 50-class task is 6.25x harder than prior work (8 pure tones). Root cause documented.
4. **"Only ESC-50"** → Standard benchmark with predefined 5-fold. UrbanSound8K as future work (or add 1-fold result).
5. **"PANNs isn't neuromorphic"** → Hybrid edge paradigm: CNN14 in cloud, SNN on edge. Precedent: Seekings et al. ICONS 2024.

---

## WHAT'S IN push/ (complete index)

| File | Content |
|------|---------|
| `BATTLE_PLAN.md` | This file — unified strategy |
| `icons_paper_strategy.md` | Detailed paper structure, titles, reviewer objections |
| `icons_acceptance_process.md` | 59% acceptance, rebuttal phase, timeline |
| `creative_extensions_brainstorm.md` | 18 experiment ideas ranked |
| `spinnaker_engineering_extensions.md` | 14 SpiNNaker-specific ideas |
| `neuromorphic_trends_2025_2026.md` | What's hot, how to position |
| `sota_snn_audio_march2026_deep.md` | Full SOTA survey, all novelty confirmed |
| `spinnaker_phd_theses_scope_comparison.md` | 13 Manchester PhDs, scope = Masters+ |
| `energy_neuromorphic_hardware_benchmarks.md` | Energy thresholds, Horowitz, commercial products |
| `conference_deadlines_2026_2027.md` | Open deadlines, strategic recommendations |
| (pre-existing files) | Previous research from earlier sessions |
