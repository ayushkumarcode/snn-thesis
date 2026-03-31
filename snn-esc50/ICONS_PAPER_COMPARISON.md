# ICONS Paper Comparison — Ours vs Top 5 Tier 1

## Papers compared
1. Seekings et al. 2024 — Hybrid SNN on Loihi+Jetson (L3, ICONS 2024)
2. Arfa et al. 2025 — SNN on SpiNNaker2 for DVS gestures (L3, ICONS 2025)
3. An et al. 2025 — SNN on DYNAP-SE for cognitive load (L3, ICONS 2025)
4. Hajizada et al. 2022 — Continual learning on Loihi (Best Paper, ICONS 2022)
5. Yarga et al. 2022 — Spike encoding for speech (ICONS 2022)

## Comparison matrix

| Dimension | Seekings | Arfa | An | Hajizada | Yarga | **Ours** |
|-----------|----------|------|----|----------|-------|---------|
| Hardware deployment | Loihi+Jetson | SpiNNaker2 | DYNAP-SE | Loihi | None | SpiNNaker 1 |
| Full or partial | Full | Full | Full | Full | N/A | **Partial (FC2)** |
| Dataset complexity | DVS Gesture | DVS Gesture | EEG (3 class) | Objects | FSDD (10 digits) | **ESC-50 (50 class)** |
| # hardware inferences | ~1000 | ~500 | ~400 | ~100 | 0 | **22,000** |
| Energy comparison | Yes (Loihi vs GPU) | Yes (32× reduction) | Yes (7.1pp gap) | No | No | **Yes (3.5× reduction)** |
| Novel finding | Hybrid paradigm | HW-aware fine-tune | Real-time ATC | CL on Loihi | Encoding matters | **Pruning regularizes HW** |
| References | ~25 | ~30 | ~35 | ~20 | ~25 | **28** |
| Pages | 8 | 6 | 8 | 8 | 8 | **6** |
| Statistical tests | Some | Minimal | Yes | Yes | Some | **Yes (p-values, 5-fold)** |

## What they do that we DON'T
- Full on-chip deployment (all 4 L3 papers)
- Measured hardware power (Seekings)
- On-chip learning (Hajizada)
- Real-time operation (An)

## What WE do that they don't
- **22,000 hardware inferences** — 10-50× more than any of them
- **Systematic pruning sweep** — no other paper does this at scale
- **Pruning-as-regularization finding** — genuinely novel for HW deployment
- **50-class task** — harder than DVS gestures (11 class) or digits (10 class)
- **Lottery ticket hypothesis connection** — first to connect pruning for neuromorphic to Frankle & Carlin

## Assessment
Our paper's main weakness (partial deployment) is offset by the unprecedented scale of hardware evaluation and the novel pruning finding. The 50-class complexity exceeds all comparison papers. We are competitive with ICONS L3 papers despite using older hardware (SpiNNaker 1 vs Loihi 2, SpiNNaker 2, DYNAP-SE).
