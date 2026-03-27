# Neuromorphic Computing & SNN Research Trends: 2025-2026

## Deep Research Intelligence Report
**Date:** 15 March 2026
**Purpose:** Position SNN-ESC50 thesis project for maximum relevance to the neuromorphic community
**Target:** ICONS 2026 (deadline: April 1, 2026)

---

## EXECUTIVE SUMMARY

The neuromorphic computing field is at a **commercial inflection point** in 2025-2026. Three macro-trends dominate: (1) the AI energy crisis is driving unprecedented interest in neuromorphic solutions, (2) SNN papers at top-tier ML conferences have exploded (29 at ICLR 2026, 23 at NeurIPS 2025, 11 at ICML 2025), and (3) hardware is finally shipping commercially (Innatera Pulsar, BrainChip AKD1500, SpiNNaker2 at Sandia).

**For our project specifically:** The SNN-ESC50 work is positioned at a remarkable sweet spot. There is ZERO prior SNN work on full ESC-50 (confirmed), and the Larroza et al. March 2025 paper on spike encoding for environmental sound only covers ESC-10. The field is hungry for: (a) actual hardware deployment results (not just simulation), (b) honest energy analysis with real numbers, (c) encoding comparisons beyond vision, and (d) transfer learning bridges between ANNs and SNNs. Our project delivers ALL of these. The adversarial robustness finding (SNN 26% vs ANN 1.75% at eps=0.1) directly aligns with a November 2025 Nature Communications paper showing SNNs achieve 2x ANN robustness. The PANNs+SNN transfer learning finding (gap collapses from 17pp to 1pp) addresses the hottest question in the field: is the SNN-ANN gap a fundamental limitation or a feature-learning problem?

---

## 1. WHAT DOMINATED ICONS 2024-2025

### ICONS 2025 (Seattle, July 29 - Aug 1, 2025)

**Best Paper Award:** "A Comparison of Custom and Standard Neuron Model Random Walks on the Ornstein-Uhlenbeck Equation for Simplified Turbulence" (Taylor et al.) -- an unconventional application demonstrating neuromorphic computing beyond classification.

**Key themes from the accepted papers (31 presentations):**

| Theme | Paper Count | Examples |
|-------|------------|---------|
| Hardware deployment & deployment pipelines | 5 | SpiNNaker2 fine-tuning, Loihi 2 deployment pipeline, code gen for embedded |
| Robotics & control | 4 | Motor neuron models, robotic perception, drone tracking |
| Privacy & security | 3 | Privacy in SNNs, model inversion attacks, cybersecurity |
| Energy-efficient edge AI | 3 | Vibration sensing, predictive maintenance, tactile sensing |
| Continual/online learning | 3 | Unsupervised continual learning, lifelong learning, GRASP |
| Novel applications | 4 | Turbulence modeling, artificial pancreas, chemical sensors, air traffic control |
| Architectures & training | 5 | Spiking transformers, oscillator Ising machines, dendrites, reservoir computing |
| Simulation frameworks | 2 | VRISP simulator, SNN uncertainty estimation |
| Vision/event-driven | 2 | Event-based action recognition, high-speed perception |

**Critical observation:** ICONS 2025 strongly favored papers with **real hardware deployment** (SpiNNaker2, Loihi 2, embedded systems) and **novel application domains** (turbulence, medical, industrial). Pure algorithm papers without hardware or application context were less prominent.

**Audio/sound papers at ICONS 2025:** ZERO. There were no audio or sound classification papers. This is a wide-open gap.

### ICONS 2024 (Arlington, VA, July 30 - Aug 2, 2024)
- Focus on RF fingerprinting, cerebellar neuron detection
- Superconducting optoelectronic neuromorphic computing

**Takeaway for our paper:** An SNN audio paper with actual SpiNNaker deployment would be genuinely novel at ICONS. The conference has never seen this combination.

---

## 2. HOTTEST TOPICS AT TOP ML CONFERENCES (2024-2026)

### SNN Paper Counts at Major Conferences

| Conference | Year | SNN Papers | Trend |
|-----------|------|-----------|-------|
| ICLR | 2026 | **29** | Strongest showing ever |
| NeurIPS | 2025 | **23** | Significant presence |
| ICML | 2025 | **11** | Growing |
| CVPR | 2025 | **5** | Vision-focused |
| ICLR | 2025 | 11 | Steady |
| NeurIPS | 2024 | ~15 | Baseline |
