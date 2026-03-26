# Deep Research: Making SNNs Dramatically More Energy-Efficient Than ANNs

**Date:** 25 March 2026
**Investigator:** Claude Research Agent
**Context:** SNN for ESC-50 audio classification, targeting 10x-100x energy advantage

---

## Executive Summary

Your SNN currently consumes 968 nJ/sample (1.08M ACs at 0.9 pJ) while the ANN uses 454 nJ (99K MACs at 4.6 pJ) -- the ANN is 2.1x MORE efficient in software simulation. The root cause is clear: your 26.4% spike rate is 4x above the 6.4% break-even threshold identified by Dampfhoffer et al. (2023) and confirmed by Yan et al. (2024). Furthermore, your T=25 timesteps multiply all operations by 25x compared to a single-pass ANN.

**The good news:** Your own Pareto experiments already show that L1 regularization at lambda=0.001 can reduce output spike rates to 0.4-0.8% while retaining 85-95% of baseline accuracy. Combined with temporal reduction (T=7 gives 90% accuracy), structured pruning, and SpiNNaker's actual hardware costs, achieving 10x energy advantage is realistic. A 100x advantage requires more aggressive techniques but is theoretically achievable.

**The three most impactful techniques for your setup are:**
1. **Spike rate regularization** (already partially validated in your Pareto experiments)
2. **Temporal reduction** (T=7 already validated, T=5 worth training for)
3. **Weight pruning + structured sparsity** (91x efficiency gains demonstrated at ICLR 2024)

---

## 1. Your Current Energy Budget: Where The Operations Come From

### Operation Breakdown (from NeuroBench 5-fold analysis)

| Component | SNN Operations | ANN Operations | Notes |
|-----------|---------------|---------------|-------|
| Conv1 (1->32, 3x3) | ~36K ACs/step * T | ~36K MACs | 9 * 32 * 32 * 108 = ~1M per step; sparsity helps |
| Conv2 (32->64, 3x3) | ~580K ACs/step * T | ~580K MACs | Dominates computation |
| FC1 (2304->256) | ~590K ACs/step * T | ~590K MACs | Large but sparse input |
| FC2 (256->50) | ~12.8K ACs/step * T | ~12.8K MACs | Small |
| **Total per sample** | **~1.08M ACs** | **~99K MACs** | SNN has 25x timestep multiplier |
| **Energy** | **968 nJ** | **454 nJ** | ANN wins by 2.1x |

### Key Insight: The Timestep Multiplier
Your ANN does ~99K MACs once. Your SNN does ~43K ACs per timestep * 25 timesteps = ~1.08M ACs total. The 73.6% sparsity helps (reduces per-step operations), but 25 timesteps overwhelms the AC-vs-MAC cost advantage (0.9 pJ vs 4.6 pJ = 5.1x).

**Break-even calculation:**
- ANN energy: 99K * 4.6 pJ = 455 nJ
- For SNN to match: 455 nJ / 0.9 pJ = 505K ACs allowed
- At T=25 with 99K dense ops per layer: need sparsity > 1 - (505K / (99K * 25)) = 79.6%
- Your current sparsity: 73.6% -- just below break-even
- At T=7: need sparsity > 1 - (505K / (99K * 7)) = 27.1% -- easily met

---

## 2. Spike Rate Reduction / Activity Regularization

### 2.1 L1 Spike Rate Regularization (YOUR DATA ALREADY SHOWS THIS WORKS)

**Your Pareto Results (5 folds):**

| Lambda | Avg Accuracy | Spike Rate | % of Baseline Acc | SpiNNaker Compatible |
|--------|-------------|------------|-------------------|---------------------|
| 0.0    | ~45%        | 2-5%       | 100%              | Yes |
| 1e-5   | ~46%        | 1.7-4.3%   | ~100%             | Yes |
