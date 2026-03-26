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
