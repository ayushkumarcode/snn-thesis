# Top SNN/Neuromorphic Computing Papers 2024-2025: Comprehensive Research Report

**Date compiled:** 2026-02-25
**Scope:** Most impactful SNN papers from 2024-2025, trends, open-source code, conference papers, benchmarks, and low-hanging fruit research directions.

---

## 1. Executive Summary

The SNN field experienced a significant acceleration in 2024-2025, with three dominant trends: (1) **Spiking Transformers** achieving ImageNet accuracy above 85% for the first time, closing the gap with ANNs; (2) **SNNs scaling to language models**, with SpikeLLM and SpikeLM demonstrating that spiking architectures can handle 7-70B parameter LLMs; and (3) **new application domains** including graph reasoning, time-series forecasting, and continual learning becoming mature research areas with conference-level papers. The number of SNN papers at CVPR alone jumped from 3 (2024) to 14 (2025), indicating explosive growth in the field.

Key takeaway for an undergraduate thesis: The field is ripe with accessible research directions. Many top papers have open-source code, the frameworks (snnTorch, SpikingJelly) are mature and well-documented, and several "gap-filling" problems remain unaddressed.

---

## 2. Top 10-15 Most Cited/Influential SNN Papers (2024-2025)

### Tier 1: Highest Impact Papers

| # | Paper | Venue | Key Contribution | Code Available? |
|---|-------|-------|-----------------|-----------------|
| 1 | **QKFormer: Hierarchical Spiking Transformer using Q-K Attention** | NeurIPS 2024 (Spotlight, top 3%) | First SNN to exceed 85% top-1 accuracy on ImageNet-1k (85.65%). Novel spike-form Q-K attention with linear complexity. | Yes: [github.com/zhouchenlin2096/QKFormer](https://github.com/zhouchenlin2096/QKFormer) |
