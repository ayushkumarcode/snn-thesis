# SNN vs ANN Energy Efficiency: Comprehensive Research Report

**Date:** 5 March 2026
**Scope:** Energy comparisons, NeuroBench benchmarks, neuromorphic hardware measurements, AC vs MAC costs, spike sparsity thresholds, edge deployments
**Research depth:** 40+ sources consulted across academic papers, hardware documentation, and industry reports

---

## Executive Summary

The energy efficiency narrative for SNNs is far more nuanced than commonly presented. The literature reveals that **SNNs are NOT automatically more energy-efficient than ANNs** on conventional digital hardware. The advantage depends critically on three factors: (1) spike sparsity rates, which must exceed ~92-93% for moderate time windows; (2) the hardware platform, with neuromorphic hardware being essential for realizing theoretical gains; and (3) whether memory access and data movement costs are included in the analysis. When these are properly accounted for, the bar for SNN energy superiority rises dramatically.

The key threshold from Dampfhoffer et al. (2023) and Yan et al. (2024) is that SNNs need spike rates below approximately 6-8% (i.e., sparsity above 92-94%) at time window T=6 to beat quantized ANNs. Most real-world SNN implementations on vision tasks report spike rates of 20-40%, well above this threshold. This makes the energy claim questionable for many practical deployments on digital hardware, though neuromorphic hardware changes the calculus significantly.

For your thesis, your measured 74.16% activation sparsity (NeuroBench) translates to a ~25.84% spike rate -- well above the 6-8% threshold needed for software-level energy superiority. However, on neuromorphic hardware with native AC operations, the per-operation cost advantage (0.9 pJ/AC vs 4.6 pJ/MAC) does provide a genuine 5.1x per-operation benefit. The honest framing for your thesis is: "SNNs achieve energy efficiency advantages on neuromorphic hardware through sparse AC operations, but require specialized hardware to realize these gains."

---

## 1. Energy Comparisons: SNN vs ANN
