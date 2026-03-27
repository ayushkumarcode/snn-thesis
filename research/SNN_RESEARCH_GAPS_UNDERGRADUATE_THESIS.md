# SNN Research Gaps: Achievable Undergraduate Thesis Opportunities

**Research Date:** 2026-02-25
**Purpose:** Identify the lowest-effort paths to a genuine novel contribution in SNN research for a 3rd-year undergraduate thesis.
**Methodology:** Exhaustive web search across arXiv, IEEE Xplore, PMC, Nature, Springer, conference proceedings, GitHub, and community resources (Open Neuromorphic, snnTorch docs, Tonic library).

---

## Executive Summary

The SNN field is in a peculiar state: it is mature enough that good tools and datasets exist, but immature enough that enormous gaps remain in basic empirical coverage. Most SNN papers focus on image classification (MNIST, CIFAR-10, ImageNet) with surrogate gradient training. Entire application domains, datasets, and framework comparisons remain untouched or have only 1-2 papers. This creates a rich landscape for undergraduate contributions that are technically novel without requiring PhD-level ambition.

The single lowest-effort strategy for a genuine contribution is: **take an existing SNN architecture/method and apply it to a dataset or domain where nobody has tried it yet.** The second lowest-effort strategy is: **run the same experiment across multiple frameworks and report the differences.** Both of these are essentially "engineering" contributions -- running experiments and reporting results -- rather than "invention" contributions, but they are genuinely valuable to the community and count as novel work.

---

## Table of Contents

1. [Application Domains Where SNNs Have Not Been Tried](#1-untried-domains)
2. [Datasets Not Yet Benchmarked with SNNs](#2-unbenchmarked-datasets)
3. [Missing Framework/Method Comparison Studies](#3-missing-comparisons)
4. [Future Work Sections from Recent SNN Papers](#4-future-work-leads)
5. [Single-Paper Domains (Easy Second Data Point)](#5-single-paper-domains)
6. [Cross-Domain Application Opportunities](#6-cross-domain)
7. [Ranked Thesis Project Ideas by Effort/Novelty Ratio](#7-ranked-ideas)
8. [Sources](#8-sources)

---

<a name="1-untried-domains"></a>
## 1. Application Domains Where SNNs Have Not Been Tried (or Barely Tried)

### 1.1 Completely Untouched or Near-Untouched

| Domain | Status | Why SNNs Could Work | Effort |
|--------|--------|-------------------|--------|
| **Plant disease detection from leaf images** | Zero SNN papers found. Entire agricultural CV field uses CNNs/transformers. | Standard image classification; direct transfer of existing SNN architectures. | LOW |
| **Wildlife camera trap classification** | No SNN papers found. | Sparse, event-like data (animals appear briefly). SNNs could exploit temporal sparsity. | LOW-MEDIUM |
| **Satellite/remote sensing land cover** | One paper (SNN4Space, ESA) on EuroSAT and UC Merced. No follow-ups. | Standard image classification with large datasets. Energy efficiency argument strong for satellite edge computing. | LOW |
| **Document/OCR classification** | No SNN papers found beyond MNIST digits. | Character recognition is a natural extension of digit recognition. | LOW |
| **Food recognition/calorie estimation** | No SNN papers found. | Standard image classification. Food-101, Food-2K datasets available. | LOW |
| **Weather/climate prediction from sensor data** | No SNN papers found. | Time-series data naturally maps to temporal spike encoding. | MEDIUM |
| **Music genre classification** | One undergraduate thesis (mrahtz, 2016) on musical pattern recognition. No genre classification. | Audio temporal patterns are a natural fit for SNNs. | LOW-MEDIUM |
| **Sports action recognition** | No SNN papers on standard sports datasets (UCF-101, HMDB-51). | Temporal dynamics of actions suit SNNs. | MEDIUM |

