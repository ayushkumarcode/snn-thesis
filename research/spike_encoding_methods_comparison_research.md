# Spike Encoding Methods: Systematic Comparison as a Thesis Topic

**Research Date:** 2026-02-25
**Scope:** Comprehensive investigation of spike encoding methods for SNNs, assessment of existing comparison studies, and viability as an undergraduate thesis topic

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Complete Taxonomy of Spike Encoding Methods](#2-complete-taxonomy-of-spike-encoding-methods)
3. [Impact of Encoding Choice on SNN Performance](#3-impact-of-encoding-choice-on-snn-performance)
4. [Existing Systematic Comparison Studies](#4-existing-systematic-comparison-studies)
5. [Which Encoding Works Best for Which Data Type](#5-which-encoding-works-best-for-which-data-type)
6. [Implementation in snnTorch](#6-implementation-in-snntorch)
7. [Thesis Viability Assessment](#7-thesis-viability-assessment)
8. [Research Gaps and Novel Contribution Opportunities](#8-research-gaps-and-novel-contribution-opportunities)
9. [Proposed Thesis Structure](#9-proposed-thesis-structure)
10. [Key Papers Reference Table](#10-key-papers-reference-table)
11. [Sources](#11-sources)

---

## 1. Executive Summary

Spike encoding -- the process of converting real-valued data into spike trains for processing by spiking neural networks -- is a fundamental and still actively researched problem in neuromorphic computing. There are at least 6-8 major encoding families (rate, latency/TTFS, delta/temporal contrast, phase, burst, population/Gaussian receptive field, direct/learned, and binary), each with distinct trade-offs in accuracy, latency, energy efficiency, noise robustness, and hardware suitability.

**The critical finding from this research: several comparison studies already exist, but none are truly comprehensive.** Each existing study compares a subset of encodings on a narrow set of tasks (usually just MNIST/Fashion-MNIST, or just one sensor modality). No single study has systematically compared all major encoding methods across multiple data modalities (images, audio, time-series, event-driven) using a unified framework and consistent evaluation metrics. This gap represents a genuine and achievable undergraduate thesis contribution.

The encoding choice demonstrably matters -- accuracy differences of 3-5% between methods on the same task are common, while latency and energy consumption can differ by 4-7.5x. This is not a trivial question with a known answer; it is a live research area where a well-executed systematic study would be valued.

**Verdict: "Systematic Evaluation of Spike Encoding Methods for Spiking Neural Networks" is a strong, feasible undergraduate thesis topic** with clear novelty potential if scoped correctly (more data modalities, more encoding methods, unified framework, consistent metrics).

---

## 2. Complete Taxonomy of Spike Encoding Methods

Based on the comprehensive survey by Auge, Hille, Mueller, and Knoll (2021) in Neural Processing Letters, and supplemented by multiple other sources, here is the complete taxonomy.

### 2.1 Rate-Based Encoding

Information is embedded in the firing frequency of neurons. Robust against noise, simple to implement, but requires many timesteps and many spikes (energy-expensive).

| Method | Description | Key Property |
|--------|-------------|--------------|
| **Poisson Rate Coding** | Each input value is treated as the probability of a spike at each timestep (Bernoulli process). Higher values = more spikes on average. | Most common baseline; stochastic; high spike count |
| **Regular Rate Coding** | Deterministic variant where spikes are evenly spaced with frequency proportional to input value. | Lower variance than Poisson; easier to analyse |
| **Population Rate Coding** | A group of neurons collectively encodes a value through their combined firing rate. | Higher information capacity; uses more neurons |

### 2.2 Temporal/Latency-Based Encoding

Information is in the precise timing of spikes. A single spike carries much more meaning than in rate codes. Much fewer spikes needed, but more susceptible to noise.

| Method | Description | Key Property |
|--------|-------------|--------------|
| **Time-to-First-Spike (TTFS)** | Each neuron fires exactly once. Stronger inputs fire earlier, weaker inputs fire later. Based on LIF neuron RC model. | Very low spike count; fast inference; ~4x lower latency than rate coding |
| **Rank-Order Coding** | Only the relative ordering of spike times matters, not absolute times. | Robust to time distortions; loses amplitude info |
| **Inter-Spike Interval (ISI)** | Information encoded in the time gap between consecutive spikes from the same neuron. | Compact encoding; good for periodic signals |

### 2.3 Delta Modulation / Temporal Contrast

