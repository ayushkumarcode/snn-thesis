# SNN Energy Efficiency: How to Measure and Report It in a Thesis

> **Research Date:** 2026-02-25
> **Purpose:** Comprehensive guide for an undergraduate thesis on measuring, estimating, and reporting SNN energy efficiency -- without access to neuromorphic hardware.
> **Key finding:** Yes, an undergraduate can credibly include energy analysis. The methodology is well-established in published literature and does not require hardware.

---

## Executive Summary

Almost every SNN paper claims energy efficiency, but few measure it properly. This report documents exactly how the research community estimates SNN energy consumption, what tools exist, what formulas to use, and how to make fair comparisons with ANNs. The core approach used by the vast majority of published papers (CVPR, ICLR, NeurIPS, ECCV) is a **theoretical/analytical energy estimation** based on counting synaptic operations and multiplying by known energy-per-operation constants from the Horowitz 2014 ISSCC reference. This approach does **not** require neuromorphic hardware and is considered the standard methodology for SNN energy analysis in software-based research.

---

## 1. How Researchers Estimate SNN Energy Without Neuromorphic Hardware

There are three tiers of energy estimation, ordered from simplest to most complex:

### Tier 1: Operation-Count Based Estimation (Most Common -- Use This)

This is what 90%+ of published SNN papers use. The method:

1. **Count synaptic operations** during inference (how many times a spike causes a weight to be accumulated)
2. **Multiply by energy-per-operation constants** from known hardware characterisations
