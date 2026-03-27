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
3. **Compare the total energy** between SNN and ANN on the same task

**The fundamental insight:** In an ANN, every synapse performs a Multiply-Accumulate (MAC) operation during every forward pass. In an SNN, a synapse only performs an Accumulate (AC) operation when it receives a spike. Since spikes are sparse and binary, SNNs potentially use far fewer and cheaper operations.

**Core formulas used across the literature:**

```
E_ANN = FLOPs_ANN * E_MAC

E_SNN = SOP_SNN * E_AC + FLOPs_non_spiking * E_MAC

where:
  SOP (Synaptic Operations) = sum over all layers of:
    (spike_count_per_neuron * fan_out_connections) * T_timesteps

  E_MAC = 4.6 pJ   (32-bit float multiply-accumulate at 45nm)
  E_AC  = 0.9 pJ   (32-bit float accumulate/addition at 45nm)
```

These energy constants come from Horowitz's 2014 ISSCC keynote "Computing's Energy Problem (and what we can do about it)." This is the single most cited source for operation energy costs in the entire neural network efficiency literature.

**Source:** [Horowitz 2014, ISSCC](https://www.researchgate.net/publication/271463146_11_Computing's_energy_problem_and_what_we_can_do_about_it)

### Tier 2: Analytical Model with Memory Access Costs (More Rigorous)

Lemaire et al. (2022) proposed a more comprehensive model that accounts for three cost components:

1. **Synaptic operations** (AC for SNN, MAC for ANN)
