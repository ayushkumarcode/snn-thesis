# STDP (Spike-Timing-Dependent Plasticity) as a Thesis Focus: Deep Research Report

**Research Date:** 2026-02-25
**Scope:** Comprehensive investigation of STDP for unsupervised feature learning with biological plausibility -- feasibility, implementations, results, novel angles, and thesis framing.

---

## Executive Summary

STDP remains a vibrant and publishable research area in 2026, far from "old news." The field has experienced a significant resurgence driven by three converging forces: (1) the NeurIPS 2024 acceptance of the Neuronal Competition Groups (NCG) paper demonstrating that STDP-based local learning can achieve competitive results on CIFAR-10/100 when properly architected, (2) the growing demand for on-device, privacy-preserving learning that cannot use backpropagation due to its non-local nature, and (3) the emergence of neuromorphic hardware (Loihi 2, SpiNNaker2, memristive chips) that natively implements STDP in silicon. The biological plausibility narrative is compelling for a thesis: STDP is the dominant experimentally-observed synaptic learning rule in the brain, and framing a project as "bridging neuroscience and machine learning" gives strong narrative coherence.

The practical reality is nuanced. Pure STDP on MNIST achieves approximately 95% accuracy (Diehl and Cook, 2015), while the state-of-the-art hybrid approach (unsupervised STDP feature extraction + supervised STDP classifier with NCG) reaches 98.92% on MNIST, 88.72% on Fashion-MNIST, and 66.41% on CIFAR-10 using a STDP-trained convolutional feature extractor (NeurIPS 2024). These numbers are respectable but lag behind surrogate-gradient-trained SNNs by 5-15 percentage points on complex datasets. However, the thesis angle should not be "beat backpropagation" -- it should be "what can local, biologically plausible learning achieve, and where does it have fundamental advantages?"

For an undergraduate thesis, the hybrid approach (STDP unsupervised feature extraction + simple supervised classifier) is the sweet spot: it is implementable in one semester using BindsNET or SpykeTorch, produces visually interpretable learned features, and offers multiple dimensions for experimental investigation. The strongest novel angles for 2026 would be: (a) STDP on event-camera/DVS data where the temporal coding matches the learning rule naturally, (b) three-factor learning rules (reward-modulated STDP) for reinforcement learning tasks, or (c) STDP for continual/lifelong learning where its local nature provides natural resistance to catastrophic forgetting.
