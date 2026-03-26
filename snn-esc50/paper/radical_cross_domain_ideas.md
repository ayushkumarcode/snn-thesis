# Radical Cross-Domain Research: Novel Ideas for SNN Audio Classification

**Research Date:** 25 March 2026
**Context:** Convolutional SNN for ESC-50, 47.15% accuracy (vs ANN 63.85%), deployed on SpiNNaker
**Goal:** Identify ONE big novel finding that justifies a publication

---

## Executive Summary

After exhaustive cross-domain research spanning neuroscience, signal processing, physics, NLP, information theory, and hardware co-design, I identified **18 radical ideas** ranked by their potential to create a genuinely novel contribution. The top three candidates are:

1. **Cochlear-Inspired Learnable Auditory Frontend (Spiking-LEAF / IHC-LIF)** -- Replace mel spectrogram with a biologically-inspired cochlear model using two-compartment spiking neurons as the front-end. Published at ICASSP 2024 but NEVER applied to environmental sound classification.

2. **Dendritic Computation with Multi-Compartment Neurons** -- Replace LIF point neurons with dendritic spiking neurons that have nonlinear branches, enabling each neuron to perform richer temporal computations. Published in Nature Communications 2023/2024 but NEVER applied to audio SNNs.

3. **Spiking State Space Model (S6 / SpikingSSM)** -- Reconceptualize LIF dynamics as state space models with expanded hidden states, enabling parallel training and better long-range temporal processing. Published at AAAI 2025 / NeurIPS 2024 but NEVER applied to environmental sound.

Each of these represents a genuine gap in the literature where our work would be the FIRST to combine these ideas with ESC-50 environmental sound classification.

---

## TIER 1: Highest Impact, Most Novel (Publication-Worthy on Their Own)

---

### IDEA 1: Cochlear-Inspired Learnable Frontend (Spiking-LEAF / IHC-LIF Neurons)

