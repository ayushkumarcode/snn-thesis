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

**Source:** Song et al., "Spiking-LEAF: A Learnable Auditory front-end for Spiking Neural Networks," ICASSP 2024
**Domain:** Neuroscience + Signal Processing

**Core Idea:**
Replace our fixed mel spectrogram preprocessing with a learnable auditory front-end inspired by the biological cochlea. Spiking-LEAF uses:
- 40 learnable Gabor 1D-convolution filters (replacing fixed mel filterbank)
- Per-Channel Energy Normalization (PCEN) with learnable per-channel parameters
- IHC-LIF neurons: a TWO-COMPARTMENT spiking neuron (dendritic + somatic) inspired by inner hair cells, capturing multi-scale temporal dynamics

**How It Applies to ESC-50:**
Currently we compute mel spectrograms offline (fixed 64 mel bins, n_fft=1024, hop=512) and feed them to the SNN. Instead:
1. Feed RAW WAVEFORMS (22050 Hz, 5 seconds = 110,250 samples) directly to learnable Gabor filters
2. IHC-LIF neurons convert filtered signals to spike trains with biologically-plausible dynamics
3. The entire front-end is jointly trained with the SNN classifier end-to-end
4. The cochlear front-end LEARNS the optimal frequency decomposition for environmental sounds

**Expected Impact:**
- Spiking-LEAF improved keyword spotting accuracy by +9.21 percentage points over mel spectrograms (83.03% -> 92.24%)
- Cochleagram representations consistently outperform mel spectrograms by ~5% on sound event datasets
- For ESC-50: estimated 3-8 pp improvement (47.15% -> 50-55%) based on task difficulty
- The end-to-end learnable pipeline is more biologically plausible AND SpiNNaker-compatible

**Implementation Feasibility:** MEDIUM
- Spiking-LEAF code exists (ICASSP 2024 paper)
- Requires adapting from 16kHz speech to 22kHz environmental audio
- Gabor filters and PCEN are standard operations
- IHC-LIF neuron needs custom implementation but is essentially two coupled LIF equations
- Training is end-to-end with surrogate gradients (we already do this)

**Novelty Assessment:** VERY HIGH
- No one has applied Spiking-LEAF or IHC-LIF to environmental sound classification
- No one has applied ANY learnable cochlear front-end to ESC-50 with SNNs
- Combines two frontier areas: bio-inspired audio + spiking neural networks
- Direct comparison with our existing mel spectrogram results provides clean ablation

**SpiNNaker Angle:** The cochlear front-end could potentially run on SpiNNaker itself (cascade of LIF neurons mimicking basilar membrane), creating a FULLY neuromorphic audio pipeline: cochlea -> SNN classifier, all in spikes.

---

### IDEA 2: Dendritic Spiking Neurons (DendSN / Multi-Compartment LIF)

**Source:** Multiple recent papers:
- "Flexible and Scalable Deep Dendritic SNNs with Multiple Nonlinear Branching" (arXiv Dec 2024)
- "Temporal dendritic heterogeneity in SNNs for multi-timescale dynamics" (Nature Communications 2024)
- "Dendrites endow ANNs with accurate, robust, parameter-efficient learning" (Nature Communications 2025)
- "Spiking world model with multicompartment neurons" (PNAS 2025)

**Domain:** Neuroscience (Dendritic Computation)

**Core Idea:**
Standard LIF neurons are "point neurons" that linearly sum all inputs. Real neurons have DENDRITES -- tree-like branches that perform nonlinear local computations before signals reach the cell body (soma). Each dendrite is essentially a mini-processor. A single dendritic neuron can compute functions that require a multi-layer network of point neurons.

Replace our LIF neurons with DendSN (Dendritic Spiking Neurons):
- Each neuron has K dendritic branches (K=2-5 typically)
- Each branch has its own membrane dynamics (different time constants)
- Branches perform nonlinear gating on their inputs
