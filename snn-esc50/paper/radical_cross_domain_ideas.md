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
- Soma integrates branch outputs
- Different branches can attend to different temporal scales

**How It Applies to ESC-50:**
Environmental sounds have MULTI-SCALE temporal structure:
- Fine-grained: pitch, harmonics (milliseconds)
- Medium: syllable-like events, repetitions (100ms-1s)
- Coarse: overall texture, onset/offset (seconds)

Current LIF neurons process all timescales with a single beta=0.95. Dendritic neurons with heterogeneous time constants (e.g., branch 1: beta=0.99 for slow, branch 2: beta=0.80 for fast) can simultaneously capture multiple temporal scales. This is exactly what environmental sound needs.

Implementation: Replace `snn.Leaky(beta=0.95)` with `DendriticLeaky(branches=3, betas=[0.99, 0.95, 0.80])`.

**Expected Impact:**
- DendSNNs consistently outperform point SNNs on FashionMNIST, CIFAR-10, SHD
- Multi-compartment neurons in PNAS 2025 surpass other SNN architectures on speech datasets
- Improved robustness to noise and adversarial attacks
- Better few-shot learning (relevant for ESC-50's small training set of 1600 samples)
- Estimated 3-6 pp improvement for ESC-50

**Implementation Feasibility:** HIGH
- Only requires modifying the neuron model, not the training pipeline
- DendSN code is available (arXiv Dec 2024) with Triton GPU kernels
- Computational overhead is negligible (paper demonstrates this)
- 2-5 branches per neuron, learnable time constants and gating parameters
- Compatible with surrogate gradient training

**Novelty Assessment:** VERY HIGH
- No one has applied dendritic spiking neurons to audio classification
- No one has applied them to environmental sound
- Multi-timescale temporal processing is a natural fit for audio
- Story: "Biologically-inspired dendritic computation enables multi-scale temporal processing in audio SNNs"

**SpiNNaker Angle:** Each dendritic branch maps to a separate neuron on SpiNNaker connected to the soma neuron. The multi-core architecture handles this naturally. DenRAM (Nature Communications 2024) already demonstrates neuromorphic dendritic architectures.

---

### IDEA 3: Spiking State Space Model (SpikingSSM / S6)

**Source:** Multiple recent papers:
- "P-SpikeSSM: Harnessing Probabilistic Spiking SSMs" (arXiv Jun 2024)
- "SpikingSSMs: Learning Long Sequences with Sparse and Parallel Spiking SSMs" (AAAI 2025)
- "SPikE-SSM: Sparse, Precise, Efficient Spiking SSM" (arXiv Oct 2024)

**Domain:** NLP/Sequence Modeling (S4/Mamba lineage) adapted for spikes

**Core Idea:**
State space models (S4, Mamba) have revolutionized sequence modeling. The key insight: LIF neurons are ALREADY a state space model! The membrane potential is a 1D hidden state with linear dynamics + nonlinear output (spike). But LIF uses only a SCALAR hidden state. SSMs use an N-DIMENSIONAL hidden state vector.

SpikingSSM expands the LIF neuron to have a rich multi-dimensional state, enabling:
- Parallel training (convert recurrence to convolution)
- Better long-range dependency modeling
- 90% network sparsity (still spike-based, energy efficient)

**How It Applies to ESC-50:**
Our spectrograms are 216 time frames long, and we simulate 25 timesteps. Environmental sounds like "train" or "helicopter" have long temporal structure that benefits from better sequence modeling. SpikingSSM would replace our LIF layers with richer state dynamics that can capture these patterns.

**Expected Impact:**
- S6 achieves 95.6% on Speech Commands (raw 16000-length signals)
- SpikingSSM achieves competitive performance with 90% sparsity
- psMNIST: 98.4% (state-of-the-art for spiking models)
- For ESC-50: potentially large improvement due to better temporal modeling

**Implementation Feasibility:** MEDIUM-LOW
- Requires significant architecture changes
- Less compatible with SpiNNaker (the parallel training trick doesn't transfer to hardware)
- More of a research prototype than a deployable system
- But the INFERENCE can still be sequential and spike-based

**Novelty Assessment:** VERY HIGH
- No one has applied spiking SSMs to environmental sound classification
- Combines two hot research areas (SSMs + SNNs)
- Would be among the first audio applications of spiking SSMs

**SpiNNaker Angle:** WEAK. The parallel training is the main benefit and doesn't help on SpiNNaker. Inference is sequential but the expanded state requires more memory per neuron.

---

## TIER 2: High Impact, Strong Novelty

---

### IDEA 4: Oscillatory Modulation of Spiking Neurons (Rhythm-SNN)

