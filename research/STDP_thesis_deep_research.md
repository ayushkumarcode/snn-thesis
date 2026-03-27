# STDP (Spike-Timing-Dependent Plasticity) as a Thesis Focus: Deep Research Report

**Research Date:** 2026-02-25
**Scope:** Comprehensive investigation of STDP for unsupervised feature learning with biological plausibility -- feasibility, implementations, results, novel angles, and thesis framing.

---

## Executive Summary

STDP remains a vibrant and publishable research area in 2026, far from "old news." The field has experienced a significant resurgence driven by three converging forces: (1) the NeurIPS 2024 acceptance of the Neuronal Competition Groups (NCG) paper demonstrating that STDP-based local learning can achieve competitive results on CIFAR-10/100 when properly architected, (2) the growing demand for on-device, privacy-preserving learning that cannot use backpropagation due to its non-local nature, and (3) the emergence of neuromorphic hardware (Loihi 2, SpiNNaker2, memristive chips) that natively implements STDP in silicon. The biological plausibility narrative is compelling for a thesis: STDP is the dominant experimentally-observed synaptic learning rule in the brain, and framing a project as "bridging neuroscience and machine learning" gives strong narrative coherence.

The practical reality is nuanced. Pure STDP on MNIST achieves approximately 95% accuracy (Diehl and Cook, 2015), while the state-of-the-art hybrid approach (unsupervised STDP feature extraction + supervised STDP classifier with NCG) reaches 98.92% on MNIST, 88.72% on Fashion-MNIST, and 66.41% on CIFAR-10 using a STDP-trained convolutional feature extractor (NeurIPS 2024). These numbers are respectable but lag behind surrogate-gradient-trained SNNs by 5-15 percentage points on complex datasets. However, the thesis angle should not be "beat backpropagation" -- it should be "what can local, biologically plausible learning achieve, and where does it have fundamental advantages?"

For an undergraduate thesis, the hybrid approach (STDP unsupervised feature extraction + simple supervised classifier) is the sweet spot: it is implementable in one semester using BindsNET or SpykeTorch, produces visually interpretable learned features, and offers multiple dimensions for experimental investigation. The strongest novel angles for 2026 would be: (a) STDP on event-camera/DVS data where the temporal coding matches the learning rule naturally, (b) three-factor learning rules (reward-modulated STDP) for reinforcement learning tasks, or (c) STDP for continual/lifelong learning where its local nature provides natural resistance to catastrophic forgetting.

---

## 1. What Can STDP Actually Learn? What Tasks Is It Good At?

### 1.1 Core Mechanism

STDP is a biologically observed synaptic plasticity rule that adjusts synaptic weights based on the relative timing of pre- and post-synaptic spikes:
- **Pre fires before post (causal):** synapse is strengthened (Long-Term Potentiation, LTP)
- **Post fires before pre (anti-causal):** synapse is weakened (Long-Term Depression, LTD)

This creates an unsupervised, Hebbian-like learning rule that extracts temporal correlations in input spike patterns without any labels or global error signal.

### 1.2 What STDP Learns Well

| Task Domain | What STDP Extracts | Quality | Evidence |
|---|---|---|---|
| **Edge/Gabor-like filters** | Oriented edge detectors from natural images | Excellent | Masquelier & Thorpe (2007), Kheradpisheh et al. (2018) |
| **Digit prototypes** | Template-like representations of handwritten digits | Very Good | Diehl & Cook (2015) -- 95% MNIST |
| **Object parts/prototypes** | Intermediate visual features in deep CSNN | Good | Kheradpisheh et al. (2018) -- 99.1% Caltech face/motorbike |
| **Temporal patterns** | Repeating spike sequences, coincidence detection | Excellent | Foundational STDP property |
| **Audio/speech features** | Spectrotemporal patterns in audio | Good | 93.3% Spoken-MNIST (2024) |
| **Event-camera features** | Motion-sensitive filters from DVS data | Good | Paredes-Valles et al., cuSNN |
| **Spatial navigation** | Place/grid cell representations | Good | SpiNNaker implementations |

### 1.3 What STDP Struggles With

- **Fine-grained classification on complex datasets:** CIFAR-10 accuracy caps around 66% with pure STDP pipelines vs. 95%+ for surrogate gradient methods
- **Deep network training:** STDP has difficulty propagating useful learning signals through many layers
- **Precise categorical boundaries:** Without supervision, learned features cluster by visual similarity, not semantic category
- **Scalability to high-resolution images:** Computational cost grows significantly; convergence slows

### 1.4 Key Insight for the Thesis

STDP is fundamentally a feature extraction mechanism, not a classifier. Its strength is unsupervised representation learning -- discovering the statistical structure of input data. The classification step should be handled by a separate (potentially supervised) mechanism. This is directly analogous to how unsupervised pre-training (autoencoders, contrastive learning) works in deep learning, giving the thesis a clean conceptual framework.

---

## 2. Best Implementations Available

### 2.1 Framework Comparison Table

| Framework | Backend | STDP Support | GPU | Best For | Maturity | Active? |
|---|---|---|---|---|---|---|
| **BindsNET** | PyTorch | Extensive (pair, post-pre, MSTDP, MSTDPET) | Yes | ML-oriented STDP experiments | High | Moderate (last release ~2023) |
| **Brian2** | Code generation (C++/Cython) | Fully customizable (any equation) | No (CPU only) | Neuroscience-accurate simulations | Very High | Yes |
| **SpykeTorch** | PyTorch | STDP + R-STDP for convolutional SNNs | Yes | Deep convolutional STDP | Medium | Low (archived) |
| **ngc-learn** | JAX | Trace STDP, event STDP, R-STDP | Yes | Biologically plausible models | Medium | Yes (v3.0.1) |
| **SpikeNN** | CPU Python | S2-STDP, SSTDP, NCG architecture | No | NeurIPS 2024 NCG paper code | New | Yes |
| **Norse** | PyTorch | Limited (focus on surrogate gradients) | Yes | Modern deep SNN training | High | Yes |
| **SpikingJelly** | PyTorch/CuPy | Limited STDP (focus on surrogate gradients) | Yes | High-performance deep SNNs | Very High | Yes |
| **snnTorch** | PyTorch | Minimal STDP | Yes | Educational + surrogate gradients | High | Yes |
| **Lava** (Intel) | Custom | Three-factor learning, R-STDP | CPU | Loihi deployment | High | Yes |
| **Custom (from scratch)** | Python/NumPy | Whatever you build | No | Deep understanding | N/A | N/A |

### 2.2 Recommended Stack for This Thesis

**Primary recommendation: BindsNET**

Reasons:
- Built on PyTorch, so GPU acceleration works out of the box
- Ships with a near-replication of Diehl & Cook 2015 (`eth_mnist.py`) that achieves ~95% on MNIST
- Supports multiple STDP variants: standard pair-based, post-pre only, reward-modulated (MSTDP, MSTDPET)
- Well-documented with examples for unsupervised, supervised, and RL tasks
- The `DiehlAndCook2015` network class provides a ready-made baseline
- Running time: ~1 hour on Intel i7 CPU for full MNIST training; faster on GPU
- Repository: https://github.com/BindsNET/bindsnet
- Paper: Hazan et al., "BindsNET: A Machine Learning-Oriented Spiking Neural Networks Library in Python," Frontiers in Neuroinformatics, 2018

**Secondary recommendation: SpykeTorch (for convolutional STDP)**

If the thesis focuses on deep convolutional STDP feature extraction:
- Implements STDP and R-STDP for convolutional layers with at-most-one-spike-per-neuron constraint
- Comes with a reimplementation of Kheradpisheh et al. (2018) deep CSNN
- Repository: https://github.com/miladmozafari/SpykeTorch

**For the NCG/S2-STDP state-of-the-art results:**

SpikeNN is the official code from the NeurIPS 2024 paper:
- Repository: https://github.com/ggoupy/SpikeNN
- CPU-only, Python 3.8+
- Implements S2-STDP, SSTDP, and the NCG architecture

**For neuroscience accuracy: Brian2**
