# Comprehensive SNN Research Report for Thesis Project Selection

**Prepared: 2026-02-23**
**Purpose: Thesis project direction selection for a third-year undergraduate in neuromorphic computing**

---

## TABLE OF CONTENTS

1. [Paper 1: Yamazaki et al. -- Detailed Analysis](#paper-1)
2. [Paper 2: Han et al. -- Detailed Analysis](#paper-2)
3. [Paper 3: Malcolm & Casco-Rodriguez -- Detailed Analysis](#paper-3)
4. [SNN Frameworks Comparison (2024-2025)](#frameworks)
5. [Neuromorphic Datasets Guide](#datasets)
6. [snnTorch Tutorials and Resources](#snntorch)
7. [ANN-to-SNN Conversion Tools Assessment](#conversion)
8. [Realistic Undergraduate Thesis Scopes](#thesis-scopes)
9. [Low-Barrier SNN Applications](#low-barrier)
10. [Example Theses and Projects](#examples)
11. [Synthesis and Project Recommendation](#recommendation)

---

<a name="paper-1"></a>
## PAPER 1: "Spiking Neural Networks and Their Applications: A Review"

**Authors:** Kashu Yamazaki, Viet-Khoa Vo-Ho, Darshan Bulsara, Ngan Le
**Published:** Brain Sciences (MDPI), July 2022, PMC9313413
**Scope:** Comprehensive review covering biological foundations, neuron models, training mechanisms, and applications in computer vision and robotics.

### 1.1 Biological Foundations (Sections 1-2)

The paper begins with detailed biological neuron anatomy:
- **Dendrites**: Input receivers from other neurons
- **Soma**: Cell body that integrates incoming signals
- **Axon**: Signal carrier transmitting action potentials
- **Synapses**: Connections between neurons (chemical via neurotransmitters, electrical via gap junctions)

Key biological constants cited:
- Resting membrane potential: approximately -70.15 mV
- Action potential peak voltage: approximately 38.43 mV
- Goldman-Hodgkin-Katz equation governs ion channel behavior
- Permeability ratios at rest: K:Na:Cl = 1:0.04:0.45

### 1.2 Spiking Neuron Models (Section 3)

**Hodgkin-Huxley (HH) Model:**
- Highest biological accuracy among all models
- Computationally intensive (differential equations for K+ and Na+ channels)
- Uses gating variables (n, m, h) for ion channel dynamics
- Rarely used in machine learning due to computational cost

**Leaky Integrate-and-Fire (LIF):**
- Most widely used model in SNN research
- Includes "leak" term accounting for ion diffusion through the membrane
- Firing rate formula: f = [tau_ref + tau_m * ln(RmI / (RmI - v_theta))]^{-1}
- Threshold v_theta = 1 (normalized), reset to 0 after firing
- Typical refractory periods: tau_ref <= 5 ms
- Computationally simple; suitable for large-scale networks

**Izhikevich Model:**
- Balances biological plausibility with computational efficiency
- Uses 2D system of ordinary differential equations with adaptation variable u
- Can reproduce diverse cortical neuron firing patterns (regular spiking, bursting, chattering, fast spiking, etc.)
- More expressive than LIF but less computationally demanding than HH

**Adaptive Exponential Integrate-and-Fire (AdEx):**
- Features exponential voltage dependence and slow adaptation variable w
- Can reproduce diverse cortical neuron firing patterns
- Good balance between biological realism and computational tractability

### 1.3 Spike Encoding Schemes (Section 4)

**Rate Encoding:**
- Information encoded as spike frequency over time windows
- Uses point processes like Poisson distributions
- Robust to noise but requires longer time windows
- Higher energy consumption due to many spikes

**Temporal Encoding:**
- Information represented by exact spike timing
- Produces sparser activity than rate encoding
- Sensitive to noise
- Lower energy consumption
- Input intensity (0-255) mapped to timing within 0-1 time window

### 1.4 Learning Mechanisms (Section 5)

**Spike-Based Backpropagation Methods:**
- **SpikeProp**: Uses van Rossum distance as loss function; early work on gradient-based SNN training
- **SuperSpike**: Approximates spike derivatives using smooth temporal convolution
- **SLAYER**: Distributes error credit backward in time; enables simultaneous learning of weights AND axonal delays (unique feature)

