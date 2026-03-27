# SNN Research Report for Thesis Project Selection

put this together on 2026-02-23 while trying to pick a thesis direction. read through three main survey papers, looked at all the frameworks, datasets, and tried to figure out what's actually achievable for a 3rd-year undergrad project.

---

## table of contents

1. [Paper 1: Yamazaki et al.](#paper-1)
2. [Paper 2: Han et al.](#paper-2)
3. [Paper 3: Malcolm & Casco-Rodriguez](#paper-3)
4. [SNN Frameworks Comparison](#frameworks)
5. [Neuromorphic Datasets Guide](#datasets)
6. [snnTorch Tutorials and Resources](#snntorch)
7. [ANN-to-SNN Conversion Tools](#conversion)
8. [Realistic Undergraduate Thesis Scopes](#thesis-scopes)
9. [Low-Barrier SNN Applications](#low-barrier)
10. [Example Theses and Projects](#examples)
11. [Synthesis and Recommendation](#recommendation)

---

<a name="paper-1"></a>
## Paper 1: "Spiking Neural Networks and Their Applications: A Review"

Kashu Yamazaki, Viet-Khoa Vo-Ho, Darshan Bulsara, Ngan Le. Brain Sciences (MDPI), July 2022, PMC9313413. covers biological foundations, neuron models, training mechanisms, and applications in computer vision and robotics.

### biological foundations (Sections 1-2)

starts with detailed biological neuron anatomy:
- **Dendrites**: input receivers from other neurons
- **Soma**: cell body that integrates incoming signals
- **Axon**: signal carrier transmitting action potentials
- **Synapses**: connections between neurons (chemical via neurotransmitters, electrical via gap junctions)

key biological constants:
- resting membrane potential: approximately -70.15 mV
- action potential peak voltage: approximately 38.43 mV
- Goldman-Hodgkin-Katz equation governs ion channel behavior
- permeability ratios at rest: K:Na:Cl = 1:0.04:0.45

### spiking neuron models (Section 3)

**Hodgkin-Huxley (HH) Model:**
- highest biological accuracy of all models
- computationally intensive (differential equations for K+ and Na+ channels)
- uses gating variables (n, m, h) for ion channel dynamics
- rarely used in machine learning because it's too expensive

**Leaky Integrate-and-Fire (LIF):**
- most widely used model in SNN research
- includes "leak" term accounting for ion diffusion through the membrane
- firing rate formula: f = [tau_ref + tau_m * ln(RmI / (RmI - v_theta))]^{-1}
- threshold v_theta = 1 (normalized), reset to 0 after firing
- typical refractory periods: tau_ref <= 5 ms
- computationally simple, suitable for large-scale networks
