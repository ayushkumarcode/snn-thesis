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

**Izhikevich Model:**
- balances biological plausibility with computational efficiency
- 2D system of ODEs with adaptation variable u
- can reproduce diverse cortical neuron firing patterns (regular spiking, bursting, chattering, fast spiking, etc.)
- more expressive than LIF but less demanding than HH

**Adaptive Exponential Integrate-and-Fire (AdEx):**
- exponential voltage dependence and slow adaptation variable w
- can reproduce diverse cortical firing patterns
- good balance between biological realism and computational tractability

### spike encoding schemes (Section 4)

**Rate Encoding:**
- information encoded as spike frequency over time windows
- uses point processes like Poisson distributions
- robust to noise but requires longer time windows
- higher energy consumption due to many spikes

**Temporal Encoding:**
- information represented by exact spike timing
- produces sparser activity than rate encoding
- sensitive to noise
- lower energy consumption
- input intensity (0-255) mapped to timing within 0-1 time window

### learning mechanisms (Section 5)

**Spike-Based Backpropagation Methods:**
- **SpikeProp**: uses van Rossum distance as loss function; early work on gradient-based SNN training
- **SuperSpike**: approximates spike derivatives using smooth temporal convolution
- **SLAYER**: distributes error credit backward in time; enables simultaneous learning of weights AND axonal delays (unique feature)

**Spike-Time-Dependent Plasticity (STDP):**
- classic STDP follows exponential temporal dependence
- if pre-synaptic spike arrives before post-synaptic: Long-Term Potentiation (LTP, strengthening)
- if pre-synaptic spike arrives after post-synaptic: Long-Term Depression (LTD, weakening)
- variants: anti-Hebbian (aSTDP), mirrored (mSTDP), probabilistic, reward-modulated (R-STDP)
- Stable STDP (S-STDP): combines weight-dependent exponential rules with spike traces for stability

**Other Learning Rules:**
- **Prescribed Error Sensitivity (PES)**: supervised online learning used in Nengo
- **Intrinsic Plasticity**: regulates neuron firing rates within optimal ranges
- **ANN-to-SNN Conversion**: transfers pre-trained parameters from artificial neural networks

### computer vision applications with specific results

**Image Classification:**
| Method | Year | Dataset | Accuracy | Notes |
|--------|------|---------|----------|-------|
| DCSNN | 2018 | MNIST | 97.2% | Conv SNN using STDP + R-STDP |
| LM-SNNs | 2020 | MNIST | Not specified | Lattice map with unsupervised learning |
| Medical SNN | 2020 | ISIC 2018 (melanoma) | 87.7% | 6,705 images, feature selection |
| EEG SNN | - | SEED dataset | 96.67% | Emotion recognition from EEG |

**Object Detection:**
| Method | Year | Dataset | Performance |
|--------|------|---------|-------------|
| Spiking YOLO | 2019 | PASCAL VOC | mAP 51.83% |
| Spiking YOLO | 2019 | MS COCO | mAP 25.66% |
| Deep SCNN | 2020 | KITTI | 56.24% mean sparsity, 0.247 mJ energy |

**Object Tracking:**
- SiamSNN (2020): first SNN for tracking, achieving 50 FPS on TrueNorth with low precision loss

**Optical Flow:**
- Spike-FlowNet (2020): hybrid SNN-ANN architecture for event camera data
- Hierarchical cuSNN (2019): uses stable STDP on Event Camera Dataset

**Segmentation:**
| Method | Year | Dataset | Accuracy | Notes |
|--------|------|---------|----------|-------|
| UNet-SNN | 2021 | ISBI 2D EM | 92.13% | lower than 94.98% ANN baseline but energy-efficient on Loihi |
| SpikeSEG | 2021 | Synthetic | 97% accuracy, 74% mIoU | semantic segmentation |

### robotics applications

**Pattern Generation:**
- NeuroPod (2019): first real-time neuromorphic CPG on SpiNNaker controlling hexapod locomotion
- Lamprey robot (2014): analog/digital VLSI with ~60 ms periodic bursting and 35 Hz spiking frequency

**Motor Control:**
- Loihi drone control (2020): root-mean-square error of 0.005 g in thrust setpoint with 99.8% spike sequence matching
- event-based PID controller improved Loihi performance by reducing saturation issues

**Navigation and SLAM:**
- Spiking RatSLAM (2012): place and grid cells on SpiNNaker for landmark detection
- Gridbot (2018): robot with 1,321 spiking neurons for autonomous environment mapping
- SLAM SNN (2019): 100x less energy than GMapping with comparable accuracy
- SDDPG (2020): spiking actor with deep critic network for energy-efficient mapless navigation

### software frameworks identified

| Framework | Training Methods | Focus Area |
|-----------|-----------------|------------|
| Brian2 | STDP | General-purpose simulator |
| NEST | STDP/R-STDP | Biological/medical applications |
| Nengo | STDP/PES | Large-scale neural models |
| NengoDL | ANN conversion | TensorFlow integration |
| SpykeTorch | STDP/R-STDP | PyTorch-based, rank-order encoding |
| BindsNet | STDP/R-STDP/conversion | Machine learning focus |
| SLAYER PyTorch | Backpropagation | Temporal credit assignment |
| Norse | BPTT | Sparse event-driven hardware |
| snn_toolbox | ANN conversion | Multi-framework compatibility |
| GeNN | General | NVIDIA GPU acceleration |
| CARLsim | STDP/STP | Multi-GPU/CPU large-scale simulation |

### key challenges
1. training complexity: non-differentiable spike operations cause gradient vanishing/explosion
2. large-scale performance: only ANN-to-SNN conversion + residual architectures match ANNs on ImageNet
3. computational overhead: many timesteps required, creating latency-accuracy tradeoffs
