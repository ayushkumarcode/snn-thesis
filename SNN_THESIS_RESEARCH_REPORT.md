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

**Spike-Time-Dependent Plasticity (STDP):**
- Classic STDP follows exponential temporal dependence
- If pre-synaptic spike arrives before post-synaptic: Long-Term Potentiation (LTP, strengthening)
- If pre-synaptic spike arrives after post-synaptic: Long-Term Depression (LTD, weakening)
- Variants: anti-Hebbian (aSTDP), mirrored (mSTDP), probabilistic, reward-modulated (R-STDP)
- Stable STDP (S-STDP): Combines weight-dependent exponential rules with spike traces for stability

**Other Learning Rules:**
- **Prescribed Error Sensitivity (PES)**: Supervised online learning used in Nengo
- **Intrinsic Plasticity**: Regulates neuron firing rates within optimal ranges
- **ANN-to-SNN Conversion**: Transfers pre-trained parameters from artificial neural networks

### 1.5 Computer Vision Applications with Specific Results

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
- SiamSNN (2020): First SNN for tracking, achieving 50 FPS on TrueNorth with low precision loss

**Optical Flow:**
- Spike-FlowNet (2020): Hybrid SNN-ANN architecture for event camera data
- Hierarchical cuSNN (2019): Uses stable STDP on Event Camera Dataset

**Segmentation:**
| Method | Year | Dataset | Accuracy | Notes |
|--------|------|---------|----------|-------|
| UNet-SNN | 2021 | ISBI 2D EM | 92.13% | Lower than 94.98% ANN baseline but energy-efficient on Loihi |
| SpikeSEG | 2021 | Synthetic | 97% accuracy, 74% mIoU | Semantic segmentation |

### 1.6 Robotics Applications

**Pattern Generation:**
- NeuroPod (2019): First real-time neuromorphic central pattern generator (CPG) on SpiNNaker controlling hexapod locomotion
- Lamprey robot (2014): Analog/digital VLSI with ~60 ms periodic bursting and 35 Hz spiking frequency

**Motor Control:**
- Loihi drone control (2020): Root-mean-square error of 0.005 g in thrust setpoint with 99.8% spike sequence matching
- Event-based PID controller improved Loihi performance by reducing saturation issues

**Navigation and SLAM:**
- Spiking RatSLAM (2012): Place and grid cells on SpiNNaker for landmark detection
- Gridbot (2018): Robot with 1,321 spiking neurons for autonomous environment mapping
- SLAM SNN (2019): 100x less energy than GMapping with comparable accuracy
- SDDPG (2020): Spiking actor with deep critic network for energy-efficient mapless navigation

### 1.7 Software Frameworks Identified

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

### 1.8 Key Challenges Identified
1. **Training complexity**: Non-differentiable spike operations cause gradient vanishing/explosion
2. **Large-scale performance**: Only ANN-to-SNN conversion + residual architectures match ANNs on ImageNet
3. **Computational overhead**: Many timesteps required, creating latency-accuracy tradeoffs
4. **Architecture design**: Limited theoretical guidance; need for neural architecture search (NAS)

### 1.9 Future Directions
- Direct SNN training using online gradient algorithms (RTRL) to move beyond ANN conversion
- Architectural innovations through meta-learning and NAS for SNN-specific designs
- Extension to large-scale datasets using residual connections

### 1.10 Key Insight for Thesis
This paper is particularly useful for understanding the FOUNDATIONS. It provides the clearest explanation of neuron models, encoding schemes, and learning rules. The robotics applications section is uniquely detailed compared to the other two papers.

---

<a name="paper-2"></a>
## PAPER 2: "Toward Large-scale Spiking Neural Networks: A Comprehensive Survey and Future Directions"

**Authors:** Cheng Han, et al.
**Published:** arXiv:2409.02111, September 2024
**Scope:** Methods for developing deep SNNs, with emphasis on Spiking Transformers as pathways toward energy-efficient large-scale models.

### 2.1 Learning Rules for Deep SNNs

#### ANN-to-SNN Conversion
**Fundamental Principle:** The ReLU activation function is functionally equivalent to the integrate-and-fire neuron through rate-coding approximation over time steps.

**Key Techniques:**
- Weight normalization and threshold balancing (addressing over/under-activation)
- Reset-by-subtraction replacing reset-by-zero
- Layer-wise calibration and potential initialization
- Quantization-aware conversion methods

**CIFAR-10 Conversion Results:**
| Method | Year | Accuracy | Time Steps |
|--------|------|----------|------------|
| clip-floor-shift | 2022 | 95.54% | 32 |
| Fast-SNN | 2023 | 95.42% | 3 |
| Parameter Calibration | 2024 | 94.75% | 4 |

**ImageNet Conversion Results:**
| Method | Year | Accuracy | Time Steps |
|--------|------|----------|------------|
| Spiking ResNet | 2021 | 73.77% | 350 |
| Fast-SNN | 2023 | 71.31% | 3 |
| clip-floor-shift | 2022 | 68.47% | 32 |

Critical observation: Time steps have dropped from 350 to just 3, representing massive latency reduction.

#### Direct Training with Surrogate Gradients
**Framework:** Backpropagation-through-time (BPTT) using surrogate gradient functions to approximate non-differentiable spike functions.

**Key Innovations:**
- Learnable surrogate gradients (LSG) adapting function width dynamically
- Information Maximization Loss (IM-Loss) optimizing surrogate shape
- Parametric LIF neurons with learnable time constants
- Membrane Potential Batch Normalization (MPBN)
- Temporal Efficient Training (TET) compensating momentum loss

**CIFAR-10 Direct Training Results:**
| Method | Year | Accuracy | Time Steps |
|--------|------|----------|------------|
| MPBN | 2023 | 96.47% | 2 |
| RecDis-SNN | 2023 | 95.55% | 6 |
| IM-Loss | 2022 | 95.49% | 6 |

**ImageNet Direct Training Results:**
| Method | Year | Accuracy | Time Steps |
|--------|------|----------|------------|
| IM-Loss | 2022 | 70.65% | 5 |
| Attention SNN | 2023 | 69.15% | 1 |
| GLIF | 2022 | 67.52% | 4 |

**DVS CIFAR-10 Results:**
| Method | Year | Accuracy | Time Steps |
|--------|------|----------|------------|
| STSC-SNN | 2022 | 81.40% | 10 |
| IM-LIF | 2024 | 80.50% | 10 |
| TET | 2022 | 77.33% | 6 |

### 2.2 Network Architectures

#### Deep Convolutional SNNs
- **SEW-ResNet**: Spike-Element-Wise with activation-before-addition; 69.26% on ImageNet (60.19M params, 5 time steps)
- **MS-ResNet**: Membrane-Shortcut preserving full-precision potentials; 74.21% on ImageNet (78.37M params, 5 time steps)
- **NAS Approaches**: AutoSNN, SNASNet, AutoST for automated architecture discovery

#### Spiking Transformer Architectures (Major Focus)

**Evolution of Spiking Self-Attention:**
1. Vanilla Self-Attention: Hybrid ANN-SNN approaches, limited event-driven benefits
2. Spikformer (2022): Replaced softmax with spike-form matrix operations
3. Spike-Driven Self-Attention (SDSA): Q-K attention with spike-driven computation
4. Dual Spike Self-Attention

**Spatio-Temporal Enhancements:**
- Spatial-Temporal Self-Attention (STSA) with relative position bias
- Temporal Interaction Module (TIM) via 1D convolution
- Frequency-Aware Token Mixer (FATM) in Spiking Wavelet Transformer

**ImageNet Spiking Transformer Leaderboard (State-of-the-Art):**
| Model | Year | Accuracy | Parameters | Time Steps | Method |
|-------|------|----------|------------|------------|--------|
| ECMT | 2024 | 88.60% | 1,074M | 4 | Conversion |
| QKFormer | 2024 | 84.22% | 64.96M | 4 | Direct |
| SpikeZIP-TF | 2024 | 83.82% | 304.33M | 64 | Conversion |
| SGLFormer | 2024 | 83.73% | 64.02M | 4 | Direct |
| Spikformer V2 | 2024 | 80.38% | 51.55M | 4 | Direct |
| Spike-driven Trans. V2 | 2024 | 79.7% | 55.4M | 4 | Direct |
| Spikformer | 2022 | 74.81% | - | 4 | Direct |

### 2.3 NLP Applications of SNNs

**Models Developed:**
- **SpikingGPT**: RWKV-based language generation
- **SpikeBERT / Spike-BERT**: BERT-adapted variants with knowledge distillation
- **SpikingMiniLM**: Lightweight BERT-based architecture
- **SpikeLLM**: 70 billion parameters via spike-driven quantization (largest SNN to date)

SpikingMiniLM tested on GLUE benchmark with accuracy, F1, and correlation metrics.

### 2.4 Beyond Image Classification

**Vision Tasks:**
- Object detection (Spike-driven Transformer V2)
- Semantic segmentation (Spike-driven Transformer V2)
- Zero-shot classification (SpikeCLIP)
- Image generation (SDiT -- Spiking Diffusion Transformer)
- Video understanding (TIM)

**Other Domains:**
- Audio-visual classification (Spiking Multi-Modal Transformer)
- Remote photoplethysmography (Spiking-PhysFormer)
- EEG seizure detection (Spiking Conformer)
- Human pose tracking (Spiking Spatiotemporal Transformer)

### 2.5 All Datasets Referenced
- CIFAR-10 / CIFAR-100 (most explored for SNNs)
- ImageNet-1k (1.2M training, 50K validation, 1K classes, 224x224)
- DVS CIFAR-10 (event-stream version of CIFAR-10)
- DVS128 Gesture (11 hand gestures, 29 subjects, 3 lighting conditions)
- N-Caltech101, N-CARS (event camera recordings)
- HAR-DVS, PokerEvents (event-based action/game recognition)
- MMHPSD, SynEventHPD, DHP19 (human pose from events)
- ImageNet-200 zero-shot variants
- GLUE benchmark (NLP)

### 2.6 Challenges and Limitations

**Training Challenges:**
- Information loss due to spike reset, gradient vanishing in deep layers
- Surrogate gradient mismatch vs. true gradient distributions
- Binary signal constraints: discrete spikes limit information vs. continuous ANNs
- Temporal complexity: recurrent nature requires BPTT across many timesteps

**Architectural Constraints:**
- Spiking attention removes softmax (non-linear), reducing expressiveness
- Real-valued shortcuts and max-pooling conflict with event-driven principles
- Batch normalization adaptation across time dimensions adds overhead

**Scalability Issues:**
- SNNs typically employ millions of parameters vs. billions in ANNs
- Knowledge distillation and conversion require pre-trained ANN teachers
- Energy benefits diminish with longer required inference latencies

### 2.7 Future Research Directions

**Learning Rules:**
- Leveraging temporal gradient information in recurrent structure
- Non-backpropagation approaches: equilibrium propagation, Forward-Forward algorithm
- Advanced biologically-plausible local plasticity

**Large-Scale Models:**
- Novel architectures for billion-parameter SNNs
- Hardware co-design with neuromorphic platforms
- Multi-modal integration (images, video, audio, text, sensors)
- Alternative attention mechanisms preserving spike properties
