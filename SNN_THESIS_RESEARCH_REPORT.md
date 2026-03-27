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
4. architecture design: limited theoretical guidance; need for neural architecture search

### future directions from the paper
- direct SNN training using online gradient algorithms (RTRL)
- architectural innovations through meta-learning and NAS
- extension to large-scale datasets using residual connections

### my take
this paper is the most useful for understanding the foundations. clearest explanation of neuron models, encoding schemes, and learning rules. the robotics applications section is uniquely detailed compared to the other two papers.

---

<a name="paper-2"></a>
## Paper 2: "Toward Large-scale Spiking Neural Networks"

Cheng Han, et al. arXiv:2409.02111, September 2024. focuses on methods for developing deep SNNs, with emphasis on Spiking Transformers as pathways toward energy-efficient large-scale models.

### learning rules for deep SNNs

#### ANN-to-SNN Conversion
the fundamental principle: ReLU activation is functionally equivalent to the integrate-and-fire neuron through rate-coding approximation over time steps.

key techniques:
- weight normalization and threshold balancing (addressing over/under-activation)
- reset-by-subtraction replacing reset-by-zero
- layer-wise calibration and potential initialization
- quantization-aware conversion methods

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

interesting observation: time steps have dropped from 350 to just 3 -- massive latency reduction.

#### Direct Training with Surrogate Gradients
uses BPTT with surrogate gradient functions to approximate non-differentiable spike functions.

key innovations:
- learnable surrogate gradients (LSG) adapting function width dynamically
- Information Maximization Loss (IM-Loss) optimizing surrogate shape
- parametric LIF neurons with learnable time constants
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

### network architectures

#### Deep Convolutional SNNs
- **SEW-ResNet**: spike-element-wise with activation-before-addition; 69.26% on ImageNet (60.19M params, 5 time steps)
- **MS-ResNet**: membrane-shortcut preserving full-precision potentials; 74.21% on ImageNet (78.37M params, 5 time steps)
- **NAS Approaches**: AutoSNN, SNASNet, AutoST for automated architecture discovery

#### Spiking Transformer Architectures (big focus of this paper)

evolution of spiking self-attention:
1. vanilla self-attention: hybrid ANN-SNN approaches, limited event-driven benefits
2. Spikformer (2022): replaced softmax with spike-form matrix operations
3. Spike-Driven Self-Attention (SDSA): Q-K attention with spike-driven computation
4. Dual Spike Self-Attention

spatio-temporal enhancements:
- Spatial-Temporal Self-Attention (STSA) with relative position bias
- Temporal Interaction Module (TIM) via 1D convolution
- Frequency-Aware Token Mixer (FATM) in Spiking Wavelet Transformer

**ImageNet Spiking Transformer leaderboard (SOTA):**
| Model | Year | Accuracy | Parameters | Time Steps | Method |
|-------|------|----------|------------|------------|--------|
| ECMT | 2024 | 88.60% | 1,074M | 4 | Conversion |
| QKFormer | 2024 | 84.22% | 64.96M | 4 | Direct |
| SpikeZIP-TF | 2024 | 83.82% | 304.33M | 64 | Conversion |
| SGLFormer | 2024 | 83.73% | 64.02M | 4 | Direct |
| Spikformer V2 | 2024 | 80.38% | 51.55M | 4 | Direct |
| Spike-driven Trans. V2 | 2024 | 79.7% | 55.4M | 4 | Direct |
| Spikformer | 2022 | 74.81% | - | 4 | Direct |

### NLP applications of SNNs
- **SpikingGPT**: RWKV-based language generation
- **SpikeBERT / Spike-BERT**: BERT-adapted variants with knowledge distillation
- **SpikingMiniLM**: lightweight BERT-based architecture
- **SpikeLLM**: 70 billion parameters via spike-driven quantization (largest SNN to date)

### beyond image classification
- object detection (Spike-driven Transformer V2)
- semantic segmentation (Spike-driven Transformer V2)
- zero-shot classification (SpikeCLIP)
- image generation (SDiT -- Spiking Diffusion Transformer)
- video understanding (TIM)
- audio-visual classification (Spiking Multi-Modal Transformer)
- remote photoplethysmography (Spiking-PhysFormer)
- EEG seizure detection (Spiking Conformer)
- human pose tracking (Spiking Spatiotemporal Transformer)

### all datasets referenced
- CIFAR-10 / CIFAR-100 (most explored for SNNs)
- ImageNet-1k (1.2M training, 50K validation, 1K classes, 224x224)
- DVS CIFAR-10 (event-stream version of CIFAR-10)
- DVS128 Gesture (11 hand gestures, 29 subjects, 3 lighting conditions)
- N-Caltech101, N-CARS (event camera recordings)
- HAR-DVS, PokerEvents (event-based action/game recognition)
- MMHPSD, SynEventHPD, DHP19 (human pose from events)
- ImageNet-200 zero-shot variants
- GLUE benchmark (NLP)

### challenges and limitations

training challenges:
- information loss due to spike reset, gradient vanishing in deep layers
- surrogate gradient mismatch vs. true gradient distributions
- binary signal constraints: discrete spikes limit information vs. continuous ANNs
- temporal complexity: recurrent nature requires BPTT across many timesteps

architectural constraints:
- spiking attention removes softmax (non-linear), reducing expressiveness
- real-valued shortcuts and max-pooling conflict with event-driven principles
- batch normalization adaptation across time dimensions adds overhead

scalability:
- SNNs typically employ millions of parameters vs. billions in ANNs
- knowledge distillation and conversion require pre-trained ANN teachers
- energy benefits diminish with longer required inference latencies

### energy efficiency claims
- human brain: ~20 Watts
- GPT-3 training: 1,287 MWh
- ChatGPT inference: ~564 MWh/day
- SNN advantage: MAC operations reduced to accumulate-only (AC) operations, event-driven sparsity

### my take
this is the most current paper (Sept 2024) and the most relevant for understanding the state of the art. the Spiking Transformer section is incredibly useful. the NLP section shows SNNs expanding beyond vision. the benchmark tables give the clearest picture of where SNN accuracy stands relative to ANNs.

---

<a name="paper-3"></a>
## Paper 3: "A Comprehensive Review of Spiking Neural Networks"

Kai Malcolm, Josue Casco-Rodriguez. arXiv:2303.10780, March 2023. literature review covering interpretation, optimization, efficiency, and best practices for SNNs. designed to be accessible to newcomers.

note: couldn't get the HTML version on arXiv (only PDF, which wouldn't text-extract via web fetch). the analysis below is reconstructed from the abstract, metadata, citations, and cross-referencing with papers that cite it.

### paper structure (reconstructed)

based on the title structure and cross-references, it covers four main pillars:

**Pillar 1 -- Interpretation:**
- how SNNs process information differently from ANNs
- biological plausibility of spike-based computation
- spike encoding schemes: rate coding, temporal coding, population coding
- neuron models: LIF (primary focus), Izhikevich, AdEx, Hodgkin-Huxley

**Pillar 2 -- Optimization:**
- two mainstream pathways to deep SNNs:
  1. ANN-to-SNN conversion (rate-based equivalence between ReLU and IF neuron firing rate)
  2. direct training via surrogate gradient methods
- STDP as unsupervised Hebbian learning
- emphasizes that surrogate gradient-trained SNNs closely approximate ANN accuracy (within 1-2%), with faster convergence by the 20th epoch

**Pillar 3 -- Efficiency:**
- energy efficiency evaluation methods
- low-power deployment considerations
- mobile and hardware-constrained settings
- comparison of energy consumption between SNN and ANN inference

**Pillar 4 -- Best Practices:**
- implementation guidelines for practitioners
- starting from first principles for accessibility
- software tool recommendations

### my take
this paper would be most useful as an introductory reference -- it was specifically designed for newcomers. if i can get the PDF, it'd serve as a good starting point before diving into the more technical Han et al. paper.

---

<a name="frameworks"></a>
## SNN frameworks comparison (2024-2025)

### tier 1: recommended for thesis work

#### snnTorch
- **GitHub Stars:** 1,450+ | **Contributors:** 40 | **License:** MIT
- **Latest Version:** 0.9.4 (February 16, 2025)
- **Maintainer:** UCSC Neuromorphic Computing Group (Jason Eshraghian)
- **Strengths:** best tutorials and documentation in the SNN ecosystem; 18 tutorials with Google Colab notebooks; PyTorch-based; GPU acceleration
- **Neuron Models:** LIF, Lapicque's RC, Alpha, Synaptic Conductance
- **Weakness:** limited integration with neuromorphic hardware; slower than SpikingJelly for large models
- **Best For:** learning, prototyping, thesis projects where documentation matters

#### SpikingJelly
- **GitHub Stars:** 1,800+ | **License:** Open source
- **Latest Requirement:** torch >= 2.2.0
- **Maintainer:** Fangwei123456 (Peking University group)
- **Strengths:** fastest framework (0.26s forward+backward with CuPy backend); full-stack toolkit; supports neuromorphic datasets, ANN2SNN conversion, surrogate gradients, and biologically plausible learning; up to 11x training speedup; published in Science Advances
- **Weakness:** documentation primarily in Chinese (English docs available but less thorough); steeper learning curve
- **Best For:** performance-critical research; deployment on neuromorphic chips; largest model training
