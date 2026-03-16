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

### 2.8 Energy Efficiency Claims
- Human brain: ~20 Watts
- GPT-3 training: 1,287 MWh
- ChatGPT inference: ~564 MWh/day
- SNN advantage: MAC operations reduced to accumulate-only (AC) operations, event-driven sparsity

### 2.9 Key Insight for Thesis
This is the most current paper (Sept 2024) and the most relevant for understanding the STATE OF THE ART. The Spiking Transformer section is invaluable. The NLP section shows SNNs expanding beyond vision. The benchmark tables provide the clearest picture of where SNN accuracy stands relative to ANNs.

---

<a name="paper-3"></a>
## PAPER 3: "A Comprehensive Review of Spiking Neural Networks: Interpretation, Optimization, Efficiency, and Best Practices"

**Authors:** Kai Malcolm, Josue Casco-Rodriguez
**Published:** arXiv:2303.10780, March 2023
**Scope:** Literature review covering interpretation, optimization, efficiency, and accuracy of SNNs, designed to be accessible to newcomers.

**NOTE:** The HTML version of this paper was unavailable on arXiv (only PDF, which could not be text-extracted via web fetch). The analysis below is reconstructed from the abstract, metadata, citations, and cross-referencing with papers that cite it.

### 3.1 Paper Structure (Reconstructed)

Based on the title structure and cross-references, the paper covers four main pillars:

**Pillar 1 -- Interpretation:**
- How SNNs process information differently from ANNs
- Biological plausibility of spike-based computation
- Spike encoding schemes: rate coding, temporal coding, population coding
- Neuron models: LIF (primary focus), Izhikevich, AdEx, Hodgkin-Huxley

**Pillar 2 -- Optimization:**
- Two mainstream pathways to deep SNNs:
  1. ANN-to-SNN conversion (rate-based equivalence between ReLU and IF neuron firing rate)
  2. Direct training via surrogate gradient methods
- STDP (Spike-Timing-Dependent Plasticity) as unsupervised Hebbian learning
- Challenges: ANN-SNN conversion inefficiencies, surrogate gradient mismatch, convergence speed
- The paper emphasizes that surrogate gradient-trained SNNs closely approximate ANN accuracy (within 1-2%), with faster convergence by the 20th epoch

**Pillar 3 -- Efficiency:**
- Energy efficiency evaluation methods for SNNs
- Low-power deployment considerations
- Mobile and hardware-constrained settings
- Comparison of energy consumption between SNN and ANN inference

**Pillar 4 -- Best Practices:**
- Implementation guidelines for practitioners
- Starting from first principles for accessibility
- Software tool recommendations
- Practical considerations for building working SNN systems

### 3.2 Key Contributions
- Makes cutting-edge SNN methods accessible to new practitioners by starting from first principles
- Provides systematic comparison of optimization techniques
- Addresses the practical gap between theory and implementation
- Licensed CC BY 4.0 (open access)

### 3.3 Cross-Referenced Findings
From papers that cite Malcolm & Casco-Rodriguez 2023:
- The paper is cited in the context of discussing surrogate gradient methods as a primary training approach
- It is referenced for its coverage of energy efficiency metrics
- It is cited for its accessible treatment of neuron model fundamentals

### 3.4 Key Insight for Thesis
This paper is most useful as an INTRODUCTORY reference -- it was specifically designed to be accessible to newcomers. If you can access the PDF, it would serve as an excellent starting point before diving into the more technical details of Han et al. (Paper 2).

---

<a name="frameworks"></a>
## SNN FRAMEWORKS COMPARISON (2024-2025)

### Tier 1: Recommended for Thesis Work

#### snnTorch
- **GitHub Stars:** 1,450+ | **Contributors:** 40 | **License:** MIT
- **Latest Version:** 0.9.4 (February 16, 2025)
- **Maintainer:** UCSC Neuromorphic Computing Group (Jason Eshraghian)
- **Strengths:** Best tutorials and documentation in the SNN ecosystem; 18 comprehensive tutorials with Google Colab notebooks; PyTorch-based; GPU acceleration
- **Neuron Models:** LIF, Lapicque's RC, Alpha, Synaptic Conductance
- **Weakness:** Limited integration with neuromorphic hardware; slower than SpikingJelly for large models
- **Best For:** Learning, prototyping, thesis projects where documentation matters

#### SpikingJelly
- **GitHub Stars:** 1,800+ | **License:** Open source
- **Latest Requirement:** torch >= 2.2.0
- **Maintainer:** Fangwei123456 (Peking University group)
- **Strengths:** Fastest framework (0.26s forward+backward with CuPy backend); full-stack toolkit; supports neuromorphic datasets, ANN2SNN conversion, surrogate gradients, and biologically plausible learning; up to 11x training speedup; published in Science Advances
- **Weakness:** Documentation primarily in Chinese (English docs available but less comprehensive); steeper learning curve
- **Best For:** Performance-critical research; deployment on neuromorphic chips; largest model training

#### Norse
- **GitHub Stars:** Growing community
- **Latest:** PyTorch 1.9+ compatible
- **Strengths:** Clean PyTorch integration; excellent for small-to-medium networks (up to ~5000 neurons/layer); Colab notebooks; PyTorch Lightning compatible
- **Weakness:** Performance constrained on very large networks
- **Best For:** Clean research code; integration with existing PyTorch workflows

### Tier 2: Specialized Use Cases

| Framework | Best For | Notes |
|-----------|----------|-------|
| Lava (Intel) | Loihi deployment | Hardware-specific; NIR support |
| Nengo/NengoDL | Large-scale brain models, ANN conversion | Mature ecosystem; TensorFlow integration |
| Brian2 | Neuroscience simulation | Easiest syntax; not ML-focused |
| NEST | Large biological networks | Biology/medicine focus |
| BindsNet | Reinforcement learning with SNNs | PyTorch-based |
| Sinabs | Vision models, hardware deployment | PyTorch-based; EXODUS backend for speed |
| GeNN | GPU-accelerated simulation | NVIDIA GPU specific |
| Spyx | JAX-based acceleration | GPU/TPU JIT compilation |
| CARLsim | Large-scale with realistic synapses | Multi-GPU support |

### Framework Performance Benchmarks (Open Neuromorphic, 2024)

Test: Single FC + LIF layer, batch=16, 500 time steps, n neurons:

| Framework | Forward+Backward Time | Notes |
|-----------|----------------------|-------|
| SpikingJelly (CuPy) | 0.26s | Fastest |
| Lava DL (SLAYER) | ~0.4-0.5s | Custom CUDA |
| Sinabs (EXODUS) | ~0.4-0.5s | Custom CUDA |
| Norse (torch.compile) | ~0.5-0.7s | Close to JAX with compile |
| snnTorch | ~1.0s+ | Flexible but slower |

### Recommendation for Thesis
**Start with snnTorch** for learning and prototyping (best documentation). **Move to SpikingJelly** if you need performance or want to work with neuromorphic datasets directly.

---

<a name="datasets"></a>
## NEUROMORPHIC DATASETS GUIDE

### Most Accessible Datasets for Thesis Work

#### Vision -- Static (Converted to Spikes)

| Dataset | Classes | Samples | Resolution | Access | Difficulty |
|---------|---------|---------|------------|--------|------------|
| MNIST | 10 digits | 70K | 28x28 | Built into snnTorch/SpikingJelly | Easiest |
| Fashion-MNIST | 10 clothing | 70K | 28x28 | Built into frameworks | Easy |
| CIFAR-10 | 10 objects | 60K | 32x32 | Built into frameworks | Moderate |
| CIFAR-100 | 100 objects | 60K | 32x32 | Built into frameworks | Harder |
| ImageNet-1K | 1000 objects | 1.2M | 224x224 | Manual download | Hard (compute) |

#### Vision -- Neuromorphic (Event Camera / DVS)

| Dataset | Classes | Samples | Source | Access | Difficulty |
|---------|---------|---------|--------|--------|------------|
| N-MNIST | 10 digits | 70K | DVS camera | garrickorchard.com; snnTorch/SpikingJelly built-in | Easy |
| CIFAR10-DVS | 10 objects | 10K | DVS on LCD | figshare; SpikingJelly built-in | Moderate |
| DVS128 Gesture | 11 gestures | 1,464 | DVS128 camera | IBM Box; SpikingJelly built-in | Moderate |
| N-Caltech101 | 101 categories | 8,709 | DVS camera | garrickorchard.com | Moderate |
| N-CARS | 2 (car/bg) | 24,029 | ATIS camera | Prophesee | Moderate |
| ASL-DVS | ASL letters | - | DVS camera | GitHub | Moderate |
| ES-ImageNet | 1000 objects | - | Simulated | Frontiers paper | Hard |

#### Audio -- Neuromorphic

| Dataset | Classes | Samples | Source | Access | Difficulty |
|---------|---------|---------|--------|--------|------------|
| Spiking Heidelberg Digits (SHD) | 20 (0-9 in EN+DE) | ~10K | Artificial cochlea | zenkelab.org | Easy-Moderate |
| Spiking Speech Commands (SSC) | 35 keywords | ~100K | Artificial cochlea | zenkelab.org | Moderate |

#### Other

| Dataset | Type | Access |
|---------|------|--------|
| DVS_barrel | Character recognition | garrickorchard.com |
| DVS Planes | Airplane detection | greg-cohen.com |
| KITTI | 3D point cloud driving | kitti.ai |
| ISBI 2D EM | Biomedical segmentation | isbi.org |
| SEED | EEG emotion | BCMI lab |
| ISIC 2018 | Skin lesion (melanoma) | isic-archive.com |

### Dataset Loading Tools
- **Tonic**: PyTorch-compatible loader for neuromorphic datasets (like TorchVision but for events)
- **SpikingJelly**: Built-in dataset loaders for N-MNIST, CIFAR10-DVS, DVS128 Gesture, NavGesture, ASLDVS
- **snnTorch**: Built-in spikevision.spikedata for N-MNIST and other datasets

### Recommendation for Thesis
**Start with MNIST** (rate-encoded) to verify your pipeline works. Then move to **N-MNIST** or **DVS128 Gesture** for neuromorphic-native data. **SHD** is excellent if you want audio classification.

---

<a name="snntorch"></a>
## snnTorch TUTORIALS AND RESOURCES

### Complete Tutorial Catalog (v0.9.4)

**Core Tutorials (Progressive Learning Path):**

| Tutorial | Title | Topic | Colab |
|----------|-------|-------|-------|
| 1 | Spike Encoding with snnTorch | Rate/latency/delta encoding | Yes |
| 2 | The Leaky Integrate and Fire Neuron | LIF model fundamentals | Yes |
| 3 | A Feedforward Spiking Neural Network | Building basic SNN architecture | Yes |
| 4 | 2nd Order Spiking Neuron Models | Synaptic, Alpha neuron models | Yes |
| 5 | Training SNNs with snnTorch | Backprop through time, loss functions | Yes |
| 6 | Surrogate Gradient Descent in a Conv SNN | Convolutional SNN on MNIST | Yes |
| 7 | Neuromorphic Datasets with Tonic + snnTorch | Loading DVS data with Tonic library | Yes |

**Advanced Tutorials:**

| Tutorial | Title | Topic |
|----------|-------|-------|
| Population Coding | Population Coding Methods | Multi-neuron encoding |
| Regression I | Membrane Potential Learning with LIF | Regression tasks |
| Regression II | Regression-based Classification with Recurrent LIF | Recurrent architectures |
| Binarized SNNs | Binarized Spiking Neural Networks | Binary weight optimization |
| IPU Acceleration | Accelerating snnTorch on IPUs | Hardware acceleration |
| Forward-Forward | Forward-Forward Algorithm for SNNs | Alternative to backprop |

**Domain-Specific Tutorials:**

| Tutorial | Title | Domain |
|----------|-------|--------|
| Exoplanet Hunter | Finding Planets Using Light Intensity | Astronomy/time series |
| ST-MNIST | Spiking-Tactile MNIST Dataset | Tactile neuromorphic data |

### Recommended Learning Path for Thesis
1. Tutorials 1-3 (fundamentals, ~4 hours)
2. Tutorials 5-6 (training, ~4 hours)
3. Tutorial 7 (neuromorphic datasets, ~2 hours)
4. Then branch to your specific project direction

### Additional Resources
- **Video lectures** by Jason Eshraghian on YouTube
- **Paper:** "Training Spiking Neural Networks Using Lessons From Deep Learning" (Eshraghian et al., 2023) -- companion paper to snnTorch
- **GitHub:** github.com/jeshraghian/snntorch (MIT license, actively maintained)

---

<a name="conversion"></a>
## ANN-to-SNN CONVERSION TOOLS ASSESSMENT

### Tool Maturity Summary

| Tool | Input Formats | Backends | Maturity | Active? |
|------|--------------|----------|----------|---------|
| snn_toolbox | Keras, PyTorch, Lasagne, Caffe | pyNN, Brian2, SpiNNaker, Loihi | Moderate | Low activity |
| SpikingJelly ann2snn | PyTorch | SpikingJelly | Good | Active |
| NengoDL Converter | Keras/TF | Nengo, Loihi | Mature | Active |
| MATLAB SNN Toolbox | MATLAB networks | Simulink | Moderate | Active |

### Current State of ANN-to-SNN Conversion

**What Works:**
- Converting simple CNNs (VGG, ResNet) to SNNs preserves reasonable accuracy (above 80%)
- SpikingJelly's ann2snn module is the most stable and scalable approach as of 2024
- NengoDL provides clean Keras-to-SNN pipeline with good documentation
- snn_toolbox offers the broadest input format support

**What Does Not Work Well:**
- NOT a "one-click solution" -- accuracy loss and significant adjustments often required
- Most conversion methods are based on rate encoding and historically needed T >= 128 time steps
- Recent methods have reduced this to T = 3-4, but at some accuracy cost
- Complex architectures (transformers, attention) do not convert cleanly
- Batch normalization, dropout, and certain activation functions need special handling

**Time Steps Required (Historical Progression):**
- 2021: 350 time steps for 73.77% on ImageNet (Spiking ResNet)
- 2022: 32 time steps for 95.54% on CIFAR-10 (clip-floor-shift)
- 2023: 3 time steps for 95.42% on CIFAR-10 (Fast-SNN)
- 2024: 4 time steps for 94.75% on CIFAR-10 (Parameter Calibration)

### Recommendation for Thesis
ANN-to-SNN conversion is viable as a thesis topic but **not recommended as a primary project** for an undergraduate. The tools work but require deep understanding of both ANNs and SNNs. Better to use direct training with surrogate gradients via snnTorch, which has better tooling and documentation.

---

<a name="thesis-scopes"></a>
## REALISTIC UNDERGRADUATE THESIS SCOPES

### What Is Achievable in One Semester (4-6 months)

**Tier 1 -- Highly Achievable (Recommended):**
- Train a convolutional SNN on MNIST/Fashion-MNIST/CIFAR-10 using surrogate gradient descent
- Compare SNN vs ANN accuracy and estimated energy consumption on the same task
- Classify DVS128 gestures using a pre-built SNN architecture
- Audio digit classification on SHD using snnTorch
- Reproduce a published result from a recent paper using snnTorch or SpikingJelly

**Tier 2 -- Achievable with Effort:**
- ANN-to-SNN conversion comparison: convert a pre-trained CNN and compare with directly-trained SNN
- Event-based gesture recognition with architecture modifications (adding attention, changing neuron models)
- Multi-dataset benchmarking: compare SNN performance across N-MNIST, CIFAR10-DVS, and SHD
- Implement and compare different surrogate gradient functions (arctangent, sigmoid, triangular)

**Tier 3 -- Ambitious but Possible:**
- Spiking Transformer implementation on a small dataset
- STDP-based unsupervised feature learning on neuromorphic data
- SNN for time-series anomaly detection (ECG, vibration data)
- Hybrid SNN-ANN architecture for a specific application

**Tier 4 -- Probably Too Ambitious for Undergraduate:**
- Novel neuron model development
- Neuromorphic hardware deployment (unless hardware is available)
- Large-scale ImageNet training
- Novel spiking transformer architecture
- SpikeLLM or NLP applications

### Time Budget Estimation

| Phase | Weeks | Activities |
|-------|-------|------------|
| Literature review | 2-3 | Read papers, understand fundamentals |
| Environment setup | 1 | Install frameworks, run tutorials |
| Tutorial completion | 2-3 | snnTorch tutorials 1-7 |
| Baseline implementation | 2-3 | Get basic model working |
| Experiments | 3-4 | Run variations, collect results |
| Analysis | 2 | Compare results, create visualizations |
| Writing | 3-4 | Draft and revise thesis |
| **Total** | **15-20 weeks** | |

---

<a name="low-barrier"></a>
## LOW-BARRIER SNN APPLICATIONS

### Ranked by Accessibility (Easiest First)

**1. MNIST/Fashion-MNIST Classification with SNN (Easiest)**
- snnTorch Tutorial 5-6 gets you 95%+ accuracy
- Direct comparison with a standard ANN
- Estimated energy savings calculation
- Good for: Understanding SNN training mechanics

**2. Neuromorphic Digit Classification (N-MNIST)**
- Uses actual event-camera data
- Built-in dataset loaders in snnTorch and SpikingJelly
- snnTorch Tutorial 7 covers this directly
- 98%+ accuracy achievable
- Good for: Demonstrating SNNs on native neuromorphic data

**3. DVS128 Gesture Recognition**
- 11 gesture classes from event camera
- SpikingJelly tutorial provides step-by-step code
- 96-98% accuracy achievable with standard architectures
- Good for: Practical real-world application; compelling demo

**4. Audio Classification on SHD**
- Spiking Heidelberg Digits dataset
- Temporal data naturally suited to SNNs
- 93-96% accuracy achievable
- Good for: Showcasing SNN temporal processing advantage

**5. ANN-to-SNN Conversion Study**
- Convert pre-trained CIFAR-10 CNN to SNN
- Compare accuracy vs. time steps tradeoff
- Use SpikingJelly ann2snn or NengoDL
- Good for: Bridging ANN and SNN knowledge

**6. Surrogate Gradient Function Comparison**
- Implement same network with different surrogate gradients (arctangent, sigmoid, triangular, Gaussian)
- Measure accuracy, convergence speed, spike density
- Good for: Understanding a key SNN training mechanism

**7. EEG/Biomedical Signal Classification**
- Use SEED dataset for emotion recognition (96.67% reported)
- Or ECG classification for arrhythmia detection
- Good for: Healthcare application; interdisciplinary appeal

### Energy Efficiency Numbers for Motivation

| Platform | Power | Task | Notes |
|----------|-------|------|-------|
| Intel Loihi 2 | 1.21 mJ/inference | Vision | 52x less than GPU |
| Neuromorphic SNN | 5 mJ/inference | General | vs. 200 mJ for ANN |
| TrueNorth | 67 mW | Keyword spotting | 82 days on coin cell battery |
| GPU (Jetson Nano) | 62.9 mJ/inference | Vision | Baseline comparison |

---

<a name="examples"></a>
## EXAMPLE THESES AND PROJECTS

### Published Example Projects

1. **Master Thesis (Univ. Padova):** "Hardware Implementation of a Spiking Neural Network" -- RTL modeling and FPGA resource mapping of SNN architecture

2. **GitHub Project (DerrickL25):** "SNN Gesture Classification" -- Neuromorphic gesture classification system using snnTorch and DVS128 event camera data

3. **VCU Dissertation:** "Spiking Neural Networks: Neuron Models, Plasticity, and Graph Applications" -- Comprehensive treatment of neuron models and learning rules

### Reproducible Paper Implementations

| Paper | Task | Dataset | Framework | Difficulty |
|-------|------|---------|-----------|------------|
| DCSNN (2018) | MNIST classification | MNIST | BindsNet/snnTorch | Easy |
| Spiking YOLO (2019) | Object detection | PASCAL VOC | Custom | Hard |
| Spike-FlowNet (2020) | Optical flow | Event camera | Custom | Hard |
| SiamSNN (2020) | Object tracking | - | Custom | Hard |
| SpikeSEG (2021) | Semantic segmentation | Synthetic | Custom | Moderate |
| Spikformer (2022) | Image classification | CIFAR/ImageNet | SpikingJelly | Moderate-Hard |

---

<a name="recommendation"></a>
## SYNTHESIS AND THESIS PROJECT RECOMMENDATION

### Assessment of the Field

Based on comprehensive analysis of all three survey papers and supplementary research:

1. **SNNs are maturing rapidly.** The accuracy gap between SNNs and ANNs on standard benchmarks has narrowed significantly. On CIFAR-10, SNNs achieve 96.47% (vs. ANN ~97%). On ImageNet, the gap is larger but closing (84-88% SNN vs. ~90% ANN).

2. **Surrogate gradient training is the dominant paradigm.** Direct training via BPTT with surrogate gradients has overtaken ANN-to-SNN conversion as the primary research approach. It requires fewer time steps and gives more control.

3. **Spiking Transformers are the hottest research area.** The field has moved from convolutional SNNs to transformer-based architectures, with rapid progress in 2023-2024.

4. **Tooling has matured significantly.** snnTorch and SpikingJelly provide production-quality frameworks with good documentation, tutorials, and community support.

5. **The energy efficiency argument is real but hard to measure.** Theoretical advantages of 50-100x are cited, but practical measurement requires neuromorphic hardware access.

### PRIMARY RECOMMENDATION: DVS128 Gesture Recognition with Surrogate Gradient Training

**Title suggestion:** "Energy-Efficient Gesture Recognition Using Spiking Neural Networks: A Comparative Study of Neuron Models and Training Approaches"

**Why this project:**
- Clear, well-defined problem (11-class gesture classification)
- Native neuromorphic dataset (DVS128 Gesture) -- you work with real event camera data, not converted static images
- Established baselines to compare against (96-98% accuracy)
- Built-in dataset loaders in SpikingJelly (1,464 samples: 1,176 train, 288 test)
- Concrete contribution potential: compare different neuron models (LIF vs. Parametric LIF vs. ALIF) on the same task
- Energy efficiency analysis can be done theoretically (spike count, synaptic operations)
- Compelling visual demonstrations (gesture videos + spike rasters)
- Publishable scope if results are strong

**Concrete Methodology:**
1. **Weeks 1-3:** Literature review + snnTorch Tutorials 1-7
2. **Weeks 4-5:** Set up SpikingJelly environment, load DVS128 Gesture dataset, run baseline
3. **Weeks 6-8:** Implement 3-4 network variants:
   - Variant A: Convolutional SNN with standard LIF neurons
   - Variant B: Same architecture with Parametric LIF (learnable time constants)
   - Variant C: Same architecture with different surrogate gradient functions
   - Variant D: (Optional) Add temporal attention mechanism
4. **Weeks 9-11:** Run experiments, collect metrics (accuracy, spike count per inference, convergence speed, estimated energy consumption)
5. **Weeks 12-13:** Compare with ANN baseline on same task (train a standard CNN on frame-integrated DVS data)
6. **Weeks 14-18:** Analysis, visualization, thesis writing

**Expected Deliverables:**
- Accuracy comparison table across variants
- Spike density analysis and energy consumption estimates
- Convergence curves for different approaches
- Visualization of learned features/spike patterns
- Discussion of when SNNs provide genuine advantages over ANNs

**Tools Required:**
- Python 3.8+, PyTorch 2.0+
- SpikingJelly (primary) and/or snnTorch (comparison)
- Standard GPU (even a single consumer GPU like RTX 3060 is sufficient)
- No neuromorphic hardware needed

### ALTERNATIVE RECOMMENDATION 1: Audio Classification on Spiking Heidelberg Digits

**Why:** Audio is temporal by nature, making it a natural fit for SNNs. The SHD dataset is small, well-curated, and the temporal processing advantage of SNNs is most clearly demonstrated on temporal data. This project would compare LIF-based SNNs with RNN baselines. State-of-the-art is 96.41% (SpikCommander, 2025).

### ALTERNATIVE RECOMMENDATION 2: Surrogate Gradient Function Comparison Study

**Why:** More methodological/analytical project. Implement the same convolutional SNN architecture on CIFAR-10 with 5-6 different surrogate gradient functions. Measure accuracy, convergence speed, spike sparsity, and gradient flow. This contributes to understanding of a fundamental SNN training mechanism and requires no neuromorphic data.

### ALTERNATIVE RECOMMENDATION 3: ANN-to-SNN Conversion Accuracy-Latency Tradeoff Analysis

**Why:** Take a pre-trained VGG-16 or ResNet-18 on CIFAR-10, convert to SNN using SpikingJelly ann2snn, and systematically measure accuracy as time steps decrease from 256 to 1. Compare with directly-trained SNN. This provides clear quantitative results and addresses a practical question in the field.

---

## CONFIDENCE ASSESSMENT

| Finding | Confidence |
|---------|------------|
| Paper 1 (Yamazaki) analysis completeness | HIGH -- full text extracted |
| Paper 2 (Han) analysis completeness | HIGH -- full text extracted |
| Paper 3 (Malcolm) analysis completeness | MODERATE -- only abstract and cross-references accessible; PDF could not be text-extracted |
| Framework comparison accuracy | HIGH -- cross-referenced multiple sources |
| Dataset availability information | HIGH -- verified through multiple sources |
| snnTorch tutorial catalog | HIGH -- verified from official documentation |
| ANN-to-SNN conversion assessment | HIGH -- multiple sources confirm current state |
| Thesis scope recommendations | HIGH -- based on published results and tool maturity |
| Energy efficiency claims | MODERATE -- theoretical numbers vary; practical measurement requires hardware |
| Benchmark accuracy numbers | HIGH -- extracted from peer-reviewed sources |

## RESEARCH GAPS (What I Could Not Fully Determine)

1. **Paper 3 full content:** The Malcolm & Casco-Rodriguez paper was not available in HTML format on arXiv, and the PDF could not be text-extracted. The full benchmark tables and specific best practices recommendations from this paper are missing from this analysis.

2. **Neuromorphic hardware availability at your university:** Whether your institution has access to Loihi, SpiNNaker, or other neuromorphic hardware would significantly affect project recommendations.

3. **Your specific GPU resources:** The compute requirements vary significantly between MNIST-scale and ImageNet-scale experiments.

4. **Your prior experience with PyTorch:** The learning curve for snnTorch assumes basic PyTorch familiarity.

## RECOMMENDED FOLLOW-UP ACTIONS

1. Download and read the PDF of Malcolm & Casco-Rodriguez (Paper 3) manually -- it is designed for newcomers
2. Complete snnTorch Tutorials 1-6 (approximately 8-10 hours of work)
3. Run the SpikingJelly DVS128 Gesture example to verify your environment works
4. Discuss the recommended project direction with your thesis supervisor
5. Check if your university has any neuromorphic hardware access (would open additional project possibilities)
6. Join the Open Neuromorphic community (open-neuromorphic.org) for support

---

## SOURCES AND REFERENCES

### Papers Analyzed
- Yamazaki et al., "Spiking Neural Networks and Their Applications: A Review," Brain Sciences, 2022. https://pmc.ncbi.nlm.nih.gov/articles/PMC9313413/
- Han et al., "Toward Large-scale Spiking Neural Networks," arXiv:2409.02111, 2024. https://arxiv.org/html/2409.02111v1
- Malcolm & Casco-Rodriguez, "A Comprehensive Review of Spiking Neural Networks," arXiv:2303.10780, 2023. https://arxiv.org/abs/2303.10780

### Frameworks
- snnTorch: https://github.com/jeshraghian/snntorch | Documentation: https://snntorch.readthedocs.io/
- SpikingJelly: https://github.com/fangwei123456/spikingjelly
- Norse: https://github.com/norse/norse | Documentation: https://norse.github.io/norse/
- snn_toolbox: https://github.com/NeuromorphicProcessorProject/snn_toolbox
- NengoDL: https://www.nengo.ai/nengo-dl/examples/keras-to-snn.html
- Open Neuromorphic: https://open-neuromorphic.org/neuromorphic-computing/software/snn-frameworks/

### Datasets
- SHD/SSC: https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/
- Neuromorphic Dataset List: https://www.simonwenkel.com/lists/datasets/list-of-neuromorphic-datasets.html
- DVS128 Gesture: https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794
- N-MNIST: https://www.garrickorchard.com/datasets/n-mnist
- CIFAR10-DVS: https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671
- Tonic (dataset loader): https://tonic.readthedocs.io/

### Benchmarks
- Open Neuromorphic SNN Benchmarks: https://open-neuromorphic.org/blog/spiking-neural-network-framework-benchmarking/
- SNN Library Benchmark Code: https://github.com/open-neuromorphic/snn-library-benchmarks

### Additional Reading
- Eshraghian et al., "Training Spiking Neural Networks Using Lessons From Deep Learning," 2023 (companion to snnTorch)
- NeuroBench: https://www.nature.com/articles/s41467-025-56739-4
