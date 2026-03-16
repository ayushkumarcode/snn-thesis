# Neuromorphic Computing & Spiking Neural Networks — Literature Review

> Initial research notes compiled from 3 key survey papers to help narrow down a thesis project direction.

---

## Paper 1: Spiking Neural Networks and Their Applications: A Review

**Source:** Yamazaki et al., Brain Sciences, 2022
**Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC9313413/
**Purpose for us:** Understanding what people are *actually building* with SNNs — helps pick a use case.

### What are SNNs?

- Third generation of neural networks, inspired by biological neurons
- Neurons communicate via discrete **spikes** (binary events) rather than continuous values
- Information is encoded in **spike timing** and **spike frequency**, not just magnitude
- Event-driven: neurons only compute when they receive a spike, making them inherently energy-efficient

### Neuron Models (simplest to most complex)

| Model | What it does | Trade-off |
|-------|-------------|-----------|
| **LIF (Leaky Integrate-and-Fire)** | Accumulates input, leaks over time, fires when threshold is reached | Simple, fast, but biologically limited |
| **Izhikevich** | 2 equations that can reproduce 20+ biological firing patterns | Good balance of realism and speed |
| **AdEx (Adaptive Exponential)** | LIF + exponential spike initiation + adaptation current | Better spike pattern fidelity |
| **Hodgkin-Huxley** | Full ion-channel dynamics (Na+, K+) | Most realistic, most expensive to simulate |

**Takeaway:** LIF is the go-to for most practical projects. Izhikevich if you need more biological realism without the cost of Hodgkin-Huxley.

### How to Encode Data as Spikes

- **Rate coding:** Higher spike frequency = stronger signal. Simple but requires many timesteps.
- **Temporal coding:** Information in precise spike *timing*. More efficient (fewer spikes) but harder to work with.

### Training Methods

| Method | Type | How it works | Pros / Cons |
|--------|------|-------------|-------------|
| **STDP** | Unsupervised | Strengthen synapse if pre fires before post, weaken if after | Biologically plausible, no labels needed / limited scalability |
| **Reward-modulated STDP** | Reinforcement | STDP + dopamine-like reward signal | Can learn from sparse rewards / slow convergence |
| **SpikeProp / SuperSpike** | Supervised | Gradient descent adapted for spikes | Can train deeper networks / needs surrogate gradients |
| **SLAYER** | Supervised | Temporal credit assignment via backprop | Good for temporal tasks / memory intensive |
| **ANN-to-SNN Conversion** | Transfer | Train a normal ANN, then convert weights to SNN | Leverages existing ANN tools / may need many timesteps |

### Applications Covered

#### Computer Vision
- **Image classification:** DCSNN achieved 97.2% on MNIST using STDP + reward learning
- **Object detection:** Spiking YOLO — energy-efficient detection with channel-wise normalization
- **Object tracking:** SiamSNN — 50 FPS on IBM TrueNorth with extremely low energy
- **Segmentation:** UNet-based SNN deployed on Intel Loihi
- **Optical flow:** Spike-FlowNet for event-camera processing
- **Medical imaging:** STDP-based melanoma detection at 87.7% accuracy

#### Robotics
- **Locomotion:** Spiking Central Pattern Generators (sCPG) for hexapod walking on SpiNNaker hardware
- **Motor control:** Event-based PID on Loihi for drone stabilization (0.005g thrust error)
- **Navigation:** Spiking RatSLAM using place cells + grid cells for SLAM
- **Autonomous mapping:** Gridbot — 1,321 LIF neurons, 100x less energy than CPU-based GMapping

### Key Software Frameworks

| Framework | Best for | Notes |
|-----------|---------|-------|
| **snnTorch** | PyTorch users | Easy to learn, active community |
| **Norse** | PyTorch users | Good for research, flexible neuron models |
| **Brian2** | Neuroscience-style simulation | Equation-based, very flexible |
| **Nengo** | Neural engineering | Good for robotics applications |
| **SpykeTorch** | Convolutional SNNs with STDP | PyTorch-based |
| **snn_toolbox** | ANN-to-SNN conversion | Multi-framework support |
| **Lava** | Intel Loihi deployment | Official Intel framework |

### Datasets / Benchmarks

- **Static images:** MNIST, CIFAR-10, CIFAR-100, ImageNet
- **Neuromorphic (event-based):** N-MNIST, N-Caltech101, DVS128 Gesture, DVS-CIFAR10
- **3D / LiDAR:** KITTI
- **Medical:** ISIC 2018 (melanoma)
- **EEG/Brain:** DEAP, SEED (emotion recognition)

---

## Paper 2: Toward Large-scale Spiking Neural Networks: A Comprehensive Survey and Future Directions

**Source:** Han et al., arXiv, 2024
**Link:** https://arxiv.org/html/2409.02111v1
**Purpose for us:** What's working *right now*, state-of-the-art results, and what the open problems are (thesis gold).

### The Energy Argument

- Human brain: ~20 watts for complex cognition
- Training GPT-3: 1,287 MWh
- SNNs promise to close this gap through event-driven, spike-based computation
- But the promise only holds if we can match ANN accuracy — otherwise the efficiency gain is meaningless

### Training Methods — Deep Dive

#### ANN-to-SNN Conversion (easiest entry point)
1. Train a standard ANN (e.g., ResNet) normally
2. Replace ReLU activations with integrate-and-fire neurons
3. Apply weight normalization and threshold balancing
4. Use **reset-by-subtraction** (not reset-to-zero) to preserve information

**Results:**
- Fast-SNN on ImageNet: 95.42% ANN → 95.51% SNN (actually *improved*)
- CIFAR-10: 95.54% with just 4 timesteps
- This is the most practical approach for a thesis — leverage existing ANN training infrastructure

#### Surrogate Gradient Training (direct SNN training)
- Problem: spikes are binary (0 or 1), so gradients are either 0 or undefined
- Solution: replace the true gradient with a smooth approximation during backprop
- Recent advances: learnable neuron parameters, batch normalization for membranes
- **CIFAR-10: 96.47% accuracy with just 2 timesteps** (better than conversion!)

#### STDP (biologically plausible but limited)
- Local learning rule, no backprop needed
- Doesn't scale well to deep networks
- Mostly used for unsupervised feature extraction in early layers

### Architectures

#### Spiking CNNs
- Spiking ResNets (SEW-ResNet, MS-ResNet) — adapted residual connections for spikes
- Key challenge: maintaining gradient flow through spiking layers

#### Spiking Transformers (cutting edge, 2023-2024)
- Replace softmax attention with spike-based matrix operations
- **Spikformer:** 74.81% ImageNet (4 timesteps)
- **QKFormer:** 84.22% ImageNet — approaching ANN-level performance
- **Spike-driven Transformer V2:** 79.7% with meta-architecture design
- This is where the field is moving rapidly

#### Spiking Language Models (very new)
- SpikingBERT, SpikingGPT, SpikingMiniLM — early explorations
- 70B parameter SpikeLLM demonstrating scalability is possible
- Very early stage but shows the direction

### State-of-the-Art Results (ImageNet)

| Method | Type | Architecture | Accuracy | Timesteps | Params |
|--------|------|-------------|----------|-----------|--------|
| ECMT | Conversion | EVA | 88.60% | 4 | 1,074M |
| QKFormer | Direct | Spiking Transformer | 84.22% | 4 | 65M |
| SGLFormer | Direct | Spiking Transformer | 83.73% | 4 | 64M |
| Spikformer V2 | Direct | Spiking Transformer | 80.38% | 4 | 52M |

### The Honest Performance Gap

- SNNs at 4 timesteps ≈ 2-bit quantized ANNs in terms of representational capacity
- Single-timestep inference is significantly worse than ANNs
- Binary spike representation is fundamentally limited — this is an open research problem

### Hardware Platforms

| Platform | Maker | Notes |
|----------|-------|-------|
| **Loihi / Loihi 2** | Intel | Most accessible for research, good software support (Lava) |
| **TrueNorth** | IBM | 1 million neurons, very low power, but limited programmability |
| **SpiNNaker** | University of Manchester | Designed for large-scale brain simulation |
| **Darwin** | Chinese Academy of Sciences | Emerging platform |

### Open Problems (potential thesis topics!)

1. **Information loss in deep SNNs** — binary spikes lose information as networks get deeper
2. **Surrogate gradient mismatch** — the smooth approximation doesn't perfectly match real spike dynamics
3. **Temporal efficiency** — more timesteps = better accuracy but slower inference; how to minimize timesteps?
4. **Spiking attention mechanisms** — current spike-based attention lacks the non-linearity of softmax
5. **Multi-modal SNNs** — combining vision, audio, text in a single spiking architecture
6. **Hardware-software co-design** — algorithms optimized for specific neuromorphic chips

---

## Paper 3: A Comprehensive Review of Spiking Neural Networks: Interpretation, Optimization, Efficiency, and Best Practices

**Source:** Malcolm & Casco-Rodriguez, arXiv, 2023
**Link:** https://arxiv.org/abs/2303.10780
**Purpose for us:** Practical guide — how to actually build and evaluate SNNs properly.

*(Full text was access-restricted; summary from abstract and metadata)*

### Key Focus Areas
1. **Interpretation** — how to understand what SNNs are doing internally
2. **Optimization** — best training strategies and hyperparameter choices
3. **Energy efficiency** — how to measure and compare power consumption fairly
4. **Best practices** — practical recommendations for SNN development

### Why This Paper Matters
- Written as an accessible entry point starting from first principles
- Emphasizes SNNs for **low-power, mobile, and hardware-constrained settings**
- Covers evaluation methodology — important for a thesis that needs rigorous results

---

## Summary: Potential Thesis Directions

Based on all three papers, here are the most viable project directions ranked by feasibility:

### Tier 1: Most achievable, clear deliverables

| Direction | What you'd build | Why it works |
|-----------|-----------------|-------------|
| **ANN-to-SNN conversion benchmark** | Convert a pre-trained CNN to SNN, measure accuracy vs efficiency on CIFAR-10/100 | Well-documented method, existing tools (snn_toolbox), clear metrics |
| **SNN on neuromorphic dataset** | Train SNN classifier on DVS128 Gesture or N-MNIST | Natural fit for spikes, public datasets, snnTorch tutorials exist |
| **SNN vs ANN energy comparison** | Same task solved by both, compare accuracy + estimated energy + latency | Examiners love comparisons, strong narrative |

### Tier 2: More ambitious, stronger thesis

| Direction | What you'd build | Why it works |
|-----------|-----------------|-------------|
| **Spiking Transformer implementation** | Reproduce or adapt Spikformer on a vision task | Cutting-edge area, but more complex to implement |
| **STDP unsupervised + supervised hybrid** | STDP for feature extraction + supervised classifier on top | Shows understanding of both learning paradigms |
| **Edge deployment demo** | SNN for a sensor task (gesture/heartbeat) optimized for low power | Practical application angle, good for presentation |

### Tier 3: Research-level (risky for undergrad thesis)

| Direction | What you'd build | Why it works |
|-----------|-----------------|-------------|
| **Multi-modal spiking network** | Combine event camera + audio in one SNN | Open problem, high novelty, but unclear if achievable |
| **Spiking NLP** | Adapt SpikingBERT for a text task | Very new area, limited existing work to build on |

---

## Next Steps

1. Read Paper 2 (applications review) — skim the use cases, pick 2-3 that interest you
2. Check dataset/tool availability for those use cases
3. Read Paper 1 (large-scale survey) — for your shortlisted areas, check state-of-the-art and what's missing
4. Commit to one direction and start scoping the MVP

---

*Last updated: 2026-02-23*
