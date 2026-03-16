# Multimodal Spiking Neural Networks: Comprehensive Research Report

**Date**: 2026-02-25
**Research Focus**: Combining different data types (vision, audio, event camera, IMU) in a single spiking neural network
**Purpose**: Assess feasibility for an undergraduate thesis project

---

## Executive Summary

Multimodal SNNs -- combining different sensory data types within a single spiking neural network -- represent an **active and rapidly growing research area** that has seen significant acceleration since 2023. The field is no longer purely theoretical: multiple working implementations exist for audio-visual classification, event camera + RGB fusion, and sensor fusion for robotics. Critically, a paper published in August 2024 demonstrates *exactly* the simplified version proposed (MNIST digits + audio digits fusion), achieving 98.43% accuracy. This means a multimodal SNN thesis project is **achievable at the undergraduate level**, provided the scope is carefully bounded. The area has enough existing work to build upon but enough open problems to contribute meaningfully.

**Verdict: This is feasible as an undergraduate project. It sits at the boundary between "well-explored for PhDs" and "emerging for undergrads" -- an ideal sweet spot for a thesis that can demonstrate both competence and novelty.**

---

## 1. Has Anyone Combined Vision + Audio in an SNN?

**Yes -- this is now an established sub-field with at least 6 major papers from 2023-2025.**

### Key Papers and Systems

| Paper/System | Year | Datasets | Accuracy | Key Innovation |
|---|---|---|---|---|
| **SMMT** (Spiking Multi-Modal Transformer) | 2023 | CREMA-D, UrbanSound8K-AV | ~66% (CREMA-D) | Spiking Cross-Attention (SCA) mechanism for audio-visual fusion |
| **MISNet** (Multimodal Interaction Spiking Network) | 2024 | 5 audio-visual datasets | Competitive | MLIF neuron that synchronizes audiovisual spikes in a single neuron |
| **Bjorndahl et al.** | Aug 2024 | N-MNIST + SHD | 98.43% | Early/middle/late fusion comparison for digit recognition |
| **S-CMRL** | Feb 2025 | CREMA-D, UrbanSound8K-AV, MNISTDVS-NTIDIGITS | 73.25% (CREMA-D), 99.28% (MNISTDVS-NTIDIGITS) | Semantic-alignment + cross-modal residual learning |
| **TAAF** | May 2025 | CREMA-D, AVE, EAD | 77.55% (CREMA-D) | Temporal attention-guided adaptive fusion for modality imbalance |
| **Oikonomou et al.** | Nov 2024 | Various | Survey | Bio-inspired multimodal perception for robotics |

### Architecture Patterns

Three dominant fusion strategies have emerged:
1. **Early fusion**: Combining raw spike trains before feature extraction
2. **Late fusion (concatenation)**: Separate unimodal branches that merge at the decision layer -- simplest to implement
3. **Cross-modal attention**: Spiking attention mechanisms that allow modalities to guide each other

The Bjorndahl et al. (2024) paper is particularly relevant: they found that **late fusion (concatenation)** achieves nearly identical results to more complex fusion strategies, and the fusion depth has minimal impact on accuracy. This means the simplest approach works well.

### Sources
- [SMMT - IEEE Xplore](https://ieeexplore.ieee.org/iel7/7274989/10552653/10293172.pdf)
- [MISNet - ACM TOMM](https://dl.acm.org/doi/10.1145/3721981)
- [Bjorndahl et al. - arXiv 2409.00552](https://arxiv.org/abs/2409.00552)
- [S-CMRL - arXiv 2502.12488](https://arxiv.org/html/2502.12488)
- [S-CMRL Code - GitHub](https://github.com/Brain-Cog-Lab/S-CMRL)
- [TAAF - arXiv 2505.14535](https://arxiv.org/abs/2505.14535)
- [Oikonomou et al. - arXiv 2411.14147](https://arxiv.org/abs/2411.14147)

---

## 2. Combining Event Camera Data + Conventional Camera Data

**Yes -- this is one of the most active areas in neuromorphic computing (2024-2025).**

### Key Papers and Systems

| Paper/System | Year | Task | Key Contribution |
|---|---|---|---|
| **SFDNet** | 2025 | Object Detection | Fully spiking RGB-event fusion detector with LIMF neuron; state-of-the-art on PKU-DAVIS-SOD |
| **SSTFormer** | 2023-2025 | Frame-Event Recognition | Hybrid SNN-ANN with Memory Support Transformer for RGB + spiking event encoding |
| **DSF-Net** | 2025 | High-Speed Detection | Dynamic sparse fusion of event-RGB via spike-triggered attention |
| **SpikeFET** | 2025 | Object Tracking | First fully spiking frame-event tracker |
| **SNNPTrack** | 2025 | RGBE Tracking | SNN-based prompt learning for RGB-Event tracking (ICASSP 2025) |
| **RGB-Event Collision Prediction** | 2025 | Collision Prediction | Self-attention fusion for UAV collision prediction (IJCNN 2025) |

### Technical Approach

The standard pattern is:
- **RGB branch**: Processes conventional camera frames (sometimes using ANN layers)
- **Event branch**: Processes DVS event streams using spiking neurons (LIF/PLIF)
- **Fusion module**: Combines features via attention, concatenation, or cross-modal mechanisms

SSTFormer is notable because it has a **publicly available codebase** on GitHub (https://github.com/Event-AHU/SSTFormer) and a custom PokerEvent dataset with 114 classes and 27,102 frame-event pairs.

### Sources
- [SFDNet - MDPI Electronics](https://www.mdpi.com/2079-9292/14/6/1105)
- [SSTFormer - arXiv](https://arxiv.org/abs/2308.04369)
- [SSTFormer Code - GitHub](https://github.com/Event-AHU/SSTFormer)
- [DSF-Net - ACM MM 2025](https://dl.acm.org/doi/10.1145/3746027.3755846)
- [SpikeFET - arXiv](https://arxiv.org/pdf/2505.20834)
- [RGB-Event Collision - IJCNN 2025](https://arxiv.org/html/2505.04258v2)

---

## 3. Sensor Fusion with SNNs (IMU + Camera, etc.)

**Yes -- emerging area, especially for robotics and autonomous navigation.**

### Key Research

| Paper/System | Year | Sensors | Platform | Key Finding |
|---|---|---|---|---|
| **Lopez-Osorio et al.** | 2024 | Neuromorphic vision + feedback sensors | SpiNNaker | Real-time robot adaptation using SNN sensor fusion |
| **Spiking Neural-Invariant Kalman Fusion** | Jan 2026 | IMU | CPU/GPU | SNN extracts motion features from noisy IMU data, fused with Kalman filter for pose estimation |
| **Loihi-2 Sensor Fusion** | Aug 2024 | Camera, LIDAR, GPS, IMU | Intel Loihi-2 | 100x more efficient than CPU, 30x more than GPU for sensor fusion |
| **Marine-Inspired Navigation** | 2025 | Multiple underwater sensors | Neuromorphic | Multi-modal sensor fusion for autonomous subaquatic navigation |

### Key Finding: Loihi-2 Sensor Fusion

The Intel Loihi-2 case study is particularly striking: deploying SNNs on neuromorphic hardware achieved **over 100x energy efficiency improvement** over CPU and **~30x over GPU** for sensor fusion tasks across AIODrive, Oxford Radar RobotCar, nuScenes, and other datasets. This demonstrates the practical motivation for neuromorphic sensor fusion.

### Sources
- [Lopez-Osorio et al. - Wiley](https://advanced.onlinelibrary.wiley.com/doi/10.1002/aisy.202300646)
- [Spiking Neural-Invariant Kalman - arXiv](https://arxiv.org/abs/2601.08248)
- [Loihi-2 Sensor Fusion - arXiv](https://arxiv.org/abs/2408.16096)
- [Marine-Inspired Navigation - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12608416/)
- [Neuromorphic Solutions Book - Springer](https://link.springer.com/book/10.1007/978-3-031-63565-6)

---

## 4. Is This an Open Research Problem?

**Yes -- multimodal SNNs are explicitly identified as an open research problem in multiple survey papers.**

### Documented Research Gaps

1. **Limited exploration of multimodal integration**: "Imaging applications remain strongly biased toward classification and regression tasks, with limited exploration of segmentation or multimodal integration" (SNN Imaging Survey, 2025)

2. **Dataset scarcity**: There is a recognized "lack of standardisation across SNN architectures and training methodologies" and limited large-scale multimodal neuromorphic datasets

3. **Modality imbalance problem**: When one modality (e.g., vision) is stronger than another (e.g., audio), the network over-relies on the dominant modality -- TAAF (2025) addresses this but it remains open

4. **Temporal alignment**: Synchronizing spike trains from modalities operating at different temporal scales is unsolved in general

5. **Scalability**: Most multimodal SNN work is limited to classification; extending to detection, segmentation, or generation is largely unexplored

6. **Biological plausibility**: Most current approaches use surrogate gradient training (not biologically plausible); combining multimodal fusion with STDP-based learning rules is almost entirely unexplored

7. **Missing modality robustness**: What happens when one modality is absent or corrupted? This is barely studied in the SNN context

### Open Research Questions Suitable for Undergrad Contribution

- How does fusion strategy (early/mid/late) affect robustness to noise in one modality?
- Can STDP-based learning (unsupervised) achieve competitive multimodal fusion?
- What is the minimum network size needed for effective multimodal SNN fusion?
- How does temporal resolution mismatch between modalities affect SNN fusion?
- Can knowledge distillation from ANN multimodal models improve SNN multimodal models?

### Sources
- [SNN Imaging Survey - MDPI Sensors](https://www.mdpi.com/1424-8220/25/21/6747)
- [SNN Multimodal Neuroimaging Review - Frontiers](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1623497/full)
- [SNN Ubiquitous Computing Survey - arXiv](https://arxiv.org/html/2506.01737v1)

---

## 5. How Complex Would a Simplified Multimodal SNN Be?

### Complexity Assessment

#### Minimum Viable Multimodal SNN (Late Fusion, Digit Classification)

Based on the Bjorndahl et al. (2024) paper architecture:

```
ARCHITECTURE OVERVIEW:
  Visual Branch (N-MNIST):
    - Input: 34x34x2 spike trains (300 timesteps)
    - Conv layers: 2-3 spiking convolutional layers with LIF neurons
    - Output: 128-dim feature vector

  Auditory Branch (SHD):
    - Input: 700 channels spike trains
    - Recurrent/FC layers: 2-3 spiking fully-connected layers with LIF neurons
    - Output: 128-dim feature vector

  Fusion:
    - Concatenation: 256-dim combined vector
    - Classification head: 1-2 FC layers -> 10 classes

ESTIMATED PARAMETERS: ~100K-500K (very small by modern standards)
TRAINING TIME: Hours on a single GPU (even a laptop GPU)
CODE COMPLEXITY: ~300-500 lines of Python (using snnTorch/SpikingJelly)
```

#### Comparison to Other Project Types

| Project Type | Approx Parameters | Training Time | Code Lines | Difficulty |
|---|---|---|---|---|
| Single-modality SNN (MNIST) | 50K-200K | <1 hour | 150-250 | Beginner |
| **Multimodal SNN (digit fusion)** | **100K-500K** | **2-6 hours** | **300-500** | **Intermediate** |
| Audio-visual SNN (CREMA-D) | 1M-5M | 1-2 days | 800-1500 | Advanced |
| RGB-Event object detection SNN | 5M-20M | Days-weeks | 2000+ | PhD-level |

### What Makes It Manageable

1. **Both datasets are small**: N-MNIST is 28x28 pixels, SHD is 700 channels -- not huge
2. **Both are pre-spiked**: No need for custom spike encoding; data is already in spike format
3. **Late concatenation fusion is trivial**: Just `torch.cat([visual_features, audio_features], dim=1)`
4. **Well-established training**: Surrogate gradient descent with standard cross-entropy loss
5. **Existing code to reference**: S-CMRL codebase on GitHub, snnTorch tutorials, multiple papers

---

## 6. Key Papers (2023-2026) -- Comprehensive Bibliography

### Core Multimodal SNN Papers

1. **Gu et al. (2023)** - "Transformer-Based Spiking Neural Networks for Multimodal Audiovisual Classification" - IEEE TCDS. [Link](https://ieeexplore.ieee.org/iel7/7274989/10552653/10293172.pdf)

2. **Wang et al. (2023-2025)** - "SSTFormer: Bridging Spiking Neural Network and Memory Support Transformer for Frame-Event based Recognition" - IEEE TCDS. [Link](https://arxiv.org/abs/2308.04369)

3. **Bjorndahl et al. (Aug 2024)** - "Digit Recognition using Multimodal Spiking Neural Networks" - Submitted to IEEE ICASSP 2025. [Link](https://arxiv.org/abs/2409.00552) -- **MOST RELEVANT FOR UNDERGRAD PROJECT**

4. **MISNet (2024)** - "Towards Energy-efficient Audio-visual Classification via Multimodal Interactive Spiking Neural Network" - ACM TOMM. [Link](https://dl.acm.org/doi/10.1145/3721981)

5. **Oikonomou et al. (Nov 2024)** - "Spiking neural networks: Towards bio-inspired multimodal perception in robotics" - arXiv survey. [Link](https://arxiv.org/abs/2411.14147)

6. **S-CMRL (Feb 2025)** - "Enhancing Audio-Visual Spiking Neural Networks through Semantic-Alignment and Cross-Modal Residual Learning" - arXiv. [Link](https://arxiv.org/abs/2502.12488)

7. **TAAF (May 2025)** - "Spiking Neural Networks with Temporal Attention-Guided Adaptive Fusion for imbalanced Multi-modal Learning" - ACM MM 2025. [Link](https://arxiv.org/abs/2505.14535)

### Event Camera + RGB Fusion Papers

8. **SFDNet (2025)** - "Efficient Spiking Neural Network for RGB-Event Fusion-Based Object Detection" - MDPI Electronics. [Link](https://www.mdpi.com/2079-9292/14/6/1105)

9. **DSF-Net (2025)** - "Dynamic Sparse Fusion of Event-RGB via Spike-Triggered Attention for High-Speed Detection" - ACM MM 2025. [Link](https://dl.acm.org/doi/10.1145/3746027.3755846)

10. **SpikeFET (2025)** - "Fully Spiking Neural Networks for Unified Frame-Event Object Tracking" - arXiv. [Link](https://arxiv.org/pdf/2505.20834)

### Sensor Fusion Papers

11. **Lopez-Osorio et al. (2024)** - "A Neuromorphic Vision and Feedback Sensor Fusion Based on SNN for Real-Time Robot Adaption" - Advanced Intelligent Systems. [Link](https://advanced.onlinelibrary.wiley.com/doi/10.1002/aisy.202300646)

12. **Loihi-2 Sensor Fusion (Aug 2024)** - "Accelerating Sensor Fusion in Neuromorphic Computing: A Case Study on Loihi-2" - arXiv. [Link](https://arxiv.org/abs/2408.16096)

13. **Spiking Kalman Fusion (Jan 2026)** - "Spiking Neural-Invariant Kalman Fusion for Accurate Localization Using Low-Cost IMUs" - arXiv. [Link](https://arxiv.org/abs/2601.08248)

### Survey Papers

14. **SNN Multimodal Neuroimaging Review (2025)** - Frontiers in Neuroscience. [Link](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1623497/full)

15. **SNN Frameworks Benchmark (2025)** - "A Practical Tutorial on Spiking Neural Networks" - MDPI. [Link](https://www.mdpi.com/2673-4117/6/11/304)

16. **SNN Sound Review (2024)** - "SNN and Sound: A Comprehensive Review" - PMC. [Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC11362401/)

---

## 7. Could a Simplified Version Work as an Undergrad Project?

### Short Answer: YES -- and a specific blueprint exists.

### The Blueprint: Multimodal Digit Recognition SNN

The Bjorndahl et al. (2024) paper provides an almost exact template:

**Objective**: Classify digits (0-9) using both visual (spiked handwritten digits) and auditory (spiked spoken digits) inputs simultaneously.

**Datasets**:
- **N-MNIST** (visual): 60K training / 10K testing, 34x34 pixels, already in spike format via Tonic library
- **SHD** (auditory): 8,156 training / 2,264 testing, 700 input channels, spoken digits 0-9 in English/German, already in spike format

Both datasets are **freely available**, **small enough to train on a laptop GPU**, and **natively supported by snnTorch and Tonic**.

### Proposed Undergraduate Project Plan

**Phase 1: Foundation (Weeks 1-3)**
- Implement single-modality SNN for N-MNIST classification (following snnTorch Tutorial 7)
- Implement single-modality SNN for SHD classification (following snnTorch SHD example)
- Establish baseline accuracies for each modality independently
- Deliverable: Two working unimodal SNN classifiers

**Phase 2: Multimodal Fusion (Weeks 4-6)**
- Implement late fusion (concatenation) of visual and auditory branches
- Train the combined network end-to-end with surrogate gradient descent
- Compare multimodal accuracy vs. unimodal baselines
- Deliverable: Working multimodal SNN with accuracy improvement over unimodal

**Phase 3: Experiments and Analysis (Weeks 7-10)**
- Compare fusion strategies: early, middle, late concatenation
- Noise robustness experiments: add noise to one modality, measure accuracy degradation
- Missing modality experiments: what happens when audio or visual input is removed?
- Energy efficiency analysis: count synaptic operations (SynOps) for each configuration
- Deliverable: Comprehensive experimental results and analysis

**Phase 4: Novel Contribution (Weeks 11-14)**
Choose ONE of these for novelty:
- (a) Implement simple STDP-based fusion instead of surrogate gradients (biologically plausible)
- (b) Add a confidence-weighted fusion mechanism (weight modalities by their reliability)
- (c) Test with a third modality (e.g., tactile/pressure data from a simple dataset)
- (d) Deploy on SpiNNaker or compare energy vs. equivalent ANN
- (e) Investigate what the network learns: visualize spike patterns and fusion representations

### Why This Is Achievable

1. **Existing template**: Bjorndahl et al. achieved 98.43% -- you know the target
2. **Small datasets**: No need for massive compute
3. **Framework support**: snnTorch has tutorials for both N-MNIST and SHD
4. **Open-source code**: S-CMRL codebase on GitHub provides reference implementation
5. **Clear metrics**: Accuracy, energy (SynOps), robustness to noise
6. **Bounded scope**: 10 classes, simple architectures, well-understood problem

### Why This Has Novelty Potential

1. Bjorndahl et al. (2024) only tested 3 fusion depths -- more systematic comparison is possible
2. No one has studied noise robustness in simple multimodal SNN digit recognition
3. STDP-based multimodal fusion is almost unexplored
4. Missing modality handling in multimodal SNNs is an open problem
5. Comparison of frameworks (snnTorch vs SpikingJelly vs BrainCog) for multimodal tasks is unstudied

---

## 8. What Frameworks Would Support This?

### Primary Recommendation: snnTorch

| Feature | snnTorch | SpikingJelly | BrainCog | Norse | Lava (Intel) |
|---|---|---|---|---|---|
| PyTorch-based | Yes | Yes | Yes | Yes | No (standalone) |
| N-MNIST support | Yes (via Tonic) | Yes (native) | Yes | Via Tonic | Via Lava-DL |
| SHD support | Yes (native) | Via custom | Yes | Via Tonic | Via custom |
| Tutorials | Excellent (8+) | Good | Moderate | Basic | Moderate |
| Documentation | Excellent | Good (some Chinese) | Moderate | Good | Good |
| Community | Large | Large | Medium | Medium | Small |
| Ease of learning | Easiest | Moderate | Moderate | Moderate | Hardest |
| GPU acceleration | PyTorch CUDA | CuPy backend (fastest) | PyTorch CUDA | PyTorch CUDA | Loihi hardware |
| Multimodal examples | SHD + N-MNIST separate | Publications list | S-CMRL built on it | No | Sensor fusion |

### Recommended Stack for Undergraduate Project

```
Framework:     snnTorch 0.9.x (easiest to learn, best tutorials)
Dataset lib:   Tonic 1.x (neuromorphic dataset loader, PyTorch compatible)
Deep learning: PyTorch 2.x
Visualization: matplotlib, seaborn
Hardware:      Single GPU (even laptop GPU is sufficient)
Optional:      SpikingJelly (for speed comparison)
Optional:      SpiNNaker access (if available at university, for deployment)
```

### Key Libraries and Installation

```bash
pip install snntorch
pip install tonic
pip install torch torchvision
pip install spikingjelly  # optional, for comparison
pip install braincog      # optional, for S-CMRL reproduction
```

### Dataset Access

```python
# N-MNIST (visual digits)
import tonic
nmnist_train = tonic.datasets.NMNIST(save_to='./data', train=True)
nmnist_test = tonic.datasets.NMNIST(save_to='./data', train=False)

# SHD (spoken digits)
from snntorch.spikevision import spikedata
shd_train = spikedata.SHD("data/shd", train=True)
shd_test = spikedata.SHD("data/shd", train=False)
```

### Sources
- [snnTorch - GitHub](https://github.com/jeshraghian/snntorch)
- [snnTorch Tutorials](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_7.html)
- [snnTorch SHD Example](https://snntorch.readthedocs.io/en/latest/examples/examples_svision/example_sv_shd.html)
- [SpikingJelly - GitHub](https://github.com/fangwei123456/spikingjelly)
- [Tonic - GitHub](https://github.com/neuromorphs/tonic)
- [BrainCog - GitHub](https://github.com/BrainCog-X/Brain-Cog)
- [SNN Frameworks Overview - Open Neuromorphic](https://open-neuromorphic.org/neuromorphic-computing/software/snn-frameworks/)
- [Lava Framework](https://lava-nc.org/)

---

## Difficulty Assessment: Undergraduate vs PhD Level

### What Makes This UNDERGRADUATE-ACHIEVABLE:

1. **Digit classification is the "Hello World" of ML** -- the problem itself is trivial
2. **Both datasets are small** and pre-processed into spike format
3. **Late fusion is just tensor concatenation** -- no complex architecture needed
4. **Surrogate gradient training** is standard and well-documented
5. **snnTorch tutorials** walk through both datasets step by step
6. **A directly comparable paper exists** (Bjorndahl et al. 2024) to validate against
7. **Training takes hours, not days** -- fast iteration cycle
8. **No special hardware needed** -- runs on any machine with a GPU

### What Would Push It to PhD LEVEL:

1. Designing novel neuron models (like MLIF in MISNet)
2. Developing new attention mechanisms (like CCSSA in S-CMRL)
3. Working on complex datasets (CREMA-D, real-world video)
4. Object detection/segmentation instead of classification
5. Deploying on neuromorphic hardware with custom optimization
6. Theoretical analysis of multimodal spike dynamics
7. Large-scale sensor fusion (LIDAR + camera + IMU + radar)

### Confidence Assessment

| Claim | Confidence |
|---|---|
| Multimodal SNN digit classification is achievable in one semester | HIGH (95%) |
| snnTorch is the best starting framework | HIGH (90%) |
| Late concatenation fusion will work well | HIGH (95%) -- confirmed by Bjorndahl et al. |
| Novel contribution is possible within scope | MEDIUM-HIGH (80%) |
| Project can be completed on laptop GPU | HIGH (90%) |
| The topic will impress examiners | HIGH (85%) -- it is cutting-edge |

---

## Research Gaps and Recommended Follow-ups

### Gaps in This Report

1. Could not determine if Bjorndahl et al. released their code (likely no, but S-CMRL code is available)
2. Exact training times for the multimodal digit classification setup are estimated, not measured
3. Did not find any undergraduate thesis specifically on multimodal SNNs to compare against
4. Limited information on BrainCog's ease of use for beginners

### Recommended Next Steps

1. **Read the Bjorndahl et al. paper in full** (arXiv 2409.00552) -- it is the direct template
2. **Clone and explore the S-CMRL codebase** (https://github.com/Brain-Cog-Lab/S-CMRL) -- reference implementation
3. **Complete snnTorch Tutorials 1-7** -- builds up to neuromorphic dataset handling
4. **Start with single-modality baselines** before attempting fusion
5. **Contact your supervisor** about whether SpiNNaker access is available at your university
