# top SNN / neuromorphic papers from 2024-2025

so i went through a ton of recent SNN papers to figure out where the field is heading and what's actually accessible for an undergrad project. the SNN space has genuinely exploded in the last two years -- spiking transformers are now hitting 85%+ on ImageNet, people are building spiking LLMs, and CVPR went from 3 SNN papers in 2024 to 14 in 2025. that's wild growth.

the good news for me is that a lot of this has open-source code and the frameworks (snnTorch, SpikingJelly) are mature enough that you can actually get stuff running without spending weeks debugging.

---

## the most impactful papers

### tier 1 -- the big ones

| # | Paper | Venue | What it does | Code? |
|---|-------|-------|-------------|-------|
| 1 | **QKFormer: Hierarchical Spiking Transformer using Q-K Attention** | NeurIPS 2024 (Spotlight, top 3%) | First SNN to break 85% top-1 on ImageNet (85.65%). New spike-form Q-K attention with linear complexity. | Yes: [github.com/zhouchenlin2096/QKFormer](https://github.com/zhouchenlin2096/QKFormer) |
| 2 | **Spike-Driven Transformer V2: Meta Spiking Neural Network Architecture** | ICLR 2024 | General transformer-based SNN ("Meta-SpikeFormer") for multiple vision tasks. Only uses sparse addition ops. | Yes: [github.com/BICLab/Spike-Driven-Transformer-V2](https://github.com/BICLab/Spike-Driven-Transformer-V2) |
| 3 | **Training Spiking Neural Networks Using Lessons From Deep Learning** (Eshraghian et al.) | Proceedings of the IEEE 2023, **2024 Best Paper Award** | The tutorial/survey that bridges DL and SNNs. Over 4,500 citations. This is the snnTorch paper. | Yes: [github.com/jeshraghian/snntorch](https://github.com/jeshraghian/snntorch) |
| 4 | **SpikeLM: Towards General Spike-Driven Language Modeling via Elastic Bi-Spiking Mechanisms** | ICML 2024 | First fully spiking mechanism for general language tasks. Novel bi-directional, elastic amplitude/frequency spike encoding. | Yes: [github.com/Xingrun-Xing/SpikeLM](https://github.com/Xingrun-Xing/SpikeLM) |
| 5 | **SpikingResformer: Bridging ResNet and Vision Transformer in Spiking Neural Networks** | CVPR 2024 | Dual Spike Self-Attention (DSSA). Gets 79.40% top-1 on ImageNet with only 4 timesteps. | Yes: [github.com/xyshi2000/SpikingResformer](https://github.com/xyshi2000/SpikingResformer) |
| 6 | **P-SpikeSSM: Harnessing Probabilistic Spiking State Space Models for Long-Range Dependency Tasks** | ICLR 2025 | Bridges SNNs with state space models. Stochastic spike generation via SpikeSampler, parallel computation. SOTA on Long Range Arena for SNNs. | Yes: [github.com/NeuroCompLab-psu/PSpikeSSMs](https://github.com/NeuroCompLab-psu/PSpikeSSMs) |
| 7 | **SpikeLLM: Scaling up Spiking Neural Network to Large Language Models** | ICLR 2025 | Scales SNNs to 7-70B parameter LLMs using saliency-based spiking. 92% decrease in perplexity vs baselines. | Yes (code with paper) |
| 8 | **Temporal Spiking Neural Networks with Synaptic Delay for Graph Reasoning** | ICML 2024 | SNNs with temporal coding + synaptic delay for knowledge graph reasoning. Estimated 20x energy savings. | Yes: [github.com/pkuxmq/GRSNN](https://github.com/pkuxmq/GRSNN) |
| 9 | **Efficient and Effective Time-Series Forecasting with Spiking Neural Networks** | ICML 2024 | Unified SNN framework for time-series forecasting that matches ANN accuracy with energy gains. | Paper with code references |
| 10 | **Advancing Spiking Neural Networks for Sequential Modeling through Central Pattern Generators** | NeurIPS 2024 | Hardware-friendly spike-form positional encoding using CPGs for sequential SNN tasks. | Paper (code links in proceedings) |

### tier 2 -- also worth knowing about

| # | Paper | Venue | What it does | Code? |
|---|-------|-------|-------------|-------|
| 11 | **TS-LIF: A Temporal Segment Spiking Neuron Network for Time Series Forecasting** | ICLR 2025 | Dual-compartment architecture (dendritic + somatic) that captures distinct frequency components. Also robust to missing data. | Yes: [github.com/kkking-kk/TS-LIF](https://github.com/kkking-kk/TS-LIF) |
| 12 | **SpikeGCL: A Graph is Worth 1-bit Spikes** | ICLR 2024 | Graph contrastive learning with SNNs. Shows binary spikes are enough for effective graph representation learning. | Yes: [github.com/EdisonLeeeee/SpikeGCL](https://github.com/EdisonLeeeee/SpikeGCL) |
| 13 | **Brain-Inspired Spiking Neural Networks for Energy-Efficient Object Detection** | CVPR 2025 | SNN-based object detection bridging event-driven vision and practical deployment. | Paper with code |
| 14 | **Continual Learning with Neuromorphic Computing: Foundations, Methods, and Emerging Applications** | arXiv survey, Oct 2024 | Maps the entire neuromorphic continual learning subfield. | Survey (references multiple code repos) |
| 15 | **Learning Long Sequences in Spiking Neural Networks** | Scientific Reports 2024 | SSM-based SNNs outperform Transformers on long-range sequence tasks with fewer parameters. | Paper with code references |

---

## where the field is heading -- main trends

### 1. spiking transformers (hottest area right now)

this is where most of the action is. QKFormer hit 85.65% on ImageNet and SGLFormer got 83.73%. the basic idea is making transformers spike-driven so they can run on neuromorphic hardware while keeping high accuracy. key techniques are spike-form Q-K attention, dual spike self-attention, and spike-driven softmax alternatives.

still ~5-7% below ANN transformer accuracy on ImageNet though. and scaling to bigger datasets/models hasn't really been explored.

### 2. spiking LLMs

this is new and honestly kind of crazy. SpikeLM (ICML 2024) and SpikeLLM (ICLR 2025) are the founding papers. the motivation is obvious -- LLMs consume absurd amounts of energy and spiking versions could theoretically cut that by orders of magnitude. they use elastic bi-spiking mechanisms and saliency-based spiking, plus ANN-to-SNN conversion for transformers.

still early though. performance lags behind ANN LLMs and nobody's tried scaling past 70B.

### 3. SNN + state space models (SSMs/Mamba)

this is a new and exciting intersection. P-SpikeSSM (ICLR 2025) is basically the only major paper so far. SSMs give you linear-time sequence modeling and combining that with event-driven spiking efficiency makes a lot of sense. uses probabilistic spike generation, SpikeSampler layers, SpikeMixer blocks.

very few papers exist here -- tons of room for novel work.

### 4. spiking graph neural networks

growing subfield, now has its own benchmark (SGNNBench). graphs show up everywhere in real data and making them spike-driven means energy-efficient processing. uses synaptic delay for relation encoding and spiking graph contrastive learning.

performance still lags behind standard GNNs on lots of benchmarks though.

### 5. time-series and temporal processing

this was kind of a breakthrough year for SNN time-series work, with multiple top-venue papers. arguably the most natural application domain for SNNs since they're inherently temporal. key techniques include temporal segment neurons, derivative spike encoding, dual-compartment architectures.

limited application to real-world time-series so far (finance, weather, IoT sensor data).

### 6. continual/online learning

natural fit for SNNs because of biological plausibility. edge devices need to learn continuously without catastrophic forgetting, and SNNs can potentially do this well. uses Hebbian learning, sleep-enhanced latent replay, energy-aware spike budgeting.

standardized benchmarks for SNN continual learning are still lacking.

### 7. ANN-to-SNN conversion

this is mature but still actively researched. the focus is shifting to converting transformers specifically. key idea is you can leverage pre-trained ANN models and deploy them on neuromorphic hardware. training-free conversion and precision spiking neurons are the main techniques.

nobody's tried converting Mamba/SSM models or mixture-of-experts architectures yet.

### 8. object detection and dense prediction

growing fast -- CVPR 2025 had papers on SNN object detection. spiking-YOLO variants, event-camera fusion, spiking U-Net for segmentation. but SNN performance on COCO-level benchmarks is still way below ANN counterparts.

---

## papers with open-source code that i could actually use

### tier A -- most accessible for an undergrad project

| Paper | Framework | GitHub Stars | Difficulty | What you can build on |
|-------|-----------|-------------|------------|----------------------|
| **snnTorch** (Eshraghian) | PyTorch | 2,900+ | Beginner-friendly | Classification, tutorials, tons of examples |
| **SpikingJelly** | PyTorch | 3,500+ | Beginner to intermediate | Full SNN pipeline: datasets, training, deployment |
| **QKFormer** | SpikingJelly/PyTorch | Active | Intermediate | Image classification with spiking transformers |
| **Spike-Driven Transformer V2** | PyTorch | 200+ | Intermediate | Multi-task vision with spiking transformers |
| **SpikingResformer** | PyTorch | Active | Intermediate | Hybrid ResNet-Transformer SNN architecture |
| **P-SpikeSSM** | PyTorch | Active | Intermediate-Advanced | Long-range sequence tasks with spiking SSMs |
| **GRSNN** (Graph Reasoning) | PyTorch/TorchDrug | Active | Intermediate | Graph reasoning with temporal spiking |
| **SpikeGCL** | PyTorch | Active | Intermediate | Graph contrastive learning with spikes |
| **SpikeLM** | PyTorch | Active | Advanced | Spiking language models |
| **TS-LIF** | PyTorch | Active | Intermediate | Time-series forecasting with SNNs |

### tier B -- useful frameworks and resource collections

| Resource | What it is | Link |
|----------|-----------|------|
| **snnTorch Tutorials** | Interactive Jupyter notebooks for learning SNN training with backprop | [github.com/snntorch/Spiking-Neural-Networks-Tutorials](https://github.com/snntorch/Spiking-Neural-Networks-Tutorials) |
| **SpikingJelly** | Full-stack SNN framework with neuromorphic dataset support | [github.com/fangwei123456/spikingjelly](https://github.com/fangwei123456/spikingjelly) |
| **Intel Lava** | Open-source framework for neuromorphic computing (works with Loihi) | [github.com/lava-nc/lava](https://github.com/lava-nc/lava) |
| **Norse** | Bio-inspired neural components for PyTorch | [github.com/norse/norse](https://github.com/norse/norse) |
| **Awesome-SNN-Conference-Paper** | Curated list of all SNN papers from top conferences with code links | [github.com/AXYZdong/awesome-snn-conference-paper](https://github.com/AXYZdong/awesome-snn-conference-paper) |
| **Awesome-Spiking-Neural-Networks** | Paper list with code and websites | [github.com/TheBrainLab/Awesome-Spiking-Neural-Networks](https://github.com/TheBrainLab/Awesome-Spiking-Neural-Networks) |
| **SGNNBench** | Benchmark for spiking graph neural networks | [github.com/Zhhuizhe/SGNNBench](https://github.com/Zhhuizhe/SGNNBench) |
| **Open Neuromorphic** | Community hub for neuromorphic software projects | [open-neuromorphic.org](https://open-neuromorphic.org) |

---

## conference papers by venue (2024-2025)

### NeurIPS 2024 (23 SNN papers)

1. QKFormer: Hierarchical Spiking Transformer using Q-K Attention **(Spotlight)**
2. Latent Diffusion for Neural Spiking Data
3. Autonomous Driving with Spiking Neural Networks
4. Spiking Graph Neural Network on Riemannian Manifolds
5. SpikeReveal: Unlocking Temporal Sequences from Real Blurry Inputs with Spike Streams
6. Advancing SNNs for Sequential Modeling through Central Pattern Generators
7. Take A Shortcut Back: Mitigating the Gradient Vanishing for Training SNNs
8. FEEL-SNN: Robust SNNs with Frequency Encoding and Evolutionary Leak Factor
9. Spiking Neural Network as Adaptive Event Stream Slicer
10. Spiking Token Mixer

### ICLR 2024 (~10 SNN papers)

1. Spike-driven Transformer V2: Meta Spiking Neural Network Architecture
2. SpikeGCL: A Graph is Worth 1-bit Spikes
3. SpikePoint: Efficient Point-based SNN for Event Cameras Action Recognition
4. LMUFormer: Low Complexity Spiking Model With Legendre Memory Units
5. Online Stabilization of Spiking Neural Networks
6. Towards Energy Efficient SNNs: An Unstructured Pruning Framework
7. Can we get the best of both Binary Neural Networks and SNNs?
8. Spatio-Temporal Approximation: Training-Free SNN Conversion for Transformers
9. Hebbian Learning based Orthogonal Projection for Continual Learning of SNNs
10. Adaptive deep SNN with global-local learning via balanced excitatory/inhibitory mechanism

### ICLR 2025 (11+ SNN papers)

1. SpikeLLM: Scaling up SNN to Large Language Models via Saliency-based Spiking
2. P-SpikeSSM: Probabilistic Spiking State Space Models for Long-Range Tasks
3. Quantized Spike-driven Transformer
4. TS-LIF: Temporal Segment Spiking Neuron Network for Time Series Forecasting
5. Spiking Vision Transformer with Saccadic Attention
6. DeepTAGE: Deep Temporal-Aligned Gradient Enhancement for Optimizing SNNs
7. SPAM: Spike-Aware Adam with Momentum Reset for Stable LLM Training
8. Rethinking Spiking Neural Networks from an Ensemble Learning Perspective

### ICML 2024 (3+ SNN papers)

1. SpikeLM: Towards General Spike-Driven Language Modeling via Elastic Bi-Spiking
2. Temporal Spiking Neural Networks with Synaptic Delay for Graph Reasoning
3. Efficient and Effective Time-Series Forecasting with Spiking Neural Networks

### CVPR 2024 (3 SNN papers)

1. SpikingResformer: Bridging ResNet and Vision Transformer in SNNs
2. SFOD: Spiking Fusion Object Detector
