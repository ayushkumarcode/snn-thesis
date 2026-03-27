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
3. Are Conventional SNNs Really Efficient? A Perspective from Network Quantization

### CVPR 2025 (14 SNN papers -- massive jump!)

1. Brain-Inspired SNNs for Energy-Efficient Object Detection
2. Spiking Transformer: Accurate Addition-Only Spiking Self-Attention
3. STAA-SNN: Spatial-Temporal Attention Aggregator for SNNs
4. USP-Gaussian: Unifying Spike-based Image Reconstruction, Pose Correction and Gaussian Splatting
5. Spiking Transformer with Spatial-Temporal Attention
6. Spk2SRImgNet: Super-Resolve Dynamic Scene from Spike Stream

### ECCV 2024
- DailyDVS-200: A Comprehensive Benchmark Dataset for Event-Based Action Recognition

### AAAI 2025
- SpikingYOLOX: Improved YOLOX with FFT convolution and SNNs
- FSTA-SNN: Frequency-based Spatial-Temporal Attention Module for SNNs

---

## new benchmarks and datasets (2024-2025)

### new datasets

| Dataset | Year | What it is | Size | Application |
|---------|------|-----------|------|-------------|
| **DailyDVS-200** | ECCV 2024 | Event-based action recognition. 200 action categories, 47 participants, 22,000+ event sequences, 14 attributes per sequence. | Large | Action recognition |
| **SGNNBench** | 2025 | Benchmark for spiking GNNs on large-scale graphs. 9 SGNN baselines vs 4 classic GNNs. | Multiple graph datasets | Graph learning |
| **eTraM** | 2024 | Event camera traffic dataset. 10+ hours of event data, 2M+ bounding boxes, varied weather/lighting. | Large | Object detection |
| **EvDET200K** | 2024 | Large-scale event-based detection dataset. | 200K+ annotations | Object detection |
| **NYC-Event-VPR** | 2024 | Event-based visual place recognition in urban environments. | Large | Place recognition |

### new benchmarks/evaluation frameworks

| Benchmark | Year | What it tests | Key finding |
|-----------|------|--------------|-------------|
| **Multimodal SNN Framework Benchmark** | 2025 | Evaluates SpikingJelly, BrainCog, Sinabs, SNNGrow, Lava across image, text, and neuromorphic data | SpikingJelly leads in energy efficiency; BrainCog strong on complex tasks |
| **Long Range Arena for SNNs** | 2024-2025 | Tests SNN long-range dependency handling | SSM-based SNNs (P-SpikeSSM) now SOTA for SNNs |
| **SNN Framework Benchmark** (Open Neuromorphic) | 2024 | Performance comparison of SNN libraries | SpikingJelly fastest; snnTorch most accessible |

---

## "low-hanging fruit" research directions

i went through the future work sections of a bunch of these papers and here's what seems doable, roughly sorted by difficulty.

### easy -- can be done with existing frameworks, clear methodology

1. **apply spiking transformers to new domains** -- QKFormer and Spike-Driven Transformer V2 are built for image classification. adapting them to medical imaging, satellite imagery, or agricultural data hasn't been done. just fine-tuning on domain-specific datasets could be publishable. (from QKFormer and SpikingResformer future work sections)

2. **SNN time-series forecasting on new application domains** -- TS-LIF and the ICML 2024 time-series paper use standard forecasting benchmarks. applying to IoT sensor data, financial markets, or energy grid prediction with real-world data is an open gap. (from TS-LIF ICLR 2025 and ICML 2024 time-series paper)

3. **benchmark SNN frameworks on event-camera datasets** -- DailyDVS-200 (ECCV 2024) is new. running existing SNN models on it and comparing performance is straightforward. (from DailyDVS-200 paper, SGNNBench paper)

4. **SNN continual learning with new replay strategies** -- SESLR paper shows binary spike features reduce memory for continual learning. testing different replay buffer strategies (reservoir sampling, surprise-based selection) is a fairly easy extension. (from SESLR 2025, continual learning survey)

5. **compare snnTorch vs SpikingJelly vs Norse on identical tasks** -- the 2025 multimodal benchmark compared 5 frameworks but didn't deeply analyze training curves, hyperparameter sensitivity, or user experience. a focused comparison study is doable. (from SNN Framework Benchmarks 2024-2025)

### moderate -- needs some novel implementation but builds on existing code

6. **spiking neural architecture search (NAS)** -- a recent survey (arXiv:2510.14235) maps the SNN NAS landscape but identifies many untried search spaces. adapting existing ANN NAS methods (DARTS, etc.) for SNNs is tractable.

7. **SNN for audio/speech with event-driven encoding** -- Spiking-LEAF proposes a learnable auditory front-end. combining with recent architectures (e.g., P-SpikeSSM) for speech command recognition is a clear next step.

8. **ANN-to-SNN conversion for modern architectures** -- converting Mamba/SSM models or mixture-of-experts architectures to SNNs is completely unexplored. even a negative result showing what fails would be publishable.

9. **pruning and compression of spiking transformers** -- pruning has been studied for CNN-based SNNs, but pruning the new spiking transformers (QKFormer, SpikingResformer) hasn't been tried. (from QP-SNN ICLR 2025, various pruning papers)

10. **SNN + reinforcement learning for simple robotics tasks** -- recent papers show SNN-based RL for locomotion and navigation. applying to simulated robotic manipulation (OpenAI Gym or MuJoCo) with SNNs is feasible with snnTorch. (from Zanatta et al. 2024, Kumar et al. 2025)

### advanced -- novel research contributions, but feasible for a strong undergrad

11. **spiking state space models for new sequence tasks** -- P-SpikeSSM opened this area. applying spiking SSMs to genomics, protein sequences, or music generation would be novel.

12. **multi-modal spiking fusion (event camera + RGB)** -- most SNN papers use either event data or RGB frames. fusing both in a spiking architecture is an active gap. (from SFOD CVPR 2024, neuromorphic vision surveys)

13. **energy-aware training/inference for edge deployment** -- measuring actual energy consumption of SNN models on real neuromorphic hardware (Intel Loihi 2 via Lava) versus GPU simulation is valuable empirical work.

14. **spiking diffusion models** -- the NeurIPS 2024 paper on "Latent Diffusion for Neural Spiking Data" opens a new direction. adapting diffusion models to operate with spiking dynamics for image generation is largely unexplored.

---

## confidence notes

things i'm pretty sure about:
- QKFormer achieving 85.65% on ImageNet (NeurIPS 2024 Spotlight)
- all the GitHub repos listed here exist and are accessible
- the explosion of SNN papers at CVPR (3 in 2024 to 14 in 2025)
- SpikeLM being the first fully spiking language model at ICML 2024
- P-SpikeSSM bridging SNNs and SSMs at ICLR 2025

things i'm less certain about:
- exact citation counts (these change all the time, rankings are approximate)
- SpikeLLM's exact performance numbers across all benchmarks
- whether i got every NeurIPS 2024 SNN paper (found 23 but there might be more)

couldn't fully verify:
- complete list of ICML 2025 SNN papers (might be after submission deadline)
- whether BrainTransformers-3B has publicly released code
- exact GitHub star counts (fluctuate daily obviously)

---

## what i should probably do next

1. **start with snnTorch tutorials** -- the interactive Jupyter notebooks at [github.com/snntorch/Spiking-Neural-Networks-Tutorials](https://github.com/snntorch/Spiking-Neural-Networks-Tutorials) are the best entry point. tutorials 1-7 build the foundation.

2. **clone and run QKFormer or SpikingResformer** -- most accessible codebases for image classification. start with CIFAR-10/100 (faster), then scale to ImageNet subsets.

3. **explore the Awesome-SNN-Conference-Paper repo** -- [axyzdong.github.io/awesome-snn-conference-paper/](https://axyzdong.github.io/awesome-snn-conference-paper/) has the most up-to-date paper listings by venue.

4. **pick a low-hanging fruit direction** -- items 1-5 from the easy list above are probably most suitable for an undergrad thesis. they're about applying existing methods to new domains/datasets rather than inventing new architectures.
