# Top SNN/Neuromorphic Computing Papers 2024-2025: Comprehensive Research Report

**Date compiled:** 2026-02-25
**Scope:** Most impactful SNN papers from 2024-2025, trends, open-source code, conference papers, benchmarks, and low-hanging fruit research directions.

---

## 1. Executive Summary

The SNN field experienced a significant acceleration in 2024-2025, with three dominant trends: (1) **Spiking Transformers** achieving ImageNet accuracy above 85% for the first time, closing the gap with ANNs; (2) **SNNs scaling to language models**, with SpikeLLM and SpikeLM demonstrating that spiking architectures can handle 7-70B parameter LLMs; and (3) **new application domains** including graph reasoning, time-series forecasting, and continual learning becoming mature research areas with conference-level papers. The number of SNN papers at CVPR alone jumped from 3 (2024) to 14 (2025), indicating explosive growth in the field.

Key takeaway for an undergraduate thesis: The field is ripe with accessible research directions. Many top papers have open-source code, the frameworks (snnTorch, SpikingJelly) are mature and well-documented, and several "gap-filling" problems remain unaddressed.

---

## 2. Top 10-15 Most Cited/Influential SNN Papers (2024-2025)

### Tier 1: Highest Impact Papers

| # | Paper | Venue | Key Contribution | Code Available? |
|---|-------|-------|-----------------|-----------------|
| 1 | **QKFormer: Hierarchical Spiking Transformer using Q-K Attention** | NeurIPS 2024 (Spotlight, top 3%) | First SNN to exceed 85% top-1 accuracy on ImageNet-1k (85.65%). Novel spike-form Q-K attention with linear complexity. | Yes: [github.com/zhouchenlin2096/QKFormer](https://github.com/zhouchenlin2096/QKFormer) |
| 2 | **Spike-Driven Transformer V2: Meta Spiking Neural Network Architecture** | ICLR 2024 | General Transformer-based SNN ("Meta-SpikeFormer") for multiple vision tasks. Supports spike-driven paradigm with only sparse addition operations. | Yes: [github.com/BICLab/Spike-Driven-Transformer-V2](https://github.com/BICLab/Spike-Driven-Transformer-V2) |
| 3 | **Training Spiking Neural Networks Using Lessons From Deep Learning** (Eshraghian et al.) | Proceedings of the IEEE 2023, **2024 Best Paper Award** | Comprehensive tutorial/survey bridging deep learning and SNNs. Over 4,500 citations. Companion tool: snnTorch. | Yes: [github.com/jeshraghian/snntorch](https://github.com/jeshraghian/snntorch) |
| 4 | **SpikeLM: Towards General Spike-Driven Language Modeling via Elastic Bi-Spiking Mechanisms** | ICML 2024 | First fully spiking mechanism for general language tasks (discriminative and generative). Novel bi-directional, elastic amplitude/frequency spike encoding. | Yes: [github.com/Xingrun-Xing/SpikeLM](https://github.com/Xingrun-Xing/SpikeLM) |
| 5 | **SpikingResformer: Bridging ResNet and Vision Transformer in Spiking Neural Networks** | CVPR 2024 | Novel Dual Spike Self-Attention (DSSA). Achieves 79.40% top-1 on ImageNet with 4 timesteps. | Yes: [github.com/xyshi2000/SpikingResformer](https://github.com/xyshi2000/SpikingResformer) |
| 6 | **P-SpikeSSM: Harnessing Probabilistic Spiking State Space Models for Long-Range Dependency Tasks** | ICLR 2025 | Bridges SNNs with state space models (SSMs). Stochastic spike generation via SpikeSampler while allowing parallel computation. SOTA on Long Range Arena benchmark for SNNs. | Yes: [github.com/NeuroCompLab-psu/PSpikeSSMs](https://github.com/NeuroCompLab-psu/PSpikeSSMs) |
| 7 | **SpikeLLM: Scaling up Spiking Neural Network to Large Language Models** | ICLR 2025 | Scales SNNs to 7-70B parameter LLMs using saliency-based spiking. 92% decrease in perplexity compared to baselines. | Yes (code with paper) |
| 8 | **Temporal Spiking Neural Networks with Synaptic Delay for Graph Reasoning** | ICML 2024 | SNNs with temporal coding + synaptic delay for knowledge graph reasoning. Estimated 20x energy savings over non-spiking models. | Yes: [github.com/pkuxmq/GRSNN](https://github.com/pkuxmq/GRSNN) |
| 9 | **Efficient and Effective Time-Series Forecasting with Spiking Neural Networks** | ICML 2024 | Unified SNN framework for time-series forecasting matching ANN accuracy with substantial energy gains. | Paper with code references |
| 10 | **Advancing Spiking Neural Networks for Sequential Modeling through Central Pattern Generators** | NeurIPS 2024 | Hardware-friendly spike-form positional encoding using CPGs for sequential SNN tasks. | Paper (code links in proceedings) |

### Tier 2: Highly Notable Papers

| # | Paper | Venue | Key Contribution | Code Available? |
|---|-------|-------|-----------------|-----------------|
| 11 | **TS-LIF: A Temporal Segment Spiking Neuron Network for Time Series Forecasting** | ICLR 2025 | Dual-compartment architecture (dendritic + somatic) capturing distinct frequency components. Outperforms traditional SNNs with missing data robustness. | Yes: [github.com/kkking-kk/TS-LIF](https://github.com/kkking-kk/TS-LIF) |
| 12 | **SpikeGCL: A Graph is Worth 1-bit Spikes** | ICLR 2024 | Graph contrastive learning with spiking neural networks. Shows binary spikes suffice for effective graph representation learning. | Yes: [github.com/EdisonLeeeee/SpikeGCL](https://github.com/EdisonLeeeee/SpikeGCL) |
| 13 | **Brain-Inspired Spiking Neural Networks for Energy-Efficient Object Detection** | CVPR 2025 | SNN-based object detection bridging event-driven vision and practical deployment. | Paper with code |
| 14 | **Continual Learning with Neuromorphic Computing: Foundations, Methods, and Emerging Applications** | arXiv survey, Oct 2024 | Comprehensive survey on Neuromorphic Continual Learning (NCL). Maps the entire subfield. | Survey (references multiple code repos) |
| 15 | **Learning Long Sequences in Spiking Neural Networks** | Scientific Reports 2024 | SSM-based SNNs outperform Transformers on long-range sequence tasks with fewer parameters. | Paper with code references |

---

## 3. Key Trends and Hot Research Problems

### Trend 1: Spiking Transformers (HOTTEST AREA)
- **Status:** Rapidly maturing. QKFormer (85.65% ImageNet) and SGLFormer (83.73% ImageNet) represent the current frontier.
- **Why it is hot:** Transformers dominate deep learning; making them spike-driven enables neuromorphic deployment while maintaining high accuracy.
- **Key techniques:** Spike-form Q-K attention, dual spike self-attention (DSSA), spike-driven softmax alternatives.
- **Gap:** Still ~5-7% below ANN Transformer accuracy on ImageNet. Scaling to larger datasets/models is underexplored.

### Trend 2: SNNs for Large Language Models
- **Status:** Emerging and rapidly evolving. SpikeLM (ICML 2024) and SpikeLLM (ICLR 2025) are the founding works.
- **Why it is hot:** LLMs consume enormous energy. Spiking LLMs promise orders-of-magnitude energy reduction.
- **Key techniques:** Elastic bi-spiking mechanisms, saliency-based spiking, ANN-to-SNN conversion for Transformers.
- **Gap:** Still early. Performance lags behind ANN LLMs on many benchmarks. Scaling beyond 70B is unexplored.

### Trend 3: SNN + State Space Models (SSMs/Mamba)
- **Status:** New and exciting intersection. P-SpikeSSM (ICLR 2025) is the flagship paper.
- **Why it is hot:** SSMs offer linear-time sequence modeling, and spiking SSMs combine this with event-driven efficiency.
- **Key techniques:** Probabilistic spike generation, SpikeSampler layers, SpikeMixer blocks.
- **Gap:** Very few papers exist. Enormous room for novel contributions.

### Trend 4: Spiking Graph Neural Networks
- **Status:** Growing subfield with its own benchmark (SGNNBench).
- **Why it is hot:** Graphs are ubiquitous in real-world data. Spiking GNNs offer energy-efficient processing.
- **Key techniques:** Synaptic delay for relation encoding, spiking graph contrastive learning.
- **Gap:** Performance still significantly lags behind standard GNNs on many benchmarks.

### Trend 5: Time-Series and Temporal Processing
- **Status:** Breakthrough year in 2024-2025 with multiple top-venue papers.
- **Why it is hot:** SNNs are inherently temporal; this is arguably their most natural application domain.
- **Key techniques:** Temporal segment neurons, derivative spike encoding, dual-compartment architectures.
- **Gap:** Limited application to real-world time-series (finance, weather, IoT sensor data).

### Trend 6: Continual/Online Learning
- **Status:** Natural fit for SNNs due to biological plausibility. Active research area.
- **Why it is hot:** Edge devices need to learn continuously without catastrophic forgetting.
- **Key techniques:** Hebbian learning, sleep-enhanced latent replay, energy-aware spike budgeting.
- **Gap:** Standardized benchmarks for SNN continual learning are lacking.

### Trend 7: ANN-to-SNN Conversion
- **Status:** Mature but still actively researched. Focus shifting to converting Transformers.
- **Why it is hot:** Allows leveraging pre-trained ANN models on neuromorphic hardware.
- **Key techniques:** Training-free conversion, precision spiking neurons, SpikedAttention for Transformers.
- **Gap:** Conversion of modern architectures (Mamba, mixture-of-experts) is unexplored.

### Trend 8: Object Detection and Dense Prediction
- **Status:** Growing rapidly. CVPR 2025 had papers on SNN object detection.
- **Key techniques:** Spiking-YOLO variants, event-camera fusion, spiking U-Net for segmentation.
- **Gap:** SNN performance on COCO-level benchmarks still far below ANN counterparts.

---

## 4. Papers with Open-Source Code (Undergraduate-Accessible)

### Tier A: Best for Undergrad Projects (Well-documented, active repos, clear instructions)

| Paper | Framework Used | GitHub Stars | Difficulty Level | What You Can Build On |
|-------|---------------|-------------|-----------------|----------------------|
| **snnTorch** (Eshraghian) | PyTorch | 2,900+ | Beginner-friendly | Classification, tutorials, many examples |
| **SpikingJelly** | PyTorch | 3,500+ | Beginner to intermediate | Full SNN pipeline: datasets, training, deployment |
| **QKFormer** | SpikingJelly/PyTorch | Active | Intermediate | Image classification with spiking Transformers |
| **Spike-Driven Transformer V2** | PyTorch | 200+ | Intermediate | Multi-task vision with spiking Transformers |
| **SpikingResformer** | PyTorch | Active | Intermediate | Hybrid ResNet-Transformer SNN architecture |
| **P-SpikeSSM** | PyTorch | Active | Intermediate-Advanced | Long-range sequence tasks with spiking SSMs |
| **GRSNN** (Graph Reasoning) | PyTorch/TorchDrug | Active | Intermediate | Graph reasoning with temporal spiking |
| **SpikeGCL** | PyTorch | Active | Intermediate | Graph contrastive learning with spikes |
| **SpikeLM** | PyTorch | Active | Advanced | Spiking language models |
| **TS-LIF** | PyTorch | Active | Intermediate | Time-series forecasting with SNNs |

### Tier B: Useful SNN Frameworks and Collections

| Resource | Description | Link |
|----------|-------------|------|
| **snnTorch Tutorials** | Interactive Jupyter notebooks for learning SNN training with backprop | [github.com/snntorch/Spiking-Neural-Networks-Tutorials](https://github.com/snntorch/Spiking-Neural-Networks-Tutorials) |
| **SpikingJelly** | Full-stack SNN framework with neuromorphic dataset support | [github.com/fangwei123456/spikingjelly](https://github.com/fangwei123456/spikingjelly) |
| **Intel Lava** | Open-source framework for neuromorphic computing (works with Loihi) | [github.com/lava-nc/lava](https://github.com/lava-nc/lava) |
| **Norse** | Bio-inspired neural components for PyTorch | [github.com/norse/norse](https://github.com/norse/norse) |
| **Awesome-SNN-Conference-Paper** | Curated list of all SNN papers from top conferences with code links | [github.com/AXYZdong/awesome-snn-conference-paper](https://github.com/AXYZdong/awesome-snn-conference-paper) |
| **Awesome-Spiking-Neural-Networks** | Comprehensive paper list with code and websites | [github.com/TheBrainLab/Awesome-Spiking-Neural-Networks](https://github.com/TheBrainLab/Awesome-Spiking-Neural-Networks) |
| **SGNNBench** | Benchmark for spiking graph neural networks | [github.com/Zhhuizhe/SGNNBench](https://github.com/Zhhuizhe/SGNNBench) |
| **Open Neuromorphic** | Community hub for neuromorphic software projects | [open-neuromorphic.org](https://open-neuromorphic.org) |

---

## 5. Top Conference Papers by Venue (2024-2025)

### NeurIPS 2024 (23 SNN papers)
Key papers:
1. QKFormer: Hierarchical Spiking Transformer using Q-K Attention **(Spotlight)**
2. Latent Diffusion for Neural Spiking Data
3. Autonomous Driving with Spiking Neural Networks
4. Spiking Graph Neural Network on Riemannian Manifolds
5. SpikeReveal: Unlocking Temporal Sequences from Real Blurry Inputs with Spike Streams
6. Advancing Spiking Neural Networks for Sequential Modeling through Central Pattern Generators
7. Take A Shortcut Back: Mitigating the Gradient Vanishing for Training SNNs
8. FEEL-SNN: Robust SNNs with Frequency Encoding and Evolutionary Leak Factor
9. Spiking Neural Network as Adaptive Event Stream Slicer
10. Spiking Token Mixer

### ICLR 2024 (~10 SNN papers)
Key papers:
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
Key papers:
1. SpikeLLM: Scaling up SNN to Large Language Models via Saliency-based Spiking
2. P-SpikeSSM: Probabilistic Spiking State Space Models for Long-Range Tasks
3. Quantized Spike-driven Transformer
4. TS-LIF: Temporal Segment Spiking Neuron Network for Time Series Forecasting
5. Spiking Vision Transformer with Saccadic Attention
6. DeepTAGE: Deep Temporal-Aligned Gradient Enhancement for Optimizing SNNs
7. SPAM: Spike-Aware Adam with Momentum Reset for Stable LLM Training
8. Rethinking Spiking Neural Networks from an Ensemble Learning Perspective

### ICML 2024 (3+ SNN papers)
Key papers:
1. SpikeLM: Towards General Spike-Driven Language Modeling via Elastic Bi-Spiking
2. Temporal Spiking Neural Networks with Synaptic Delay for Graph Reasoning
3. Efficient and Effective Time-Series Forecasting with Spiking Neural Networks

### CVPR 2024 (3 SNN papers)
1. SpikingResformer: Bridging ResNet and Vision Transformer in SNNs
2. SFOD: Spiking Fusion Object Detector
3. Are Conventional SNNs Really Efficient? A Perspective from Network Quantization

### CVPR 2025 (14 SNN papers - massive increase!)
Key papers:
1. Brain-Inspired Spiking Neural Networks for Energy-Efficient Object Detection
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

## 6. New Benchmarks and Datasets (2024-2025)

### New Datasets

| Dataset | Year | Description | Size | Application |
|---------|------|-------------|------|-------------|
| **DailyDVS-200** | ECCV 2024 | Event-based action recognition. 200 action categories, 47 participants, 22,000+ event sequences, 14 attributes per sequence. | Large | Action recognition |
| **SGNNBench** | 2025 | Holistic benchmark for spiking graph neural networks on large-scale graphs. 9 SGNN baselines vs. 4 classic GNNs. | Multiple graph datasets | Graph learning |
| **eTraM** | 2024 | Event camera traffic dataset. 10+ hours of event data, 2M+ bounding boxes, varied weather/lighting. | Large | Object detection |
| **EvDET200K** | 2024 | Large-scale event-based detection dataset. | 200K+ annotations | Object detection |
| **NYC-Event-VPR** | 2024 | Event-based visual place recognition in urban environments. | Large | Place recognition |

### New Benchmarks/Evaluation Frameworks

| Benchmark | Year | What It Tests | Key Finding |
|-----------|------|---------------|-------------|
| **Comprehensive Multimodal SNN Framework Benchmark** | 2025 | Evaluates SpikingJelly, BrainCog, Sinabs, SNNGrow, Lava across image, text, and neuromorphic data | SpikingJelly leads in energy efficiency; BrainCog strong on complex tasks |
| **Long Range Arena for SNNs** | 2024-2025 | Tests SNN long-range dependency handling | SSM-based SNNs (P-SpikeSSM) now SOTA for SNNs |
| **SNN Framework Benchmark** (Open Neuromorphic) | 2024 | Performance comparison of SNN libraries | SpikingJelly fastest; snnTorch most accessible |

---

## 7. "Low-Hanging Fruit" Research Directions

These are research gaps explicitly or implicitly suggested in the future work sections of recent top papers, ranked by accessibility for an undergraduate:

### Accessibility Level: EASY (Can be done with existing frameworks, clear methodology)

1. **Apply spiking Transformers to new domains**
   - QKFormer and Spike-Driven Transformer V2 are designed for image classification. Adapting them to medical imaging, satellite imagery, or agricultural data has not been done. Simply fine-tuning these models on domain-specific datasets is publishable.
   - *Source:* QKFormer and SpikingResformer future work sections.

2. **SNN time-series forecasting on new application domains**
   - TS-LIF and the ICML 2024 time-series paper demonstrate SNNs for standard forecasting benchmarks. Applying these to IoT sensor data, financial markets, or energy grid prediction with real-world data is an open gap.
   - *Source:* TS-LIF (ICLR 2025) and ICML 2024 time-series paper.

3. **Benchmark SNN frameworks on event-camera datasets**
   - The DailyDVS-200 dataset (ECCV 2024) is new. Running existing SNN models on it and comparing performance is straightforward and publishable.
   - *Source:* DailyDVS-200 paper, SGNNBench paper.

4. **SNN continual learning with new replay strategies**
   - The SESLR paper shows that binary spike features reduce memory for continual learning. Testing different replay buffer strategies (reservoir sampling, surprise-based selection) is an easy extension.
   - *Source:* SESLR (2025), continual learning survey.

5. **Compare snnTorch vs. SpikingJelly vs. Norse on identical tasks**
   - The 2025 multimodal benchmark compared 5 frameworks but did not deeply analyze training curves, hyperparameter sensitivity, or user experience. A focused comparison study is accessible.
   - *Source:* SNN Framework Benchmarks (2024-2025).

### Accessibility Level: MODERATE (Requires some novel implementation but builds on existing code)

6. **Spiking Neural Architecture Search (NAS)**
   - A recent survey (arXiv:2510.14235) maps the SNN NAS landscape but identifies many untried search spaces and strategies. Adapting existing ANN NAS methods (DARTS, etc.) for SNNs is tractable.
   - *Source:* SNN Architecture Search Survey (2024).

7. **SNN for audio/speech with event-driven encoding**
   - Spiking-LEAF proposes a learnable auditory front-end. Combining this with recent SNN architectures (e.g., P-SpikeSSM) for speech command recognition is a clear next step.
   - *Source:* Spiking-LEAF paper, P-SpikeSSM paper.

8. **ANN-to-SNN conversion for modern architectures**
   - Converting Mamba/SSM models or mixture-of-experts architectures to SNNs is completely unexplored. Even a negative result (showing what fails) would be publishable.
   - *Source:* ANN-to-SNN conversion survey, training-free conversion papers.

9. **Pruning and compression of spiking Transformers**
   - While pruning has been studied for CNN-based SNNs, pruning the new spiking Transformers (QKFormer, SpikingResformer) is unexplored.
   - *Source:* QP-SNN (ICLR 2025), various pruning papers.

10. **SNN + reinforcement learning for simple robotics tasks**
    - Recent papers show SNN-based RL for locomotion and navigation. Applying this to simulated robotic manipulation (e.g., OpenAI Gym or MuJoCo environments) with SNNs is feasible with frameworks like snnTorch.
    - *Source:* Zanatta et al. (2024), Kumar et al. (2025).

### Accessibility Level: ADVANCED (Novel research contributions, but feasible for a strong undergrad)

11. **Spiking state space models for new sequence tasks**
    - P-SpikeSSM opened this area. Applying spiking SSMs to genomics, protein sequences, or music generation would be novel.
    - *Source:* P-SpikeSSM (ICLR 2025).

12. **Multi-modal spiking fusion (event camera + RGB)**
    - Most SNN papers use either event data or RGB frames. Fusing both modalities in a spiking architecture is an active gap.
    - *Source:* SFOD (CVPR 2024), neuromorphic vision surveys.

13. **Energy-aware training/inference for edge deployment**
    - Measuring actual energy consumption of SNN models on real neuromorphic hardware (Intel Loihi 2, via Lava framework) versus GPU simulation is valuable empirical work.
    - *Source:* Neuromorphic hardware surveys, energy-aware spike budgeting paper.

14. **Spiking diffusion models**
    - The NeurIPS 2024 paper on "Latent Diffusion for Neural Spiking Data" opens a new direction. Adapting diffusion models to operate with spiking dynamics for image generation is largely unexplored.
    - *Source:* NeurIPS 2024 latent diffusion paper.

---

## 8. Research Gaps and Confidence Assessment

### What I Am Highly Confident About (verified across multiple sources):
- QKFormer achieving 85.65% on ImageNet (NeurIPS 2024 Spotlight)
- The existence and availability of all GitHub repositories listed
- The explosion of SNN papers at CVPR (3 in 2024 to 14 in 2025)
- SpikeLM being the first fully spiking language model at ICML 2024
- P-SpikeSSM bridging SNNs and SSMs at ICLR 2025

### What I Am Moderately Confident About:
- Exact citation counts (these change rapidly; rankings are approximate)
- SpikeLLM's exact performance numbers on all benchmarks
- The completeness of the NeurIPS 2024 paper list (23 papers found but there may be more)

### What I Could Not Fully Verify:
- Complete list of all ICML 2025 SNN papers (conference may be post-submission as of this date)
- Whether BrainTransformers-3B has publicly released code
- Exact GitHub star counts (these fluctuate daily)

---

## 9. Recommended Follow-Up Actions

1. **Start with snnTorch tutorials**: The interactive Jupyter notebooks at [github.com/snntorch/Spiking-Neural-Networks-Tutorials](https://github.com/snntorch/Spiking-Neural-Networks-Tutorials) are the best entry point. Complete tutorials 1-7 to build foundational understanding.

2. **Clone and run QKFormer or SpikingResformer**: These have the most accessible codebases for image classification tasks. Use CIFAR-10/100 first (faster training), then scale to ImageNet subsets.

3. **Explore the Awesome-SNN-Conference-Paper repo**: Visit [axyzdong.github.io/awesome-snn-conference-paper/](https://axyzdong.github.io/awesome-snn-conference-paper/) for the most up-to-date paper listings organized by venue.

4. **Pick a "low-hanging fruit" direction from Section 7**: Items 1-5 (EASY level) are particularly suitable for an undergraduate thesis. They involve applying existing methods to new domains/datasets rather than inventing new architectures.

5. **Read the key surveys**:
   - "Training Spiking Neural Networks Using Lessons From Deep Learning" (Eshraghian et al.) -- foundational
   - "Toward Large-scale Spiking Neural Networks: A Comprehensive Survey" (2024) -- most recent overview
   - "Continual Learning with Neuromorphic Computing" (Oct 2024) -- if continual learning interests you

---

## 10. Sources and References

### Primary Research Papers
- [QKFormer - NeurIPS 2024](https://arxiv.org/abs/2403.16552)
- [Spike-Driven Transformer V2 - ICLR 2024](https://openreview.net/forum?id=1SIBN5Xyw7)
- [SpikeLM - ICML 2024](https://arxiv.org/abs/2406.03287)
- [SpikeLLM - ICLR 2025](https://arxiv.org/abs/2407.04752)
- [SpikingResformer - CVPR 2024](https://github.com/xyshi2000/SpikingResformer)
- [P-SpikeSSM - ICLR 2025](https://arxiv.org/abs/2406.02923)
- [GRSNN (Graph Reasoning with SNNs) - ICML 2024](https://arxiv.org/abs/2405.16851)
- [SpikeGCL - ICLR 2024](https://github.com/EdisonLeeeee/SpikeGCL)
- [TS-LIF - ICLR 2025](https://github.com/kkking-kk/TS-LIF)
- [DailyDVS-200 - ECCV 2024](https://github.com/QiWang233/DailyDVS-200)
- [SGNNBench](https://github.com/Zhhuizhe/SGNNBench)
- [Training SNNs Using Lessons From Deep Learning - Eshraghian et al.](https://arxiv.org/abs/2109.12894)

### Curated Paper Lists and Community Resources
- [Awesome SNN Conference Paper](https://github.com/AXYZdong/awesome-snn-conference-paper)
- [Awesome Spiking Neural Networks](https://github.com/TheBrainLab/Awesome-Spiking-Neural-Networks)
- [Open Neuromorphic](https://open-neuromorphic.org)
- [SNN Framework Benchmarks](https://open-neuromorphic.org/blog/spiking-neural-network-framework-benchmarking/)

### Surveys and Reviews
- [Neuromorphic Computing 2025: Current SotA](https://humanunsupervised.com/papers/neuromorphic_landscape.html)
- [The Road to Commercial Success for Neuromorphic Technologies - Nature Communications 2025](https://www.nature.com/articles/s41467-025-57352-1)
- [Toward Large-scale Spiking Neural Networks - Survey 2024](https://arxiv.org/abs/2409.02111)
- [SNN Architecture Search Survey 2024](https://arxiv.org/abs/2510.14235)
- [Continual Learning with Neuromorphic Computing 2024](https://arxiv.org/abs/2410.09218)
- [SNNs for Ubiquitous Computing - Survey 2025](https://arxiv.org/abs/2506.01737)

### Frameworks and Tools
- [snnTorch](https://github.com/jeshraghian/snntorch)
- [SpikingJelly](https://github.com/fangwei123456/spikingjelly)
- [Intel Lava](https://github.com/lava-nc/lava)
- [Norse](https://github.com/norse/norse)
