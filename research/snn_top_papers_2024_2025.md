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
