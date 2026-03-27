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
