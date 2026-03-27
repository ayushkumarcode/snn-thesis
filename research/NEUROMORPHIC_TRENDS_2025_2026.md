# Neuromorphic Computing & SNN Research Trends: 2025-2026

## Deep Research Intelligence Report
**Date:** 15 March 2026
**Purpose:** Position SNN-ESC50 thesis project for maximum relevance to the neuromorphic community
**Target:** ICONS 2026 (deadline: April 1, 2026)

---

## EXECUTIVE SUMMARY

The neuromorphic computing field is at a **commercial inflection point** in 2025-2026. Three macro-trends dominate: (1) the AI energy crisis is driving unprecedented interest in neuromorphic solutions, (2) SNN papers at top-tier ML conferences have exploded (29 at ICLR 2026, 23 at NeurIPS 2025, 11 at ICML 2025), and (3) hardware is finally shipping commercially (Innatera Pulsar, BrainChip AKD1500, SpiNNaker2 at Sandia).

**For our project specifically:** The SNN-ESC50 work is positioned at a remarkable sweet spot. There is ZERO prior SNN work on full ESC-50 (confirmed), and the Larroza et al. March 2025 paper on spike encoding for environmental sound only covers ESC-10. The field is hungry for: (a) actual hardware deployment results (not just simulation), (b) honest energy analysis with real numbers, (c) encoding comparisons beyond vision, and (d) transfer learning bridges between ANNs and SNNs. Our project delivers ALL of these. The adversarial robustness finding (SNN 26% vs ANN 1.75% at eps=0.1) directly aligns with a November 2025 Nature Communications paper showing SNNs achieve 2x ANN robustness. The PANNs+SNN transfer learning finding (gap collapses from 17pp to 1pp) addresses the hottest question in the field: is the SNN-ANN gap a fundamental limitation or a feature-learning problem?

---

## 1. WHAT DOMINATED ICONS 2024-2025

### ICONS 2025 (Seattle, July 29 - Aug 1, 2025)

**Best Paper Award:** "A Comparison of Custom and Standard Neuron Model Random Walks on the Ornstein-Uhlenbeck Equation for Simplified Turbulence" (Taylor et al.) -- an unconventional application demonstrating neuromorphic computing beyond classification.

**Key themes from the accepted papers (31 presentations):**

| Theme | Paper Count | Examples |
|-------|------------|---------|
| Hardware deployment & deployment pipelines | 5 | SpiNNaker2 fine-tuning, Loihi 2 deployment pipeline, code gen for embedded |
| Robotics & control | 4 | Motor neuron models, robotic perception, drone tracking |
| Privacy & security | 3 | Privacy in SNNs, model inversion attacks, cybersecurity |
| Energy-efficient edge AI | 3 | Vibration sensing, predictive maintenance, tactile sensing |
| Continual/online learning | 3 | Unsupervised continual learning, lifelong learning, GRASP |
| Novel applications | 4 | Turbulence modeling, artificial pancreas, chemical sensors, air traffic control |
| Architectures & training | 5 | Spiking transformers, oscillator Ising machines, dendrites, reservoir computing |
| Simulation frameworks | 2 | VRISP simulator, SNN uncertainty estimation |
| Vision/event-driven | 2 | Event-based action recognition, high-speed perception |

**Critical observation:** ICONS 2025 strongly favored papers with **real hardware deployment** (SpiNNaker2, Loihi 2, embedded systems) and **novel application domains** (turbulence, medical, industrial). Pure algorithm papers without hardware or application context were less prominent.

**Audio/sound papers at ICONS 2025:** ZERO. There were no audio or sound classification papers. This is a wide-open gap.

### ICONS 2024 (Arlington, VA, July 30 - Aug 2, 2024)
- Focus on RF fingerprinting, cerebellar neuron detection
- Superconducting optoelectronic neuromorphic computing

**Takeaway for our paper:** An SNN audio paper with actual SpiNNaker deployment would be genuinely novel at ICONS. The conference has never seen this combination.

---

## 2. HOTTEST TOPICS AT TOP ML CONFERENCES (2024-2026)

### SNN Paper Counts at Major Conferences

| Conference | Year | SNN Papers | Trend |
|-----------|------|-----------|-------|
| ICLR | 2026 | **29** | Strongest showing ever |
| NeurIPS | 2025 | **23** | Significant presence |
| ICML | 2025 | **11** | Growing |
| CVPR | 2025 | **5** | Vision-focused |
| ICLR | 2025 | 11 | Steady |
| NeurIPS | 2024 | ~15 | Baseline |

### Dominant Topics Across These Conferences

1. **Spiking Transformers** (HOTTEST): SpikFormer, Spike-driven Transformer V2, Binary Event-Driven Spiking Transformer. ImageNet SOTA for SNNs now at 83.73% (SGLFormer). This is the "transformer moment" for SNNs.

2. **SNN-LLM intersection** (EMERGING, HIGH IMPACT): SpikeGPT, SpikeLLM (ICLR 2025), Neuromorphic Spike-Based LLM (National Science Review 2025), MatMul-free LLM on Loihi 2. The field is racing to apply neuromorphic principles to language models.

3. **Knowledge Distillation / Self-Distillation** (NeurIPS 2025): "SNNs are Inherently Self-Distillers," Enhanced Self-Distillation Framework. ANN-to-SNN knowledge transfer is a major research direction.

4. **Adversarial Robustness** (NATURE COMMUNICATIONS 2025): "Neuromorphic computing paradigms enhance robustness through spiking neural networks." SNNs achieve 2x ANN robustness on CIFAR-10. BUT: Wang et al. (2025) warns robustness may be overestimated due to vanishing gradients in spike activations.

5. **ANN-to-SNN Conversion** (CVPR 2025, IJCAI 2025): Training-free conversion for transformers, negative spike methods, inference-scale complexity reduction. Gap closing to <0.04% in some settings.

6. **Temporal Processing** (NeurIPS 2025): "SNNs Need High-Frequency Information," Temporal Shift modules, Spiking NeRF. Exploiting temporal dynamics is a key differentiator.

7. **Multimodal SNNs** (2025): Audio-visual spiking transformers, cross-modal residual learning, temporal attention-guided fusion.

8. **State Space Models + SNNs** (EMERGING): SpikySpace (first fully spiking SSM), delays in SNNs via state variables, Mamba-inspired spiking architectures.

9. **Federated Learning + SNNs** (2025): Privacy-preserving properties of spike-based communication, energy-efficient distributed training.

10. **Neural Architecture Search for SNNs** (arXiv survey 2025): Hardware-aware NAS without training on neuromorphic platforms.

---

## 3. HARDWARE LANDSCAPE: WHAT THE BIG PLAYERS ARE DOING

### Intel Loihi 2
- **Hala Point** system: 1.15 billion neurons, 1,152 Loihi 2 processors
- **Performance**: 75x lower latency, 1000x higher energy efficiency vs NVIDIA Jetson Orin Nano on SSM workloads
- **New direction**: First LLM on neuromorphic hardware (March 2025, arXiv:2503.18002)
- **Key demo**: Continual learning on-chip (CLP-SNN, November 2025)
- **Software**: Lava framework (open-source but still maturing)

### IBM NorthPole
- **Architecture**: 256 cores, 224MB on-chip SRAM, 12nm process
- **Performance**: 25x more energy efficient than GPU on ResNet-50; 46.9x faster, 72.7x more energy efficient on 3B-parameter LLM inference
- **Roadmap**: Future 4nm versions planned for higher density
- **Key insight**: Not truly spiking -- uses on-chip memory architecture inspired by brain, runs conventional neural networks efficiently
- **Status**: Inference-only chip, not training

### SpiNNaker 2
- **Scale**: Targeting 10 million ARM cores (10x SpiNNaker 1)
- **Deployment**: Sandia National Laboratories deployed SpiNNaker2 system (June 2025) for AI and national security
- **New capability**: Now supports both SNNs AND conventional DNNs (event-based deep learning)
- **Commercial entity**: SpiNNcloud marketing 5-million-core systems
- **Relevance**: Our project uses SpiNNaker 1 -- SpiNNaker 2 deployment would be a natural extension

### BrainChip Akida
- **AKD1500**: 800 GOPS at <300mW, samples available now, volume production Q3 2026
- **Funding**: $25M raised (December 2025) for Akida 2 and Akida GenAI
- **AkidaTag**: Wearable reference platform with Nordic nRF5340, evaluation May 2026
- **Cloud**: Akida Developer Cloud launched August 2025
- **Key differentiator**: Only commercially shipping neuromorphic IP for edge AI

### Innatera Pulsar (NEW, SIGNIFICANT)
