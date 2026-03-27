# Neuromorphic computing & SNN research trends: 2025-2026

looking into what's trending in the neuromorphic community so i can position the SNN-ESC50 project well, especially for ICONS 2026.

The field is at a **commercial inflection point** in 2025-2026. Three big trends: (1) the AI energy crisis is driving unprecedented interest in neuromorphic solutions, (2) SNN papers at top-tier ML conferences have exploded (29 at ICLR 2026, 23 at NeurIPS 2025, 11 at ICML 2025), and (3) hardware is finally shipping commercially (Innatera Pulsar, BrainChip AKD1500, SpiNNaker2 at Sandia).

For my project specifically: the SNN-ESC50 work sits at a pretty remarkable sweet spot. Zero prior SNN work on full ESC-50 (confirmed), and the Larroza et al. March 2025 paper on spike encoding for environmental sound only covers ESC-10. The field seems to want: (a) actual hardware deployment results, (b) honest energy analysis with real numbers, (c) encoding comparisons beyond vision, and (d) transfer learning bridges between ANNs and SNNs. My project does all of these. The adversarial robustness finding (SNN 26% vs ANN 1.75% at eps=0.1) directly aligns with a November 2025 Nature Comms paper showing SNNs achieve 2x ANN robustness. The PANNs+SNN transfer learning finding (gap collapses from 17pp to 1pp) addresses what seems to be the hottest question: is the SNN-ANN gap fundamental or a feature-learning problem?

---

## 1. What dominated ICONS 2024-2025

### ICONS 2025 (Seattle, July 29 - Aug 1, 2025)

**Best Paper Award:** "A Comparison of Custom and Standard Neuron Model Random Walks on the Ornstein-Uhlenbeck Equation for Simplified Turbulence" (Taylor et al.) -- an unconventional application showing neuromorphic computing beyond classification.

**Key themes from accepted papers (31 presentations):**

| Theme | Paper Count | Examples |
|-------|------------|---------|
| Hardware deployment & pipelines | 5 | SpiNNaker2 fine-tuning, Loihi 2 deployment pipeline, code gen for embedded |
| Robotics & control | 4 | Motor neuron models, robotic perception, drone tracking |
| Privacy & security | 3 | Privacy in SNNs, model inversion attacks, cybersecurity |
| Energy-efficient edge AI | 3 | Vibration sensing, predictive maintenance, tactile sensing |
| Continual/online learning | 3 | Unsupervised continual learning, lifelong learning, GRASP |
| Novel applications | 4 | Turbulence modeling, artificial pancreas, chemical sensors, air traffic control |
| Architectures & training | 5 | Spiking transformers, oscillator Ising machines, dendrites, reservoir computing |
| Simulation frameworks | 2 | VRISP simulator, SNN uncertainty estimation |
| Vision/event-driven | 2 | Event-based action recognition, high-speed perception |

ICONS 2025 strongly favored papers with **real hardware deployment** (SpiNNaker2, Loihi 2, embedded systems) and **novel application domains** (turbulence, medical, industrial). Pure algorithm papers without hardware or application context were less prominent.

**Audio/sound papers at ICONS 2025:** Zero. None. This is a wide-open gap.

### ICONS 2024 (Arlington, VA)
- Focus on RF fingerprinting, cerebellar neuron detection
- Superconducting optoelectronic neuromorphic computing

So an SNN audio paper with actual SpiNNaker deployment would be genuinely novel at ICONS. The conference has never seen this combination.

---

## 2. Hottest topics at top ML conferences (2024-2026)

### SNN paper counts at major conferences

| Conference | Year | SNN Papers | Trend |
|-----------|------|-----------|-------|
| ICLR | 2026 | **29** | Strongest showing ever |
| NeurIPS | 2025 | **23** | Significant presence |
| ICML | 2025 | **11** | Growing |
| CVPR | 2025 | **5** | Vision-focused |
| ICLR | 2025 | 11 | Steady |
| NeurIPS | 2024 | ~15 | Baseline |

### What's hot

1. **Spiking Transformers** (HOTTEST): SpikFormer, Spike-driven Transformer V2, Binary Event-Driven Spiking Transformer. ImageNet SOTA for SNNs now at 83.73% (SGLFormer). This is basically the "transformer moment" for SNNs.

2. **SNN-LLM intersection** (EMERGING): SpikeGPT, SpikeLLM (ICLR 2025), neuromorphic LLM (National Science Review 2025), MatMul-free LLM on Loihi 2. The field is racing to apply neuromorphic principles to language models.

3. **Knowledge Distillation / Self-Distillation** (NeurIPS 2025): "SNNs are Inherently Self-Distillers," Enhanced Self-Distillation Framework. ANN-to-SNN knowledge transfer is a major research direction.

4. **Adversarial Robustness** (Nature Communications 2025): formal proof of SNN adversarial advantage. SNNs achieve 2x ANN robustness on CIFAR-10. BUT: Wang et al. (2025) warns robustness may be overestimated due to vanishing gradients in spike activations.

5. **ANN-to-SNN Conversion** (CVPR 2025, IJCAI 2025): training-free conversion for transformers, negative spike methods. Gap closing to <0.04% in some settings.

6. **Temporal Processing** (NeurIPS 2025): "SNNs Need High-Frequency Information," temporal shift modules, spiking NeRF. Exploiting temporal dynamics is a key differentiator.

7. **Multimodal SNNs** (2025): Audio-visual spiking transformers, cross-modal residual learning.

8. **State Space Models + SNNs** (EMERGING): SpikySpace (first fully spiking SSM), delays via state variables, Mamba-inspired spiking architectures.

9. **Federated Learning + SNNs** (2025): Privacy-preserving properties of spike-based communication.

10. **Neural Architecture Search for SNNs** (arXiv survey 2025): Hardware-aware NAS without training.

---

## 3. Hardware landscape: what the big players are doing

### Intel Loihi 2
- **Hala Point** system: 1.15 billion neurons, 1,152 Loihi 2 processors
- 75x lower latency, 1000x higher energy efficiency vs Jetson Orin Nano on SSM workloads
- First LLM on neuromorphic hardware (March 2025, arXiv:2503.18002)
- Continual learning on-chip (CLP-SNN, November 2025)
- Software: Lava framework (open-source but still maturing)

### IBM NorthPole
- 256 cores, 224MB on-chip SRAM, 12nm
- 25x more energy efficient than GPU on ResNet-50; 46.9x faster, 72.7x more efficient on 3B-parameter LLM inference
- Not truly spiking -- uses on-chip memory architecture inspired by brain
- Inference-only chip, not training

### SpiNNaker 2
- Targeting 10 million ARM cores (10x SpiNNaker 1)
- Deployed at Sandia National Laboratories (June 2025) for AI and national security
- Now supports both SNNs AND conventional DNNs
- Commercial entity: SpiNNcloud marketing 5-million-core systems
- My project uses SpiNNaker 1 -- SpiNNaker 2 would be a natural extension

### BrainChip Akida
- AKD1500: 800 GOPS at <300mW, volume production Q3 2026
- $25M raised (December 2025) for Akida 2 and GenAI
- Cloud: Akida Developer Cloud launched August 2025
- Only commercially shipping neuromorphic IP for edge AI

### Innatera Pulsar (new, significant)
- Hybrid SNN + RISC-V CPU + CNN/DSP accelerators
- Audio classification at ~400 microW (!), 500x lower than traditional MCUs
- Demos at CES 2026: smart home, industrial IoT, wearables, healthcare
- First mass-produced neuromorphic processor (Computex 2025)
- Sub-millisecond keyword spotting, audio scene recognition

The hardware landscape is maturing fast. Papers that show actual deployment on real neuromorphic hardware (like my SpiNNaker work) have outsized credibility. Innatera's audio focus validates the audio domain choice.

---

## 4. Most impactful SNN papers 2024-2025

### Award winners
1. **"Training Spiking Neural Networks Using Lessons From Deep Learning"** (Eshraghian et al., 2023) -- 2024 Proceedings of the IEEE Best Paper Award. The foundational tutorial paper for modern SNN training. My project uses snnTorch (the companion software).

2. **"NeuroBench: A Framework for Benchmarking Neuromorphic Computing"** (Yik et al., 2025) -- Nature Communications. My project uses NeuroBench metrics.

### High-impact publications

| Paper | Venue | Significance |
|-------|-------|-------------|
| Neuromorphic Computing at Scale | Nature (Jan 2025) | First Nature review calling the field at a "pivotal moment" |
| Road to Commercial Success | Nature Communications (Apr 2025) | Commercial viability assessment |
| Neuromorphic computing enhances robustness through SNNs | Nature Communications (Nov 2025) | Formal proof of SNN adversarial advantage |
| Can neuromorphic computing reduce AI's energy cost? | PNAS (2025) | AI energy crisis + neuromorphic solutions |
| SpiNNaker2: Large-Scale Neuromorphic System | arXiv (Jan 2024) | SpiNNaker2 architecture paper |
| Spike-driven Transformer V2 | ICLR 2024 | Next-gen spiking transformers |
| SpikeLLM | ICLR 2025 | First large-scale spiking language model |

### Key surveys
- "Toward Large-scale Spiking Neural Networks" (2024)
- "SNN Architecture Search: A Survey" (Oct 2025)
