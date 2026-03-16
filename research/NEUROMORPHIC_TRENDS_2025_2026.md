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
- **Architecture**: Hybrid SNN + RISC-V CPU + CNN/DSP accelerators
- **Power**: Audio classification at ~400 microW (!), 500x lower than traditional MCUs
- **Demos at CES 2026**: Smart home, industrial IoT, wearables, healthcare
- **Shipping**: First mass-produced neuromorphic processor launched at Computex 2025
- **Audio significance**: Sub-millisecond keyword spotting, audio scene recognition

**Takeaway for our paper:** The hardware landscape is maturing rapidly. Papers that demonstrate actual deployment on real neuromorphic hardware (like our SpiNNaker work) have outsized credibility. Innatera's audio focus validates our audio domain choice.

---

## 4. MOST-CITED AND MOST IMPACTFUL SNN PAPERS 2024-2025

### Award Winners
1. **"Training Spiking Neural Networks Using Lessons From Deep Learning"** (Eshraghian et al., 2023) -- **2024 Proceedings of the IEEE Best Paper Award**. The foundational tutorial paper for modern SNN training. Our project uses snnTorch (the companion software). Most-cited SNN paper of the era.

2. **"NeuroBench: A Framework for Benchmarking Neuromorphic Computing"** (Yik et al., 2025) -- Nature Communications. Our project uses NeuroBench metrics. This paper standardized neuromorphic benchmarking.

### High-Impact Publications (2024-2025)

| Paper | Venue | Significance |
|-------|-------|-------------|
| Neuromorphic Computing at Scale | Nature (Jan 2025) | First Nature review calling the field at a "pivotal moment" |
| Road to Commercial Success for Neuromorphic Technologies | Nature Communications (Apr 2025) | Commercial viability assessment |
| Neuromorphic computing paradigms enhance robustness through SNNs | Nature Communications (Nov 2025) | Formal proof of SNN adversarial advantage |
| Can neuromorphic computing help reduce AI's high energy cost? | PNAS (2025) | AI energy crisis + neuromorphic solutions |
| SpiNNaker2: Large-Scale Neuromorphic System | arXiv (Jan 2024) | SpiNNaker2 architecture paper |
| Spike-driven Transformer V2 | ICLR 2024 | Next-gen neuromorphic chip design via SNN transformers |
| SpikeLLM: Scaling SNNs to LLMs | ICLR 2025 | First large-scale spiking language model |

### Key Survey Papers
- "Toward Large-scale Spiking Neural Networks: A Comprehensive Survey" (2024)
- "SNN Architecture Search: A Survey" (Oct 2025)
- "SNNs on FPGA: A Survey" (Neural Networks, 2025)
- "SNN and Sound: A Comprehensive Review" (Aug 2024)
- "Continual Learning with Neuromorphic Computing" (Oct 2024)

---

## 5. TRENDING ON arXiv (cs.NE and cs.SD)

### cs.NE (Neural and Evolutionary Computing)
- **Spiking transformers** dominate submissions
- **Hardware-aware optimization** (quantization, pruning, SpikeFit)
- **Theoretical understanding** of SNNs (stability, robustness, generalization)
- **SSM-SNN hybrids** (state space models meet spiking)
- **Neuromorphic LLMs** (MatMul-free inference)

### cs.SD (Sound) + Neuromorphic
- **Larroza et al. (March 2025)**: "Spike Encoding for Environmental Sound: A Comparative Benchmark" -- ESC-10 only, FC network only, no hardware deployment. **This is our closest competitor and we substantially exceed their scope.**
- **Spiking Vocos** (2025): Energy-efficient neural vocoder using SNNs
- **Audio-visual multimodal SNNs** (Feb 2025): Cross-modal spiking transformers
- **Neuromorphic keyword spotting** with PDM microphones (Interspeech 2024): 91.54% on Google Speech Commands
- **Hilbert Transform encoding** for audio source localization (Nature Communications Engineering, 2025)
- **HPCNeuroNet**: Transformer-enhanced SNN for audio (2023, still cited)

### Key arXiv Trends (last 6 months)
1. Spike-driven everything (transformers, LLMs, NeRF, graph networks)
2. Energy-accuracy Pareto analysis becoming mandatory
3. Neuromorphic + embodied intelligence / robotics
4. Privacy as an inherent SNN property
5. Temporal coding gaining over rate coding

---

## 6. OPEN PROBLEMS AND GRAND CHALLENGES

### Tier 1: Critical Unsolved Problems
1. **Software ecosystem gap**: No PyTorch/TensorFlow equivalent for neuromorphic. Lava, snnTorch, Norse, SpikingJelly all fragmented. The Open Neuromorphic community explicitly calls this the #1 barrier.

2. **Scaling SNNs**: Current directly-trained SNNs top out at ~85M parameters (SpikeLLM). Brain has ~86 billion neurons. Multiple orders of magnitude to bridge.

3. **The accuracy gap**: On standard benchmarks (ImageNet), best SNN (83.73% SGLFormer) still trails best ANN (90%+). Gap narrowing but not closed.

4. **Killer application**: No single application has proven neuromorphic superiority definitively in real-world deployment at scale.

5. **Energy claims need hardware verification**: Most "energy efficiency" claims are theoretical (counting ACs/MACs with assumed pJ costs). Very few papers do actual hardware power measurement.

### Tier 2: Active Research Challenges
6. **Spike rate break-even**: SNNs only beat quantized ANNs when spike rate < 6.4% (Dampfhoffer 2023). Most practical SNNs have 20-30% spike rates. Closing this gap is critical.

7. **Hardware mapping efficiency**: Naive mappings achieve only 30-50% neuromorphic hardware utilization. Algorithm-hardware co-design is essential.

8. **Continual learning at scale**: On-chip learning demonstrated in small settings (Loihi 2) but not at production scale.

9. **Temporal coding advantages**: Theory says temporal codes should be more efficient, but rate coding still dominates in practice. Why?

10. **Standardized benchmarking**: NeuroBench is helping but adoption is incomplete. No consensus on how to fairly compare SNN and ANN energy.

### Tier 3: Emerging Frontiers
11. **Neuromorphic + LLMs**: Can spiking principles make transformer inference radically more efficient?
12. **Photonic neuromorphic computing**: All-optical SNNs for speed-of-light processing
13. **2D materials for neuromorphic devices**: Sub-100mV switching, femtojoule energy
14. **Neuromorphic sensing end-to-end**: From event camera/mic directly to SNN inference

---

## 7. IS THERE A "NEUROMORPHIC AUDIO" COMMUNITY?

### Assessment: YES, emerging but small and underserved

**Key Research Groups:**

| Group/Researcher | Affiliation | Focus | Key Work |
|-----------------|-------------|-------|----------|
| Jimenez-Fernandez et al. | University of Seville | SNN audio on SpiNNaker | Dominguez-Morales et al. 2016 (pure tones on SpiNNaker) |
| Larroza et al. | IVACE/Spain | Spike encoding for environmental sound | March 2025 ESC-10 benchmark |
| Zenke Lab | Friedrich Miescher Institute | SHD/SSC datasets | Spiking Heidelberg Digits (the standard audio SNN benchmark) |
| Wu/Chua | NTU Singapore | Robust sound classification | 2018 Frontiers paper |
| Yarga et al. | Multiple | Neuromorphic KWS with PDM mics | Interspeech 2024 |
| Innatera | Netherlands (commercial) | Always-on audio sensing | Pulsar chip, 400microW audio classification |
| SpiNNcloud | Dresden | SpiNNaker2 applications | Including audio potential |

**Key Datasets:**
- **SHD/SSC** (Spiking Heidelberg Digits / Spiking Speech Commands) -- the de facto standard
- **Google Speech Commands** -- for keyword spotting
- **ESC-10/ESC-50** -- environmental sound (our domain, ZERO prior SNN work on full ESC-50)
- **UrbanSound8K** -- urban environmental audio

**Community Gaps (our opportunities):**
1. **No SNN on full ESC-50** -- we are first
2. **No systematic 7-encoding comparison on audio** -- Larroza only tested 3 methods on ESC-10
3. **No PANNs+SNN transfer learning for audio** -- entirely novel
4. **No adversarial robustness study for SNN audio** -- entirely novel
5. **Very few actual hardware deployments for audio SNNs** -- Dominguez-Morales 2016 (pure tones only) is the closest

---

## 8. WHAT WOULD MAKE A PAPER "GO VIRAL" IN THE NEUROMORPHIC COMMUNITY

Based on analysis of high-impact recent papers and community discourse:

### Formula for Maximum Impact

1. **Real hardware + real numbers**: The community is TIRED of theoretical energy estimates. Papers with actual hardware deployment (Loihi, SpiNNaker, FPGA) get 3-5x more attention. Our SpiNNaker deployment is genuine gold.

2. **Address the AI energy narrative**: The PNAS 2025 paper on AI energy crisis + neuromorphic solutions hit mainstream news. Any paper that credibly contributes data to the "can neuromorphic computing save AI from its energy problem?" narrative gets amplified.

3. **Honest about limitations**: The community has developed allergy to overclaiming. Papers that honestly report where SNNs fall short (like our 47.15% vs 63.85% gap) while explaining WHY are more trusted than papers claiming SNN superiority.

4. **Bridge SNN and mainstream ML**: Papers connecting SNNs to transformers, LLMs, or pretrained models generate excitement because they lower the barrier for mainstream ML researchers to engage with neuromorphic.

5. **Novel domain + systematic methodology**: Audio is underexplored. Seven encoding methods is thorough. The combination of systematic rigor with a novel application domain is exactly what reviewers want.

6. **Reproducible with code**: Open Neuromorphic's new peer review program specifically rewards reproducibility. Publishing code is now table stakes.

7. **Clear narrative / insight**: The most cited papers tell a story. Our story -- "the SNN-ANN gap is a feature-learning problem, not a spiking computation problem" -- is exactly the kind of insight that gets remembered and cited.

### What NOT to do:
- Don't claim SNNs are "better" than ANNs without qualification
- Don't ignore the energy break-even threshold literature
- Don't present simulation-only results as "neuromorphic deployment"
- Don't compare unfairly (different parameter counts, training budgets)

---

## 9. NEW BENCHMARKS, CHALLENGES, AND COMPETITIONS

### NeuroBench (Active, Growing)
- Published in Nature Communications (February 2025)
- 100+ researchers from 50+ institutions
- **Algorithm Track v1.0**: Few-shot continual learning, object detection (event cameras), sensorimotor decoding, predictive modeling
- **System Track**: Deployed execution time, throughput, efficiency
- Our project already uses NeuroBench metrics (Effective_ACs, Effective_MACs, Dense, ActivationSparsity)

### Open Neuromorphic Research (ONR)
- Community-driven peer review for open, reproducible neuromorphic research
- Badges for recognized projects
- Launched 2025, actively accepting submissions

### Innatera/BrainChip Developer Programs
- Akida Developer Cloud (August 2025)
- AkidaTag evaluation platform (May 2026)

### Conference Workshops (2025-2026)
- CVPR 2025: Fifth International Workshop on Event-based Vision
- ICCV 2025: 2nd Workshop on Neuromorphic Vision (NeVi)
- IROS 2025: Workshop on Neuromorphic Perception for Real World Robotics
- NeurIPS 2025: Machine Learning and the Physical Sciences (neuromorphic angle)

### No dedicated "neuromorphic audio challenge" exists yet
This is actually an opportunity -- proposing one based on ESC-50 + SNN could be high-impact future work.

---

## 10. THE SNN vs ANN NARRATIVE: HAS THE COMMUNITY VIEW SHIFTED?

### 2020 View: "SNNs will replace ANNs"
Overoptimistic. Many papers claimed energy efficiency advantages without hardware evidence.

### 2023 View: "SNNs are interesting but not practical"
Skepticism peaked. ImageNet gap was large. No commercial deployments.

### 2025-2026 View: "SNNs are complementary specialized accelerators"
The consensus has settled into a **mature, nuanced position:**

1. **Energy efficiency is CONDITIONAL, not automatic**:
   - SNNs need >93% sparsity to beat ANNs on most hardware (Dampfhoffer 2023)
   - Hardware-agnostic estimates show ~50-60% SNN advantage
   - Hardware-aware estimates on conventional hardware show NO advantage
   - On actual neuromorphic hardware, 100-1000x advantage IS real (but task-specific)

2. **Accuracy gap is narrowing but not closed**:
   - Best SNN on ImageNet: 83.73% (SGLFormer, 2025) vs best ANN: 90%+
   - The gap is SMALLEST when: pretrained features are used, the task is temporal, or the data is event-driven
   - Our PANNs finding (gap collapses from 17pp to 1pp with pretrained features) perfectly illustrates this

3. **The killer app is EDGE, not cloud**:
   - Always-on sensing (audio, vibration, vision)
   - Battery-powered devices (wearables, IoT, implants)
   - Real-time control (robotics, drones, autonomous vehicles)
   - Privacy-preserving on-device processing

4. **Hybrid architectures are the pragmatic path**:
   - ANN feature extraction + SNN classification (exactly our PANNs approach)
   - ANN training + SNN inference
   - Heterogeneous systems with both conventional and neuromorphic processors

5. **The "biological plausibility" argument has weakened**:
   - The community increasingly values practical performance over bio-plausibility
   - Surrogate gradients (not biologically plausible) are accepted as standard
   - "Brain-inspired" is a design philosophy, not a constraint

### Key Quote from Nature Communications (2025):
"After several false starts, advances now promise widespread commercial adoption."

---

## STRATEGIC POSITIONING FOR OUR ICONS 2026 PAPER

### Our Project's Alignment with Trends

| Trend | Our Coverage | Alignment |
|-------|-------------|-----------|
| Hardware deployment | SpiNNaker FC2-only hybrid | STRONG -- real hardware, honest about challenges |
| Energy analysis | NeuroBench metrics, AC/MAC comparison | STRONG -- uses accepted framework |
| Encoding comparison | 7 methods (most comprehensive for audio) | UNIQUE -- no one has done this |
| Transfer learning bridge | PANNs+SNN (gap 17pp -> 1pp) | HOT TOPIC -- directly addresses #1 community question |
| Adversarial robustness | FGSM+PGD, SNN dramatically more robust | HOT TOPIC -- aligns with Nature Comms 2025 |
| Novel application domain | ESC-50 (first SNN work on full dataset) | UNIQUE -- fills a clear gap |
| Honest analysis | Report where SNNs fail, explain why | MATCHES community preference for honesty |
| Continual learning | 5-task sequential with BWT | RELEVANT -- active area |
| Surrogate gradient analysis | 8-surrogate ablation | NOVEL -- provides practical guidance |

### Recommended Framing for ICONS 2026

**Title suggestions (emphasizing hardware + systematic comparison):**
1. "Spiking Neural Networks for Environmental Sound Classification: A Systematic Encoding Comparison with SpiNNaker Deployment"
2. "From Spectrograms to Spikes: Encoding, Training, and Deploying SNNs for Audio on Neuromorphic Hardware"
3. "Bridging the Gap: Transfer Learning and Hardware Deployment for Spiking Audio Classification"

**Key narratives to emphasize:**
1. **First SNN work on ESC-50** -- novelty claim is watertight
2. **7-encoding comparison reveals hierarchy** -- practical guidance for researchers
3. **PANNs+SNN insight** -- the gap is feature-learning, not spiking-computation
4. **Actual SpiNNaker deployment** -- not simulation, real hardware
5. **Adversarial robustness** -- aligns with Nature Communications 2025 finding
6. **Honest energy analysis** -- software simulation shows ANN cheaper; neuromorphic hardware flips the result

**What NOT to claim:**
- Don't claim our SNN beats ANN accuracy (it doesn't, and that's the point)
- Don't claim energy advantage without qualifying software vs hardware
- Don't overstate SpiNNaker accuracy (33.1% with 12.8pp gap from snnTorch)

### Competitive Landscape for ICONS 2026

Our closest competitors in the audio SNN space:

| Work | Dataset | Methods | Hardware | Our Advantage |
|------|---------|---------|----------|---------------|
| Larroza et al. 2025 | ESC-10 only | 3 encodings, FC only | None | Full ESC-50, 7 encodings, CNN, SpiNNaker |
| Dominguez-Morales 2016 | Pure tones | 1 method | SpiNNaker 1 | Complex audio, multiple methods, modern training |
| Yarga et al. 2024 | GSC (speech) | PDM mic + SNN | Simulation | Different domain (env. sound vs speech) |
| Wu & Chua 2018 | RWCP/TIMIT | Robust SNN | None | Modern methods, hardware, comprehensive |

---

## RESEARCH GAPS AND RECOMMENDED FOLLOW-UPS

1. **Loihi 2 deployment**: If we could also deploy on Loihi 2, comparing SpiNNaker vs Loihi would be a separate high-impact paper.

2. **Neuromorphic audio challenge**: Proposing an ESC-50 SNN benchmark/challenge through Open Neuromorphic or NeuroBench could establish us as community leaders.

3. **Event-driven audio encoding**: Moving from frame-based spectrograms to true event-driven audio (like Hilbert Transform encoding) would be a natural next step.

4. **Innatera Pulsar evaluation**: If we could get our model onto Innatera's platform, the power numbers would be compelling for the always-on audio use case.

5. **SpiNNaker 2 deployment**: SpiNNaker 2 at Sandia/SpiNNcloud would be the obvious hardware upgrade path.

---

## CONFIDENCE ASSESSMENT

| Finding | Confidence | Basis |
|---------|-----------|-------|
| No prior SNN work on full ESC-50 | HIGH | Multiple searches, Larroza only covers ESC-10 |
| ICONS has no audio papers (2024-2025) | HIGH | Full schedule analysis |
| Adversarial robustness is hot topic | HIGH | Nature Communications 2025, multiple NeurIPS papers |
| Transfer learning is hot topic | HIGH | Knowledge distillation papers at NeurIPS 2025, ICLR 2026 |
| Hardware deployment valued by reviewers | HIGH | ICONS 2025 schedule analysis, community discourse |
| SNN paper counts at top conferences | HIGH | Cross-referenced GitHub lists and official programs |
| Energy break-even threshold (~6.4% spike rate) | MEDIUM-HIGH | Dampfhoffer 2023, cited in multiple 2025 papers |
| Commercial neuromorphic market projections | MEDIUM | Multiple sources but projections vary widely |
| Community view on SNN vs ANN | HIGH | Synthesized from surveys, Nature papers, community discussions |

---

## SOURCES

### Conference Proceedings
- [ICONS 2025 Schedule](https://iconsneuromorphic.cc/Schedule-2025/)
- [ICONS 2026 Call for Papers](https://iconsneuromorphic.cc/calls-2026/)
- [ICONS 2024 Proceedings](https://www.computer.org/csdl/proceedings/icons/2024/22lE6EOwpkA)
- [ICML 2025 Papers](https://icml.cc/virtual/2025/papers.html)
- [NeurIPS 2025 Paper List](https://papercopilot.com/paper-list/neurips-paper-list/neurips-2025-paper-list/)
- [ICLR 2026 Papers](https://iclr.cc/virtual/2026/papers.html)

### High-Impact Publications
- [Neuromorphic computing paradigms enhance robustness through SNNs (Nature Communications, 2025)](https://www.nature.com/articles/s41467-025-65197-x)
- [NeuroBench framework (Nature Communications, 2025)](https://www.nature.com/articles/s41467-025-56739-4)
- [Road to commercial success for neuromorphic technologies (Nature Communications, 2025)](https://www.nature.com/articles/s41467-025-57352-1)
- [Neuromorphic computing at scale (Nature, 2025)](https://www.nature.com/articles/s41586-024-08253-8)
- [Can neuromorphic computing help reduce AI's high energy cost? (PNAS, 2025)](https://www.pnas.org/doi/10.1073/pnas.2528654122)
- [IBM NorthPole (Science, 2023)](https://www.science.org/doi/10.1126/science.adh1174)

### SNN Audio and Encoding
- [Spike Encoding for Environmental Sound (Larroza et al., March 2025)](https://arxiv.org/abs/2503.11206)
- [SNN and Sound: A Comprehensive Review (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11362401/)
- [Neuromorphic Keyword Spotting with PDM MEMS Microphones (Interspeech 2024)](https://arxiv.org/abs/2408.05156)
- [Low-power SNN Audio Source Localization (Nature Comms Eng, 2025)](https://www.nature.com/articles/s44172-025-00359-9)
- [Spiking Vocos: Energy-Efficient Neural Vocoder (2025)](https://arxiv.org/html/2509.13049v1)
- [Audio-Visual SNN with Cross-Modal Residual Learning (2025)](https://arxiv.org/abs/2502.12488)

### Hardware
- [Intel Loihi 2 / Hala Point](https://www.intel.com/content/www/us/en/research/neuromorphic-computing.html)
- [SpiNNaker2 at Sandia (June 2025)](https://www.nextplatform.com/2025/06/16/sandia-deploys-spinnaker2-neuromorphic-system/)
- [BrainChip AKD1500 Launch](https://www.edge-ai-vision.com/2025/11/brainchip-unveils-breakthrough-akd1500-edge-ai-co-processor-at-embedded-world-north-america/)
- [Innatera Pulsar at CES 2026](https://innatera.com/press-releases/redefining-the-cutting-edge-innatera-debuts-real-world-neuromorphic-edge-ai-at-ces-2026)
- [Neuromorphic LLM on Loihi 2 (March 2025)](https://arxiv.org/abs/2503.18002)

### Training and Methods
- [Training SNNs Using Lessons From Deep Learning (Eshraghian et al., 2023 -- 2024 Best Paper)](https://arxiv.org/abs/2109.12894)
- [Surrogate Gradient Theoretical Underpinnings (Gygax & Zenke, 2025)](https://direct.mit.edu/neco/article/37/5/886/128506)
- [Adaptive Surrogate Gradients (2025)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2026.1795946/full)
- [Enhanced Self-Distillation for SNNs (NeurIPS 2025)](https://arxiv.org/abs/2510.06254)
- [Towards Reliable Adversarial Robustness Evaluation for SNNs (Wang et al., 2025)](https://arxiv.org/abs/2512.22522)

### Energy and Efficiency
- [Are SNNs Really More Energy-Efficient? (Dampfhoffer et al., IEEE TECI 2023)](https://ieeexplore.ieee.org/document/9927729/)
- [Reconsidering the Energy Efficiency of SNNs (2024)](https://arxiv.org/abs/2409.08290)
- [SpikeFit: Optimal SNN Deployment on Neuromorphic Hardware (2025)](https://arxiv.org/html/2510.15542)
- [Energy-Efficient SNN for Edge AI (2025)](https://arxiv.org/html/2602.02439v1)

### Surveys and Roadmaps
- [Roadmap to Neuromorphic Computing with Emerging Technologies (2024)](https://arxiv.org/html/2407.02353v1)
- [Solving Neuromorphic Computing's Key Challenges (Open Neuromorphic)](https://open-neuromorphic.org/getting-involved/solving-neuromorphic-computings-key-challenges/)
- [Neuromorphic Computing 2025: Current SotA](https://humanunsupervised.com/papers/neuromorphic_landscape.html)
- [Neuromorphic Computing for Embodied Intelligence (2025)](https://arxiv.org/abs/2507.18139)
- [Awesome SNN Conference Papers (GitHub)](https://github.com/AXYZdong/awesome-snn-conference-paper)
- [Awesome Spiking Neural Networks (GitHub)](https://github.com/TheBrainLab/Awesome-Spiking-Neural-Networks)

### Community and Ecosystem
- [Open Neuromorphic Community](https://open-neuromorphic.org/)
- [NeuroBench Official Website](https://neurobench.ai/)
- [SpiNNcloud Systems](https://spinncloud.com/)
