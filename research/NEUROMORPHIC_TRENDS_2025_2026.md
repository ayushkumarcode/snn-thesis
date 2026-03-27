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
- "SNNs on FPGA: A Survey" (Neural Networks, 2025)
- "SNN and Sound: A Comprehensive Review" (Aug 2024)
- "Continual Learning with Neuromorphic Computing" (Oct 2024)

---

## 5. Trending on arXiv

### cs.NE (Neural and Evolutionary Computing)
- Spiking transformers dominate submissions
- Hardware-aware optimization (quantization, pruning, SpikeFit)
- Theoretical understanding of SNNs (stability, robustness, generalization)
- SSM-SNN hybrids (state space models meet spiking)
- Neuromorphic LLMs (MatMul-free inference)

### cs.SD (Sound) + Neuromorphic
- **Larroza et al. (March 2025)**: "Spike Encoding for Environmental Sound" -- ESC-10 only, FC network only, no hardware. This is my closest competitor and i substantially exceed their scope.
- Spiking Vocos (2025): energy-efficient neural vocoder
- Audio-visual multimodal SNNs (Feb 2025): cross-modal spiking transformers
- Neuromorphic keyword spotting with PDM microphones (Interspeech 2024): 91.54% on GSC
- Hilbert Transform encoding for audio localization (Nature Comms Eng, 2025)
- HPCNeuroNet: transformer-enhanced SNN for audio (2023, still being cited)

### Key arXiv trends (last 6 months)
1. Spike-driven everything (transformers, LLMs, NeRF, graph networks)
2. Energy-accuracy Pareto analysis becoming mandatory
3. Neuromorphic + embodied intelligence / robotics
4. Privacy as an inherent SNN property
5. Temporal coding gaining over rate coding

---

## 6. Open problems and grand challenges

### Tier 1: Critical unsolved problems
1. **Software ecosystem gap**: No PyTorch/TensorFlow equivalent for neuromorphic. Lava, snnTorch, Norse, SpikingJelly all fragmented. Open Neuromorphic explicitly calls this the #1 barrier.

2. **Scaling SNNs**: Directly-trained SNNs top out at ~85M parameters (SpikeLLM). Brain has ~86 billion neurons. Multiple orders of magnitude to bridge.

3. **The accuracy gap**: Best SNN on ImageNet (83.73% SGLFormer) still trails best ANN (90%+). Narrowing but not closed.

4. **Killer application**: No single application has proven neuromorphic superiority definitively in real-world deployment at scale.

5. **Energy claims need hardware verification**: Most "energy efficiency" claims are theoretical (counting ACs/MACs). Very few papers do actual hardware power measurement.

### Tier 2: Active research challenges
6. **Spike rate break-even**: SNNs only beat quantized ANNs when spike rate < 6.4% (Dampfhoffer 2023). Most practical SNNs run 20-30%. Closing this gap is critical.

7. **Hardware mapping efficiency**: Naive mappings achieve only 30-50% neuromorphic hardware utilization.

8. **Continual learning at scale**: On-chip learning demonstrated in small settings but not production scale.

9. **Temporal coding advantages**: Theory says temporal codes should be more efficient, but rate coding still dominates in practice. Not clear why.

10. **Standardized benchmarking**: NeuroBench is helping but adoption is incomplete.

### Tier 3: Emerging frontiers
11. Neuromorphic + LLMs: can spiking principles make transformer inference radically more efficient?
12. Photonic neuromorphic computing: all-optical SNNs
13. 2D materials for neuromorphic devices: sub-100mV switching, femtojoule energy
14. Neuromorphic sensing end-to-end: from event camera/mic directly to SNN inference

---

## 7. Is there a "neuromorphic audio" community?

### Assessment: yes, emerging but small and underserved

**Key research groups:**

| Group/Researcher | Affiliation | Focus | Key Work |
|-----------------|-------------|-------|----------|
| Jimenez-Fernandez et al. | University of Seville | SNN audio on SpiNNaker | Dominguez-Morales et al. 2016 (pure tones) |
| Larroza et al. | IVACE/Spain | Spike encoding for env. sound | March 2025 ESC-10 benchmark |
| Zenke Lab | Friedrich Miescher Institute | SHD/SSC datasets | Spiking Heidelberg Digits |
| Wu/Chua | NTU Singapore | Robust sound classification | 2018 Frontiers paper |
| Yarga et al. | Multiple | Neuromorphic KWS with PDM mics | Interspeech 2024 |
| Innatera | Netherlands (commercial) | Always-on audio sensing | Pulsar chip, 400microW audio |
| SpiNNcloud | Dresden | SpiNNaker2 applications | Including audio potential |

**Key datasets:**
- SHD/SSC (Spiking Heidelberg Digits / Spiking Speech Commands) -- the de facto standard
- Google Speech Commands -- for keyword spotting
- ESC-10/ESC-50 -- environmental sound (my domain, ZERO prior SNN work on full ESC-50)
- UrbanSound8K -- urban environmental audio

**Gaps that are my opportunities:**
1. No SNN on full ESC-50 -- i'm first
2. No 7-encoding comparison on audio -- Larroza only tested 3 methods on ESC-10
3. No PANNs+SNN transfer learning for audio -- entirely novel
4. No adversarial robustness study for SNN audio -- entirely novel
5. Very few hardware deployments for audio SNNs -- Dominguez-Morales 2016 (pure tones only) is the closest

---

## 8. What makes a paper "go viral" in the neuromorphic community

Based on looking at high-impact recent papers and community discussion:

### What works

1. **Real hardware + real numbers**: The community is tired of theoretical energy estimates. Papers with actual hardware deployment get way more attention. My SpiNNaker deployment is genuine gold.

2. **Address the AI energy narrative**: The PNAS 2025 paper on AI energy crisis + neuromorphic hit mainstream news. Anything that contributes data to "can neuromorphic computing save AI from its energy problem?" gets amplified.

3. **Honest about limitations**: The community has developed an allergy to overclaiming. Papers that honestly report where SNNs fall short (like my 47.15% vs 63.85% gap) while explaining WHY are more trusted.

4. **Bridge SNN and mainstream ML**: Papers connecting SNNs to transformers, LLMs, or pretrained models generate excitement because they lower the barrier for mainstream ML researchers.

5. **Novel domain + rigorous methodology**: Audio is underexplored. Seven encoding methods is thorough. The combination works well.

6. **Reproducible with code**: Open Neuromorphic's peer review program specifically rewards reproducibility. Publishing code is now table stakes.

7. **Clear narrative / insight**: The most cited papers tell a story. My story -- "the SNN-ANN gap is a feature-learning problem, not a spiking computation problem" -- is the kind of insight that gets remembered and cited.

### What NOT to do:
- Don't claim SNNs are "better" than ANNs without qualification
- Don't ignore the energy break-even threshold literature
- Don't present simulation-only results as "neuromorphic deployment"
- Don't compare unfairly (different parameter counts, training budgets)

---

## 9. New benchmarks, challenges, and competitions

### NeuroBench (active, growing)
- Published in Nature Communications (February 2025)
- 100+ researchers from 50+ institutions
- Algorithm Track v1.0: few-shot continual learning, object detection (event cameras), sensorimotor decoding, predictive modeling
- System Track: deployed execution time, throughput, efficiency
- My project already uses NeuroBench metrics

### Open Neuromorphic Research (ONR)
- Community-driven peer review for open, reproducible neuromorphic research
- Badges for recognized projects
- Launched 2025, actively accepting submissions

### Conference workshops (2025-2026)
- CVPR 2025: Fifth International Workshop on Event-based Vision
- ICCV 2025: 2nd Workshop on Neuromorphic Vision (NeVi)
- IROS 2025: Workshop on Neuromorphic Perception for Real World Robotics
- NeurIPS 2025: Machine Learning and the Physical Sciences (neuromorphic angle)

### No dedicated "neuromorphic audio challenge" exists yet
This is actually an opportunity -- proposing one based on ESC-50 + SNN could be high-impact future work.

---

## 10. The SNN vs ANN narrative: has the community view shifted?

### 2020 view: "SNNs will replace ANNs"
Overoptimistic. Many papers claimed energy efficiency without hardware evidence.

### 2023 view: "SNNs are interesting but not practical"
Skepticism peaked. ImageNet gap was large. No commercial deployments.

### 2025-2026 view: "SNNs are complementary specialized accelerators"
The consensus has settled into a mature, nuanced position:

1. **Energy efficiency is conditional, not automatic**:
   - SNNs need >93% sparsity to beat ANNs on most hardware (Dampfhoffer 2023)
   - Hardware-agnostic estimates: ~50-60% SNN advantage
   - Hardware-aware on conventional hardware: no advantage
   - On actual neuromorphic hardware: 100-1000x advantage IS real (but task-specific)

2. **Accuracy gap is narrowing but not closed**:
   - Best SNN on ImageNet: 83.73% (SGLFormer, 2025) vs best ANN: 90%+
   - Gap is smallest when: pretrained features are used, the task is temporal, or the data is event-driven
   - My PANNs finding (gap collapses from 17pp to 1pp) perfectly illustrates this

3. **The killer app is edge, not cloud**:
   - Always-on sensing (audio, vibration, vision)
   - Battery-powered devices (wearables, IoT, implants)
   - Real-time control (robotics, drones)
   - Privacy-preserving on-device processing

4. **Hybrid architectures are the pragmatic path**:
   - ANN feature extraction + SNN classification (exactly the PANNs approach)
   - ANN training + SNN inference
   - Heterogeneous systems with both conventional and neuromorphic processors

5. **The "biological plausibility" argument has weakened**:
   - Community increasingly values practical performance over bio-plausibility
   - Surrogate gradients (not biologically plausible) are accepted as standard
   - "Brain-inspired" is a design philosophy, not a constraint

Key quote from Nature Communications (2025): "After several false starts, advances now promise widespread commercial adoption."

---

## How my project aligns with these trends

| Trend | My Coverage | Alignment |
|-------|-------------|-----------|
| Hardware deployment | SpiNNaker FC2-only hybrid | STRONG -- real hardware, honest about challenges |
| Energy analysis | NeuroBench metrics, AC/MAC comparison | STRONG -- uses accepted framework |
| Encoding comparison | 7 methods (most for audio) | UNIQUE -- nobody has done this |
| Transfer learning bridge | PANNs+SNN (gap 17pp -> 1pp) | HOT TOPIC -- directly addresses #1 community question |
| Adversarial robustness | FGSM+PGD, SNN dramatically more robust | HOT TOPIC -- aligns with Nature Comms 2025 |
| Novel application domain | ESC-50 (first SNN work on full dataset) | UNIQUE -- fills a clear gap |
| Honest analysis | Report where SNNs fail, explain why | MATCHES community preference |
| Continual learning | 5-task sequential with BWT | RELEVANT -- active area |
| Surrogate gradient analysis | 8-surrogate ablation | NOVEL -- practical guidance |

### Recommended framing for ICONS 2026

**Title ideas:**
1. "Spiking Neural Networks for Environmental Sound Classification: A Systematic Encoding Comparison with SpiNNaker Deployment"
2. "From Spectrograms to Spikes: Encoding, Training, and Deploying SNNs for Audio on Neuromorphic Hardware"
3. "Bridging the Gap: Transfer Learning and Hardware Deployment for Spiking Audio Classification"

**Narratives to emphasize:**
1. First SNN work on ESC-50 -- novelty claim is watertight
2. 7-encoding comparison reveals hierarchy -- practical guidance
3. PANNs+SNN insight -- the gap is feature-learning, not spiking-computation
4. Actual SpiNNaker deployment -- real hardware, not simulation
5. Adversarial robustness -- aligns with Nature Comms 2025
6. Honest energy analysis -- software simulation shows ANN cheaper; neuromorphic hardware flips it

**What NOT to claim:**
- Don't claim SNN beats ANN accuracy (it doesn't, and that's the point)
- Don't claim energy advantage without qualifying software vs hardware
- Don't overstate SpiNNaker accuracy (33.1% with 12.8pp gap)

### Competitive landscape

| Work | Dataset | Methods | Hardware | My Advantage |
|------|---------|---------|----------|---------------|
| Larroza et al. 2025 | ESC-10 only | 3 encodings, FC only | None | Full ESC-50, 7 encodings, CNN, SpiNNaker |
| Dominguez-Morales 2016 | Pure tones | 1 method | SpiNNaker 1 | Complex audio, multiple methods, modern training |
| Yarga et al. 2024 | GSC (speech) | PDM mic + SNN | Simulation | Different domain (env. sound vs speech) |
| Wu & Chua 2018 | RWCP/TIMIT | Robust SNN | None | Modern methods, hardware, more thorough |

---

## Research gaps and follow-up ideas

1. **Loihi 2 deployment**: comparing SpiNNaker vs Loihi would be a separate high-impact paper.

2. **Neuromorphic audio challenge**: proposing an ESC-50 SNN benchmark through Open Neuromorphic or NeuroBench could establish community leadership.

3. **Event-driven audio encoding**: moving from frame-based spectrograms to true event-driven audio (like Hilbert Transform encoding).

4. **Innatera Pulsar evaluation**: getting my model onto Innatera's platform -- the power numbers would be compelling for always-on audio.

5. **SpiNNaker 2 deployment**: SpiNNaker 2 at Sandia/SpiNNcloud would be the obvious hardware upgrade.

---

## Confidence assessment

| Finding | Confidence | Basis |
|---------|-----------|-------|
| No prior SNN work on full ESC-50 | HIGH | Multiple searches, Larroza only covers ESC-10 |
| ICONS has no audio papers (2024-2025) | HIGH | Full schedule analysis |
| Adversarial robustness is hot topic | HIGH | Nature Communications 2025, multiple NeurIPS papers |
| Transfer learning is hot topic | HIGH | KD papers at NeurIPS 2025, ICLR 2026 |
| Hardware deployment valued by reviewers | HIGH | ICONS 2025 schedule analysis, community discourse |
| SNN paper counts at top conferences | HIGH | Cross-referenced GitHub lists and official programs |
| Energy break-even threshold (~6.4% spike rate) | MEDIUM-HIGH | Dampfhoffer 2023, cited in multiple 2025 papers |
| Commercial neuromorphic market projections | MEDIUM | Multiple sources but projections vary widely |
| Community view on SNN vs ANN | HIGH | Synthesized from surveys, Nature papers, discussions |

---

## Sources

### Conference proceedings
- [ICONS 2025 Schedule](https://iconsneuromorphic.cc/Schedule-2025/)
- [ICONS 2026 Call for Papers](https://iconsneuromorphic.cc/calls-2026/)
- [ICONS 2024 Proceedings](https://www.computer.org/csdl/proceedings/icons/2024/22lE6EOwpkA)
- [ICML 2025 Papers](https://icml.cc/virtual/2025/papers.html)
- [NeurIPS 2025 Paper List](https://papercopilot.com/paper-list/neurips-paper-list/neurips-2025-paper-list/)
- [ICLR 2026 Papers](https://iclr.cc/virtual/2026/papers.html)

### High-impact publications
- [Neuromorphic computing enhances robustness through SNNs (Nature Comms, 2025)](https://www.nature.com/articles/s41467-025-65197-x)
- [NeuroBench framework (Nature Comms, 2025)](https://www.nature.com/articles/s41467-025-56739-4)
- [Road to commercial success (Nature Comms, 2025)](https://www.nature.com/articles/s41467-025-57352-1)
- [Neuromorphic computing at scale (Nature, 2025)](https://www.nature.com/articles/s41586-024-08253-8)
- [Can neuromorphic computing reduce AI's energy cost? (PNAS, 2025)](https://www.pnas.org/doi/10.1073/pnas.2528654122)
- [IBM NorthPole (Science, 2023)](https://www.science.org/doi/10.1126/science.adh1174)

### SNN audio and encoding
- [Spike Encoding for Environmental Sound (Larroza et al., March 2025)](https://arxiv.org/abs/2503.11206)
- [SNN and Sound review (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11362401/)
- [Neuromorphic KWS with PDM MEMS (Interspeech 2024)](https://arxiv.org/abs/2408.05156)
- [Low-power SNN Audio Source Localization (Nature Comms Eng, 2025)](https://www.nature.com/articles/s44172-025-00359-9)
- [Spiking Vocos (2025)](https://arxiv.org/html/2509.13049v1)
- [Audio-Visual SNN (2025)](https://arxiv.org/abs/2502.12488)

### Hardware
- [Intel Loihi 2 / Hala Point](https://www.intel.com/content/www/us/en/research/neuromorphic-computing.html)
- [SpiNNaker2 at Sandia (June 2025)](https://www.nextplatform.com/2025/06/16/sandia-deploys-spinnaker2-neuromorphic-system/)
- [BrainChip AKD1500](https://www.edge-ai-vision.com/2025/11/brainchip-unveils-breakthrough-akd1500-edge-ai-co-processor-at-embedded-world-north-america/)
- [Innatera Pulsar at CES 2026](https://innatera.com/press-releases/redefining-the-cutting-edge-innatera-debuts-real-world-neuromorphic-edge-ai-at-ces-2026)
- [Neuromorphic LLM on Loihi 2 (March 2025)](https://arxiv.org/abs/2503.18002)

### Training and methods
- [Training SNNs Using Lessons From Deep Learning (Eshraghian et al., 2023)](https://arxiv.org/abs/2109.12894)
- [Surrogate Gradient Theory (Gygax & Zenke, 2025)](https://direct.mit.edu/neco/article/37/5/886/128506)
- [Adaptive Surrogate Gradients (2025)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2026.1795946/full)
- [Enhanced Self-Distillation for SNNs (NeurIPS 2025)](https://arxiv.org/abs/2510.06254)
- [Reliable Adversarial Robustness Evaluation for SNNs (Wang et al., 2025)](https://arxiv.org/abs/2512.22522)

### Energy and efficiency
- [Are SNNs Really More Energy-Efficient? (Dampfhoffer et al., IEEE TECI 2023)](https://ieeexplore.ieee.org/document/9927729/)
- [Reconsidering SNN Energy Efficiency (2024)](https://arxiv.org/abs/2409.08290)
- [SpikeFit (2025)](https://arxiv.org/html/2510.15542)
- [Energy-Efficient SNN for Edge AI (2025)](https://arxiv.org/html/2602.02439v1)

### Surveys and roadmaps
- [Roadmap to Neuromorphic Computing (2024)](https://arxiv.org/html/2407.02353v1)
- [Solving Neuromorphic Computing's Key Challenges (Open Neuromorphic)](https://open-neuromorphic.org/getting-involved/solving-neuromorphic-computings-key-challenges/)
- [Neuromorphic Computing 2025 Landscape](https://humanunsupervised.com/papers/neuromorphic_landscape.html)
- [Neuromorphic Computing for Embodied Intelligence (2025)](https://arxiv.org/abs/2507.18139)
- [Awesome SNN Conference Papers (GitHub)](https://github.com/AXYZdong/awesome-snn-conference-paper)
- [Awesome Spiking Neural Networks (GitHub)](https://github.com/TheBrainLab/Awesome-Spiking-Neural-Networks)

### Community and ecosystem
- [Open Neuromorphic Community](https://open-neuromorphic.org/)
- [NeuroBench](https://neurobench.ai/)
- [SpiNNcloud Systems](https://spinncloud.com/)
