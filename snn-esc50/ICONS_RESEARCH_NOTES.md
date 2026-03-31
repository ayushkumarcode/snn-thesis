# ICONS Research Notes -- 80 Papers Organized by Reading Priority

**Date:** 31 March 2026
**Purpose:** Structured reading list for ICONS 2026 submission and thesis background. Papers drawn from ICONS 2022-2025 proceedings, project bibliography, and related SNN/audio/neuromorphic literature.

**Scoring:** Relevance 1-10 where 10 = directly about SNN audio classification or SpiNNaker deployment, 1 = tangentially neuromorphic.

---

## Tier 1 -- Deep Read (20 papers)

These are the papers most directly relevant to our work: SNN audio, SpiNNaker deployment, spike encoding comparisons, energy benchmarking, adversarial robustness in SNNs, and ICONS best papers from 2022-2025. Read fully, take detailed notes, cite in paper/thesis.

| # | Title | Authors (abbreviated) | Year | Venue | Relevance | Key Takeaway |
|---|-------|-----------------------|------|-------|-----------|--------------|
| 1 | Spiking Neural Networks for Environmental Sound Classification | Larroza et al. | 2025 | arXiv:2503.11206 | 10 | Only prior SNN work on ESC-10; FC-only, ~60% with direct encoding -- our direct predecessor, we extend to ESC-50 with conv architecture and hardware deployment. |
| 2 | Multilayer Spiking Neural Network for Audio Samples Classification Using SpiNNaker | Dominguez-Morales et al. | 2016 | ICANN (LNCS 9886) | 10 | Only prior SpiNNaker audio deployment; used pure synthetic tones with silicon cochlea, not comparable to ESC-50 difficulty but establishes SpiNNaker-audio precedent. |
| 3 | The SpiNNaker Project | Furber, Galluppi, Temple, Plana | 2014 | Proceedings of the IEEE 102(5) | 10 | Foundational SpiNNaker architecture paper: ARM968 cores, packet-switched spike routing, fixed-point arithmetic -- essential for understanding our deployment constraints. |
| 4 | NeuroBench: A Framework for Benchmarking Neuromorphic Computing Algorithms and Systems | Yik et al. | 2025 | Nature Communications 16:1589 | 10 | Standardized SynapticOperations metrics (Effective_ACs, MACs, Dense) we use for all energy comparisons; 45nm reference: AC=0.9pJ, MAC=4.6pJ. |
| 5 | Training Spiking Neural Networks Using Lessons from Deep Learning | Eshraghian et al. | 2023 | Proceedings of the IEEE 111(9) | 10 | snnTorch framework paper; covers LIF neurons, surrogate gradients, population coding, loss functions -- our primary training framework. |
| 6 | Evaluating Encoding and Decoding Approaches for Spiking Neuromorphic Systems | Schuman, Rizzo, McDonald-Carmack, Skuda, Plank | 2022 | ICONS 2022 | 10 | Direct ICONS predecessor for encoding comparison; evaluates rate, temporal, population encoding + decoding on TENNLab -- we extend to 7 encodings on audio with hardware. |
| 7 | Efficient Spike Encoding Algorithms for Neuromorphic Speech Recognition | Yarga, Rouat, Wood | 2022 | ICONS 2022 | 9 | Only ICONS audio encoding paper; compares 4 methods (Send-on-Delta, TTFS, LIF, BSA) on FSDD (10 digits) -- closest audio work at ICONS, simulation only. |
| 8 | PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition | Kong et al. | 2020 | IEEE/ACM TASLP 28 | 9 | CNN14 pretrained on AudioSet provides 2048-d embeddings; our PANNs+SNN head experiment shows SNN-ANN gap collapses from 16.7pp to <1pp with these features. |
| 9 | Surrogate Gradient Learning in Spiking Neural Networks | Neftci, Mostafa, Zenke | 2019 | IEEE Signal Processing Magazine 36(6) | 9 | Foundational surrogate gradient paper; explains fast sigmoid, ATan, and STE substitutions for discontinuous spike derivative -- core of our training method. |
| 10 | Are SNNs Really More Energy-Efficient than ANNs? An In-Depth Hardware-Aware Study | Dampfhoffer et al. | 2023 | IEEE TETCI 7(3) | 9 | Shows SNNs need <6.4% spike rate to beat quantized ANNs on CPU; essential nuance for our energy discussion where SNN costs more in simulation but less on neuromorphic HW. |
| 11 | Towards Efficient Deployment of Hybrid SNNs on Neuromorphic and Edge AI Hardware | Seekings, Chandarana, Ardakani, Mohammadi, Zand | 2024 | ICONS 2024 | 9 | Only L3 paper at ICONS 2024 (Loihi + Jetson); DVS gesture classification with full energy comparison -- similar deployment philosophy to ours. |
| 12 | Hardware-Aware Fine-Tuning of Spiking Q-Networks on SpiNNaker2 | Arfa, Vogginger, Mayr | 2025 | ICONS 2025 | 9 | Only SpiNNaker2 paper at ICONS; achieves 32x energy reduction for Q-learning -- shows SpiNNaker platform relevance continues at ICONS. |
| 13 | A Complete Pipeline for Deploying SNNs with Synaptic Delays on Loihi 2 | Meszaros, Knight, Timcheck, Nowotny | 2025 | ICONS 2025 | 9 | L3 deployment on Loihi 2 with ~0% accuracy loss and 250x energy reduction; benchmark for deployment pipeline quality. |
| 14 | A Comprehensive Analysis on Adversarial Robustness of Spiking Neural Networks | Sharmin, Rathi, Panda, Roy | 2019 | IJCNN 2019 | 9 | Shows SNNs with Poisson encoding have higher adversarial accuracy in black-box; rate coding stochasticity as defense mechanism -- supports our FGSM/PGD findings. |
| 15 | Towards Reliable Evaluation of Adversarial Robustness for Spiking Neural Networks | Wang et al. | 2025 | arXiv:2512.22522 | 9 | SA-PGD shows standard PGD underestimates SNN robustness due to vanishing surrogate gradients -- critical caveat for our adversarial results. |
| 16 | sPyNNaker: A Software Package for Running PyNN Simulations on SpiNNaker | Rhodes et al. | 2018 | Frontiers in Neuroscience 12 | 9 | Our deployment interface; IF_curr_exp neuron model, FromListConnector for weight transfer, SpikeSourceArray for input encoding. |
| 17 | Black-Box Adversarial Attacks on SNN for Time Series Data | Hutchins, Ferrer, Fillers, Schuman | 2024 | ICONS 2024 | 8 | Shows SNNs ARE vulnerable via LSTM surrogate (black-box); complementary to our white-box results showing SNN ADVANTAGE -- different attack types, different conclusions. |
| 18 | Interactive Continual Learning for Robots: A Neuromorphic Approach [BEST PAPER] | Hajizada, Berggold, Iacono, Glover, Sandamirskaya | 2022 | ICONS 2022 | 8 | ICONS 2022 best paper; Loihi deployment for continual learning in robotics -- sets the bar for ICONS best paper quality and hardware deployment expectation. |
| 19 | A High-Throughput Low-Latency Interface Board for SpiNNaker-in-the-loop | Romero, Plana, Rowley et al. | 2023 | ICONS 2023 | 8 | Only SpiNNaker 1 paper at ICONS (2022-2025); SPIF interface board for DVS cameras -- infrastructure paper, not classification, but demonstrates SpiNNaker presence at ICONS. |
| 20 | Neuromorphic SNN Deployment for Cognitive Load Classification in ATC | An, Cai, Gallou, Fabrikant, Indiveri, Donati | 2025 | ICONS 2025 | 8 | L3 on DYNAP-SE; 80.6% sim to 73.5% HW = 7.1pp gap -- our 3.8pp gap for rhythm SNN is better; classification deployment benchmark. |

---

## Tier 2 -- Focused Read (30 papers)

SNN energy and efficiency, pruning for neuromorphic, encoding theory, NeuroBench ecosystem, ICONS 2023-2025 proceedings papers with methodological relevance. Read methods and results sections, skim rest.

| # | Title | Authors (abbreviated) | Year | Venue | Relevance | Key Takeaway |
|---|-------|-----------------------|------|-------|-----------|--------------|
| 21 | The Remarkable Robustness of Surrogate Gradient Learning for Instilling Complex Function in SNNs | Zenke, Vogels | 2021 | Neural Computation 33(4) | 8 | Surrogate choice (fast sigmoid, ATan, triangular) has limited impact on final accuracy; supports our surrogate ablation finding. |
| 22 | DIET-SNN: A Low-Latency SNN with Direct Input Encoding and Leakage and Threshold Optimization | Rathi, Roy | 2021/2023 | AAAI / IEEE TNNLS 34(6) | 8 | Direct encoding + learnable leak/threshold achieves strong performance with few timesteps; informs our direct encoding being the best method. |
| 23 | ESC: Dataset for Environmental Sound Classification | Piczak | 2015 | ACM Multimedia | 8 | ESC-50 dataset paper; 2000 recordings, 50 classes, 5-fold CV, human baseline 81.3% -- the benchmark we evaluate on. |
| 24 | Reducing the Spike Rate in Deep Spiking Neural Networks | Fontanini, Esseni, Loghi | 2022 | ICONS 2022 | 8 | Lower spike rate = better energy efficiency in SNNs; directly relevant to our NeuroBench spike rate analysis. |
| 25 | Scalable Event-by-event Processing with Deep State-Space Models [AWARD TALK] | Schone et al. | 2024 | ICONS 2024 | 7 | ICONS 2024 probable best paper; state-space models as alternative to SNNs for event processing -- represents competition/complementary approach. |
| 26 | IM-SNN: Memory-Efficient SNN with Low-Precision Membrane Potentials [AWARD TALK] | Hassan, Meng, Anupreetham, Seo | 2024 | ICONS 2024 | 7 | Low-precision membrane potentials reduce memory with minimal accuracy loss; relevant to our fixed-point SpiNNaker deployment constraints. |
| 27 | How Activity Regularization Harms Pruned SNNs | Krausse, Neher, Knobloch, Becker | 2025 | ICONS 2025 | 7 | Activity regularization (common for energy reduction) actually hurts pruned SNN performance; important caveat for our energy optimization discussion. |
| 28 | Sparsifying Spiking Networks through Local Rhythms | Olin-Ammentorp | 2023 | ICONS 2023 | 7 | Local oscillatory rhythms induce sparsity in SNNs; potentially relevant to our phase/rhythm encoding experiments. |
| 29 | Weight Sparsity Complements Activity Sparsity in Neuromorphic Language Models | Mukherji et al. | 2024 | ICONS 2024 | 7 | Weight pruning + spike sparsity are complementary for energy reduction; applicable to our pruning experiments. |
| 30 | Performance and Energy Simulation of Spiking Neuromorphic Architectures | Boyle, Plagge, Cardwell, Chance, Gerstlauer | 2023 | ICONS 2023 | 7 | SANA-FE simulator for neuromorphic energy estimation; alternative framework to NeuroBench for architectural exploration. |
| 31 | Continuous Learning for Real-Time Auditory Blind Source Separation | Schmitt, Gupta, Abbs | 2024 | ICONS 2024 | 7 | Only audio paper at ICONS 2024; SNN for BSS with continual learning -- shows audio SNN work exists but is rare at ICONS. |
| 32 | Scalable Energy-Efficient Low-Latency Implementations of Trained Spiking DBNs on SpiNNaker | Stromatias et al. | 2015 | IJCNN 2015 | 8 | 784-500-500-10 DBN on SpiNNaker with only 0.06% accuracy gap using Q3.8 fixed-point; proves large fan-in works on SpiNNaker 1. |
| 33 | Event-Driven Implementation of Deep Spiking ConvNets on SpiNNaker | Patino-Saucedo et al. | 2020 | Neural Networks | 8 | Full LeNet on 103-chip SpiNNaker; 98.20% MNIST via SNN toolbox conversion -- best SpiNNaker classification result. |
| 34 | Opportunities for Neuromorphic Computing Algorithms and Applications | Schuman et al. | 2022 | Nature Computational Science 2 | 7 | Survey of neuromorphic opportunities; positions our work within the broader landscape of SNN applications. |
| 35 | Networks of Spiking Neurons: The Third Generation of Neural Network Models | Maass | 1997 | Neural Networks 10(9) | 7 | Foundational theory: SNNs as third-generation networks with temporal coding capability; theoretical underpinning of our work. |
| 36 | Spiking Neuron Models: Single Neurons, Populations, Plasticity | Gerstner, Kistler | 2002 | Cambridge University Press | 7 | Textbook covering LIF model, population coding, STDP -- reference for neuron dynamics background. |
| 37 | Loihi: A Neuromorphic Manycore Processor with On-Chip Learning | Davies et al. | 2018 | IEEE Micro 38(1) | 7 | Intel's Loihi architecture; primary comparison platform to SpiNNaker in our discussion of neuromorphic hardware landscape. |
| 38 | SpiNNaker 2: A 10 Million Core Processor System for Brain Simulation Research | Mayr, Hoeppner, Furber | 2019 | arXiv:1911.02385 | 7 | Next-gen SpiNNaker with digital accelerators; contextualizes our SpiNNaker 1 deployment as stepping stone to SpiNNaker 2. |
| 39 | 1.1 Computing's Energy Problem (and What We Can Do About It) | Horowitz | 2014 | ISSCC Digest | 7 | Seminal reference for MAC vs AC energy costs at different process nodes; 45nm figures underpin our energy argument. |
| 40 | An Energy Budget for Signaling in the Grey Matter of the Brain | Attwell, Laughlin | 2001 | J Cerebral Blood Flow Metab 21(10) | 6 | Biological neuron energy budget (~20fJ/spike); motivates why neuromorphic hardware aims for low per-spike energy. |
| 41 | Elucidating the Theoretical Underpinnings of Surrogate Gradient Learning in SNNs | Gygax, Zenke | 2025 | arXiv:2404.14964 | 7 | Theoretical analysis of why surrogate gradients work; relevant to our surrogate ablation experiments across 8 functions. |
| 42 | Environmental Sound Classification with Convolutional Neural Networks | Piczak | 2015 | IEEE MLSP | 7 | First CNN on ESC-50; established the mel-spectrogram + CNN paradigm we follow for our ANN baseline. |
| 43 | Audio Set: An Ontology and Human-Labeled Dataset for Audio Events | Gemmeke et al. | 2017 | ICASSP | 6 | AudioSet is the pretraining corpus for PANNs; 1.8M clips, 527 tags -- relevant context for why PANNs features are so powerful. |
| 44 | AST: Audio Spectrogram Transformer | Gong, Chung, Glass | 2021 | Interspeech | 6 | Transformer-based ESC-50 SOTA (95.6% at time); represents the accuracy ceiling our SNN is compared against. |
| 45 | Dendritic Learning in Superconducting Optoelectronic Networks [BEST PAPER] | O'Loughlin, Primavera, Shainline | 2023 | ICONS 2023 | 6 | ICONS 2023 best paper; superconducting neuromorphic hardware with dendritic computation -- calibrates ICONS best paper expectations. |
| 46 | Stochastic Spiking Neural Networks with First-to-Spike Coding | Jiang, Lu, Sengupta | 2024 | ICONS 2024 | 6 | First-to-spike coding as alternative temporal encoding; relevant to our latency encoding analysis. |
| 47 | Custom Neuron Model Random Walks on Ornstein-Uhlenbeck [BEST PAPER] | Taylor, Smith, Wang, Kolla, Schmidt | 2025 | ICONS 2025 | 6 | ICONS 2025 best paper; physics application of neuromorphic -- shows ICONS values novel application domains. |
| 48 | On the Role of Time-Constant Hierarchy in SNNs | Moro, Payvand | 2024 | ICONS 2024 | 6 | Multiple time constants improve SNN temporal processing; relevant to our beta/decay parameter choices. |
| 49 | Generating SNN Code Libraries for Embedded Systems [NOMINEE] | Gullett, Mowry, Plank, Whatley, Rizzo, Schuman | 2025 | ICONS 2025 | 6 | SNN deployment on embedded MCUs; alternative deployment target to our SpiNNaker approach. |
| 50 | Optimal Conversion of Conventional ANNs to Spiking Neural Networks | Deng, Gu | 2021 | ICLR | 7 | ANN-to-SNN conversion theory; alternative to our surrogate gradient training -- relevant for discussing why we chose direct training. |

---

## Tier 3 -- Skim (30 papers)

Other neuromorphic applications, other audio ML, cross-domain ICONS proceedings papers. Read abstract and conclusion only; cite if needed for breadth.

| # | Title | Authors (abbreviated) | Year | Venue | Relevance | Key Takeaway |
|---|-------|-----------------------|------|-------|-----------|--------------|
| 51 | A Million Spiking-Neuron Integrated Circuit with a Scalable Communication Network and Interface | Merolla et al. | 2014 | Science 345 | 5 | IBM TrueNorth; 1M neurons, 256M synapses at 70mW -- landmark neuromorphic chip, contextualizes hardware landscape. |
| 52 | The BrainScaleS-2 Accelerated Neuromorphic System with Hybrid Plasticity | Pehle et al. | 2022 | Frontiers in Neuroscience 16 | 5 | Analog neuromorphic accelerator from Heidelberg; another hardware comparison point alongside SpiNNaker and Loihi. |
| 53 | Efficient Neuromorphic Signal Processing with Loihi 2 | Orchard et al. | 2021 | arXiv:2111.03746 | 5 | Loihi 2 capabilities paper; shows what next-gen Loihi can do vs our SpiNNaker 1 deployment. |
| 54 | Spikformer: When Spiking Neural Network Meets Transformer | Zhou et al. | 2023 | ICLR | 5 | Spiking attention mechanism; represents frontier of SNN architecture design beyond our conv approach. |
| 55 | The Heidelberg Spiking Data Sets for the Systematic Evaluation of SNNs | Cramer et al. | 2022 | IEEE TNNLS 33(7) | 6 | SHD/SSC audio spike datasets; alternative audio SNN benchmark -- we chose ESC-50 for broader applicability. |
| 56 | SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition | Park et al. | 2019 | Interspeech | 5 | Time/frequency masking augmentation we apply to mel spectrograms during SNN training. |
| 57 | Overcoming Catastrophic Forgetting in Neural Networks (EWC) | Kirkpatrick et al. | 2017 | PNAS 114(13) | 5 | Elastic Weight Consolidation; foundational continual learning method -- context for our SNN forgetting experiments. |
| 58 | Spiking Neural Networks Resist Catastrophic Forgetting | Golden et al. | 2022 | arXiv:2206.10678 | 6 | Claims SNNs have intrinsic forgetting resistance via offline consolidation; relevant to our CL experiment findings. |
| 59 | Continual Learning in Spiking Neural Networks | Zhang et al. | 2023 | Neural Networks 161 | 5 | NACA method for online SNN continual learning; provides context for our simpler sequential fine-tuning protocol. |
| 60 | Explaining and Harnessing Adversarial Examples (FGSM) | Goodfellow, Shlens, Szegedy | 2015 | ICLR | 5 | Foundational FGSM attack paper; we use this attack in our adversarial robustness evaluation. |
| 61 | Towards Deep Learning Models Resistant to Adversarial Attacks (PGD) | Madry et al. | 2018 | ICLR | 5 | PGD attack definition; our stronger attack (40 steps) used alongside FGSM for robustness evaluation. |
| 62 | OmniVec2: A Novel Transformer Based Network for Large Scale Multimodal and Multitask Learning | Srivastava, Sharma | 2024 | CVPR | 4 | Current ESC-50 SOTA (98.25%); the accuracy ceiling we reference for context. |
| 63 | Adam: A Method for Stochastic Optimization | Kingma, Ba | 2015 | arXiv:1412.6980 | 4 | Our optimizer; standard reference. |
| 64 | librosa: Audio and Music Signal Analysis in Python | McFee et al. | 2015 | SciPy Conference | 4 | Our audio processing library for mel spectrogram extraction. |
| 65 | Torchattacks: A PyTorch Repository for Adversarial Attacks | Kim | 2020 | arXiv:2010.01950 | 4 | Library we use for FGSM and PGD implementation on our SNN/ANN. |
| 66 | A Dataset and Taxonomy for Urban Sound Research | Salamon, Jacoby, Bello | 2014 | ACM Multimedia | 5 | UrbanSound8K dataset; related ESC benchmark, sometimes compared to ESC-50. |
| 67 | Acoustic Scene Classification: Classifying Environments from the Sounds They Produce | Barchiesi et al. | 2015 | IEEE SPM 32(3) | 5 | Survey of acoustic scene classification; broader context for ESC task positioning. |
| 68 | SONYC: A System for Monitoring, Analyzing, and Mitigating Urban Noise Pollution | Bello et al. | 2019 | Communications of the ACM 62(2) | 4 | Real-world urban sound monitoring system; application motivation for efficient audio classification at the edge. |
| 69 | Automatic Acoustic Detection of Birds Through Deep Learning | Stowell et al. | 2019 | Methods in Ecology and Evolution 10(3) | 4 | Bird audio detection; another ESC application domain motivating edge-efficient SNN audio. |
| 70 | An Oscillatory Hierarchy Controlling Neuronal Excitability and Stimulus Processing in the Auditory Cortex | Lakatos et al. | 2005 | J Neurophysiology 94 | 5 | Theta-gamma oscillatory coupling in auditory cortex; biological basis for our phase encoding scheme. |
| 71 | Phase Relationship Between Hippocampal Place Units and the EEG Theta Rhythm | O'Keefe, Recce | 1993 | Hippocampus 3(3) | 4 | Theta-phase precession; original biological evidence for phase coding that our encoding implements. |
| 72 | Phase-of-Firing Coding of Natural Visual Stimuli in Primary Visual Cortex | Montemurro et al. | 2008 | Current Biology 18 | 4 | Phase-of-firing carries information in visual cortex; extends phase coding evidence beyond hippocampus. |
| 73 | Neuronal Population Coding of Movement Direction | Georgopoulos, Schwartz, Kettner | 1986 | Science 233 | 4 | Original population coding paper; our population output decoding is inspired by this biological principle. |
| 74 | Bursts as a Unit of Neural Information: Selective Communication via Resonance | Izhikevich et al. | 2003 | Trends in Neurosciences 26 | 5 | Burst firing as information unit; biological basis for our burst encoding scheme. |
| 75 | Bursts as a Unit of Neural Information: Making Unreliable Synapses Reliable | Lisman | 1997 | Trends in Neurosciences 20 | 4 | Burst coding improves synaptic reliability; complements Izhikevich for burst encoding rationale. |
| 76 | The Impulses Produced by Sensory Nerve-endings: Part II | Adrian, Zotterman | 1926 | J Physiology 61(2) | 3 | Original rate coding discovery; historical context for our rate encoding implementation. |
| 77 | Speed of Processing in the Human Visual System | Thorpe, Fize, Marlot | 1996 | Nature 381 | 4 | Evidence for ultra-fast single-spike coding in biological vision; motivates latency/TTFS encoding. |
| 78 | A 128x128 120dB 15us Latency Asynchronous Temporal Contrast Vision Sensor | Lichtsteiner, Posch, Delbruck | 2008 | IEEE JSSC 43(2) | 4 | Original DVS sensor paper; delta/temporal contrast encoding in our work mirrors this event-driven principle. |
| 79 | Attentive Decision-making and Dynamic Resetting of SRNNs for Streaming Keyword Spotting | Yin, Guo, Corradi, Bohte | 2022 | ICONS 2022 | 6 | Recurrent SNN for audio keyword spotting at ICONS; one of only two audio papers at ICONS 2022. |
| 80 | Design and Implementation of Neuromorphic PID for MAVs [BEST STUDENT PAPER] | Stroobants, Dupeyroux, de Croon | 2022 | ICONS 2022 | 5 | ICONS 2022 best student paper; Loihi deployment for drone control -- L2 hardware deployment benchmark at ICONS. |

---

## Summary Statistics

| Tier | Count | Purpose | Time Estimate |
|------|-------|---------|---------------|
| Tier 1 | 20 | Full read, detailed notes, must-cite | ~40 hours |
| Tier 2 | 30 | Methods + results focus, cite as needed | ~20 hours |
| Tier 3 | 30 | Abstract + conclusion, cite for breadth | ~8 hours |
| **Total** | **80** | | **~68 hours** |

### By Source

- ICONS 2022 proceedings: 5 papers (#6, #7, #18, #24, #79, #80)
- ICONS 2023 proceedings: 4 papers (#19, #28, #30, #45)
- ICONS 2024 proceedings: 7 papers (#11, #17, #25, #26, #29, #31, #46, #48)
- ICONS 2025 proceedings: 6 papers (#12, #13, #20, #27, #47, #49)
- Project bibliography (non-ICONS): 58 papers

### By Topic

- SNN audio / environmental sound: 7 papers (#1, #2, #7, #31, #55, #79, #23)
- SpiNNaker deployment: 7 papers (#2, #3, #12, #16, #19, #32, #33, #38)
- Spike encoding: 5 papers (#6, #7, #22, #46, #50)
- Energy / NeuroBench: 6 papers (#4, #10, #24, #29, #30, #39)
- Adversarial robustness in SNNs: 5 papers (#14, #15, #17, #60, #61)
- SNN fundamentals and training: 8 papers (#5, #9, #21, #35, #36, #41, #50, #54)
- ICONS best papers: 5 papers (#18, #45, #25, #47, #20)
- Neuromorphic hardware (non-SpiNNaker): 5 papers (#37, #51, #52, #53, #49)
- Audio ML / datasets: 7 papers (#23, #42, #43, #44, #56, #62, #64)
- Continual learning: 3 papers (#57, #58, #59)
- Biological coding references: 9 papers (#40, #70, #71, #72, #73, #74, #75, #76, #77, #78)

### Priority Reading Order (first 10)

1. Larroza et al. 2025 -- our direct predecessor
2. Dominguez-Morales et al. 2016 -- only prior SpiNNaker audio
3. Schuman et al. 2022 (ICONS) -- encoding comparison at ICONS
4. Yarga et al. 2022 (ICONS) -- audio encoding at ICONS
5. Seekings et al. 2024 (ICONS) -- L3 deployment benchmark
6. Arfa et al. 2025 (ICONS) -- SpiNNaker2 at ICONS
7. Meszaros et al. 2025 (ICONS) -- Loihi 2 deployment pipeline
8. Dampfhoffer et al. 2023 -- SNN energy reality check
9. Wang et al. 2025 -- adversarial evaluation caveat
10. An et al. 2025 (ICONS) -- hardware accuracy gap benchmark
