# Publication Strategy: SNN-ESC50 Paper

**Date:** 5 March 2026
**Author:** Research Analysis for COMP30040 Thesis Project
**Primary Target:** ICONS 2026 (ACM International Conference on Neuromorphic Systems)

---

## Executive Summary

ICONS 2026 is the ideal primary venue for this paper. The conference explicitly welcomes benchmark studies, hardware deployment work, and SNN algorithm research -- all of which this paper delivers. The paper's unique combination of (a) first-ever SNN study on full 50-class ESC-50, (b) comprehensive 7-encoding comparison, (c) SpiNNaker hardware deployment, and (d) adversarial/continual learning analysis gives it a strong multi-contribution profile that aligns perfectly with ICONS's scope. The 47.15% accuracy on ESC-50 is not a weakness at ICONS -- the conference values methodology, hardware integration, and scientific insight over raw accuracy numbers. Papers at ICONS 2024 and 2025 regularly feature modest accuracy results when accompanied by meaningful neuromorphic contributions.

The EUSIPCO 2026 deadline has already passed (13 February 2026). ICASSP 2026 deadline has also passed (September 2025). The realistic alternative venues are AICAS 2026 (deadline 22 March 2026 -- very soon), ICNCE 2026 (abstracts due 3 April 2026), and journal submissions to Frontiers in Neuroscience (Neuromorphic Engineering section) or Neuromorphic Computing and Engineering (IOP).

---

## Part 1: ICONS 2026 -- Primary Target

### Conference Details

| Field | Details |
|-------|---------|
| **Full Name** | ACM International Conference on Neuromorphic Systems (ICONS 2026) |
| **Year** | 10th edition |
| **Location** | Chicago, Illinois, USA |
| **Dates** | August 4--6, 2026 |
| **Publisher** | ACM (fully open access from 2026) |
| **Submission Portal** | EasyChair: https://easychair.org/conferences?conf=icons26 |
| **Website** | https://iconsneuromorphic.cc/ |

### Key Deadlines

| Milestone | Date |
|-----------|------|
| Paper submission deadline | **April 1, 2026 (AoE)** |
| Reviews due | May 13, 2026 |
| Reviews to authors | May 18, 2026 |
| Author rebuttals due | May 25, 2026 |
| Final notification | June 5, 2026 |
| Camera-ready papers due | June 24, 2026 |
| Conference | August 4--6, 2026 |

**Note:** Rebuttals are permitted, which is favorable -- it gives an opportunity to address reviewer concerns.

### Formatting Requirements

- **Full papers:** 8 pages maximum (priority for 20-minute talks; otherwise posters)
- **Short papers:** 4 pages maximum (eligible for 10-minute talks; otherwise posters)
- **Template:** ACM Conference Proceedings Template (available on Overleaf)
- **ORCID:** All authors must have ORCID IDs
- **Tutorials/Special Sessions:** 3 pages (not in proceedings)

### Publication and Cost

- ACM open access from January 2026 onward
- **University of Manchester IS a participating ACM Open institution** -- confirmed via Manchester library's ACM Open Access Agreement page. This means **zero APC (Article Processing Charge)** for corresponding authors affiliated with UoM.
- Even without institutional participation, the 2026 subsidized APC would be $250 (ACM member) or $350 (non-member)

### Acceptance Rate

Historical data is limited. The only publicly available figure is from ICONS 2018: 13 of 22 submissions accepted (59%). This is a relatively high acceptance rate compared to top-tier ML conferences (NeurIPS ~20%, ICLR ~30%), but consistent with a specialized community conference. ICONS is growing -- the 2024 and 2025 editions had significantly more papers than 2018, so the rate may have decreased.

### Conference Prestige and Positioning

**Tier:** ICONS is a specialized, niche conference -- not a top-tier venue like NeurIPS or ICLR, but the **premier dedicated venue for neuromorphic systems**. It is the go-to conference for researchers working specifically on neuromorphic hardware, SNN algorithms, and brain-inspired computing.

**Strengths of publishing here:**
- ACM proceedings with DOI -- proper indexed publication
- Directly targets the neuromorphic community who will appreciate the work
- Hardware deployment papers (SpiNNaker, Loihi) are first-class citizens
- Benchmark and methodology papers are welcomed
- Modest accuracy results are normal and accepted
- Community is growing rapidly given industry interest in neuromorphic computing

**Limitations:**
- Not widely recognized outside the neuromorphic community
- Lower citation impact compared to ICASSP, NeurIPS, ICLR
- Google Scholar h5-index is relatively low for a conference

### Topics Explicitly Welcomed by ICONS 2026

The call for papers lists four main areas, all of which this paper addresses:

1. **Systems and Architecture:** "Neuromorphic circuits or sensors," non-von Neumann architectures -- *our SpiNNaker deployment directly fits here*
2. **Algorithms and Training:** "Supervised, unsupervised and self-supervised learning methods," biologically-inspired approaches, continual learning -- *our encoding comparison, surrogate gradient ablation, and continual learning experiment fit here*
3. **Applications:** Energy-efficient edge AI, **benchmark tasks**, neuromorphic datasets, domain-specific implementations -- *ESC-50 as a benchmark for SNN audio is exactly this*
4. **Software and Tools:** Efficient simulation techniques -- *NeuroBench integration fits here*

---

## Part 2: Analysis of ICONS Accepted Papers (2022--2025)

### ICONS 2024 -- Full Paper List (from DBLP)

The following papers were accepted as full or short papers at ICONS 2024 (Arlington, VA, July 30--Aug 2, 2024):

1. "Scalable Event-by-event Processing of Neuromorphic Sensory Signals With Deep State-Space Models" -- Schone et al. **[BEST PAPER AWARD]** (99.2% DVS Gestures)
2. "IM-SNN: Memory-Efficient Spiking Neural Network with Low-Precision Membrane Potentials and Weights" -- Hassan et al.
3. "Programmable Synapses and Dendritic Circuits for Superconducting Optoelectronic Neuromorphic Computing" -- Primavera et al.
4. "Stochastic Spiking Neural Networks with First-to-Spike Coding" -- Jiang, Lu, Sengupta
5. "Edge Device CNN Classification Using Eventized RF Fingerprints" -- Smith et al.
6. "Real-Time Supervised SNN for Cerebellar Purkinje Cells Spike Detection and Classification" -- Raisiardali et al.
7. "Neuromorphic Wireless Device-Edge Co-Inference via Directed Information Bottleneck" -- Ke et al.
8. "Neuromorphic Computing for the Masses" -- Matinizadeh et al.
9. "Solving Minimum Spanning Tree Problem in Spiking Neural Networks" -- Janssen et al.
10. "Asynchronous Multi-Fidelity Hyperparameter Optimization of SNNs" -- Firmin et al.
11. "Neuro-Spark: Submicrosecond SNN Architecture for In-Sensor Filtering" -- Miniskar et al.
12. "Towards Efficient Deployment of Hybrid SNNs on Neuromorphic and Edge AI Hardware" -- Seekings et al.
13. "Temporal and Spatial Reservoir Ensembling for Liquid State Machines" -- Biswas et al.
14. "Timing Actions in Games Through Bio-Inspired Reinforcement Learning" -- Ambrosini et al.
15. **"Continuous Learning for Real-Time Auditory Blind Source Separation Applications"** -- Schmitt et al. *[AUDIO-RELATED]*
16. "TRIP: Trainable Region-of-Interest Prediction for Hardware-Efficient Neuromorphic Processing" -- Arjmand et al.
17. "Supervised Radio Frequency Interference Detection with SNNs" -- (authors truncated)
18. "Variation-Aware Non-linear Mapping for Honey-Memristor Based Neuromorphic System" -- Uppaluru et al.

**Key observation:** Paper #15 is an audio-related neuromorphic paper at ICONS 2024. Audio/sound processing IS represented at ICONS, though it is a minority topic.

### ICONS 2025 -- Full Paper List (from schedule)

ICONS 2025 (Bellevue/Seattle, July 29--31, 2025) accepted the following:

**Best Paper Award:** "A Comparison of Custom and Standard Neuron Model Random Walks on the Ornstein-Uhlenbeck Equation for Simplified Turbulence" -- Taylor et al.

**Full Talks (11 papers):**
1. "Neuromorphic Closed-Loop Control with Spiking Motor Neuron and Muscle Spindle Models" -- Stoll et al.
2. Best Paper (above)
3. "Generating Spiking Neural Network Code Libraries for Embedded Systems" -- Gullett et al.
4. "Optimizing generalized feedback paths for credit assignment" -- Western et al.
5. "Izhikevich-Inspired Temporal Dynamics for Privacy in SNNs" -- Moshruba et al.
6. "EEvAct: Early Event-Based Action Recognition with Two-Stream SNNs" -- Neumeier et al.
7. "Quantitative evaluation of brain-inspired vision sensors" -- Wang et al.
8. "Quantizing Small-Scale State-Space Models for Edge AI" -- Zhao et al.
9. "Neuromorphic Deployment of SNNs for Cognitive Load Classification in Air Traffic Control" -- An et al.
10. "How to Train an Oscillator Ising Machine using Equilibrium Propagation" -- Gower
11. "Uncertainty-Aware Spiking Neural Networks for Regression" -- Sun & Bohte

**Lightning Talks (15+ papers):**
12. "GRASP: Dynamic and Priority-Aware Gradient Sparsification" -- Swaminathan & Sampson
13. "Do Spikes Protect Privacy? Black-Box Model Inversion Attacks in SNNs" -- Poursiami et al.
14. "Model-Free Multiplexed Gradient Descent: Neuromorphic Learning" -- O'Loughlin et al.
15. "Vibe2Spike: Wireless, Batteryless Vibration Sensing with Event Cameras and SNNs" -- Scott et al.
16. "Constant Depth Threshold Circuits For Exhaustive Epistasis Detection" -- Ribeiro et al.
17. "Synaptic Sampling Networks with True Random Number Generation" -- Aimone et al.
18. "How Activity Regularization Harms Pruned SNNs" -- Krausse et al.
19. "An Empirical Study on Input Distribution Impact on Reservoir Computer Performance" -- Thelen & Ravindra
20. **"Hardware-Aware Fine-Tuning of Spiking Q-Networks on the SpiNNaker2 Neuromorphic Platform"** -- Arfa et al. *[SPINNAKER-RELATED]*
21. "Exploring Dendrites in Large-Scale Neuromorphic Architectures" -- Boyle et al.
22. "A Complete Pipeline for deploying SNNs with Synaptic Delays on Loihi 2" -- Meszaros et al.
23. "Propeller-Based Drone Tracking with a Moving Neuromorphic Camera" -- Murray & Nowzari
24. "NAP: Neuromorphic Artificial Pancreas" -- Rizzo et al.
25. "DESTformer: Energy-Efficient Monocular Depth Estimation with Spiking Transformer" -- Tumpa et al.
26. **"Unsupervised continual learning of complex sequences in spiking neuronal networks"** -- Bouhadjar et al. *[CONTINUAL LEARNING]*
27. **"Spiking Neural Networks for Low-Power Vibration-Based Predictive Maintenance"** -- Vasilache et al. *[APPLICATION BENCHMARK]*
28. "SpikeRL: Scalable and Energy-efficient Deep Spiking Reinforcement Learning" -- Tahmid et al.
29. "Energy-Efficient Adiabatic Circuits for Neuromorphic Tactile Sensing with E-Prop Learning" -- Muller-Cleve et al.
30. "VRISP: A Vectorized Open-Source Simulator for Neuromorphic Computing" -- Mowry & Plank
31. "Neuromorphic Cybersecurity with Semi-supervised Lifelong Learning" -- Mia et al.

**Key observations for our paper's positioning:**
- Paper #20: SpiNNaker2 paper accepted at ICONS 2025 -- hardware deployment papers welcome
- Paper #26: Continual learning in SNNs -- directly related to our continual learning experiment
- Paper #27: Application benchmark paper with modest results -- benchmark-style work is accepted
- Multiple papers with application-focused work where methodology matters more than raw accuracy
- The conference accepts a broad range of work, from theoretical to deeply applied

### ICONS 2022 -- Directly Comparable Audio Paper

**"Efficient Spike Encoding Algorithms for Neuromorphic Speech Recognition"** (Yarga, Rouat, Wood -- Universite de Sherbrooke)
- Venue: ICONS 2022 (Knoxville, TN)
- Topic: Compared 4 spike encoding methods for speaker-independent digit classification
- Results: Send-on-Delta variants matched state-of-the-art CNN baseline while reducing spike bit rate
- This is the closest existing ICONS paper to ours -- and our paper is significantly more comprehensive (7 encodings vs 4, ESC-50 vs speech digits, hardware deployment, adversarial/continual)

---

## Part 3: Competitive Landscape -- Most Comparable Papers

### Direct Competitors (SNN + Environmental Sound)

| Paper | Year | Venue | Dataset | Classes | Accuracy | Hardware |
|-------|------|-------|---------|---------|----------|----------|
| **Ours** | **2026** | **ICONS 2026** | **ESC-50** | **50** | **47.15% (direct), 92.5% (PANNs+SNN)** | **SpiNNaker (33.1%)** |
| Larroza et al. | 2025 | arXiv (submitted to EUSIPCO) | ESC-10 | 10 | 69.0% (TAE), 40.9% (SF), 35.4% (MW) | None |
| Dominguez-Morales et al. | 2016 | ICANN (Springer LNCS) | Pure tones (130-1397 Hz) | ~10 frequencies | High (simple task) | SpiNNaker |
| Yarga et al. | 2022 | ICONS 2022 | Speech digits | 10 digits | Matched CNN baseline | None |
| Speech2Spikes | 2023 | NICE 2023 | Google Speech Commands | 35 commands | 88.5% | Intel Loihi (demo) |
| Xylo SNN audio | 2022 | ESSCIRC 2022 | Ambient sounds | ~5 classes | 98% | Xylo (sub-mW) |

### Why Our Paper Is Unique

1. **First SNN on full ESC-50:** Every prior SNN audio paper uses simpler datasets (ESC-10, speech commands, pure tones). Nobody has tackled the full 50-class environmental sound challenge.

2. **Most comprehensive encoding comparison:** 7 encodings is the largest comparison in SNN audio literature. Larroza compared 3 encodings on ESC-10. Yarga compared 4 on speech digits.

3. **Hardware deployment:** Only Dominguez-Morales (2016) has done SpiNNaker audio, but on pure tones. We deploy on a real 50-class task.

4. **Multi-dimensional analysis:** Adversarial robustness, continual learning, energy benchmarking, surrogate gradient ablation -- no single prior paper covers all of these.

### What About The 47.15% Accuracy?

This is a critical question. Here is why 47.15% is publishable at ICONS:

1. **ICONS values methodology over accuracy:** The conference scope explicitly welcomes "benchmark tasks for neuromorphic computing." The scientific contribution is the systematic comparison, not hitting SOTA.

2. **Context matters:** 47.15% on 50 classes (random baseline = 2%) is a meaningful result. ESC-50 human performance is 81.3%. ANN SOTA is 98.25%. The gap IS the scientific finding.

3. **The PANNs result rehabilitates the SNN:** 92.5% with PANNs+SNN shows the gap is in feature learning, not spiking computation. This is a key scientific insight.

4. **Comparable precedents exist:** The Larroza et al. paper (submitted to EUSIPCO) reports 69% on ESC-10 (10 classes, simpler task). The Yarga ICONS 2022 paper focused on encoding comparison quality, not absolute accuracy. The ICONS 2025 best paper was about turbulence modeling with neuron random walks -- not about achieving high classification accuracy at all.

5. **Hardware deployment adds a separate contribution dimension:** The SpiNNaker results (33.1%) are about demonstrating feasibility and analyzing the hardware gap, not about beating software performance.

---

## Part 4: Alternative Publication Venues

