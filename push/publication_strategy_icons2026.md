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

### Venues Where Deadlines Have Passed

| Venue | Deadline | Status | Notes |
|-------|----------|--------|-------|
| EUSIPCO 2026 | Feb 13, 2026 | **PASSED** | Larroza et al. submitted their SNN-ESC10 paper to EUSIPCO 2025 |
| ICASSP 2026 | Sep 17, 2025 | **PASSED** | Premiere IEEE audio/speech venue, Barcelona May 2026 |
| ISCAS 2026 | Oct 26, 2025 | **PASSED** | IEEE Circuits & Systems, Shanghai May 2026 |
| NICE 2026 | ~Jan 2026 | **PASSED** | ACM Neuro-Inspired Computational Elements, March 24-26, 2026 |

### Venues Still Open

#### ICONS 2026 -- PRIMARY TARGET
- **Deadline:** April 1, 2026
- **Fit:** PERFECT -- neuromorphic systems, SNN, hardware deployment, benchmarking
- **Format:** 8-page ACM full paper
- **Recommendation:** SUBMIT HERE

#### AICAS 2026 (IEEE AI Circuits and Systems)
- **Deadline:** March 22, 2026 (VERY SOON -- likely passed or imminent)
- **Location:** Ha Long Bay, Vietnam, September 16-18, 2026
- **Fit:** Good -- AI circuits and systems, energy-efficient computing
- **Format:** IEEE conference paper
- **Challenge:** Deadline may have already passed by the time you read this. Also requires IEEE formatting rather than ACM.

#### ICNCE 2026 (International Conference on Neuromorphic Computing and Engineering)
- **Deadline:** April 3, 2026 (abstracts only -- one-page PDF)
- **Location:** Aachen, Germany, June 29 -- July 2, 2026
- **Fit:** Good -- neuromorphic computing, hardware, algorithms
- **Note:** This is abstract-based, not a full paper venue. Good for visibility/networking but not a primary publication. Actively supports student participation.
- **Publisher:** IOP Science (Neuromorphic Computing and Engineering journal)

#### INTERSPEECH 2026
- **Deadline:** February 25, 2026 -- **PASSED**
- **Location:** Sydney, Australia, September 28 -- October 1, 2026
- **Fit:** Moderate -- audio/speech focused but not neuromorphic-oriented
- **Note:** Deadline already passed

#### NeurIPS 2026 Workshops
- **Deadline:** ~October 2026 (workshop proposals); ~September 2026 (main conference)
- **Fit:** Could target a neuromorphic/efficient ML workshop
- **Note:** Main conference is extremely competitive (~20% acceptance). Workshops are more accessible. The Machine Learning with New Compute Paradigms (MLNCP) workshop at NeurIPS has featured neuromorphic work.
- **Recommendation:** Consider as a secondary target for late 2026

#### ICLR 2027
- **Deadline:** ~September 2026
- **Fit:** SNN papers do get accepted (e.g., SMixer at ICLR 2026). However, ICLR expects SOTA-competitive results or strong theoretical contributions.
- **Note:** Very competitive (~30% acceptance). The 47.15% accuracy would need to be framed very carefully. The PANNs+SNN result and the encoding analysis would be the strongest angles.
- **Recommendation:** Reach target -- would need significant framing work

#### ICASSP 2027
- **Deadline:** ~September 2026
- **Fit:** Good -- premiere audio/speech/signal processing venue with SNN presence
- **Note:** Has featured SNN/neuromorphic papers. Very competitive.
- **Recommendation:** Strong secondary target if ICONS acceptance is secured first

### Journal Alternatives

#### Frontiers in Neuroscience -- Neuromorphic Engineering Section
- **Deadline:** Rolling (no deadline)
- **Fit:** Excellent -- dedicated neuromorphic engineering section
- **Open Access:** Yes (APC applies, typically ~$2,950 but some institutional coverage)
- **Impact Factor:** ~3.2
- **Recommendation:** Good fallback if conference submission is rejected. Can submit expanded version anytime.

#### Neuromorphic Computing and Engineering (IOP Science)
- **Deadline:** Rolling
- **Fit:** Excellent -- dedicated journal for exactly this type of work
- **Impact Factor:** New journal, growing rapidly
- **Note:** Published in partnership with ICNCE conference
- **Recommendation:** Strong journal option for expanded version

#### IEEE Transactions on Neural Networks and Learning Systems (TNNLS)
- **Fit:** Good but very competitive
- **Impact Factor:** ~10.4
- **Recommendation:** Reach target for expanded version

---

## Part 5: Competitive Positioning Assessment

### Strengths of Our Paper for ICONS

| Contribution | Strength | Comparable at ICONS? |
|---|---|---|
| First SNN on full ESC-50 (50 classes) | **Very Strong** -- novelty is watertight, confirmed by literature review | No prior ESC-50 SNN paper exists anywhere |
| 7-encoding comparison | **Very Strong** -- most comprehensive encoding comparison for SNN audio | Yarga (ICONS 2022) did 4 encodings on speech; we do 7 on ESC-50 |
| SpiNNaker hardware deployment | **Strong** -- ICONS explicitly values hardware work | SpiNNaker papers appear regularly at ICONS (e.g., 2025 lightning talk) |
| PANNs+SNN (92.5%) | **Strong** -- shows SNN can match ANN with good features | Novel transfer learning approach for neuromorphic audio |
| Adversarial robustness (SNN 26% vs ANN 1.75%) | **Strong** -- dramatic result with practical implications | Adversarial SNN papers appear at ICONS (2024 lightning talk on adversarial attacks) |
| Continual learning (SNN 74.4% vs ANN 81.3% forgetting) | **Moderate** -- incremental finding but adds breadth | Continual learning SNN paper at ICONS 2025 (Bouhadjar et al.) |
| NeuroBench energy benchmarking | **Strong** -- standardized benchmarking valued | NeuroBench is the community standard (Nature Communications 2025) |
| Surrogate gradient ablation (8 functions) | **Moderate** -- useful reference but not novel methodology | Surrogate gradient work published at IJCAI 2023, ICLR 2024 |

### Weaknesses to Address

| Concern | Mitigation |
|---|---|
| 47.15% accuracy on ESC-50 seems low | Frame as: (1) first-ever SNN result on this task, (2) gap analysis is the contribution, (3) PANNs+SNN achieves 92.5% |
| SpiNNaker accuracy (33.1%) is low | Frame as: hardware gap analysis contribution, FC2-only hybrid approach, quantify the gap explicitly |
| Only 1,600 training samples | Acknowledge as a limitation; show that PANNs pre-training overcomes it |
| 8-page limit constrains content | Prioritize: encoding comparison + PANNs + SpiNNaker + adversarial. Move surrogate/continual to supplementary if needed |

### Paper Structure Recommendation for 8 Pages

Given the 8-page limit, prioritize contributions by novelty and impact:

**Must include (core contributions):**
1. Introduction + motivation (0.75 pages)
2. Related work (0.75 pages)
3. Methodology: architecture, dataset, 7 encodings, training (1.5 pages)
4. Results: encoding comparison + analysis (1.5 pages)
5. PANNs+SNN transfer learning result (0.5 pages)
6. SpiNNaker hardware deployment + gap analysis (0.75 pages)
7. Adversarial robustness (SNN vs ANN) (0.5 pages)
8. NeuroBench energy benchmarking (0.5 pages)
9. Conclusion + future work (0.5 pages)
10. References (0.75 pages)

**Include if space permits:**
- Surrogate gradient ablation (can be condensed to a single table)
- Continual learning result (one paragraph + table)

**Move to supplementary / extended version:**
- Per-fold detailed results tables
- Confusion matrices
- t-SNE visualizations
- SpiNNaker calibration details

---

## Part 6: Comparable Papers at Recent Venues

### Papers with Modest Accuracy but Strong Methodology

These demonstrate that accuracy alone does not determine acceptance:

1. **ICONS 2025 Best Paper** -- "Custom and Standard Neuron Model Random Walks on Ornstein-Uhlenbeck Equation" -- This is a theoretical/computational neuroscience paper about turbulence, not a classification task at all. No accuracy metric. Won best paper based on novelty and rigor.

2. **ICONS 2024** -- "Stochastic Spiking Neural Networks with First-to-Spike Coding" (Sengupta group) -- Methodology-focused paper on a novel SNN coding scheme, valued for the approach not the absolute numbers.

3. **ICONS 2024** -- "Continuous Learning for Real-Time Auditory Blind Source Separation" -- Audio application paper at ICONS that focuses on methodology and real-time capability, not SOTA accuracy.

4. **ICONS 2025** -- "SNN for Low-Power Vibration-Based Predictive Maintenance" -- Application benchmark paper where the contribution is demonstrating SNN feasibility in a new domain, not beating ANNs.

5. **Larroza et al. 2025** (submitted to EUSIPCO) -- Reports 69% on ESC-10 (10 classes). Our 47.15% on ESC-50 (50 classes) is arguably more impressive given the 5x harder task. Their best encoding (TAE) gets only F1=0.661.

### Papers Demonstrating Hardware Gap Is Acceptable

1. **SpiNNaker2 DVS Gesture Recognition** (arXiv 2504.06748, 2025) -- Reports 94.13% on SpiNNaker2 using quantization-aware training. But this is SpiNNaker**2** (vastly more capable) on a simpler task (11 gesture classes). Our 33.1% on SpiNNaker1 with 50 classes and FC2-only hybrid approach is a valid contribution documenting real hardware constraints.

2. **Loihi SNN deployments** -- Multiple papers at ICONS/NICE document Loihi accuracy gaps. The community understands and values honest hardware characterization.

### Adversarial Robustness Context

Our SNN adversarial result (SNN 26% vs ANN 1.75% at FGSM eps=0.1) is strong:
- Nature Communications 2025 paper confirms SNN robustness advantage under FGSM
- NeurIPS 2024 FEEL-SNN paper works on improving SNN robustness
- Our finding of 14.8x robustness advantage at eps=0.1 is a striking, publishable result

---

## Part 7: Final Recommendations

### Primary Recommendation: Submit to ICONS 2026

**Deadline: April 1, 2026 (27 days from now)**

**Reasons:**
1. Perfect topical fit -- neuromorphic systems, SNN, hardware, benchmarking
2. Novelty is clear -- first SNN on full ESC-50
3. Multi-contribution paper aligns with what ICONS publishes
4. Hardware deployment (SpiNNaker) is a first-class contribution at ICONS
5. Moderate acceptance rate (historically ~59%, likely slightly lower now) -- very achievable
6. ACM proceedings, properly indexed, open access via UoM institutional agreement (zero APC)
7. Rebuttal process allows addressing reviewer concerns
8. Conference is in Chicago, August 2026

### Secondary Recommendations

| Priority | Venue | Timeline | Action |
|----------|-------|----------|--------|
| 1st | **ICONS 2026** | Submit by April 1, 2026 | 8-page ACM paper |
| 2nd | **ICNCE 2026 abstract** | Submit by April 3, 2026 | 1-page abstract for poster/talk (non-proceedings) |
| 3rd | **NeurIPS 2026 MLNCP Workshop** | Submit ~Oct 2026 | 4-page workshop paper if ICONS succeeds |
| 4th | **Frontiers in Neuroscience** | Anytime | Expanded journal version (~15 pages) |
| 5th | **ICASSP 2027** | ~Sep 2026 | Condensed audio-focused version |

### Formatting Checklist for ICONS 2026 Submission

- [ ] Use ACM Conference Proceedings Template from Overleaf
- [ ] Ensure all authors have ORCID IDs
- [ ] 8 pages maximum (including references)
- [ ] Submit via EasyChair: https://easychair.org/conferences?conf=icons26
- [ ] Follow ACM Publications Policies
- [ ] Title should emphasize neuromorphic/SNN angle, not just "sound classification"

### Suggested Title Options

1. "Spiking Neural Networks for Environmental Sound Classification: A Comprehensive Benchmark on ESC-50 with SpiNNaker Deployment"
2. "From Spike Encoding to Neuromorphic Hardware: A Complete Pipeline for SNN-Based Environmental Sound Classification"
3. "Bridging the SNN-ANN Gap in Audio Classification: Encoding Comparison, Transfer Learning, and SpiNNaker Deployment on ESC-50"

---

## Appendix A: All Venue Deadlines Summary

| Venue | Type | Deadline | Status | Location | Fit |
|-------|------|----------|--------|----------|-----|
| ICONS 2026 | Conference | Apr 1, 2026 | **OPEN** | Chicago, USA | Perfect |
| ICNCE 2026 | Conference (abstracts) | Apr 3, 2026 | **OPEN** | Aachen, Germany | Good |
| AICAS 2026 | Conference | Mar 22, 2026 | Likely passed | Ha Long Bay, Vietnam | Good |
| EUSIPCO 2026 | Conference | Feb 13, 2026 | Passed | Bruges, Belgium | Good |
| INTERSPEECH 2026 | Conference | Feb 25, 2026 | Passed | Sydney, Australia | Moderate |
| ICASSP 2026 | Conference | Sep 17, 2025 | Passed | Barcelona, Spain | Good |
| ISCAS 2026 | Conference | Oct 26, 2025 | Passed | Shanghai, China | Moderate |
| NICE 2026 | Conference | ~Jan 2026 | Passed | TBD | Good |
| NeurIPS 2026 WS | Workshop | ~Oct 2026 | Future | TBD | Good |
| ICASSP 2027 | Conference | ~Sep 2026 | Future | TBD | Good |
| ICLR 2027 | Conference | ~Sep 2026 | Future | TBD | Reach |
| Frontiers NeuroSci | Journal | Rolling | Always open | N/A | Excellent |
| NCE (IOP) | Journal | Rolling | Always open | N/A | Excellent |

## Appendix B: ICONS Papers Most Relevant to Our Work

### Audio/Sound Papers at ICONS and Related Venues

| Paper | Venue/Year | Topic | Relevance |
|-------|-----------|-------|-----------|
| Yarga et al. "Efficient Spike Encoding for Speech Recognition" | ICONS 2022 | 4 encoding comparison on speech digits | **Direct predecessor** -- we extend to 7 encodings on harder task |
| Schmitt et al. "Continuous Learning for Auditory Source Separation" | ICONS 2024 | Audio + continual learning | Audio neuromorphic accepted at ICONS |
| Speech2Spikes | NICE 2023 | Audio-to-spike pipeline, 88.5% on Speech Commands | Related encoding work, Loihi deployment |
| Xylo SNN audio (sub-mW) | ESSCIRC 2022 | Ambient audio, 98% on simple task | Hardware deployment for audio -- different chip |
| Larroza et al. "Spike Encoding for Environmental Sound" | arXiv 2025 (EUSIPCO) | 3 encodings on ESC-10, best 69% | **Closest competitor** -- our ESC-50 work supersedes |
| Dominguez-Morales et al. "SNN Audio on SpiNNaker" | ICANN 2016 | Pure tones on SpiNNaker | Only prior SpiNNaker audio paper |

### Hardware Gap Documentation Papers

| Paper | Venue/Year | Hardware | Gap |
|-------|-----------|----------|-----|
| SpiNNaker2 DVS Gesture | arXiv 2025 | SpiNNaker2 | 94.13% on-chip (close to software) |
| Brian2Loihi emulator | Frontiers 2022 | Loihi | Reasonable discrepancy documented |
| **Ours** | **ICONS 2026** | **SpiNNaker1** | **33.1% vs 46.0% (12.8pp gap)** |

---

## Appendix C: Source URLs

### ICONS Conference
- ICONS 2026 Call for Papers: https://iconsneuromorphic.cc/calls-2026/
- ICONS 2026 EasyChair CFP: https://easychair.org/cfp/ACM-ICONS-2026
- ICONS 2025 Schedule: https://iconsneuromorphic.cc/Schedule-2025/
- ICONS 2024 Schedule: https://iconsneuromorphic.cc/icons-2024/schedule/
- ICONS 2024 DBLP: https://dblp.org/db/conf/icons2/icons2024.html
- ICONS 2024 IEEE Proceedings: https://www.computer.org/csdl/proceedings/icons/2024/22lE6EOwpkA
- ICONS 2023 ACM Proceedings: https://dl.acm.org/doi/proceedings/10.1145/3589737
- ICONS 2022 ACM Proceedings: https://dl.acm.org/doi/proceedings/10.1145/3546790

### ACM Open Access
- ACM Open Access Transition: https://www.acm.org/special-interest-groups/volunteer-resources/acms-transition-to-open-access
- ACM Open Participants: https://libraries.acm.org/acmopen/open-participants
- UoM ACM Open Access Agreement: https://manchester-uk.libanswers.com/OOR/faq/279168

### Comparable Papers
- Yarga et al. ICONS 2022: https://dl.acm.org/doi/10.1145/3546790.3546803
- Larroza et al. 2025: https://arxiv.org/abs/2503.11206
- SpiNNaker2 DVS Gesture: https://arxiv.org/abs/2504.06748
- ICONS 2024 Best Paper (Schone et al.): https://arxiv.org/abs/2404.18508
- Speech2Spikes: https://dl.acm.org/doi/10.1145/3584954.3584995
- NeuroBench: https://www.nature.com/articles/s41467-025-56739-4
- SNN Adversarial Robustness (Nature Comms 2025): https://www.nature.com/articles/s41467-025-65197-x
- Neuromorphic Audio Survey 2025: https://arxiv.org/abs/2502.15056

### Alternative Venues
- EUSIPCO 2026: https://eusipco2026.org/
- ICASSP 2026: https://2026.ieeeicassp.org/
- ISCAS 2026: https://2026.ieee-iscas.org/
- ICNCE 2026: https://icnce-2026.de/
- AICAS 2026: https://2026.ieee-aicas.org/
- INTERSPEECH 2026: https://interspeech2026.org/
- Frontiers Neuromorphic Engineering: https://www.frontiersin.org/journals/neuroscience/sections/neuromorphic-engineering
