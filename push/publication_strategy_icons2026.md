# Publication Strategy for the SNN-ESC50 Paper

ICONS 2026 is the target. here's my analysis of why it's right and what else is available.

---

## Part 1: ICONS 2026 -- the main target

### conference details

| Field | Details |
|-------|---------|
| Full Name | ACM International Conference on Neuromorphic Systems (ICONS 2026) |
| Edition | 10th |
| Location | Chicago, Illinois, USA |
| Dates | August 4-6, 2026 |
| Publisher | ACM (fully open access from 2026) |
| Submission | EasyChair: https://easychair.org/conferences?conf=icons26 |
| Website | https://iconsneuromorphic.cc/ |

### key deadlines

| Milestone | Date |
|-----------|------|
| Paper submission | **April 1, 2026 (AoE)** |
| Reviews due | May 13, 2026 |
| Reviews to authors | May 18, 2026 |
| Author rebuttals due | May 25, 2026 |
| Final notification | June 5, 2026 |
| Camera-ready | June 24, 2026 |
| Conference | August 4-6, 2026 |

rebuttals are permitted which is nice -- opportunity to address reviewer concerns.

### formatting

- full papers: 8 pages max (priority for 20-min talks, otherwise posters)
- short papers: 4 pages max (10-min talks, otherwise posters)
- ACM Conference Proceedings Template (Overleaf)
- all authors need ORCID IDs

### cost

ACM went open access from January 2026. UoM IS a participating ACM Open institution (checked via Manchester library's ACM Open Access Agreement page), so **zero APC** for corresponding authors affiliated with UoM. even without this, the subsidized APC would be $250 (ACM member) or $350 (non-member).

### acceptance rate

only public figure is ICONS 2018: 13/22 accepted (59%). relatively high compared to top ML conferences (NeurIPS ~20%, ICLR ~30%). it's a specialized community conference. growing though -- 2024 and 2025 had way more papers than 2018.

### why ICONS fits us

it's not a top-tier venue like NeurIPS but it's the **premier dedicated venue for neuromorphic systems**. the go-to conference for neuromorphic hardware, SNN algorithms, brain-inspired computing.

strengths of publishing here:
- ACM proceedings with DOI, properly indexed
- directly targets the neuromorphic community who will actually appreciate our work
- hardware deployment papers (SpiNNaker, Loihi) are first-class citizens
- benchmark and methodology papers are welcome
- modest accuracy results are normal and accepted
- community is growing rapidly

limitations:
- not widely recognized outside neuromorphic community
- lower citation impact vs ICASSP, NeurIPS, ICLR
- relatively low h5-index

### topics they explicitly welcome (and how we fit)

1. **Systems and Architecture:** neuromorphic circuits, non-von Neumann -- *our SpiNNaker deployment fits here*
2. **Algorithms and Training:** supervised/unsupervised learning, continual learning -- *our encoding comparison, surrogate ablation, CL experiment*
3. **Applications:** energy-efficient edge AI, **benchmark tasks**, domain-specific implementations -- *ESC-50 as SNN audio benchmark is exactly this*
4. **Software and Tools:** efficient simulation -- *NeuroBench integration fits*

---

## Part 2: What's Been Accepted at ICONS Before

### ICONS 2024 papers

full list from DBLP. key ones:

1. "Scalable Event-by-event Processing with Deep State-Space Models" -- Schone et al. **[BEST PAPER]** (99.2% DVS Gestures)
2. "Stochastic Spiking Neural Networks with First-to-Spike Coding" -- Sengupta group
3. "Towards Efficient Deployment of Hybrid SNNs on Neuromorphic and Edge AI Hardware" -- Seekings et al.
4. **"Continuous Learning for Real-Time Auditory Blind Source Separation"** -- Schmitt et al. *[AUDIO]*
5. multiple application-focused papers where methodology matters more than raw accuracy

paper #4 is audio-related neuromorphic work at ICONS. audio/sound processing IS represented, though it's a minority topic.

### ICONS 2025 papers

**Best Paper:** "A Comparison of Custom and Standard Neuron Model Random Walks on the Ornstein-Uhlenbeck Equation for Simplified Turbulence" -- not a classification task at all. won best paper on novelty and rigor.

notable for our positioning:
- **Paper #20:** "Hardware-Aware Fine-Tuning of Spiking Q-Networks on SpiNNaker2" -- Arfa et al. SpiNNaker papers are welcome.
- **Paper #26:** "Unsupervised continual learning of complex sequences in SNNs" -- Bouhadjar et al. CL in SNNs is represented.
- **Paper #27:** "SNNs for Low-Power Vibration-Based Predictive Maintenance" -- application benchmark with modest results. benchmark-style work is accepted.

broad range of work from theoretical to deeply applied.

### ICONS 2022 -- directly comparable audio paper

"Efficient Spike Encoding Algorithms for Neuromorphic Speech Recognition" (Yarga, Rouat, Wood -- Sherbrooke). compared 4 spike encoding methods for digit classification. our paper is significantly more comprehensive: 7 encodings vs 4, ESC-50 vs speech digits, hardware deployment, adversarial/continual.

---

## Part 3: Competitive Landscape

### direct competitors

| Paper | Year | Venue | Dataset | Classes | Accuracy | Hardware |
|-------|------|-------|---------|---------|----------|----------|
| **ours** | **2026** | **ICONS** | **ESC-50** | **50** | **47.15% (direct), 92.5% (PANNs+SNN)** | **SpiNNaker (33.1%)** |
| Larroza et al. | 2025 | arXiv (EUSIPCO) | ESC-10 | 10 | 69.0% (TAE) | None |
| Dominguez-Morales et al. | 2016 | ICANN | Pure tones | ~10 | High (trivial task) | SpiNNaker |
| Yarga et al. | 2022 | ICONS | Speech digits | 10 | Matched CNN baseline | None |
| Speech2Spikes | 2023 | NICE | GSC | 35 | 88.5% | Intel Loihi |
| Xylo SNN audio | 2022 | ESSCIRC | Ambient sounds | ~5 | 98% | Xylo (sub-mW) |

### why 47.15% is publishable at ICONS

this is the obvious concern. here's why i think it's fine:

1. **ICONS values methodology over accuracy.** their scope explicitly welcomes "benchmark tasks for neuromorphic computing." the scientific contribution is the comparison, not SOTA.

2. **context matters.** 47.15% on 50 classes (random = 2%) is meaningful. ESC-50 human performance is 81.3%. the gap IS the finding.

3. **PANNs result rehabilitates the SNN.** 92.5% with PANNs+SNN shows the gap is feature learning, not spiking computation. that's a key insight.

4. **comparable precedents.** Larroza reports 69% on 10 classes (simpler task). Yarga at ICONS 2022 focused on encoding quality not absolute accuracy. ICONS 2025 best paper was about turbulence modeling with neuron random walks -- no classification accuracy at all.

5. **hardware deployment is a separate contribution.** SpiNNaker results (33.1%) are about demonstrating feasibility and analyzing the hardware gap.

---

## Part 4: Alternative Venues

### deadlines that have passed

| Venue | Deadline | Notes |
|-------|----------|-------|
| EUSIPCO 2026 | Feb 13, 2026 | Larroza submitted their paper to EUSIPCO |
| ICASSP 2026 | Sep 17, 2025 | premiere IEEE audio venue |
| ISCAS 2026 | Oct 26, 2025 | circuits & systems |
| NICE 2026 | ~Jan 2026 | ACM neuromorphic |

### still open

**ICONS 2026 (April 1)** -- primary target, perfect fit, draft already exists.

**AICAS 2026 (March 22)** -- IEEE AI Circuits and Systems, Ha Long Bay Vietnam. good fit. might have already passed.

**ICNCE 2026 (April 3)** -- Aachen, Germany. abstracts only (1-page PDF), non-proceedings. good for visibility/networking, low effort. student participation sponsored.

**NeurIPS 2026 Workshops (~Oct 2026)** -- could target MLNCP workshop (neuromorphic/efficient ML). workshops are more accessible than main conf (~20% acceptance). secondary target for late 2026.

**ICLR 2027 (~Sep 2026)** -- very competitive (~30%). 47.15% would need careful framing. PANNs+SNN and encoding analysis would be the strongest angles. reach target.

**ICASSP 2027 (~Sep 2026)** -- premiere audio/speech/signal processing. has featured SNN papers. strong secondary target.

### journal alternatives

**Frontiers in Neuroscience (Neuromorphic Engineering section)** -- rolling deadline, IF ~3.2, excellent fit. good fallback if conference rejected. expanded ~15 page version anytime.

**Neuromorphic Computing and Engineering (IOP)** -- rolling, new journal growing rapidly. dedicated to exactly this work. strong option.

**IEEE TNNLS** -- IF ~10.4. good but very competitive. reach target for expanded version.

---

## Part 5: Strengths and Weaknesses for ICONS

### strengths

| Contribution | Strength | Comparable at ICONS? |
|---|---|---|
| First SNN on full ESC-50 | Very strong -- novelty watertight | No prior ESC-50 SNN paper exists anywhere |
| 7-encoding comparison | Very strong -- most comprehensive for SNN audio | Yarga (ICONS 2022) did 4 on speech; we do 7 on ESC-50 |
| SpiNNaker deployment | Strong -- ICONS values hardware | SpiNNaker papers appear regularly |
| PANNs+SNN (92.5%) | Strong -- shows SNN can match ANN | Novel transfer learning for neuromorphic audio |
| Adversarial robustness | Strong -- dramatic result | SNN adversarial papers appear at ICONS |
| Continual learning | Moderate -- adds breadth | CL SNN paper at ICONS 2025 |
| NeuroBench energy | Strong -- standardized benchmarking | NeuroBench is the community standard |
| Surrogate gradient ablation | Moderate -- useful reference | Surrogate work at IJCAI 2023, ICLR 2024 |

### weaknesses to address

| Concern | Mitigation |
|---|---|
| 47.15% looks low | Frame as first-ever + gap analysis is the contribution + PANNs gets 92.5% |
| SpiNNaker 33.1% looks low | Hardware gap analysis contribution, FC2-only hybrid, explicit gap quantification |
| Only 1,600 training samples | Acknowledge as limitation; PANNs overcomes it |
| 8-page limit | Prioritize encoding + PANNs + SpiNNaker + adversarial; supplementary for rest |

### paper structure for 8 pages

must include:
1. Introduction + motivation (0.75 pages)
2. Related work (0.75 pages)
3. Methodology: architecture, dataset, 7 encodings, training (1.5 pages)
4. Results: encoding comparison + analysis (1.5 pages)
5. PANNs+SNN transfer learning (0.5 pages)
6. SpiNNaker deployment + gap analysis (0.75 pages)
7. Adversarial robustness (0.5 pages)
8. NeuroBench energy (0.5 pages)
9. Conclusion + future work (0.5 pages)
10. References (0.75 pages)

include if space: surrogate gradient ablation (condensed to one table), continual learning (one paragraph + table).

move to supplementary: per-fold tables, confusion matrices, t-SNE, SpiNNaker calibration details.

---

## Part 6: Final Recommendations

### submit to ICONS 2026

**deadline: April 1, 2026**

reasons:
1. perfect topical fit
2. clear novelty -- first SNN on full ESC-50
3. multi-contribution paper matches what ICONS publishes
