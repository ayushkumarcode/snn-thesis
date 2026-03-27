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
