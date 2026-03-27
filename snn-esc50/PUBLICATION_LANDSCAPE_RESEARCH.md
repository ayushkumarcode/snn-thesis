# publication landscape notes -- SNN & neuromorphic computing

research date: 25 march 2026
purpose: figure out the competitive landscape and what angle to push for ICONS 2026

---

## where we stand

our paper already has multiple genuine "firsts":
- first convolutional SNN on ESC-50 (confirmed, no prior work exists)
- first neuromorphic hardware deployment for environmental sound classification
- first adversarial robustness analysis of SNNs on audio
- first 7-encoding systematic comparison on a 50-class audio task
- first cross-encoding transfer analysis on any benchmark

the gap-collapse finding (16.7pp to 0.95pp) is probably the single most publishable result. it's not just a number -- it's a mechanistic insight about WHERE the SNN bottleneck is (feature learning, not spiking computation). nobody's demonstrated this cleanly for audio before, and it has direct implications for hybrid neuromorphic system design.

for ICONS specifically: 59% acceptance rate (13/22 in 2018), accepts papers across a wide quality range (undergrad work to national lab research), and explicitly wants "applications" and "neuromorphic benchmarks." we have hardware deployment (SpiNNaker), novel application domain (ESC-50), and multiple quantified findings. should be within the acceptance bar.

what my supervisor probably means by "one big finding": the paper currently presents many small-to-medium findings. the risk is it reads as a survey/benchmark paper rather than having a clear thesis. the fix is framing, not new experiments.

---

## top impactful SNN papers (2023-2025)

### tier 1: landmark papers (500+ citations or best paper)

Eshraghian et al. "Training SNNs Using Lessons from Deep Learning" -- Proc. IEEE 2023, 500+ citations, 100K+ snnTorch downloads. bridge paper that made SNNs accessible to DL community.

QKFormer -- NeurIPS 2024 Spotlight (top 3%). first directly-trained SNN to exceed 85% on ImageNet (85.65%, +10.84pp over Spikformer). classic "first X to exceed Y" framing.

SpikeLLM -- ICLR 2025. first spiking LLM (7-70B params). 11% perplexity reduction on WikiText2. "first X applied to Y" on the hottest topic.

NeuroBench -- Nature Communications 2025. standardized benchmarking for neuromorphic computing. 60+ institutions.

### tier 2: high-impact conference papers

Spikformer v2 (AAAI 2024) - 81.1% ImageNet with 1 timestep
SpikingResformer (CVPR 2024) - bridge between ResNet and ViT in SNNs
SpikedAttention (NeurIPS 2024) - training-free transformer-to-SNN conversion
Spike-based Sound Source Localization (NeurIPS 2024) - RF neurons, SOTA + robustness
Nature Comms 2025 robustness paper - SNNs 2x robustness of ANNs via temporal processing
Speech2Spikes (ACM NICE 2023) - 109x lower energy than GPU, 23x lower than CPU
SATRN (Electronics 2025) - spiking audio tagging, comparable to CNN, better noise robustness
Spikingformer (AAAI 2026) - pure event-driven spiking transformer, 75.85% ImageNet, 57.34% less energy
Chen et al. (Nature Comms 2024) - 0.3 spikes/neuron, minimal accuracy loss
"Exploiting Noise as Resource" (Patterns 2023) - stochastic resonance improves generalization
SpiNNaker2 (arXiv 2024) - 22nm, 153 ARM cores/chip, 10x capacity per watt

the key observation about "wow" numbers -- the most impactful papers always have ONE clear number:
- QKFormer: 85.65% (first SNN > 85% on ImageNet)
- Speech2Spikes: 109x energy reduction
- Nature Comms robustness: 2x more robust

our best "wow" numbers:
- gap collapse: 16.7pp to 0.95pp
- 6.0x adversarial robustness (first audio analysis)
- 7 encodings (most comprehensive audio comparison)
- first SNN on ESC-50

---

## ICONS specifically (2022-2024)

### ICONS 2024 (Arlington, Virginia)

award-winning papers: "Scalable Event-by-event Processing with Deep State-Space Models" (Schone et al.), "IM-SNN: Memory-Efficient SNNs with Low-Precision Membrane Potentials" (Hassan et al.), "Programmable Synapses and Dendritic Circuits for Superconducting Neuromorphic Computing" (Primavera et al.)

80+ presentations covering algorithms (visualization, dendritic processing), hardware (analog neurons, CMOS, memristors), and applications (gaming, audio processing, materials science, gait analysis, RF fingerprinting, swarm robotics).

turns out ICONS 2024 accepted audio processing papers as lightning talks in the Applications session. our paper covers THREE of their tracks (algorithms, hardware, applications).

### ICONS 2023 (Santa Fe, NM)

best paper: "Dendritic Learning in Superconducting Optoelectronic Networks" (O'Loughlin, Primavera, Shainline). published in ACM proceedings. papers included continual learning for robots, prosthetic vision, dictionary learning.

### ICONS 2022 (Knoxville, TN)

directly relevant: "Evaluating Encoding and Decoding Approaches for Spiking Neuromorphic Systems" -- this is the Yarga et al. paper we already cite. they compared 4 encodings on speech digits. our 7-encoding comparison on ESC-50 is a direct and significant extension. ICONS reviewers will recognize this lineage and see us building on their community's research. this is a strong positioning strategy.

### ICONS 2026 (our target)

Chicago, IL. August 4-6, 2026. deadline: April 1 (maybe April 8 AoE). 8-page full papers (20-min talks), 4-page short (10-min).

explicit topics: neuromorphic circuits/sensors, non-von Neumann computing, event/spike-based systems, brain-inspired architectures, supervised/unsupervised/self-supervised learning, continual learning, energy-efficient edge AI, neuromorphic benchmarks, domain-specific implementations, simulation techniques.

we map to: #3 (spike-based), #5 (supervised learning), #8 (energy-efficient edge AI), #10 (neuromorphic benchmarks), #11 (domain-specific: environmental sound)

special emphasis 2026: "welcoming work at the intersection of neuromorphic computing and energy-efficient AI, including ultra-low-power spike-based machine learning..."

our temporal truncation result (90% accuracy at T=7 = 72% energy saving) directly addresses this.

---

## SNN + audio: complete landscape (2016-2025)

every known SNN audio/sound paper i could find:

| year | authors | dataset | accuracy | hardware | notes |
|------|---------|---------|----------|----------|-------|
| 2016 | Dominguez-Morales et al. | 8 pure tones | >85% at SNR>3dB | SpiNNaker | first SNN audio on SpiNNaker |
| 2018 | Wu et al. | SHD, TIMIT | 97.5% TIDIGITS | none | SOM + SNN |
| 2018 | Dong et al. | TIDIGITS | 97.5% / 93.8% | none | unsupervised STDP |
| 2021 | Amin | TIDIGITS, RWCP | 97.64% / 99.50% | none | adaptive threshold |
| 2021 | Bensimon et al. | RWCP | 98.73% | none | spiking continuous time neuron |
| 2022 | Yarga et al. | speech digits | variable | none | 4-encoding comparison at ICONS |
| 2022 | Blouw & Choo | speech commands | ~95% | Loihi | real-time |
| 2023 | Speech2Spikes (Intel) | Google Speech Cmds | SOTA | Loihi | 109x energy |
| 2023 | Xiang et al. | TIDIGITS | 93.75% | photonic | photonic neuromorphic |
| 2024 | Yang & Chang | TIMIT | PER 22.6% | hw accelerator (71.2 uW) | low-power accelerator |
| 2024 | NeurIPS paper | sound sources | SOTA localization | none | resonate-and-fire |
| 2025 | Larroza et al. | ESC-10 (10 classes) | 69% | none | 3 encodings, FC only |
| 2025 | SATRN (Gao et al.) | UrbanSound8K, FSD50K | mAP 0.455 | none | spiking attention |
| 2026 | ours | ESC-50 (50 classes) | 47.15% | SpiNNaker | 7 encodings, gap-collapse, adversarial |

the gap our paper fills: no prior paper has used convolutional SNNs on ESC-50 (50 classes), compared more than 4 encodings for audio, deployed any SNN for environmental sound on neuromorphic hardware, analyzed adversarial robustness of SNNs on audio, quantified encoding specificity via transfer analysis, or shown the gap-collapse phenomenon on audio.

our closest competitor (Larroza et al., arXiv March 2025) only does ESC-10, FC-only, 3 encodings, no hardware. we're strictly superior on every axis.

---

## common winning formulas in SNN publications

formula 1: "first X on Y"
- QKFormer: first directly-trained SNN exceeding 85% on ImageNet
- SpikeLLM: first spiking LLM
- ours: "first convolutional SNN evaluated on ESC-50" + "first neuromorphic hardware deployment for environmental sound"
- legitimate and verified. strong.

formula 2: "X orders of magnitude improvement"
- Speech2Spikes: 109x energy reduction
- ours: "6.0x adversarial robustness advantage" + "gap collapses from 16.7pp to 0.95pp" (17.6x reduction in the gap)
- the 17.6x gap reduction framing is novel and dramatic. worth using.

formula 3: "novel analysis revealing Z"
- Zenke & Vogels 2021: surrogate shape matters less than slope
- ours: "the SNN-ANN gap is a feature-learning problem, not a spiking limitation" + "encoding specificity: transfer ratio 0.255"
- gap-collapse insight is genuinely novel for audio. encoding specificity has never been quantified before.

formula 4: "hardware deployment showing W"
- ours: "first SpiNNaker deployment for 50-class environmental sound, with documented root-cause analysis of deployment failures and validated hybrid approach"
- the honesty about failure modes actually strengthens the paper. reviewers value reproduciblity over inflated claims.

formula 5: "comprehensive benchmark/comparison"
- ours: "most comprehensive SNN encoding comparison for audio: 7 methods, 50 classes, 5-fold validated"
- no prior audio SNN paper compares more than 4 encodings. this alone could be a short paper at ICONS.

---

## hot topics in neuromorphic computing (2025-2026)

ranked by how excited reviewers seem to be about them:

1. neuromorphic + LLMs (hottest) - SpikeLLM at ICLR 2025 was a landmark. not our domain though.

2. event-driven sensing + edge AI - Innatera T1 processor, event cameras + SNN. our "always-on audio sensing" framing targets this. HIGH relevance.

3. spiking transformers - QKFormer, Spikformer v2, race to close SNN-ANN gap on ImageNet. our gap-collapse relates. MEDIUM.

4. neuromorphic robustness - Nature Comms 2025 paper, growing interest. our 6x adversarial robustness is directly relevant. HIGH.

5. hardware-software co-design - SpiNNaker2 at Sandia, SpikeFit. our SpiNNaker deployment with documented failures is a case study. HIGH.

6. standardized benchmarking (NeuroBench) - community push for reproducibility. we use NeuroBench metrics and provide 5-fold validated results with p-values. HIGH.

7. continual/online learning - survey in IEEE Access 2025, explicit ICONS topic. our result is preliminary but present. MEDIUM.

8. new application domains - SNNs applied to speech, EEG, medical imaging. "first SNN on [domain]" papers consistently publishable. HIGHEST relevance for us.

---

## gap analysis and framing recommendations

what we have vs what top papers have:

| dimension | top papers | ours | gap |
|-----------|-----------|------|-----|
| novelty | "first X on Y" | first SNN on ESC-50 | no gap |
| hardware | full pipeline on Loihi/SpiNNaker | FC2-only hybrid | moderate (honest about it) |
| wow number | 85.65% ImageNet, 109x energy | 6x robustness, 16.7pp to 0.95pp | no gap |
| statistical rigor | varies (many papers lack it) | 5-fold, p-values, Cohen's d | we exceed most |
| depth of analysis | usually 1-2 analyses | 7 enc + adversarial + transfer + temporal + CL + noise + surrogate + energy | we exceed most |
| clarity of thesis | one clear message | multiple findings, unclear which is central | THIS is our gap |

### the specific recommended framing

current title: "Spiking Neural Networks for Environmental Sound Classification: From Seven Encodings to SpiNNaker Deployment"

problem: sounds like a survey/benchmark. "from X to Y" suggests breadth not depth. reviewers may think "lots of experiments but what's the insight?"

option A (gap-collapse as central thesis):

title: "The SNN-ANN Gap is a Feature-Learning Problem: Evidence from Environmental Sound Classification with Seven Encodings and SpiNNaker Deployment"

one-sentence thesis: when given equal-quality features, SNNs match ANNs within 0.95pp on 50-class audio -- the 16.7pp scratch-training gap reflects a feature-learning bottleneck, not a spiking computation limitation.

this works because it positions everything else as supporting evidence for a single scientific claim:
- 7-encoding comparison shows encoding matters enormously (direct = continuous features win)
- PANNs gap-collapse is the central proof
- adversarial robustness shows SNNs have unique advantages despite the accuracy gap
- SpiNNaker deployment demonstrates practical viability of the hybrid approach implied by the thesis
- encoding transfer analysis shows WHY encoding matters
- temporal truncation shows path to energy efficiency

option B (hardware co-design):

title: "From Software to SpiNNaker: A Complete Pipeline for Neuromorphic Audio Classification on ESC-50"

fine for ICONS (it's a systems conference) but limits the paper to a systems audience.

**Assessment:** The gap-collapse insight is genuinely novel for audio. The encoding specificity finding (transfer ratio) has never been quantified before. Both are strong analytical contributions.

### Formula 4: "Hardware deployment showing W"
**Examples:**
- Dominguez-Morales 2016: SpiNNaker for pure tones
- Speech2Spikes: Loihi for keyword spotting
- Yang & Chang 2024: Custom accelerator at 71.2 uW

**YOUR VERSION:** "First SpiNNaker deployment for 50-class environmental sound, with documented root-cause analysis of deployment failures and validated hybrid approach"

**Assessment:** The honesty about failure modes actually strengthens the paper. Reviewers value reproducibility and honest reporting over inflated claims.

### Formula 5: "Comprehensive benchmark/comparison"
**Examples:**
- NeuroBench: Standardized framework for all neuromorphic computing
- Yarga et al. ICONS 2022: 4-encoding comparison for speech digits
- Neural Coding comparison (Frontiers 2021): 4 coding schemes on MNIST

**YOUR VERSION:** "Most comprehensive SNN encoding comparison for audio: 7 methods, 50 classes, 5-fold validated, with statistical tests and cross-encoding transfer analysis"

**Assessment:** No prior audio SNN paper compares more than 4 encodings. Your 7-encoding comparison with statistical rigor is genuinely the most comprehensive. This alone could be a short paper at ICONS.

---

## 6. HOT TOPICS IN NEUROMORPHIC COMPUTING (2025-2026)

Ranked by reviewer excitement (based on publication trends, conference emphasis, and community direction):

### 1. Neuromorphic + LLMs (HOTTEST)
- SpikeLLM at ICLR 2025 was a landmark
- MatMul-free LLM on Loihi 2 (3x less energy than edge GPU)
- ICONS 2024 had language model talks
- **Relevance to you:** Low. Not your domain.

### 2. Event-Driven Sensing + Edge AI
- Innatera T1 SNN processor at CES 2024
- Event cameras + SNN processing
- Always-on sensing with ultra-low power
- **Relevance to you:** HIGH. Your "always-on audio sensing" framing in the introduction directly targets this.

### 3. Spiking Transformers
- QKFormer, Spikformer v2, SpikingResformer all at top venues
- Race to close the SNN-ANN gap on ImageNet
- **Relevance to you:** Medium. Your gap-collapse finding relates to this narrative.

### 4. Neuromorphic Robustness
- Nature Communications 2025 paper on SNN robustness via temporal processing
- Growing interest in adversarial + noise robustness of SNNs
- **Relevance to you:** HIGH. Your 6x adversarial robustness finding is directly relevant.

### 5. Hardware-Software Co-Design
- SpiNNaker2 deployment at Sandia National Labs (June 2025)
- SpikeFit (EurIPS 2025) on optimal SNN deployment
- Software-to-hardware gap quantification
- **Relevance to you:** HIGH. Your SpiNNaker deployment with documented failure modes is a case study in co-design challenges.

### 6. Standardized Benchmarking (NeuroBench)
- NeuroBench in Nature Communications 2025
- Community push for reproducible, comparable results
- **Relevance to you:** HIGH. You use NeuroBench metrics. You provide 5-fold validated results with p-values.

### 7. Continual/Online Learning on Neuromorphic Hardware
- Survey in IEEE Access 2025
- ICONS has "continual learning systems" as explicit topic
- **Relevance to you:** Medium. Your continual learning result is preliminary but present.

### 8. New Application Domains
- SNNs being applied to speech enhancement, EEG, medical imaging, structural mechanics
- "First SNN on [domain]" papers are consistently publishable
- **Relevance to you:** HIGHEST. Your "first SNN on ESC-50" claim is exactly this pattern.

---

## 7. GAP ANALYSIS AND SPECIFIC RECOMMENDATIONS

### What You Have vs. What Top Papers Have

| Dimension | Top Papers | Your Paper | Gap |
|-----------|-----------|------------|-----|
| Novelty of domain | "First X on Y" | First SNN on ESC-50 | **No gap -- you have this** |
| Hardware deployment | Full pipeline on Loihi/SpiNNaker | FC2-only hybrid on SpiNNaker | Moderate gap (honest about it) |
| Wow number | 85.65% ImageNet, 109x energy | 6x robustness, 16.7pp to 0.95pp gap collapse | **No gap -- your numbers are dramatic** |
| Statistical rigor | Varies (many papers lack it) | 5-fold CV, p-values, Cohen's d | **You exceed most papers** |
| Depth of analysis | Usually 1-2 analyses | 7 encodings + adversarial + transfer + temporal + CL + noise + surrogate + energy | **You exceed most papers** |
| Clarity of thesis | One clear message | Multiple findings, unclear which is central | **This is your gap** |

### THE SPECIFIC RECOMMENDED FRAMING

**Current title:** "Spiking Neural Networks for Environmental Sound Classification: From Seven Encodings to SpiNNaker Deployment"

**Problem with current title:** It sounds like a survey/benchmark. "From X to Y" framing suggests breadth, not depth. Reviewers may think "lots of experiments but what is the insight?"

**Recommended reframe -- Option A (Gap-Collapse as Central Thesis):**

> **Title:** "The SNN-ANN Gap is a Feature-Learning Problem: Evidence from Environmental Sound Classification with Seven Encodings and SpiNNaker Deployment"
>
> **One-sentence thesis:** When given equal-quality features, SNNs match ANNs within 0.95pp on 50-class audio -- the 16.7pp scratch-training gap reflects a feature-learning bottleneck, not a spiking computation limitation.
>
> **Why this works:** This positions all other results as SUPPORTING EVIDENCE for a single scientific claim:
> - 7-encoding comparison: shows encoding matters enormously (direct = continuous features win)
> - PANNs gap-collapse: the central proof of the thesis
> - Adversarial robustness: shows SNNs have unique ADVANTAGES despite the accuracy gap
> - SpiNNaker deployment: demonstrates practical viability of the hybrid approach implied by the thesis
> - Encoding transfer analysis: shows WHY encoding matters (encoding-specific representations)
> - Temporal truncation: shows path to energy efficiency once feature-learning is solved

**Recommended reframe -- Option B (Hardware Co-Design as Central Thesis):**

> **Title:** "From Software to SpiNNaker: A Complete Pipeline for Neuromorphic Audio Classification on ESC-50"
>
> **One-sentence thesis:** We present the first end-to-end pipeline for deploying SNNs for environmental sound classification on neuromorphic hardware, documenting the design space (7 encodings), software-hardware gap (12.8pp), and co-design constraints (AvgPool incompatibility).
>
> **Why this works for ICONS:** ICONS is a systems conference. Hardware deployment stories resonate. The honest root-cause analysis of failure modes is valuable to the community.

**RECOMMENDATION: Use Option A.** The gap-collapse finding is more scientifically significant and applies beyond ICONS. Option B is fine but limits the paper to a systems audience.

### Specific Structural Suggestions for the Paper

1. **Lead with the gap-collapse in the abstract** (you already do this -- good)
2. **Make Section 4.4 (Transfer Learning: Gap Collapse) the FIRST result, not fourth.** Reorder to: Gap-Collapse -> Encoding Comparison -> Encoding Transfer -> SpiNNaker -> Adversarial -> Energy. This puts the thesis front and center.
3. **Frame encoding comparison as "why the gap exists"** rather than just reporting numbers. The encoding hierarchy (direct >> rate = phase > latency > delta > burst) is evidence that preserving continuous information is what matters -- which supports the gap-collapse thesis.
4. **Frame SpiNNaker deployment as "implications of the gap-collapse"** -- the hybrid CNN14 (features) + SNN (classification on SpiNNaker) is exactly the architecture implied by the thesis.
5. **Reduce emphasis on noise robustness and continual learning** -- these are weak results (not statistically significant for noise, preliminary for CL) that dilute the paper's strength. Keep them as one-paragraph mentions or move to supplementary.

### What NOT To Do

1. **Do NOT add more experiments.** You have enough. The risk is breadth without depth, not insufficient data.
2. **Do NOT chase SOTA accuracy.** 47.15% on ESC-50 is scientifically valid and interesting precisely BECAUSE it reveals the gap.
3. **Do NOT apologize for the gap.** The gap IS the finding. Frame it as "we identified the bottleneck" not "our SNN is worse."
4. **Do NOT hide the SpiNNaker failure.** The FC1 cancellation root-cause analysis is genuinely valuable to the community. Papers that document what went wrong are cited more than papers that only report successes.

---

## 8. CONFIDENCE ASSESSMENT

| Finding | Confidence | Basis |
|---------|-----------|-------|
| No prior SNN work on full ESC-50 | **Very High** | Confirmed by Larroza et al. 2025 arXiv paper, comprehensive literature search, SNN+sound review (PMC 2024) |
| 7-encoding comparison is most comprehensive for audio | **Very High** | Yarga et al. (4 encodings, digits), Larroza et al. (3 encodings, ESC-10) are the closest; no one has done 7 |
| Gap-collapse finding is novel for audio | **High** | ANN-to-SNN gap closure via pretrained features is known conceptually for images, but never quantified for audio with PANNs |
| ICONS 2026 acceptance likelihood | **Medium-High** | 59% acceptance rate, paper fits 5+ explicit topics, has hardware deployment + novel domain |
| Cross-encoding transfer analysis is first of its kind | **High** | Exhaustive search found no prior quantification of encoding specificity via transfer matrix |
| 6x adversarial robustness is first for audio | **High** | Sharmin et al. 2020 (images), Wang et al. 2025 (images) -- no audio adversarial SNN work found |

---

## 9. RESEARCH GAPS (What I Could Not Find)

1. **ICONS 2024 complete paper list:** The proceedings are behind IEEE paywall. I could see the schedule (80+ presentations) but not every paper title.
2. **ICONS acceptance rate since 2018:** Only 2018 data (59%) is publicly available. May have changed.
3. **Direct comparison of SNN vs ANN on ESC-50 in any prior work:** None exists. You are creating the baseline.
4. **SNN + PANNs in any prior work:** Found no paper combining pretrained audio neural networks with SNN classifier heads. Your PANNs+SNN experiment appears to be genuinely novel.
5. **Detailed reviewer criteria for ICONS:** No public rubric exists. Inference based on accepted papers and call for papers.

---

## 10. RECOMMENDED FOLLOW-UP ACTIONS

1. **Reframe the paper around the gap-collapse thesis** (Option A above) -- this is the highest-impact change and requires only editing, not new experiments.
2. **Consider reordering results** to lead with the gap-collapse finding.
3. **Tighten the paper to 6-7 strong pages** rather than cramming everything into 8 pages. Move noise robustness and continual learning details to a brief mention or cut entirely -- they dilute the message.
4. **Submit as a FULL paper (8 pages)** to ICONS 2026. Your work easily justifies a 20-minute talk.
5. **If the paper is rejected from ICONS, immediately submit to EUSIPCO 2026 or similar.** The "first SNN on ESC-50" claim has a shelf life -- Larroza et al. may extend to ESC-50 in their next paper.
6. **Consider a parallel arXiv preprint** to establish priority. The Larroza et al. paper is already on arXiv (March 2025). Getting your paper on arXiv before ICONS submission locks in your "first" claims.

---

## Sources

### SNN Conference Papers (curated lists)
- [Awesome SNN Conference Papers - GitHub](https://github.com/AXYZdong/awesome-snn-conference-paper)
- [Awesome Spiking Neural Networks - GitHub](https://github.com/TheBrainLab/Awesome-Spiking-Neural-Networks)

### ICONS Conference
- [ICONS 2026 Call for Papers](https://iconsneuromorphic.cc/calls-2026/)
- [ICONS 2026 EasyChair CFP](https://easychair.org/cfp/ACM-ICONS-2026)
- [ICONS 2024 Schedule](https://iconsneuromorphic.cc/icons-2024/schedule/)
- [ICONS 2024 IEEE Proceedings](https://www.computer.org/csdl/proceedings/icons/2024/22lE6EOwpkA)
- [ICONS 2023 ACM Proceedings](https://dl.acm.org/doi/proceedings/10.1145/3589737)
- [ICONS 2022 ACM Proceedings](https://dl.acm.org/doi/proceedings/10.1145/3546790)

### Key SNN Papers
- [Eshraghian et al. 2023 - Training SNNs Using Lessons from Deep Learning (Proc. IEEE Best Paper)](https://arxiv.org/abs/2109.12894)
- [QKFormer - NeurIPS 2024 Spotlight](https://arxiv.org/abs/2403.16552)
- [SpikeLLM - ICLR 2025](https://arxiv.org/abs/2407.04752)
- [NeuroBench Framework - Nature Communications 2025](https://www.nature.com/articles/s41467-025-56739-4)
- [Neuromorphic Robustness - Nature Communications 2025](https://www.nature.com/articles/s41467-025-65197-x)
- [Sharmin et al. 2020 - Inherent Adversarial Robustness of SNNs (ECCV)](https://arxiv.org/abs/2003.10399)

### SNN + Audio
- [Larroza et al. 2025 - Spike Encoding for Environmental Sound (arXiv)](https://arxiv.org/abs/2503.11206)
- [SNN and Sound Comprehensive Review (PMC 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11362401/)
- [SATRN: Spiking Audio Tagging Robust Network (MDPI 2025)](https://www.mdpi.com/2079-9292/14/4/761)
- [Speech2Spikes - ACM NICE 2023](https://dl.acm.org/doi/10.1145/3584954.3584995)
- [Spike-based Sound Source Localization - NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/ce953d71deeb33d9ffa2c879b518d273-Paper-Conference.pdf)
- [Dominguez-Morales et al. 2016 - SpiNNaker Audio (Springer)](https://link.springer.com/chapter/10.1007/978-3-319-44778-0_6)

### Encoding Comparisons
- [Neural Coding Comparative Study (Frontiers 2021)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.638474/full)
- [Spike Encoding for IoT Signals (Frontiers 2022)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.999029/full)
- [Survey of Encoding Techniques (Neural Processing Letters 2021)](https://link.springer.com/article/10.1007/s11063-021-10562-2)

### Neuromorphic Hardware
- [SpiNNaker2 System (arXiv 2024)](https://arxiv.org/abs/2401.04491)
- [Neuromorphic Computing 2025 Landscape](https://humanunsupervised.com/papers/neuromorphic_landscape.html)
- [Neuromorphic LLM on Loihi 2 (arXiv 2025)](https://arxiv.org/html/2503.18002v2)

### Robustness and Energy
- [SNN Adversarial Robustness with Local Learning (2025)](https://arxiv.org/html/2504.08897v2)
- [Reconsidering SNN Energy Efficiency (arXiv 2024)](https://arxiv.org/abs/2409.08290)
- [SpikeFit: Optimal SNN Deployment (EurIPS 2025)](https://arxiv.org/html/2510.15542)
- [Stochastic Resonance in SNNs (MDPI 2025)](https://www.mdpi.com/1099-4300/27/3/219)
