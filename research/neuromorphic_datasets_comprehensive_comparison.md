# neuromorphic dataset comparison for thesis

i looked at eight major neuromorphic datasets plus some newer ones from 2024-2025. they span three modalities: visual event streams (DVS128 Gesture, N-MNIST, N-Caltech101, CIFAR10-DVS, DVS-Lip), audio spike trains (SHD, SSC), and newer large-scale benchmarks (DailyDVS-200, eTraM, EvDET200K).

for an undergrad thesis, the best balance of difficulty, tooling, narrative strength, and achievable contribution is: **DVS128 Gesture** (strong narrative, mature tooling, achievable), **SHD** (great for audio-domain novelty, small and fast), or **CIFAR10-DVS** (good difficulty, well-benchmarked). N-MNIST is too easy. DVS-Lip and SSC are too hard. N-Caltech101 has poor class balance and awkward splits.

---

## dataset-by-dataset analysis

### 1. DVS128 Gesture

| Property | Details |
|---|---|
| Domain | Visual / hand gesture recognition |
| Sensor | DVS128 (iniVation, 128x128) |
| Classes | 11 hand gestures |
| Samples | 1,464 (1,176 train / 288 test) |
| Subjects | 29 subjects, 3 illumination conditions |
| Format | AEDAT 3.1 (raw events: x, y, timestamp, polarity) |
| Output tensor | [T x 2 x 128 x 128] |
| Year | 2017 (IBM, Amir et al., CVPR 2017) |

**preprocessing:** convert AEDAT 3.1 to event arrays, segment recordings into gesture clips, integrate events into frames (fixed time-bin, fixed event-count, or voxel grid). commonly T=16 or T=20 time bins. optional denoising. both SpikingJelly and Tonic handle this automatically.

**framework support:** SpikingJelly (native, built-in loader + model + scripts), snnTorch (via Tonic), Tonic (full support), Norse (via Tonic).

**SOTA SNN accuracy:**
| Method | Year | Accuracy |
|---|---|---|
| Self-organizing Glial SNN | 2024 | 99.3% |
| TCJA-SNN | 2023 | 99.59% |
| SpikePoint | 2024 | 98.74% |
| Typical baseline | -- | ~95-97% |

**difficulty:** MODERATE. small enough to iterate fast, 11 classes, near-perfect accuracy with modern methods. an undergrad can realistically hit 95-97% and still explore architecture choices, timestep ablations, efficiency analysis.

**thesis narrative:** STRONG. gesture recognition maps to real applications -- sign language, touchless control, AR/VR, accessibility. "brain-inspired computing for HCI" is compelling and easy for examiners.

---

### 2. N-MNIST (Neuromorphic MNIST)

| Property | Details |
|---|---|
| Domain | Visual / handwritten digits |
| Sensor | ATIS |
| Classes | 10 (digits 0-9) |
| Samples | 70,000 (60K train / 10K test) |
| Resolution | 34 x 34 |
| Year | 2015 (Orchard et al.) |

**preprocessing:** read binary events, temporal binning (T=10-30 steps). simple.

**critical caveat:** a 2021 paper ("Is Neuromorphic MNIST Neuromorphic?") showed N-MNIST can be classified 99%+ by collapsing time into a static image and using a standard CNN. the temporal dynamics add almost no value. so the dataset doesn't actually test whether your SNN uses temporal spike patterns -- it's just MNIST with extra steps.

**difficulty:** TOO EASY. basically solved. even basic SNNs hit >99%.

**thesis narrative:** WEAK. "i classified handwritten digits" isn't a compelling thesis. examiners familiar with the field will see this as insufficient scope.

---

### 3. N-Caltech101

| Property | Details |
|---|---|
| Classes | 101 |
| Samples | ~8,709 |
| Resolution | Variable |
| Year | 2015 |

**problems:** severe class imbalance (some classes have 31 images, others 800+), no standard train/test split, variable spatial resolution.

**SOTA SNN:** 84.35% (NeuroMoCo 2024). the lower accuracy ceiling means results can look "bad" even if the method is good.

**difficulty:** MODERATE-HARD. might be overly ambitious for undergrad scope.

**thesis narrative:** MODERATE. artificial creation (sensor viewing LCD) weakens the "real neuromorphic data" story.

---

### 4. CIFAR10-DVS

| Property | Details |
|---|---|
| Domain | Visual / object classification |
| Classes | 10 (CIFAR-10 categories) |
| Samples | 10,000 |
| Resolution | 128 x 128 |
| Year | 2017 |
| Download | [Figshare](https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671) |

**preprocessing:** AEDAT to events, frame integration (commonly T=10, 128x128), two-polarity channels. data augmentation matters (random crop, flip, EventDrop). no official train/test split.

**SOTA SNN:** 92.5% (with transfer learning from synthetic events, 2025). more typical SNN SOTA is ~81-84%.

**difficulty:** MODERATE-HARD. harder than DVS128 (lower-res CIFAR images), ~78% is perfectly respectable for an undergrad.

**thesis narrative:** MODERATE. works if framed as "can SNNs match ANNs on event-converted data while using fewer operations?"

---

### 5. SHD (Spiking Heidelberg Digits)

| Property | Details |
|---|---|
| Domain | Audio / spoken digit classification |
| Sensor | Artificial cochlea model (software spike generation) |
| Classes | 20 (digits 0-9 in English and German) |
| Samples | ~10,420 (8,332 train / 2,088 test) |
| Speakers | 12 (2 only in test) |
| Input channels | 700 (cochlear frequency channels) |
| Format | HDF5 |
| Year | 2019 (Cramer et al., IEEE TNNLS) |

**preprocessing:** load HDF5, bin 700 channels to 140 (5:1 spatial binning), discretize time into ~100 steps at 10ms each. no spatial preprocessing needed (1D audio). clean pipeline.

**framework support:** SpikingJelly, Tonic, snnTorch (via Tonic), sparch toolkit, Rockpool.

**SOTA SNN:** 96.41% (SpikCommander 2025). typical achievable: ~85-92% with basic recurrent SNN.

**difficulty:** MODERATE. small (fast to train), clean splits with held-out speakers, 20-class problem is interesting. the main thing is that temporal dynamics GENUINELY MATTER here unlike N-MNIST. true test of SNN temporal processing.

**thesis narrative:** STRONG. "mimicking how the brain processes speech using spike-based computation." cochlear model input makes the bio-plausibility argument strong. applications: voice commands, hearing aids, smart speakers. the ONLY widely-used neuromorphic audio benchmark, so it has novelty value. few undergrads tackle audio SNNs.

---

### 6. SSC (Spiking Speech Commands)

| Property | Details |
|---|---|
| Classes | 35 speech commands |
| Samples | ~105,829 |
| Input channels | 700 |
| Year | 2019 |

same pipeline as SHD but 10x larger and 35 classes. SOTA only ~86%. training takes much longer.

**difficulty:** HARD. results can look mediocre even with good methods.

**thesis narrative:** MODERATE. similar to SHD but less clean. better referenced as "future work" if you do SHD.

---

### 7. DVS-Lip

| Property | Details |
|---|---|
| Domain | Visual / lip reading (word-level) |
| Classes | 100 words |
| Samples | 19,871 |
| Subjects | 40 volunteers (no speaker overlap between train/test) |
| Year | 2022 (CVPR) |

**difficulty:** VERY HARD. 100-class fine-grained visual recognition from events. SNN accuracy in the 55-65% range. needs sophisticated multi-scale temporal processing and lip region extraction. computationally expensive.

**thesis narrative:** VERY STRONG (but impractical). lip reading has an incredible application story -- privacy, silent speech, hearing impairment. but the difficulty makes this unsuitable for undergrad. better for a PhD.

---

## newer datasets (2024-2025)

### DailyDVS-200 (ECCV 2024)
200 daily actions, >22K sequences, 47 participants. **too large and complex for a thesis.** insufficient baselines and tooling.

### eTraM (CVPR 2024)
traffic monitoring, 10 hours, 2M bounding boxes. **not suitable.** object detection is much harder than classification.

### EvDET200K
object detection, 200K+ bounding boxes. **not suitable.** same reasons as eTraM.

### other
| Dataset | Year | Domain | Suitable? |
|---|---|---|---|
| HARDVS | 2023 | Action recognition (300 classes, 100K seqs) | No |
| E-POSE | 2025 | Pose estimation | No |
| LIPSFUS | 2023 | Audio-visual lip reading | No |
| Prophesee Gen4 | 2020 | Automotive detection | No |

---

## comparison table

| Dataset | Classes | Samples | Modality | SOTA SNN | Difficulty | Narrative | Training Time (est.) |
|---|---|---|---|---|---|---|---|
| **DVS128 Gesture** | 11 | 1,464 | Vision | 99.3% | Moderate | Strong | ~30 min |
| **N-MNIST** | 10 | 70,000 | Vision | 99.6% | Too Easy | Weak | ~1 hr |
| **N-Caltech101** | 101 | ~8,709 | Vision | 84.4% | Mod-Hard | Moderate | ~2-4 hrs |
| **CIFAR10-DVS** | 10 | 10,000 | Vision | 92.5%* | Mod-Hard | Moderate | ~2-4 hrs |
| **SHD** | 20 | 10,420 | Audio | 96.4% | Moderate | Strong | ~15-30 min |
| **SSC** | 35 | 105,829 | Audio | 86.0% | Hard | Moderate | ~4-8 hrs |
| **DVS-Lip** | 100 | 19,871 | Vision | ~60-65% | Very Hard | Very Strong | ~6-12 hrs |

*92.5% uses transfer learning; typical SNN SOTA is ~81-84%.

training times assume single consumer GPU (RTX 3060-3090 tier), ~100-200 epochs.

---

## framework support matrix

| Dataset | SpikingJelly | Tonic | snnTorch (via Tonic) | Norse (via Tonic) | sparch |
|---|---|---|---|---|---|
| DVS128 Gesture | Built-in + model | Yes | Yes | Yes | No |
| N-MNIST | Built-in | Yes | Yes | Yes | No |
| N-Caltech101 | Built-in | Yes | Yes | Yes | No |
| CIFAR10-DVS | Built-in | Yes | Yes | Yes | No |
| SHD | Built-in | Yes | Yes | Yes | Yes |
| SSC | Built-in | Yes | Yes | Yes | Yes |
| DVS-Lip | Listed | Yes | Yes | Yes | No |

---

## preprocessing comparison

### vision datasets
1. load raw events (x, y, timestamp, polarity)
2. optional denoising (Tonic provides `Denoise`)
3. temporal binning: event stream to T frames
4. output: [T x C x H x W] where C=2 (on/off polarity)
5. optional augmentation: random crop, flip, EventDrop

### audio datasets
1. load HDF5 (spike_times and spike_units)
2. channel binning: 700 -> 140 (optional but common)
3. temporal discretization: T steps (e.g., T=100 at 10ms)
4. output: [T x C] where C=140 or 700
5. zero-padding to 1 second

### complexity ranking (easiest to hardest):
1. SHD/SSC -- 1D data, clean HDF5, simple binning
2. N-MNIST -- small 2D, standardized format
3. DVS128 Gesture -- needs AEDAT parsing + clip segmentation, but well-tooled
4. CIFAR10-DVS -- AEDAT format, no official split
5. N-Caltech101 -- variable resolution, class imbalance, no standard split
6. DVS-Lip -- multi-scale temporal processing, lip region extraction

---

## thesis narrative ranking

### tier 1: best stories

**DVS128 Gesture -- "brain-inspired gesture recognition"**
- clear real-world application: touchless interfaces, AR/VR, accessibility
- natural efficiency argument: event cameras use less data than RGB
- easy to explain to non-specialists
- enables SNN vs ANN comparison with energy analysis

**SHD -- "biologically plausible speech processing"**
- unique angle: audio SNNs are rare in undergrad work
- strong biological motivation: cochlear model mimics inner ear
- applications: voice assistants, hearing aids, edge audio
- dataset specifically designed to require temporal processing (unlike N-MNIST)
- fast iteration = thorough experimentation

### tier 2: acceptable

**CIFAR10-DVS** -- well-known classes, good for SNN vs ANN comparison, but artificial creation weakens "real neuromorphic" claim

**N-Caltech101** -- 101 classes provides complexity, but class imbalance and artificial creation

### tier 3: not recommended

**N-MNIST** -- too easy, weak narrative
**SSC** -- too large/hard for undergrad
**DVS-Lip** -- fascinating but impractical difficulty

---

## my recommended strategy

### option A: DVS128 Gesture (safest)

achievable baseline 95-97%. contribution angles: architecture comparison (LIF vs PLIF), timestep ablation, SNN vs ANN energy analysis, robustness across illumination, latency-accuracy tradeoff. low risk, well-documented.

### option B: SHD (strongest novel narrative)

achievable baseline 88-92%. contribution angles: feedforward vs recurrent SNN comparison, temporal resolution study, SNN vs RNN comparison, speaker generalization, delay learning. low-medium risk, dataset is clean and fast but recurrent SNNs are trickier to train.

### option C: dual dataset (most ambitious)

use BOTH DVS128 and SHD across modalities. vision chapter + audio chapter + cross-cutting analysis of whether same SNN principles transfer. medium risk, double the implementation but each dataset trains fast.

---

## confidence notes

**confident about:** dataset sizes/classes/specs, framework support (confirmed via docs), SOTA accuracy ranges (cross-referenced), preprocessing pipelines (confirmed via tutorials/code).

**lower confidence:** DVS-Lip absolute accuracy (papers report relative improvements, estimated 55-65% SNN), exact download sizes, training time estimates (hardware-dependent), DVS128 "100%" claim (one paper, not replicated).

**couldn't determine:** exact GPU memory requirements per dataset/model, preprocessing time comparison across frameworks, community adoption metrics.

---

## sources

### DVS128 Gesture
- [IBM original (CVPR 2017)](https://openaccess.thecvf.com/content_cvpr_2017/papers/Amir_A_Low_Power_CVPR_2017_paper.pdf)
- [SpikePoint (2024)](https://arxiv.org/html/2310.07189v2)
- [CatalyzeX benchmarks](https://www.catalyzex.com/s/Dvs128%20Gesture%20Dataset)

### N-MNIST
- [Original (Orchard et al., 2015)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4644806/)
- ["Is Neuromorphic MNIST Neuromorphic?" (2021)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.608567/full)

### N-Caltech101
- [NeuroMoCo (2024)](https://arxiv.org/html/2406.06305)
- [NDA (2022)](https://arxiv.org/abs/2203.06145)

### CIFAR10-DVS
- [Original (Li et al., 2017)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2017.00309/full)
- [Papers With Code](https://paperswithcode.com/dataset/cifar10-dvs)

### SHD/SSC
- [Zenke Lab datasets](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/)
- [Original (Cramer et al., 2019)](https://arxiv.org/abs/1910.07407)
- [SpikCommander (2025)](https://arxiv.org/html/2511.07883)
- [sparch toolkit](https://github.com/idiap/sparch)

### DVS-Lip
- [CVPR 2022 MSTP](https://openaccess.thecvf.com/content/CVPR2022/papers/Tan_Multi-Grained_Spatio-Temporal_Features_Perceived_Network_for_Event-Based_Lip-Reading_CVPR_2022_paper.pdf)
- [SpikGRU-DVSLip](https://github.com/manondampfhoffer/SpikGRU-DVSLip)

### newer datasets
- [DailyDVS-200 (ECCV 2024)](https://arxiv.org/abs/2407.05106)
- [eTraM (CVPR 2024)](https://eventbasedvision.github.io/eTraM/)
- [EvDET200K](https://github.com/Event-AHU/OpenEvDET)

### frameworks
- [Tonic](https://tonic.readthedocs.io/en/latest/datasets.html)
- [SpikingJelly](https://github.com/fangwei123456/spikingjelly)
- [snnTorch](https://snntorch.readthedocs.io/)
- [Norse](https://github.com/norse/norse)
- [NeuroBench](https://neurobench.ai/)
