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

