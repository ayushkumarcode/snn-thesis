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
