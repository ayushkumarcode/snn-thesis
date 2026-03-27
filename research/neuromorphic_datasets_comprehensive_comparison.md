# Comprehensive Neuromorphic Dataset Comparison for Undergraduate Thesis

**Research Date:** 2026-02-25
**Scope:** Exhaustive comparison of all major neuromorphic datasets, their suitability for a 3rd-year thesis, framework support, preprocessing requirements, and narrative potential.

---

## Executive Summary

Eight neuromorphic datasets were investigated in depth, along with three newer datasets from 2024-2025. The datasets span three modalities: visual event streams (DVS128 Gesture, N-MNIST, N-Caltech101, CIFAR10-DVS, DVS-Lip), audio spike trains (SHD, SSC), and newer large-scale benchmarks (DailyDVS-200, eTraM, EvDET200K). For an undergraduate thesis, the best balance of difficulty, tooling, narrative strength, and achievable contribution lies with **DVS128 Gesture** (strong narrative, mature tooling, achievable but non-trivial), **SHD** (excellent for audio-domain novelty, small and fast to train), or **CIFAR10-DVS** (good difficulty level, well-benchmarked). N-MNIST is too easy to tell an interesting story. DVS-Lip and SSC are too difficult for an undergraduate scope. N-Caltech101 has poor class balance and awkward splits.

---

## Part 1: Dataset-by-Dataset Analysis

---

### 1. DVS128 Gesture

| Property | Details |
|---|---|
| **Domain** | Visual / Hand gesture recognition |
| **Sensor** | DVS128 (iniVation, 128x128 resolution) |
| **Classes** | 11 hand gestures (hand wave, right hand clockwise, etc.) |
| **Total samples** | 1,464 (1,176 train / 288 test) |
| **Subjects** | 29 subjects, 3 illumination conditions |
| **Data format** | AEDAT 3.1 (raw events: x, y, timestamp, polarity) |
| **Spatial resolution** | 128 x 128 pixels |
| **Tensor shape** | [T x 2 x 128 x 128] (T = time steps, 2 = on/off polarity) |
| **Download size** | ~2-3 GB (raw AEDAT files) |
| **Year introduced** | 2017 (IBM Research, Amir et al., CVPR 2017) |

**Preprocessing required:**
- Convert AEDAT 3.1 to event arrays (x, y, t, p)
- Segment recordings into individual gesture clips (using label timestamps)
- Integrate events into frame tensors using one of: fixed time-bin integration, fixed event-count integration, or voxel grid encoding
- Common approach: split into T=16 or T=20 time bins, accumulate events per pixel per polarity per bin
- Denoising optional (remove isolated events with no neighbors in a time window)
- Both SpikingJelly and Tonic handle this automatically

**Framework support:**
- SpikingJelly: Native support (built-in dataset loader, pre-built DVSGestureNet model, full training scripts)
- snnTorch: Via Tonic integration (deprecated spikevision also had it)
- Tonic: Full support as `tonic.datasets.DVSGesture`
- Norse: No built-in loader, but compatible via Tonic

**State-of-the-art accuracy (SNN methods):**

| Method | Year | Accuracy | Notes |
|---|---|---|---|
| Self-organizing Glial SNN | 2024 | 99.3% | Current SOTA |
| 100% claim (one paper) | 2024 | 100.0% | Reported but not widely replicated |
| TCJA-SNN | 2023 | 99.59% | 192K parameters |
| SpikePoint | 2024 | 98.74% | Point-cloud SNN approach |
| Ternarized Hybrid CNN | 2023 | 97.7% | Best embedded implementation |
| Typical baseline SNN | -- | ~95-97% | Achievable with standard training |

**Difficulty assessment:** MODERATE. The dataset is small enough to iterate quickly (minutes per epoch on a single GPU), has only 11 classes, and achieves near-perfect accuracy with modern methods. An undergraduate can realistically reach 95-97% accuracy and still have room to explore architecture choices, time-step ablations, and efficiency analysis.

**Thesis narrative strength:** STRONG. Gesture recognition maps directly to real-world applications: sign language interfaces, touchless device control, AR/VR interaction, accessibility technology. The narrative of "brain-inspired computing for human-computer interaction" is compelling and easy for examiners to understand. The efficiency angle (SNNs on edge devices) adds practical motivation.

---

### 2. N-MNIST (Neuromorphic MNIST)

| Property | Details |
|---|---|
| **Domain** | Visual / Handwritten digit recognition |
| **Sensor** | ATIS (Asynchronous Time-based Image Sensor) |
| **Classes** | 10 (digits 0-9) |
| **Total samples** | 70,000 (60,000 train / 10,000 test) |
| **Data format** | Binary event files (x, y, timestamp, polarity) |
| **Spatial resolution** | 34 x 34 pixels |
| **Creation method** | ATIS sensor mounted on motorized pan-tilt unit viewing MNIST on LCD |
| **Year introduced** | 2015 (Orchard et al., Frontiers in Neuroscience) |

**Preprocessing required:**
- Read binary event files into (x, y, t, p) arrays
- Apply temporal binning (ToFrame transformation) to create dense frame representations
- Optional denoising (remove isolated one-off events)
- Temporal collapsing to static images is possible but defeats the purpose
- Standard approach: bin events into T=10-30 time steps

**Framework support:**
- SpikingJelly: Native support
- snnTorch: Via Tonic (previously via deprecated spikevision)
- Tonic: Full support as `tonic.datasets.NMNIST`
- Norse: Compatible via Tonic

**State-of-the-art accuracy (SNN methods):**

| Method | Year | Accuracy | Notes |
|---|---|---|---|
| Sa-SNN (Spiking Attention) | 2024 | 99.63% | Current SOTA |
| ANN on collapsed frames | 2021 | 99.23% | Frame-based approach |
| Typical SNN baseline | -- | ~99.0-99.3% | Very easy to achieve |

**Critical caveat:** A seminal 2021 paper ("Is Neuromorphic MNIST Neuromorphic?") demonstrated that N-MNIST can be classified with 99%+ accuracy by simply collapsing time information into a static image and using a standard CNN. The temporal dynamics add almost no discriminative value. This means the dataset does NOT actually test whether your SNN leverages temporal spike patterns -- it is essentially just MNIST with extra steps.

**Difficulty assessment:** TOO EASY. The dataset is essentially solved. Even basic SNNs reach >99% accuracy. There is virtually no room for meaningful contribution or interesting analysis.

**Thesis narrative strength:** WEAK. "I classified handwritten digits" is not a compelling thesis narrative. The dataset exists primarily as a sanity check / tutorial exercise. The "Is Neuromorphic MNIST Neuromorphic?" criticism undermines the entire premise. Examiners familiar with the field will view this as insufficient scope.

---

### 3. N-Caltech101

| Property | Details |
|---|---|
| **Domain** | Visual / Object classification |
| **Sensor** | ATIS (same as N-MNIST) |
| **Classes** | 101 (100 object classes + 1 background class) |
| **Total samples** | ~8,709 (based on original Caltech101 minus "Faces" class) |
| **Data format** | Binary event files (x, y, timestamp, polarity) |
| **Spatial resolution** | Variable (ATIS sensor moved across images of varying sizes) |
| **Creation method** | ATIS sensor on motorized pan-tilt viewing Caltech101 on LCD |
| **Train/test split** | No standard split; researchers use various splits (commonly 80/20) |
| **Year introduced** | 2015 (Orchard et al., same paper as N-MNIST) |

**Preprocessing required:**
- Read binary event files (same format as N-MNIST)
- Handle variable spatial resolution (images are different sizes)
- Resize/crop to fixed dimensions (commonly 180x240 or 128x128)
- Temporal binning into frames
- Handle class imbalance (some classes have only 31 images, others have 800+)

**Framework support:**
- SpikingJelly: Native support
- Tonic: Full support as `tonic.datasets.NCALTECH101`
- snnTorch: Via Tonic
- Norse: Compatible via Tonic

**State-of-the-art accuracy (SNN methods):**

| Method | Year | Accuracy | Notes |
|---|---|---|---|
| NeuroMoCo (SEW-ResNet-18) | 2024 | 84.35% | Self-supervised + fine-tuning |
| RPLIF | 2024 | 83.35% | Low latency SOTA |
| NeuroMoCo (Spikformer) | 2024 | 81.62% | Transformer-based SNN |
| ANN on collapsed frames | 2021 | 78.01% | Frame-based approach |
| NDA-augmented SNN | 2022 | ~78% | With neuromorphic data augmentation |

**Difficulty assessment:** MODERATE-HARD. 101 classes with severe class imbalance. No standard train/test split (you have to justify your own). The accuracy ceiling is significantly lower than DVS128 or N-MNIST, meaning results can look "bad" even if the method is sound. Variable spatial resolution adds preprocessing complexity.

**Thesis narrative strength:** MODERATE. Object recognition is a reasonable application, but the dataset was created artificially (sensor viewing LCD screen), which weakens the "real neuromorphic data" narrative. The class imbalance issue could itself be a thesis angle, but 101 classes may be overly ambitious for undergraduate scope.

---

### 4. CIFAR10-DVS

| Property | Details |
|---|---|
| **Domain** | Visual / Object classification |
| **Sensor** | DVS (Dynamic Vision Sensor, 128x128) |
| **Classes** | 10 (same as CIFAR-10: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) |
| **Total samples** | 10,000 (1,000 per class) |
| **Standard split** | No official split; commonly 9,000 train / 1,000 test or 8,000 / 2,000 |
| **Data format** | AEDAT (address-event representation) |
| **Spatial resolution** | 128 x 128 pixels |
| **Creation method** | DVS camera viewing CIFAR-10 images on LCD with closed-loop smooth movement |
| **Download source** | Figshare (https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671) |
| **Year introduced** | 2017 (Li et al., Frontiers in Neuroscience) |

**Preprocessing required:**
- Convert AEDAT format to event arrays
- Integrate events into frames (commonly T=10 time bins with spatial resolution 128x128)
- Two-polarity representation (positive/negative events as separate channels)
- Time surface representation possible (accumulate events weighted by recency)
- Voxel grid encoding for more advanced representations
- Data augmentation important (random crop, flip, EventDrop)

**Framework support:**
- SpikingJelly: Native support (built-in dataset loader with frame integration)
- Tonic: Full support as `tonic.datasets.CIFAR10DVS`
- snnTorch: Via Tonic
- Norse: Compatible via Tonic

**State-of-the-art accuracy (SNN methods):**

| Method | Year | Accuracy | Notes |
|---|---|---|---|
| I2E-ImageNet pretrain + finetune | 2025 | 92.5% | Transfer learning from synthetic events |
| ICLR 2024 paper | 2024 | 84.9% | Sparse connectivity (6.8%) |
| VGG-11 SNN | 2024 | 81.23% | Standard architecture |
