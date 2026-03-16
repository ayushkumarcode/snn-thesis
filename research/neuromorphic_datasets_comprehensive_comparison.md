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
| HSD (Hybrid Distillation) | 2024 | 81.10% | 5 time steps |
| TET | 2022 | 78.80% | Popular baseline |
| Typical baseline SNN | -- | ~74-78% | Standard training |

**Difficulty assessment:** MODERATE-HARD. More challenging than DVS128 Gesture due to harder object categories and lower resolution (128x128 DVS capturing 32x32 CIFAR images). The lack of an official train/test split requires care. SOTA accuracy is much lower than DVS128, meaning a ~78% result is perfectly respectable for an undergraduate. The 10,000 sample size is manageable.

**Thesis narrative strength:** MODERATE. Object classification is a standard ML task, but the "neuromorphic conversion of a classic benchmark" angle is somewhat artificial. The story improves if framed as "can SNNs match ANN performance on event-converted data while using fewer operations?" The efficiency comparison narrative works well here.

---

### 5. SHD (Spiking Heidelberg Digits)

| Property | Details |
|---|---|
| **Domain** | Audio / Spoken digit classification |
| **Sensor** | Artificial cochlea model (software-based spike generation) |
| **Classes** | 20 (digits 0-9 in English and German) |
| **Total samples** | ~10,420 (8,332 train / 2,088 test) |
| **Speakers** | 12 distinct speakers (2 only in test set) |
| **Input channels** | 700 (cochlear frequency channels) |
| **Data format** | HDF5 (spike times + neuron IDs) |
| **Download source** | https://zenkelab.org/datasets/ and IEEE DataPort |
| **Duration** | Each sample ~1 second, zero-padded |
| **Year introduced** | 2019 (Cramer et al., IEEE TNNLS) |

**Preprocessing required:**
- Load HDF5 files containing spike times and unit IDs
- Bin 700 input channels into 140 channels (5:1 spatial binning)
- Discretize time into fixed bins (commonly 100 time steps at 10ms each for 1s duration)
- Create dense spike tensor: shape [T x 140] or [T x 700]
- Optional augmentation: EventDrop (drop-by-time, drop-by-neuron), time stretching
- No spatial preprocessing needed (1D audio, not 2D vision)

**Framework support:**
- SpikingJelly: Native support
- Tonic: Full support as `tonic.datasets.SHD`
- snnTorch: Via Tonic (previously via deprecated spikevision)
- sparch toolkit: Dedicated PyTorch toolkit for SHD/SSC (https://github.com/idiap/sparch)
- Rockpool: Built-in tutorial for SHD training

**State-of-the-art accuracy (SNN methods):**

| Method | Year | Accuracy | Notes |
|---|---|---|---|
| SpikCommander | 2025 | 96.41% | 0.19M params, Transformer-based |
| SpikeSCR | 2024 | 93.60% | 40 time steps |
| SpikGRU | 2024 | ~95% | Gated recurrent SNN |
| Surrogate gradient baseline | 2022 | ~88-91% | Standard recurrent SNN |
| Typical achievable | -- | ~85-92% | With basic RSNN |

**Difficulty assessment:** MODERATE. The dataset is small (fast to train -- minutes on a GPU), has a clean train/test split with held-out speakers, and the 20-class problem is challenging enough to be interesting. The 1D nature (audio, not vision) makes it computationally lighter. An undergraduate can realistically achieve 88-92% accuracy and have meaningful ablation studies. The main challenge is that temporal dynamics GENUINELY MATTER here, unlike N-MNIST -- this is a true test of SNN temporal processing.

**Thesis narrative strength:** STRONG. Audio classification with brain-inspired computing has a great narrative: "mimicking how the brain processes speech using spike-based computation." The cochlear model input makes the biological plausibility argument strong. Applications include voice command recognition, hearing aids, smart speakers. The dataset was specifically designed to require temporal processing, making it a genuine test of SNN capabilities. Also notably, this is the ONLY widely-used neuromorphic audio benchmark, giving it novelty value.

---

### 6. SSC (Spiking Speech Commands)

| Property | Details |
|---|---|
| **Domain** | Audio / Speech command recognition |
| **Sensor** | Artificial cochlea model (same as SHD) |
| **Classes** | 35 speech commands ("yes", "no", "up", "down", etc.) |
| **Total samples** | ~105,829 (training/validation/test splits provided) |
| **Input channels** | 700 (cochlear frequency channels) |
| **Data format** | HDF5 (spike times + neuron IDs) |
| **Base dataset** | Google Speech Commands v0.2 |
| **Download source** | https://zenkelab.org/datasets/ |
| **Year introduced** | 2019 (Cramer et al., same paper as SHD) |

**Preprocessing required:**
- Same as SHD: load HDF5, bin channels (700 -> 140), discretize time (100 time steps at 10ms)
- Larger dataset means longer preprocessing but same pipeline
- Same augmentation options as SHD

**Framework support:**
- Same as SHD: SpikingJelly, Tonic (`tonic.datasets.SSC`), sparch toolkit
- Note: Tonic lists SSC as supported

**State-of-the-art accuracy (SNN methods):**

| Method | Year | Accuracy | Notes |
|---|---|---|---|
| SpikCommander (T=250) | 2025 | 85.98% | Best reported, but 250 time steps |
| SpikCommander (T=100) | 2025 | 83.49% | Standard time steps |
| SpikeSCR | 2024 | 82.54% | 100 time steps |
| DCLS | 2023 | 80.69% | Previous SOTA |
| Typical baseline | -- | ~75-80% | Standard RSNN |

**Difficulty assessment:** HARD. The dataset is 10x larger than SHD (105K vs 10K samples), has 35 classes (vs 20), and SOTA accuracy is only ~86%. Training time is significantly longer. The accuracy ceiling is lower, meaning results can look mediocre even with good methods. The 100+ time steps required for good performance increase memory usage.

**Thesis narrative strength:** MODERATE. Similar to SHD but the story is less clean because the dataset is much larger and harder, making it difficult to achieve competitive results in an undergraduate timeframe. If you use SHD, you can reference SSC as "future work" for scale-up.

---

### 7. DVS-Lip

| Property | Details |
|---|---|
| **Domain** | Visual / Lip reading (word-level) |
| **Sensor** | DAVIS346 event camera |
| **Classes** | 100 words (25 visually-confusing pairs + 50 random words from LRW) |
| **Total samples** | 19,871 (14,896 train / 4,975 test) |
| **Subjects** | 40 volunteers (30 train / 10 test, no speaker overlap) |
| **Spatial resolution** | 128 pixels (preprocessed) |
| **Data format** | Event streams |
| **Year introduced** | 2022 (Tan et al., CVPR 2022) |

**Preprocessing required:**
- Event data from DAVIS346 sensor
- Spatial cropping/resizing to 128-pixel resolution
- Frame integration (multi-rate for different temporal granularities)
- The original CVPR 2022 paper uses a multi-branch architecture at different frame rates
- Significant preprocessing pipeline for lip region extraction

**Framework support:**
- Tonic: Full support as `tonic.datasets.DVSLip`
- SpikingJelly: Supported (listed in HARDVS and related)
- SpikGRU-DVSLip: Dedicated codebase (https://github.com/manondampfhoffer/SpikGRU-DVSLip)
- Less tutorial coverage compared to DVS128 or SHD

**State-of-the-art accuracy (SNN methods):**

| Method | Year | Accuracy | Notes |
|---|---|---|---|
| SpikGRU2+ (CVPR 2024 Workshop) | 2024 | ~60-65% (est.) | 25% improvement over previous SNN SOTA |
| MSTP (ANN, CVPR 2022) | 2022 | ~70-75% (est.) | Multi-grained Spatio-Temporal Network |
| Previous SNN SOTA | pre-2024 | ~40-50% (est.) | Significant SNN-ANN gap |
| SSE-Net | 2025 | Unknown | Latest, claims SOTA |

Note: Exact accuracy percentages are difficult to confirm from abstracts alone, as papers report relative improvements. The SNN-ANN gap remains significant on this dataset.

**Difficulty assessment:** VERY HARD. 100-class fine-grained visual recognition from events is extremely challenging. The SNN accuracy is well below ANN accuracy, meaning state-of-the-art SNN results are in the 55-65% range for 100-word classification. The dataset requires sophisticated multi-scale temporal processing. Training is computationally expensive. The lip-region preprocessing pipeline is non-trivial.

**Thesis narrative strength:** VERY STRONG (but impractical). Lip reading has an incredibly compelling application narrative: privacy-preserving communication, silent speech interfaces, hearing-impaired assistance, surveillance. Event cameras are uniquely suited because they capture micro-movements without motion blur. However, the difficulty level makes this unsuitable for an undergraduate thesis -- you would struggle to achieve meaningful accuracy, and the gap to SOTA would be large. Better suited for a PhD project.

---

## Part 2: Newer Datasets (2024-2025)

### 8. DailyDVS-200 (ECCV 2024)

| Property | Details |
|---|---|
| **Domain** | Visual / Human action recognition |
| **Sensor** | DVXplorer Lite + RGB camera |
| **Classes** | 200 daily actions |
| **Total samples** | >22,000 event sequences |
| **Subjects** | 47 participants |
| **Annotations** | 14 attributes per sequence |
| **Source** | https://github.com/QiWang233/DailyDVS-200 |

**Assessment for undergraduate thesis:** TOO LARGE AND COMPLEX. 200 classes with 22K sequences from a cutting-edge 2024 paper. Insufficient baseline implementations and tooling. This is a research-frontier dataset intended for pushing SOTA, not for thesis projects.

### 9. eTraM (CVPR 2024, Poster Highlight)

| Property | Details |
|---|---|
| **Domain** | Visual / Traffic monitoring (object detection) |
| **Sensor** | Prophesee EVK4 HD event camera |
| **Classes** | 8 traffic participant classes |
| **Total data** | 10 hours of recordings, 2M bounding boxes |
| **Source** | https://eventbasedvision.github.io/eTraM/ |

**Assessment for undergraduate thesis:** NOT SUITABLE. Object detection (not classification) requires significantly more complex architectures (RVT, RED, YOLO adaptations). The 10-hour dataset is massive. Detection tasks are inherently harder than classification for a thesis project.

### 10. EvDET200K (arXiv 2024, CVPR 2025)

| Property | Details |
|---|---|
| **Domain** | Visual / Object detection |
| **Sensor** | Prophesee EVK4-HD |
| **Classes** | 10 (people, cars, bicycles, etc.) |
| **Total samples** | 10,054 sequences, 200K bounding boxes |
| **Source** | https://github.com/Event-AHU/OpenEvDET |

**Assessment for undergraduate thesis:** NOT SUITABLE. Same reasons as eTraM -- detection is a harder task than classification, and the dataset is very new with limited baseline tooling.

### 11. Other Notable Recent Datasets

| Dataset | Year | Domain | Suitable for UG thesis? |
|---|---|---|---|
| HARDVS | 2023 | Action recognition (300 classes, 100K seqs) | No -- too large, only 5 subjects |
| E-POSE | 2025 | Object pose estimation | No -- pose estimation, not classification |
| LIPSFUS | 2023 | Audio-visual lip reading | No -- multimodal fusion, very complex |
| Prophesee 1MP (Gen4) | 2020 | Automotive detection | No -- 15hrs, detection task |

---

## Part 3: Comparative Summary Table

| Dataset | Classes | Samples | Resolution | Modality | SOTA SNN Acc | Difficulty | Narrative | Framework Support | Training Time (est.) |
|---|---|---|---|---|---|---|---|---|---|
| **DVS128 Gesture** | 11 | 1,464 | 128x128 | Vision | 99.3% | Moderate | Strong | Excellent | ~30 min/full train |
| **N-MNIST** | 10 | 70,000 | 34x34 | Vision | 99.6% | Too Easy | Weak | Excellent | ~1 hr/full train |
| **N-Caltech101** | 101 | ~8,709 | Variable | Vision | 84.4% | Mod-Hard | Moderate | Good | ~2-4 hrs |
| **CIFAR10-DVS** | 10 | 10,000 | 128x128 | Vision | 92.5%* | Mod-Hard | Moderate | Excellent | ~2-4 hrs |
| **SHD** | 20 | 10,420 | 700 ch | Audio | 96.4% | Moderate | Strong | Excellent | ~15-30 min |
| **SSC** | 35 | 105,829 | 700 ch | Audio | 86.0% | Hard | Moderate | Good | ~4-8 hrs |
| **DVS-Lip** | 100 | 19,871 | 128px | Vision | ~60-65% | Very Hard | Very Strong | Limited | ~6-12 hrs |

*The 92.5% for CIFAR10-DVS uses transfer learning from synthetic events; a more typical SNN SOTA is ~81-84%.

Estimated training times assume a single consumer GPU (RTX 3060-3090 tier, 12-24GB VRAM), standard architectures, and ~100-200 epochs.

---

## Part 4: Framework Support Matrix

| Dataset | SpikingJelly | Tonic | snnTorch (via Tonic) | Norse (via Tonic) | sparch | Dedicated Code |
|---|---|---|---|---|---|---|
| DVS128 Gesture | Built-in loader + model | Yes | Yes | Yes | No | Multiple GitHub repos |
| N-MNIST | Built-in | Yes | Yes | Yes | No | Multiple |
| N-Caltech101 | Built-in | Yes | Yes | Yes | No | Few |
| CIFAR10-DVS | Built-in | Yes | Yes | Yes | No | Multiple |
| SHD | Built-in | Yes | Yes | Yes | Yes | spytorch, Rockpool |
| SSC | Built-in | Yes | Yes | Yes | Yes | sparch |
| DVS-Lip | Listed | Yes | Yes | Yes | No | SpikGRU-DVSLip |

---

## Part 5: Preprocessing Pipeline Comparison

### Vision Datasets (DVS128, N-MNIST, N-Caltech101, CIFAR10-DVS, DVS-Lip)

Common pipeline:
1. **Load raw events**: (x, y, timestamp, polarity) tuples
2. **Optional denoising**: Remove isolated events (Tonic provides `Denoise` transform)
3. **Temporal binning**: Convert event stream to T fixed-duration frames
   - `tonic.transforms.ToFrame(sensor_size, time_window=...)` -- fixed time windows
   - `tonic.transforms.ToFrame(sensor_size, n_time_bins=...)` -- fixed number of bins
4. **Output tensor**: [T x C x H x W] where C=2 (on/off polarity)
5. **Optional augmentation**: Random crop, horizontal flip, EventDrop, time jitter

### Audio Datasets (SHD, SSC)

Common pipeline:
1. **Load HDF5**: Contains spike_times and spike_units arrays
2. **Channel binning**: Reduce 700 channels to 140 (5:1 ratio) -- optional but common
3. **Temporal discretization**: Bin spike times into T steps (e.g., T=100 at 10ms resolution)
4. **Output tensor**: [T x C] where C=140 or 700
5. **Zero-padding**: All samples aligned to 1 second duration
6. **Optional augmentation**: EventDrop (drop-by-time, drop-by-neuron), time stretching

### Preprocessing Complexity Ranking (easiest to hardest):
1. **SHD/SSC** -- 1D data, clean HDF5 format, simple binning
2. **N-MNIST** -- Small 2D, standardized binary format
3. **DVS128 Gesture** -- Requires AEDAT parsing + clip segmentation, but well-tooled
4. **CIFAR10-DVS** -- AEDAT format, no official split, needs careful handling
5. **N-Caltech101** -- Variable resolution, class imbalance, no standard split
6. **DVS-Lip** -- Multi-scale temporal processing, lip region extraction

---

## Part 6: Thesis Narrative Analysis

### Tier 1: Best "Story" for Thesis

**DVS128 Gesture -- "Brain-Inspired Gesture Recognition"**
- Clear real-world application: touchless interfaces, AR/VR, accessibility
- Natural efficiency argument: event cameras use less data than RGB cameras
- Easy to explain to non-specialists
- Enables SNN vs ANN comparison with energy/efficiency analysis
- Existing work at BSc level (Filippo Ferrari at Manchester did a simpler DVS project for BSc)

**SHD -- "Biologically Plausible Speech Processing"**
- Unique angle: audio processing with SNNs (rare in undergraduate work)
- Strong biological motivation: cochlear model mimics inner ear
- Applications: voice assistants, hearing aids, edge audio processing
- Dataset specifically designed to require temporal processing (unlike N-MNIST)
- Fast iteration cycle enables thorough experimentation
- Novel for an undergraduate thesis -- few BSc students tackle audio SNNs

### Tier 2: Acceptable Story

**CIFAR10-DVS -- "Efficient Object Recognition with Neuromorphic Computing"**
- Well-known classes (CIFAR-10 categories)
- Good for SNN vs ANN comparison
- Efficiency narrative works well
- But: artificial creation (LCD display + DVS) weakens "real neuromorphic" claim

**N-Caltech101 -- "Neuromorphic Multi-Class Recognition"**
- 101 classes provides complexity
- But: class imbalance, no standard split, artificial creation

### Tier 3: Not Recommended

**N-MNIST** -- Too easy, dataset essentially "solved", weak narrative
**SSC** -- Too large and difficult for undergraduate scope
**DVS-Lip** -- Fascinating narrative but impractical difficulty for undergraduate

---

## Part 7: Recommended Strategy for Your Thesis

### Option A: DVS128 Gesture (Recommended -- aligns with your existing research)

You have already researched the DVS128 technology stack extensively (see SNN_DVS128_TECHNOLOGY_STACK_REPORT.md). This is the safest choice:

- **Achievable baseline**: 95-97% accuracy with standard SpikingJelly pipeline
- **Contribution angles**:
  - SNN architecture comparison (LIF vs PLIF vs parametric neurons)
  - Time-step ablation study (how does T affect accuracy and efficiency?)
  - SNN vs ANN comparison with energy analysis (using syops library)
  - Robustness to different illumination conditions
  - Latency-accuracy tradeoff analysis
- **Risk**: Low. Well-documented, extensive tutorial coverage.

### Option B: SHD (Strongest novel narrative)

If you want to differentiate from other students:

- **Achievable baseline**: 88-92% accuracy with recurrent SNN
- **Contribution angles**:
  - Temporal processing comparison (feedforward vs recurrent SNN)
  - Time resolution study (what temporal resolution does speech recognition need?)
  - SNN vs RNN comparison for audio classification
  - Speaker generalization analysis (2 test speakers unseen in training)
  - Delay learning in SNNs (active research area on SHD)
- **Risk**: Low-Medium. Dataset is clean and fast, but recurrent SNNs are trickier to train than feedforward ones.

### Option C: Dual Dataset (Most ambitious, strongest thesis)

Use BOTH DVS128 Gesture AND SHD to demonstrate SNN capabilities across modalities:

- **Chapter 1**: Vision -- DVS128 Gesture recognition
- **Chapter 2**: Audio -- SHD spoken digit classification
- **Cross-cutting analysis**: Do the same SNN principles transfer across domains?
- **Risk**: Medium. Double the implementation work, but each dataset is fast to train.

---

## Part 8: Research Gaps and Confidence Assessment

### Information I am confident about (verified across multiple sources):
- Dataset sizes, classes, and sensor specifications
- Framework support (confirmed via official documentation)
- SOTA accuracy ranges (cross-referenced across papers)
- Preprocessing pipelines (confirmed via tutorials and code)

### Information with lower confidence:
- DVS-Lip absolute accuracy numbers (papers report relative improvements; estimated range is 55-65% for SNN, 70-75% for ANN)
- Exact download sizes in GB (not consistently reported)
- Training time estimates (depend heavily on hardware, batch size, time steps)
- DVS128 Gesture "100% accuracy" claim (one paper, not widely replicated)

### Gaps that could not be filled:
- Exact GPU memory requirements per dataset/model combination
- Detailed comparison of preprocessing time across frameworks
- Community adoption metrics (how many papers use each dataset per year)

---

## Sources

### DVS128 Gesture
- [IBM DVS128 Gesture original paper (CVPR 2017)](https://openaccess.thecvf.com/content_cvpr_2017/papers/Amir_A_Low_Power_CVPR_2017_paper.pdf)
- [SpikePoint SNN (2024)](https://arxiv.org/html/2310.07189v2)
- [CatalyzeX DVS128 benchmarks](https://www.catalyzex.com/s/Dvs128%20Gesture%20Dataset)
- [snnTorch DVS Gesture loader](https://snntorch.readthedocs.io/en/latest/_modules/snntorch/spikevision/spikedata/dvs_gesture.html)
- [DVS128 Gesture PyTorch repo](https://github.com/wponghiran/dvs128_gesture_pytorch)

### N-MNIST
- [Original N-MNIST paper (Orchard et al., 2015)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4644806/)
- ["Is Neuromorphic MNIST Neuromorphic?" (2021)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.608567/full)
- [N-MNIST download page](https://www.garrickorchard.com/datasets/n-mnist)
- [snnTorch Tutorial 7 (Tonic + snnTorch)](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_7.html)

### N-Caltech101
- [Original paper (same as N-MNIST)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4644806/)
- [N-Caltech101 download](https://www.garrickorchard.com/datasets/n-caltech101)
- [NeuroMoCo (2024) -- SOTA on N-Caltech101](https://arxiv.org/html/2406.06305)
- [NDA -- Neuromorphic Data Augmentation (2022)](https://arxiv.org/abs/2203.06145)

### CIFAR10-DVS
- [Original paper (Li et al., 2017)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2017.00309/full)
- [Figshare download](https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671)
- [ICLR 2025 -- 92.5% accuracy](https://proceedings.iclr.cc/paper_files/paper/2025/file/6b6492cd06db22bac024506e9ed0925e-Paper-Conference.pdf)
- [ICLR 2024 -- sparse SNN](https://proceedings.iclr.cc/paper_files/paper/2024/file/dd3c889922df2112a5b1769e3c19e28e-Paper-Conference.pdf)
- [Papers With Code -- CIFAR10-DVS](https://paperswithcode.com/dataset/cifar10-dvs)

### SHD
- [Zenke Lab -- SHD and SSC datasets](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/)
- [Original paper (Cramer et al., 2019)](https://arxiv.org/abs/1910.07407)
- [IEEE DataPort](https://ieee-dataport.org/open-access/heidelberg-spiking-datasets)
- [SpikCommander (2025) -- 96.41% SOTA](https://arxiv.org/html/2511.07883)
- [sparch toolkit](https://github.com/idiap/sparch)
- [Tonic SHD documentation](https://tonic.readthedocs.io/en/latest/generated/tonic.datasets.SHD.html)
- [Rockpool SHD tutorial](https://rockpool.ai/tutorials/rockpool-shd.html)

### SSC
- [Zenke Lab (same page as SHD)](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/)
- [SpikCommander -- 85.98% SOTA](https://arxiv.org/html/2511.07883)
- [SpikeSCR (2024)](https://arxiv.org/html/2412.12858v1)
- [Surrogate gradient baseline](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.865897/full)

### DVS-Lip
- [CVPR 2022 -- MSTP paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Tan_Multi-Grained_Spatio-Temporal_Features_Perceived_Network_for_Event-Based_Lip-Reading_CVPR_2022_paper.pdf)
- [DVS-Lip project page](https://sites.google.com/view/event-based-lipreading)
- [CVPR 2024 Workshop -- SpikGRU for DVS-Lip](https://openaccess.thecvf.com/content/CVPR2024W/EVW/html/Dampfhoffer_Neuromorphic_Lip-Reading_with_Signed_Spiking_Gated_Recurrent_Units_CVPRW_2024_paper.html)
- [SpikGRU-DVSLip code](https://github.com/manondampfhoffer/SpikGRU-DVSLip)
- [Tonic DVSLip](https://tonic.readthedocs.io/en/latest/generated/tonic.datasets.DVSLip.html)

### Newer Datasets
- [DailyDVS-200 (ECCV 2024)](https://arxiv.org/abs/2407.05106)
- [eTraM (CVPR 2024)](https://eventbasedvision.github.io/eTraM/)
- [EvDET200K (2024/2025)](https://github.com/Event-AHU/OpenEvDET)
- [HARDVS](https://github.com/Event-AHU/HARDVS)

### Frameworks
- [Tonic -- all datasets](https://tonic.readthedocs.io/en/latest/datasets.html)
- [SpikingJelly](https://github.com/fangwei123456/spikingjelly)
- [snnTorch](https://snntorch.readthedocs.io/en/latest/)
- [Norse](https://github.com/norse/norse)
- [NeuroBench framework](https://neurobench.ai/)

### General
- [NeuroBench (Nature Communications, 2025)](https://www.nature.com/articles/s41467-025-56739-4)
- [Event-based vision resources (comprehensive list)](https://github.com/eventbasedvision/event-based-vision)
- [Open Neuromorphic software guide](https://open-neuromorphic.org/neuromorphic-computing/software/)
