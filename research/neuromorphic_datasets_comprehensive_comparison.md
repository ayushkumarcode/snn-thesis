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
