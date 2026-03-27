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
