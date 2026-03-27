# DVS128 gesture recognition with SNNs -- where things stand

so i looked into DVS128 Gesture as a potential thesis dataset and... it's basically approaching saturation. the dataset was introduced by IBM in 2017 with a baseline around 94%, and accuracy has now climbed to 99.6% (TENNs-PLEIADES) and even 100% (STREAM) as of 2024-2025. it only has 1,464 samples (1,176 train / 288 test) across 11 gesture classes, which is pretty small. just getting high accuracy on this isn't a meaningful contribution anymore.

that said, there are still real research opportunities here for a thesis: comparison studies across architectures/frameworks that nobody's done properly, efficiency investigations, event representation ablation, neuron model comparisons. the key is that a good undergrad project here is NOT about hitting SOTA accuracy -- it's about rigorous methodology, reproducibility, and actual analysis.

**SpikingJelly** is the recommended framework. built-in DVS128 support, faster training via CUDA kernels, and a working example that gets ~96% accuracy out of the box. snnTorch's DVS loader (spikevision) is deprecated and broken -- you'd need Tonic as a workaround, which adds pipeline complexity.

---

## 1. state-of-the-art accuracy

### current leaderboard (as of Feb 2025)

| Rank | Method | Accuracy (%) | Year | Params | Timesteps | Type |
|------|--------|-------------|------|--------|-----------|------|
| 1 | TENNs-PLEIADES | 99.59 (100 w/ filter) | 2024 | 192K | Variable | Temporal Neural Network |
| 2 | SG-SNN | 99.3 | 2025 | Not reported | Multiple | Self-Organizing Glial SNN |
| 3 | Spikeformer | 98.96 | 2024 | Large | 16 | Spiking Transformer |
| 4 | MSVIT | 98.80 | 2025 | Not reported | 16 | Multi-Scale Vision Transformer |
| 5 | SpikePoint | 98.74 | 2024 | Small (~1/21 of ANN equiv.) | 16 | Point-based SNN |
| 6 | Spike-HAR++ | 98.26 | 2024 | Lightweight | - | Spiking Transformer |
| 7 | STREAM | 100.0 | 2024 | Not reported | - | Temporal Kernel Network |
| 8 | EgoEvGesture | 96.97 | 2025 | - | - | Egocentric SNN |
| 9 | SpikingJelly baseline | 96.18 | 2021 | ~1.5M (est.) | 16 | 5-layer CSNN + LIF |
| 10 | snnTorch example | ~90.6 | 2024 | Small | 300 | 3-layer CSNN + LIF |

worth noting: the gap between SpikingJelly's tutorial baseline (96.18%) and SOTA (99.6%) is only ~3.4 percentage points. on a test set of only 288 samples, that's roughly 10 correctly classified samples.
