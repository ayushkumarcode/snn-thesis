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

### accuracy over time

- 2017 (IBM original): ~94.59% (TrueNorth EEDN)
- 2020-2021: 95-97% (PLIF, TA-SNN, DECOLLE)
- 2022-2023: 97-98% (SEW-ResNet, Spikformer, attention mechanisms)
- 2024-2025: 98.7-99.6% (SpikePoint, SG-SNN, TENNs-PLEIADES)

Sources: [Papers with Code DVS128 Benchmark](https://paperswithcode.com/sota/gesture-recognition-on-dvs128-gesture), [CatalyzeX](https://www.catalyzex.com/s/Dvs128%20Gesture%20Dataset)

---

## 2. most common architectures

### taxonomy

| Type | Description | Examples | Typical accuracy |
|------|-------------|---------|-----------------|
| **Convolutional SNN (CSNN)** | Conv layers + spiking neurons (LIF/PLIF) + pooling. The workhorse. | SpikingJelly baseline, DECOLLE | 93-97% |
| **Spiking Transformer** | Self-attention adapted for spikes (SSA). | Spikformer, Spikeformer, MSVIT, Spike-HAR++ | 97-99% |
| **Recurrent SNN** | Recurrent connections for temporal modeling. | SCRNN, ALIF-based models | 92-96% |
| **Point-based SNN** | Process events as 3D point clouds, skip frame conversion. | SpikePoint | 98.74% |
| **Temporal Kernel Networks** | Structured temporal convolutions (not strictly SNN). | TENNs-PLEIADES, STREAM | 99.6-100% |
| **Self-Organizing SNN** | Topographic maps + glial cell mechanisms. | SG-SNN | 99.3% |
| **Hybrid (ANN-SNN)** | ANN-to-SNN conversion or knowledge distillation. | HSD, BKDSNN | 95-98% |
| **Lightweight/Pruned SNN** | Focus on parameter efficiency. | NSPDI-SNN (<7K params), LightSNN | 94-97% |

### the standard starting point: CSNN

the convolutional SNN with LIF or PLIF neurons is by far the most common. typical pattern from SpikingJelly's tutorial:

```
{Conv2d-BatchNorm-LIF-MaxPool}*N -> Flatten -> FC-LIF -> FC-LIF -> Output
```

usually 5 conv blocks with 128 channels, 3x3 kernels, then 2 FC layers (512, then 11 classes).

### neuron models

| Neuron | What it is | Impact on DVS128 |
|--------|-----------|-----------------|
| **LIF** | Leaky Integrate-and-Fire. Fixed decay constant. Standard. | Baseline ~96% |
| **PLIF** | Parametric LIF. Learnable decay per layer. ICCV 2021. | ~1-2% improvement over LIF |
| **IF** | Integrate-and-Fire. No leak. Simpler but less expressive. | Lower accuracy |
| **ALIF** | Adaptive LIF. Learnable threshold adaptation. | Better temporal modeling |

PLIF (learnable membrane time constant) consistently beats fixed-parameter LIF and is less sensitive to hyperparameter initialization. backed by the SpikingJelly authors' ICCV 2021 paper.

Sources: [PLIF Paper (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/papers/Fang_Incorporating_Learnable_Membrane_Time_Constant_To_Enhance_Learning_of_Spiking_ICCV_2021_paper.pdf), [Parametric-LIF GitHub](https://github.com/fangwei123456/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron)

---

## 3. framework comparison: SpikingJelly vs snnTorch

### head-to-head

| Feature | SpikingJelly | snnTorch |
|---------|-------------|---------|
| **DVS128 Loader** | Built-in, works well. Auto-downloads AEDAT, converts to npz/frames. | DEPRECATED (spikevision broken). Must use Tonic. |
| **DVS128 Tutorial** | Complete end-to-end classification with code | Partial (Tutorial 7 uses NMNIST, not DVS128 directly) |
| **Pre-built DVS Model** | Yes: classify_dvsg example, full training script | No |
| **Training Speed** | ~18s/epoch (CUDA) to ~28s/epoch (PyTorch) on RTX 2080 Ti | Slower. No custom CUDA kernels. torch.compile doesn't help much. |
| **CUDA Acceleration** | CuPy backend: 0.26s fwd+bwd for 16K neuron benchmark. Up to 11x speedup at T=32. | No custom CUDA. Standard PyTorch only. |
| **Neuron Models** | LIF, PLIF, IF, QIF, EIF, Izhikevich | Leaky, Synaptic, Alpha, Recurrent LIF, LSTM-based |
| **Docs** | Good but partly in Chinese. English docs available. | Excellent. Very tutorial-driven, beginner-friendly. |
| **Stars** | ~3.3K | ~2.5K |
| **Publication** | Science Advances (2023) | NeurIPS Workshop |
| **Multi-step Processing** | Native. SeqToANNContainer for parallel timestep processing. | Sequential only (loop over timesteps). |
| **Surrogate Gradients** | ATan, Sigmoid, many others | ATan, Fast Sigmoid, Straight-Through, Triangular |

### framework benchmark (Open Neuromorphic, 2024)

for a 16K neuron forward+backward pass:

| Framework | Time (seconds) | Notes |
|-----------|---------------|-------|
| SpikingJelly (CuPy) | 0.26 | Fastest by a lot |
| Lava DL (SLAYER) | ~0.4-0.5 | Custom CUDA |
| Sinabs EXODUS | ~0.4-0.5 | Custom CUDA |
| Norse (torch.compile) | ~0.5-0.7 | |
| snnTorch | ~1.0+ | No significant torch.compile speedup |
| Spyx (JAX) | ~0.3-0.4 | Different ecosystem |

### my take

SpikingJelly is the clear winner for DVS128 work. working data pipeline, complete example at 96.18%, fast training, PLIF built-in.

snnTorch is better for learning fundamentals and has better docs, but its DVS128 support is broken. if you go with snnTorch, budget extra time for the Tonic data pipeline setup.

Sources: [SNN Benchmarks - Open Neuromorphic](https://open-neuromorphic.org/blog/spiking-neural-network-framework-benchmarking/), [SpikingJelly GitHub](https://github.com/fangwei123456/spikingjelly), [snnTorch Issue #285](https://github.com/jeshraghian/snntorch/issues/285)

---

## 4. data pipeline complexity

### dataset specs

| Property | Value |
|----------|-------|
| Source | IBM DVS128 camera |
| Resolution | 128 x 128 pixels |
| Classes | 11 hand/body gestures |
| Subjects | 29 people |
| Lighting | 3 conditions (natural, LED, fluorescent) |
| Train samples | 1,176 |
| Test samples | 288 |
| Total | 1,464 |
| Raw format | AEDAT 3.1 (binary) |
| Download size | ~3 GB tar, ~5 GB extracted |
| Download source | IBM Box (manual, needs login) |
| Event format | (x, y, timestamp, polarity) per event |

### SpikingJelly pipeline

**step 1: download (manual)** -- from IBM Box: https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8. login required, can't automate. annoying but one-time.

**step 2: extract and convert (automatic, first run only)**
```python
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
dataset = DVS128Gesture(root='./data/DVS128Gesture', data_type='frame',
                         frames_number=16, split_by='number')
```
SpikingJelly handles checksum verification, AEDAT extraction, binary parsing, gesture splitting, and npz saving. takes 10-30 minutes on first run.

**step 3: event-to-frame conversion** -- events need to be binned into frames for batch training. three strategies:

| Method | Description | Param |
|--------|-------------|-------|
| `split_by='number'` | Fixed number of frames (e.g., T=16). Roughly equal events per frame. | `frames_number=16` |
| `split_by='time'` | Fixed time window. Variable frame count. | `duration=time_ms` |
| Custom transforms | Via Tonic: ToFrame, Denoise, Downsample | Various |

output shape: `[T, 2, 128, 128]` -- T time bins, 2 polarity channels (ON/OFF).

**step 4: batching** -- each sample has different temporal length. use `frames_number=16` for uniform tensors (simpler) or `pad_sequence_collate` for variable length (more faithful but slower).

### snnTorch + Tonic pipeline

```python
import tonic
import tonic.transforms as transforms

sensor_size = tonic.datasets.DVSGesture.sensor_size  # (128, 128, 2)
transform = transforms.Compose([
    transforms.Denoise(filter_time=10000),
    transforms.ToFrame(sensor_size=sensor_size, time_window=1000),
])
train_ds = tonic.datasets.DVSGesture(save_to='./data', train=True, transform=transform)
from tonic import DiskCachedDataset
cached_train = DiskCachedDataset(train_ds, cache_path='./cache/dvsg/train')
train_dl = DataLoader(cached_train, batch_size=16,
                      collate_fn=tonic.collation.PadTensors(batch_first=True))
```

this requires understanding event transforms, caching strategies, and custom collation functions. more complex than SpikingJelly's integrated approach, but more flexible.

### complexity rating

| Aspect | Difficulty (1-5) |
|--------|------------------|
| Dataset download | 2/5 (manual from IBM Box, one-time) |
| Initial preprocessing | 1/5 (automatic in SpikingJelly) |
| Understanding event representation | 4/5 (events vs frames is conceptually tricky) |
| Frame conversion config | 3/5 (choosing bins, strategy) |
| Batching variable-length data | 3/5 (padding, collation) |
| Overall pipeline (SpikingJelly) | **2/5** |
| Overall pipeline (snnTorch+Tonic) | **3/5** |

---

## 5. is there anything novel left?

### the honest answer

DVS128 is **not solved in absolute terms** but it's **saturated as a pure accuracy benchmark**. when multiple methods hit 99%+ on 288 test samples, the differences are statistically meaningless. the dataset is too small (1,464 samples), too constrained (controlled lab conditions), and too well-studied (14+ papers with code on Papers with Code).

### what IS still novel and feasible for a thesis

| Direction | Novelty | Feasibility | Notes |
|-----------|---------|-------------|-------|
| **Architecture comparison** | Medium | High | nobody's done a clean controlled comparison of CSNN vs spiking transformer vs recurrent SNN vs point-based SNN in the same framework, same preprocessing, same hyperparameter budget. actually valuable. |
| **Neuron model ablation** | Medium | High | compare LIF vs PLIF vs IF vs ALIF on same architecture. study/visualize what PLIF learns. |
| **Event representation study** | Medium | High | compare event frames (fixed-count vs fixed-time), voxel grids, time surfaces on same model. |
| **Timestep-accuracy-energy tradeoff** | Medium-High | High | vary T from 4 to 64, measure accuracy, time, estimated energy. plot Pareto frontiers. |
| **Efficiency analysis** | Medium-High | Medium | accuracy per parameter, accuracy per FLOP/SynOp. compare 7K-param model vs 1.5M-param model. |
| **Framework comparison** | Medium | High | same architecture in SpikingJelly vs snnTorch vs Norse. training speed, accuracy, reproducibility. |
| **Knowledge distillation ANN-to-SNN** | High | Medium | ANN teacher on frames, SNN student. measure gap and efficiency gain. |
| **Attention integration** | Medium-High | Medium | add temporal/channel attention to baseline CSNN, measure improvement. |
| **Robustness/generalization** | High | Medium | how well does a DVS128-trained model handle unseen conditions, noise, temporal perturbation? |
| **Visualization/interpretability** | Medium | High | spike raster plots, membrane traces, t-SNE, learned features. under-explored and pedagogically valuable. |

### what would NOT be novel
- just training a CSNN on DVS128 and reporting accuracy
- reproducing SpikingJelly's tutorial and calling it a thesis
- getting 97% with a standard model
- "using SNNs for gesture recognition" without a specific research question
