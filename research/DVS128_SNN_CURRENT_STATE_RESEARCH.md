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

---

## 6. what makes a good undergrad project here

### minimum viable thesis (gets a pass)
- reproduce SpikingJelly DVS128 example
- report accuracy, plot training curves
- brief lit review
- minimal analysis

### good thesis (gets a distinction)

you'd need ALL of:

**1. a clear research question** -- not "can SNNs classify DVS128 gestures?" (obviously yes) but rather:
- "how does neuron model choice affect accuracy-efficiency tradeoffs?"
- "what's the minimum timesteps needed for 95%+ accuracy?"
- "how do different event representations affect SNN performance?"

**2. controlled experimental design**
- same random seeds across experiments
- same preprocessing and splits
- statistical significance: run each experiment 3-5 times, report mean +/- std
- ablation studies isolating single variables
- fair comparison: same parameter budget when comparing architectures

**3. multi-dimensional evaluation** -- go beyond accuracy:
- test accuracy (mean +/- std over multiple runs)
- number of parameters
- training time (wall clock, per epoch, total)
- estimated energy (SynOps)
- inference latency
- confusion matrices, per-class accuracy

**4. ANN baseline** -- same architecture with ReLU instead of LIF on frame-converted DVS data. answers: does the SNN actually bring any advantage?

**5. visualization and analysis**
- spike raster plots
- membrane potential traces
- t-SNE/UMAP of learned representations
- failure case analysis: which gestures get confused and why?
- visualization of learned PLIF time constants

**6. reproducibility** -- clean public repo, requirements.txt, clear instructions, saved checkpoints and logs.

---

## 7. comparison studies between architectures

### the gap in the literature

several papers include comparison tables, but a dedicated fair benchmark is **largely missing**. most papers compare against prior SOTA but use different preprocessing, augmentation, timesteps, hardware, and random seeds.

**nobody has published a single paper implementing 5+ architectures in the same framework with the same preprocessing, reporting accuracy, parameters, training time, and energy for all of them.** that's a genuine gap.

### compiled results from published papers

| Method | Type | Accuracy (%) | Timesteps | Year |
|--------|------|-------------|-----------|------|
| IBM EEDN (TrueNorth) | Hardware SNN | 94.59 | - | 2017 |
| DECOLLE | Online BPTT | 95.54 | 500 | 2020 |
| PLIF (SpikingJelly) | CSNN + Learnable LIF | 97.57 | 20 | 2021 |
| TA-SNN | CSNN + Temporal Attention | 98.61 | 60 | 2021 |
| SEW-ResNet | Residual SNN | 97.92 | 16 | 2022 |
| Spikformer | Spiking Transformer | 98.3 | 16 | 2023 |
| Sparse SCNN | Sparse Conv SNN | 93.40 | - | 2022 |
| SpikePoint | Point-based SNN | 98.74 | 16 | 2024 |
| Spikeformer (v2) | Spiking Transformer | 98.96 | 16 | 2024 |
| Spike-HAR++ | Spiking Transformer | 98.26 | - | 2024 |
| MSVIT | Multi-Scale ViT | 98.80 | 16 | 2025 |
| SG-SNN | Self-Organizing | 99.30 | - | 2025 |
| TENNs-PLEIADES | Temporal Kernel | 99.59 | Variable | 2024 |
| STREAM | Temporal Kernel | 100.0 | - | 2024 |
| SpikMamba | SNN + Mamba | ~99% | - | 2024 |
| STAA-SNN (CVPR 2025) | Attention Aggregator | ~98.5 (est.) | Low | 2025 |
| SpiNNaker2 deployment | Hardware SNN | 94.13 | - | 2025 |
| Embedded TCN | Ternarized CNN | 97.7 | - | 2023 |

---

## 8. training time estimates

### SpikingJelly baseline (5-layer CSNN, LIF)

| Config | GPU | Time/Epoch | Epochs to ~96% | Total |
|--------|-----|-----------|----------------|-------|
| T=16, batch=16, PyTorch backend | RTX 2080 Ti | 27.76 s | ~256 | ~2 hours |
| T=16, batch=16, CuPy backend | RTX 2080 Ti | 18.17 s | ~256 | ~1.3 hours |
| T=16, batch=16, AMP | RTX 2080 Ti | ~15-20 s | ~256 | ~1-1.5 hours |
| T=16 (estimated) | Apple M1/M2 MPS | ~40-60 s | ~256 | ~3-4 hours |

### scaling factors

| Factor | Impact |
|--------|--------|
| T: 16 -> 32 | ~2x longer |
| Batch: 16 -> 32 | ~1.3x (if GPU memory allows) |
| Adding attention layers | ~1.5-2x |
| Spiking Transformer architecture | ~2-3x vs CSNN |
| Full hyperparameter sweep (10 configs) | 10-30 hours total |

### memory constraints

- T=16, batch=16 on 12GB GPU: fine
- T=20, batch=16 on 12GB: might OOM (16GB recommended)
- T=32: needs 24GB+ or smaller batch
- Apple M-series (16-32GB unified): sufficient for most configs

### practical thesis timeline

| Phase | Duration |
|-------|----------|
| Environment setup + dataset download | 1-2 days |
| Running SpikingJelly baseline | 1 day |
| Understanding code + architecture | 3-5 days |
| Implementing comparison experiments | 2-3 weeks |
| Running all experiments | 1-2 weeks |
| Analysis and visualization | 1-2 weeks |
| Writing thesis | 3-4 weeks |

---

## 9. frontier directions (for context)

don't need to advance these, but should know about them:

- **SpikMamba (2024):** SNN + Mamba for long-range temporal dependencies. SOTA on multiple event-based action recognition datasets. Code: https://github.com/Typistchen/SpikMamba
- **STAA-SNN (CVPR 2025):** Spatial-Temporal Attention Aggregator with spike-driven self-attention.
- **ANN-to-SNN Knowledge Distillation:** HSD combines conversion with distillation. 7.50% improvement on DVS-Gesture.
- **Hardware Deployment:** SpiNNaker2 gets 94.13% on-chip. CUTIE gets 7 microJ/inference at 0.9ms latency. hardware efficiency is where DVS128 still has genuine value.
- **DVS-Gesture-Chain:** new variant testing temporal order perception rather than just gesture identity. tests whether SNNs truly use temporal info or just spatial patterns.

---

## 10. known gaps and open problems

1. **small dataset reliance** -- most SNN papers rely on DVS128, CIFAR10-DVS, NMNIST. cross-dataset evaluation is rare.
2. **reproducibility** -- few papers report standard deviations. run-to-run variance not well characterized.
3. **preprocessing sensitivity** -- results are highly sensitive to event-to-frame conversion choices but this is rarely ablated.
4. **scalability** -- SNNs are hard to scale up due to training instability, making DVS128's small size convenient but not representative.
5. **ANN vs SNN fairness** -- many comparisons aren't controlled (different architectures, preprocessing, etc.).
6. **energy claims without hardware** -- most papers estimate energy without actual neuromorphic deployment.

---

## 11. suggested thesis structures

### option A: "comparative study of SNN architectures for event-based gesture recognition"

research question: how do different architectures compare when controlling for preprocessing, parameter budget, and training procedure?

experiments:
1. Baseline CSNN (5-layer, LIF) -- reproduce SpikingJelly tutorial
2. CSNN with PLIF -- swap neuron model
3. CSNN with attention (channel/temporal)
4. Spiking Transformer (Spikformer-style)
5. ANN baseline (same conv architecture, ReLU)
6. Vary T = {4, 8, 16, 32} for each

metrics per experiment: accuracy (mean +/- std, 5 runs), parameters, SynOps, training time, per-class accuracy.

nobody has done this controlled comparison. clean, reproducible code would be genuinely useful.

### option B: "efficiency-accuracy tradeoffs in SNNs for neuromorphic gesture recognition"

research question: what's the Pareto frontier between accuracy and computational cost?

