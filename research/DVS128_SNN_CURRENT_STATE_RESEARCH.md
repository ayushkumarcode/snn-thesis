# DVS128 Gesture Recognition with SNNs: Current State of the Field

**Research Date:** 2026-02-25
**Purpose:** Comprehensive assessment for undergraduate thesis project planning

---

## Executive Summary

DVS128 Gesture recognition using Spiking Neural Networks is approaching a **saturated benchmark**. The dataset, originally introduced by IBM in 2017 with a baseline of ~94%, has seen accuracy climb to 99.6% (TENNs-PLEIADES) and even 100% (STREAM) as of 2024-2025. The dataset contains only 1,464 samples (1,176 train / 288 test) across 11 gesture classes, making it small by modern standards. This saturation means that simply achieving high accuracy is no longer a meaningful contribution.

However, several genuine research opportunities remain that are well-suited to an undergraduate thesis: (1) systematic comparison studies across architectures/frameworks that nobody has done cleanly, (2) efficiency-focused investigations (accuracy vs. parameters vs. timesteps vs. energy), (3) event representation ablation studies, and (4) neuron model comparisons (LIF vs. PLIF vs. others). The key differentiator for a good undergrad project is NOT achieving SOTA accuracy -- it is rigorous experimental methodology, reproducibility, and genuine analysis.

**SpikingJelly** is the recommended framework. It has built-in DVS128 support, faster training via CUDA kernels, and a working end-to-end example achieving ~96% accuracy out of the box. snnTorch's DVS loader (spikevision) is **deprecated and broken**, requiring the Tonic library as a workaround, adding pipeline complexity.

---

## 1. State-of-the-Art Accuracy on DVS128 Gesture

### Current Leaderboard (Papers with Code, as of Feb 2025)

| Rank | Method | Accuracy (%) | Year | Parameters | Timesteps | Type |
|------|--------|-------------|------|------------|-----------|------|
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

**Key observation:** The gap between the SpikingJelly tutorial baseline (96.18%) and SOTA (99.6%) is only ~3.4 percentage points. On a test set of only 288 samples, this difference amounts to roughly 10 correctly classified samples.

### Accuracy Progression Over Time

- 2017 (IBM original): ~94.59% (TrueNorth EEDN)
- 2020-2021: 95-97% (PLIF, TA-SNN, DECOLLE)
- 2022-2023: 97-98% (SEW-ResNet, Spikformer, attention mechanisms)
- 2024-2025: 98.7-99.6% (SpikePoint, SG-SNN, TENNs-PLEIADES)

**Source:** [Papers with Code DVS128 Gesture Benchmark](https://paperswithcode.com/sota/gesture-recognition-on-dvs128-gesture), [CatalyzeX DVS128 Gesture Dataset](https://www.catalyzex.com/s/Dvs128%20Gesture%20Dataset)

---

## 2. Most Common Architectures

### Architecture Taxonomy

| Architecture Type | Description | Representative Methods | Typical Accuracy |
|------------------|-------------|----------------------|-----------------|
| **Convolutional SNN (CSNN)** | Conv layers + spiking neurons (LIF/PLIF) + pooling. The workhorse. | SpikingJelly baseline, DECOLLE | 93-97% |
| **Spiking Transformer** | Self-attention adapted for spike-based computation (SSA). | Spikformer, Spikeformer, MSVIT, Spike-HAR++ | 97-99% |
| **Recurrent SNN (RSNN/SCRNN)** | Recurrent connections for temporal modeling. | SCRNN, ALIF-based models | 92-96% |
| **Point-based SNN** | Process events as 3D point clouds, avoiding frame conversion. | SpikePoint | 98.74% |
| **Temporal Kernel Networks** | Structured temporal convolutions (not strictly SNN). | TENNs-PLEIADES, STREAM | 99.6-100% |
| **Self-Organizing SNN** | Topographic maps + glial cell mechanisms. | SG-SNN | 99.3% |
| **Hybrid (ANN-SNN)** | ANN-to-SNN conversion or knowledge distillation. | HSD, BKDSNN | 95-98% |
| **Lightweight/Pruned SNN** | Focus on parameter efficiency. | NSPDI-SNN (<7K params), LightSNN | 94-97% |

### Most Common Architecture for New Projects: CSNN

The **Convolutional SNN** with LIF or PLIF neurons is by far the most common starting point. The standard architecture pattern used in SpikingJelly's tutorial and most papers is:

```
{Conv2d-BatchNorm-LIF-MaxPool}*N -> Flatten -> FC-LIF -> FC-LIF -> Output
```

Typical configuration: 5 convolutional blocks with 128 channels, 3x3 kernels, followed by 2 fully connected layers (512, then 11 classes).

### Neuron Models

| Neuron | Description | DVS128 Gesture Impact |
|--------|-------------|----------------------|
| **LIF** | Leaky Integrate-and-Fire. Fixed decay constant. Standard choice. | Baseline ~96% |
| **PLIF** | Parametric LIF. Learnable decay per layer. ICCV 2021. | ~1-2% improvement over LIF |
| **IF** | Integrate-and-Fire. No leak. Simpler but less expressive. | Lower accuracy |
| **ALIF** | Adaptive LIF. Learnable threshold adaptation. | Improved temporal modeling |

**Key finding:** PLIF (learnable membrane time constant) consistently outperforms fixed-parameter LIF neurons, and is less sensitive to hyperparameter initialization. This is supported by the SpikingJelly authors' own ICCV 2021 paper.

**Source:** [PLIF Paper (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/papers/Fang_Incorporating_Learnable_Membrane_Time_Constant_To_Enhance_Learning_of_Spiking_ICCV_2021_paper.pdf), [GitHub: Parametric-LIF](https://github.com/fangwei123456/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron)

---

## 3. Framework Comparison: SpikingJelly vs. snnTorch

### Head-to-Head Comparison

| Feature | SpikingJelly | snnTorch |
|---------|-------------|---------|
| **DVS128 Loader** | Built-in, works well. Auto-downloads AEDAT, converts to npz/frames. | DEPRECATED (spikevision broken). Must use Tonic library. |
| **DVS128 Tutorial** | Complete end-to-end classification tutorial with code | Partial (Tutorial 7 uses NMNIST, not DVS128 directly) |
| **Pre-built DVS Model** | Yes: classify_dvsg example with full training script | No pre-built DVS128 model |
| **Training Speed** | ~18s/epoch (CUDA backend) to ~28s/epoch (PyTorch backend) on RTX 2080 Ti | Slower. No custom CUDA kernels. torch.compile does not help snnTorch significantly. |
| **CUDA Acceleration** | CuPy backend: 0.26s fwd+bwd for 16K neuron benchmark. Up to 11x speedup at T=32. | No custom CUDA. Relies on standard PyTorch. |
| **Neuron Models** | LIF, PLIF, IF, QIF, EIF, Izhikevich | Leaky, Synaptic, Alpha, Recurrent LIF, LSTM-based |
| **Documentation Quality** | Good but partly in Chinese. English docs available. | Excellent. Very tutorial-driven, beginner-friendly. |
| **Community/Stars** | ~3.3K GitHub stars | ~2.5K GitHub stars |
| **Academic Paper** | Science Advances (2023) | NeurIPS Workshop |
| **Multi-step Processing** | Native support. SeqToANNContainer for parallel timestep processing. | Sequential processing only (loop over timesteps). |
| **Surrogate Gradients** | ATan, Sigmoid, many others | ATan, Fast Sigmoid, Straight-Through, Triangular |
| **Active Development** | Active (last commit recent) | Active (v0.9.4 latest) |
| **PyPI Package** | `pip install spikingjelly` | `pip install snntorch` |

### Framework Benchmark (Open Neuromorphic, 2024)

For a 16K neuron network forward+backward pass:

| Framework | Time (seconds) | Notes |
|-----------|---------------|-------|
| SpikingJelly (CuPy) | 0.26 | Fastest by significant margin |
| Lava DL (SLAYER) | ~0.4-0.5 | Custom CUDA |
| Sinabs EXODUS | ~0.4-0.5 | Custom CUDA |
| Norse (torch.compile) | ~0.5-0.7 | JAX-competitive with compilation |
| snnTorch | ~1.0+ | No significant torch.compile speedup |
| Spyx (JAX) | ~0.3-0.4 | JAX-based, different ecosystem |

### Verdict for Thesis Project

**SpikingJelly is the clear winner for DVS128 Gesture recognition.** It has:
- A working DVS128 data pipeline out of the box
- A complete classification example achieving 96.18% accuracy
- Significantly faster training through CUDA kernels
- The PLIF neuron model (learnable parameters) built-in

snnTorch is better for learning fundamentals and has better documentation, but its DVS128 support is broken and requires workarounds through Tonic. If you choose snnTorch, budget extra time for data pipeline setup.

**Source:** [SNN Library Benchmarks - Open Neuromorphic](https://open-neuromorphic.org/blog/spiking-neural-network-framework-benchmarking/), [SpikingJelly GitHub](https://github.com/fangwei123456/spikingjelly), [snnTorch GitHub Issue #285](https://github.com/jeshraghian/snntorch/issues/285)

---

## 4. Data Pipeline Complexity

### Dataset Specifications

| Property | Value |
|----------|-------|
| **Source** | IBM DVS128 camera |
| **Resolution** | 128 x 128 pixels |
| **Classes** | 11 hand/body gestures |
| **Subjects** | 29 people |
| **Lighting conditions** | 3 (natural, LED, fluorescent) |
| **Training samples** | 1,176 |
| **Test samples** | 288 |
| **Total samples** | 1,464 |
| **Raw format** | AEDAT 3.1 (binary) |
| **Download size** | ~3 GB (tar file), ~5 GB extracted |
| **Download source** | IBM Box (manual download required, no API) |
| **Event format** | (x, y, timestamp, polarity) per event |

### Pipeline Steps (SpikingJelly)

The data pipeline has **moderate complexity**. It is not trivial but is largely handled by the framework.

**Step 1: Download (Manual)**
- Download from IBM Box: https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8
- Login required. Cannot be automated with Python scripts.
- This is an annoyance but a one-time step.

**Step 2: Extract and Convert (Automatic, first run only)**
```python
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

# First run: extracts AEDAT files, converts to .npz events, splits by gesture
# Takes 10-30 minutes on first run depending on hardware
dataset = DVS128Gesture(root='./data/DVS128Gesture', data_type='frame',
                         frames_number=16, split_by='number')
```

SpikingJelly automatically:
1. Verifies MD5 checksums
2. Extracts AEDAT 3.1 binary files
3. Parses binary event data (64-bit structure: polarity, x, y, timestamp)
4. Cuts each recording into individual gesture samples using CSV labels
5. Saves events as .npz files
6. Converts events to frames (if `data_type='frame'`)

**Step 3: Event-to-Frame Conversion**

Events must be binned into frames for batch training. Three strategies:

| Method | Description | Param |
|--------|-------------|-------|
| `split_by='number'` | Fixed number of frames (e.g., T=16). Each frame has roughly equal events. | `frames_number=16` |
| `split_by='time'` | Fixed time window per frame. Variable number of frames. | `duration=time_ms` |
| Custom transforms | Via Tonic: ToFrame, Denoise, Downsample | Various |

Output tensor shape: `[T, 2, 128, 128]` where T = number of time bins, 2 = polarity channels (ON/OFF).

**Step 4: Batching (Non-trivial)**

Each sample has different temporal length. Two approaches:
- **Fixed T (recommended):** Use `frames_number=16` to get uniform tensors. Simpler batching.
- **Variable T:** Use `pad_sequence_collate` to pad shorter sequences. More faithful to data but slower.

### Pipeline Steps (snnTorch + Tonic)

```python
import tonic
import tonic.transforms as transforms

sensor_size = tonic.datasets.DVSGesture.sensor_size  # (128, 128, 2)
transform = transforms.Compose([
    transforms.Denoise(filter_time=10000),
    transforms.ToFrame(sensor_size=sensor_size, time_window=1000),
])

train_ds = tonic.datasets.DVSGesture(save_to='./data', train=True, transform=transform)
# Uses DiskCachedDataset for performance
from tonic import DiskCachedDataset
cached_train = DiskCachedDataset(train_ds, cache_path='./cache/dvsg/train')
train_dl = DataLoader(cached_train, batch_size=16,
                      collate_fn=tonic.collation.PadTensors(batch_first=True))
```

**Complexity assessment:** The Tonic route requires understanding event transforms, caching strategies, and custom collation functions. It is more complex than SpikingJelly's integrated approach, but more flexible.

### Summary: Data Pipeline Complexity Rating

| Aspect | Difficulty (1-5) | Notes |
|--------|------------------|-------|
| Dataset download | 2/5 | Manual download from IBM Box, one-time |
| Initial preprocessing | 1/5 | Automatic in SpikingJelly, 10-30 min wait |
| Understanding event representation | 4/5 | Events vs. frames is conceptually challenging |
| Frame conversion config | 3/5 | Choosing time bins, number of frames, bin strategy |
| Batching variable-length data | 3/5 | Padding, collation functions needed |
| Overall pipeline (SpikingJelly) | **2/5** | Mostly handled by framework |
| Overall pipeline (snnTorch+Tonic) | **3/5** | More manual setup required |

**Source:** [SpikingJelly DVS128 Dataset Code](https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/datasets/dvs128_gesture.py), [Tonic DVSGesture Docs](https://tonic.readthedocs.io/en/latest/generated/tonic.datasets.DVSGesture.html), [SpikingJelly Neuromorphic Datasets Tutorial](https://spikingjelly.readthedocs.io/zh-cn/0.0.0.0.4/clock_driven_en/13_neuromorphic_datasets.html)

---

## 5. Is There Anything Novel Left to Do?

### The Hard Truth

DVS128 Gesture is **not solved in the absolute sense**, but it is **saturated as a pure accuracy benchmark**. When multiple methods achieve 99%+ accuracy on a 288-sample test set, the differences are statistically meaningless. The dataset is:
- Too small (1,464 total samples)
- Too constrained (controlled lab conditions)
- Too well-studied (14+ papers with code on Papers with Code alone)

### What IS Still Novel and Feasible for an Undergraduate Thesis

| Research Direction | Novelty Level | Feasibility | Notes |
|-------------------|--------------|-------------|-------|
| **1. Systematic Architecture Comparison** | Medium | High | Nobody has done a clean, controlled comparison of CSNN vs. Spiking Transformer vs. Recurrent SNN vs. Point-based SNN using the same framework, same preprocessing, same hyperparameter budget. This is actually valuable. |
| **2. Neuron Model Ablation** | Medium | High | Compare LIF vs. PLIF vs. IF vs. ALIF on the same architecture. Study learned membrane constants. Visualize what PLIF learns. |
| **3. Event Representation Study** | Medium | High | Compare event frames (fixed-count vs. fixed-time binning), voxel grids, time surfaces on the same model. Analyze which preserves temporal info best. |
| **4. Timestep-Accuracy-Energy Tradeoff** | Medium-High | High | Systematically vary T from 4 to 64 and measure accuracy, training time, and estimated energy (SynOps). Plot Pareto frontiers. |
| **5. Efficiency Analysis** | Medium-High | Medium | Accuracy per parameter, accuracy per FLOP/SynOp. Compare a 7K-param model (NSPDI-SNN style) vs. a 1.5M-param model. |
| **6. Framework Comparison** | Medium | High | Same model architecture implemented in SpikingJelly vs. snnTorch vs. Norse. Compare training speed, accuracy, reproducibility. |
| **7. Knowledge Distillation ANN-to-SNN** | High | Medium | Train an ANN teacher on frames, distill to SNN student. Measure accuracy gap and efficiency gain. |
| **8. Attention Mechanism Integration** | Medium-High | Medium | Add temporal or channel attention (TCJA module) to a baseline CSNN and measure improvement. |
| **9. Robustness / Generalization** | High | Medium | Test how well a model trained on DVS128 transfers to unseen conditions, noise injection, temporal perturbation. |
| **10. Visualization/Interpretability** | Medium | High | Visualize spiking activity patterns, membrane potential dynamics, learned features across layers. This is under-explored and pedagogically valuable. |

### What Would NOT Be Novel

- Simply training a CSNN on DVS128 and reporting accuracy
- Reproducing SpikingJelly's tutorial and calling it a thesis
- Achieving 97% accuracy with a standard model
- "Using SNNs for gesture recognition" without a specific research question

**Source:** [SNN for Event-Based Action Recognition Survey](https://www.sciencedirect.com/science/article/pii/S0925231224014280), [SNN in Imaging Review](https://www.mdpi.com/1424-8220/25/21/6747), [Event Cameras in 2025 Blog](https://lenzgregor.com/posts/event-cameras-2025-part2/)

---

## 6. What Differentiates a Good Undergraduate Project

### Minimum Viable Thesis (Gets a Pass)

- Reproduce the SpikingJelly DVS128 example
- Report accuracy, plot training curves
- Brief literature review
- Minimal analysis

### Good Thesis (Gets a Distinction)

A good undergraduate thesis on this topic would include ALL of the following:

**1. A Clear Research Question**
Not "can SNNs classify DVS128 gestures?" (answer: obviously yes) but rather:
- "How does the choice of neuron model affect accuracy-efficiency tradeoffs on DVS128?"
- "What is the minimum number of timesteps needed to maintain 95%+ accuracy?"
- "How do different event representations affect SNN performance?"

**2. Controlled Experimental Design**
- Same random seeds across experiments
- Same data preprocessing and splits
- Statistical significance: run each experiment 3-5 times, report mean +/- std
- Ablation studies isolating single variables
- Fair comparison: same parameter budget when comparing architectures

**3. Multi-Dimensional Evaluation**
Go beyond accuracy. Report:
- Test accuracy (mean +/- std over multiple runs)
- Number of parameters
- Training time (wall clock, per epoch, total)
- Estimated energy consumption (SynOps via syops library)
- Inference latency
- Confusion matrices per class
- Per-class accuracy (some gestures are harder than others)

**4. ANN Baseline Comparison**
Train an equivalent ANN (same architecture but ReLU instead of LIF) on frame-converted DVS data. This answers: "Does the SNN bring any advantage over a standard ANN for this task?"

**5. Visualization and Analysis**
- Spike raster plots showing neuron activity
- Membrane potential traces
- t-SNE/UMAP of learned representations
- Analysis of failure cases: which gestures are confused and why?
- Visualization of learned PLIF time constants across layers

**6. Reproducibility**
- Clean code in a public repository
- Environment/dependency files (requirements.txt, conda env)
- Clear instructions to reproduce all experiments
- Saved model checkpoints and logs

**Source:** General academic thesis standards, [SpikingJelly classify_dvsg documentation](https://github.com/fangwei123456/spikingjelly/blob/master/docs/source/activation_based_en/classify_dvsg.rst)

---

