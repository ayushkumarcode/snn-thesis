# Comprehensive Technology Stack Report: Spiking Neural Network DVS128 Gesture Recognition

**Research Date:** 2026-02-23
**Scope:** Full-stack investigation for building an SNN-based DVS128 Gesture Recognition system

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [SpikingJelly Framework](#2-spikingjelly-framework)
3. [snnTorch Framework](#3-snntorch-framework)
4. [DVS128 Data Pipeline](#4-dvs128-data-pipeline)
5. [Tonic Library](#5-tonic-library)
6. [Training Infrastructure](#6-training-infrastructure)
7. [Evaluation and Visualization](#7-evaluation-and-visualization)
8. [End-to-End Examples](#8-end-to-end-examples)
9. [ANN Comparison and Frame Integration](#9-ann-comparison-and-frame-integration)
10. [State-of-the-Art Benchmarks](#10-state-of-the-art-benchmarks)
11. [Recommended Stack Configuration](#11-recommended-stack-configuration)
12. [Sources](#12-sources)

---

## 1. Executive Summary

Building a Spiking Neural Network for DVS128 Gesture Recognition requires integrating several key components: a neuromorphic dataset loader, an SNN framework built on PyTorch, appropriate preprocessing transforms, and visualization/evaluation tools. The two leading SNN frameworks are **SpikingJelly** (from the Peking University group, published in Science Advances) and **snnTorch** (from UC Santa Cruz, by Jason Eshraghian). Both are built on PyTorch and support surrogate gradient-based backpropagation for training deep SNNs.

SpikingJelly provides a more complete built-in pipeline for DVS128 Gesture classification, including its own dataset loader, a pre-built DVSGestureNet model, and full training scripts. snnTorch offers a more modular, tutorial-driven approach and integrates with the **Tonic** library for neuromorphic dataset loading. For a thesis project, SpikingJelly offers the fastest path to a working baseline, while snnTorch offers better educational resources and more flexible visualization tools.

The DVS128 Gesture dataset consists of 1176 training and 288 test samples across 11 gesture classes, recorded using a Dynamic Vision Sensor (DVS128) camera at 128x128 resolution. Raw data comes in AEDAT 3.1 format as polarity events (x, y, timestamp, polarity). Events must be converted to frame tensors for batch training, which both frameworks handle through various binning strategies.

Training on a single GPU (e.g., RTX 2080 Ti with 12GB VRAM) takes approximately 18-28 seconds per epoch, with 64-256 epochs typically needed to reach peak accuracy of approximately 96-97%. Apple M-series MacBooks can be used via PyTorch MPS backend, though the CuPy/Triton acceleration backends for SpikingJelly will not be available. Energy efficiency can be estimated without neuromorphic hardware using the **syops** library, which computes synaptic operations (SynOps) and estimates energy based on 45nm technology costs.

---

## 2. SpikingJelly Framework

### 2.1 Version Requirements

| Dependency | Version |
|---|---|
| Python | >= 3.6 (recommended: 3.9 - 3.11) |
| PyTorch | >= 2.2.0 (tested on 2.7.1) |
| torchvision | Required (version matching PyTorch) |
| torchaudio | Required (version matching PyTorch) |
| numpy | Required (no pinned version) |
| scipy | Required (no pinned version) |
| tensorboard | Required |
| einops | Required |
| tqdm | Required |
| h5py | Required |
| matplotlib | Required |

**Source:** https://github.com/fangwei123456/spikingjelly/blob/master/setup.py

### 2.2 Installation

```bash
# Install PyTorch first (visit pytorch.org for your platform)
pip install torch torchvision torchaudio

# Install SpikingJelly stable release
pip install spikingjelly

# Or install latest from GitHub
pip install git+https://github.com/fangwei123456/spikingjelly.git

# Optional: Triton backend (typically comes with PyTorch 2.x)
# Tested on triton==3.3.1

# Optional: CuPy backend (NVIDIA CUDA only)
# pip install cupy-cuda12x  # For CUDA 12.x
# pip install cupy-cuda11x  # For CUDA 11.x
```

### 2.3 Neuron Models

SpikingJelly provides the following spiking neuron models in `spikingjelly.activation_based.neuron`:

| Neuron Model | Description |
|---|---|
| `IFNode` | Integrate-and-Fire neuron. Simplest model: V += X; fire if V > threshold |
| `LIFNode` | Leaky Integrate-and-Fire. V += (X - (V - V_reset)) / tau. Most commonly used |
| `ParametricLIFNode` | PLIF: learnable membrane time constant tau per layer, from ICCV 2021 paper |
| `QIFNode` | Quadratic Integrate-and-Fire neuron |
| `EIFNode` | Exponential Integrate-and-Fire neuron |
| `IzhikevichNode` | Izhikevich neuron model |

All neurons support:
- **Single-step mode** (`'s'`): process one timestep at a time
- **Multi-step mode** (`'m'`): process all timesteps in parallel (faster)
- **Three backends**: `torch` (CPU/GPU), `cupy` (NVIDIA GPU only), `triton` (NVIDIA GPU only, fastest)

### 2.4 Surrogate Gradient Functions

Available in `spikingjelly.activation_based.surrogate`:

- `ATan` - Arctangent surrogate (most commonly used)
- `Sigmoid` - Sigmoid surrogate
- `PiecewiseQuadratic` - Piecewise quadratic
- `PiecewiseExponential` - Piecewise exponential
- `SoftSign` - Soft sign function
- `Erf` - Gaussian error function
- `NonzeroSignLogAbs` - Log-abs surrogate
- `PiecewiseLeakyReLU` - Piecewise leaky ReLU

### 2.5 DVS128 Gesture Dataset Loader

SpikingJelly has a built-in dataset loader at `spikingjelly.datasets.dvs128_gesture.DVS128Gesture`.

**Loading raw events:**
```python
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

# Load raw events
event_set = DVS128Gesture(root='./data/DVS128Gesture', train=True, data_type='event')
event, label = event_set[0]
# event is a dict with keys: 't', 'x', 'y', 'p'
# label is an integer 0-10
```

**Loading as frames (fixed number):**
```python
# Split events into T=16 frames, each with roughly equal number of events
train_set = DVS128Gesture(
    root='./data/DVS128Gesture',
    train=True,
    data_type='frame',
    frames_number=16,       # Number of frames per sample
    split_by='number'        # Split events equally by count
)
# Returns tensors of shape [T, C, H, W] = [16, 2, 128, 128]
```

**Loading as frames (fixed duration):**
```python
# Split events into frames of fixed duration
train_set = DVS128Gesture(
    root='./data/DVS128Gesture',
    train=True,
    data_type='frame',
    duration=1000000,       # 1 second per frame (in microseconds)
    split_by='time'
)
# Returns variable-length tensors (different T per sample)
```

**Handling variable-length sequences:**
```python
from spikingjelly.datasets.utils import pad_sequence_collate

# When using duration-based splitting, samples have different T
# Use pad_sequence_collate to handle this
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=16,
    collate_fn=pad_sequence_collate,
    shuffle=True
)
# Returns (frames, labels, mask) where mask indicates valid timesteps
```

### 2.6 Pre-built DVSGestureNet Architecture

SpikingJelly provides a pre-built model specifically for DVS128 Gesture:

```python
from spikingjelly.activation_based.model import parametric_lif_net
from spikingjelly.activation_based import neuron, surrogate, functional

# Architecture: {Conv128-BN-LIF-MaxPool}x5 -> Dropout -> FC512 -> LIF -> Dropout -> FC11 -> LIF -> AvgPool
net = parametric_lif_net.DVSGestureNet(
    channels=128,                        # Number of channels in conv layers
    spiking_neuron=neuron.LIFNode,       # Can also use ParametricLIFNode
    surrogate_function=surrogate.ATan(),  # Surrogate gradient function
    detach_reset=True                     # Detach reset for better gradient flow
)

# Enable multi-step mode (processes all timesteps in parallel)
functional.set_step_mode(net, 'm')

# Optional: Use CuPy backend for speedup (NVIDIA only)
# functional.set_backend(net, 'cupy', instance=neuron.LIFNode)
```

### 2.7 Complete Training Loop

```python
import torch
import torch.nn.functional as F
from torch.cuda import amp
from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.activation_based.model import parametric_lif_net
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

# === Configuration ===
T = 16              # Timesteps
batch_size = 16
epochs = 64
lr = 0.1
device = 'cuda:0'   # or 'mps' for Mac
channels = 128

# === Model ===
net = parametric_lif_net.DVSGestureNet(
    channels=channels,
    spiking_neuron=neuron.LIFNode,
    surrogate_function=surrogate.ATan(),
    detach_reset=True
)
functional.set_step_mode(net, 'm')
net.to(device)

# === Dataset ===
train_set = DVS128Gesture(root='./data/DVS128Gesture', train=True,
                          data_type='frame', frames_number=T, split_by='number')
test_set = DVS128Gesture(root='./data/DVS128Gesture', train=False,
                         data_type='frame', frames_number=T, split_by='number')

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           shuffle=True, num_workers=4, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                          shuffle=False, num_workers=4, pin_memory=True)

# === Optimizer ===
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
scaler = amp.GradScaler()  # For mixed precision

# === Training ===
