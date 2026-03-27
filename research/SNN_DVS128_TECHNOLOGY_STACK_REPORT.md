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
max_test_acc = 0
for epoch in range(epochs):
    # Train
    net.train()
    train_loss = train_acc = train_samples = 0
    for frame, label in train_loader:
        optimizer.zero_grad()
        frame = frame.to(device)
        # Reshape [N, T, C, H, W] -> [T, N, C, H, W] for multi-step mode
        frame = frame.transpose(0, 1)
        label = label.to(device)
        label_onehot = F.one_hot(label, 11).float()

        # Mixed precision forward pass
        with amp.autocast():
            out_fr = net(frame).mean(0)  # Average output spikes over time
            loss = F.mse_loss(out_fr, label_onehot)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_samples += label.numel()
        train_loss += loss.item() * label.numel()
        train_acc += (out_fr.argmax(1) == label).float().sum().item()

        # CRITICAL: Reset neuron states after each batch
        functional.reset_net(net)

    lr_scheduler.step()
    train_loss /= train_samples
    train_acc /= train_samples

    # Evaluate
    net.eval()
    test_loss = test_acc = test_samples = 0
    with torch.no_grad():
        for frame, label in test_loader:
            frame = frame.to(device).transpose(0, 1)
            label = label.to(device)
            label_onehot = F.one_hot(label, 11).float()
            out_fr = net(frame).mean(0)
            loss = F.mse_loss(out_fr, label_onehot)

            test_samples += label.numel()
            test_loss += loss.item() * label.numel()
            test_acc += (out_fr.argmax(1) == label).float().sum().item()
            functional.reset_net(net)

    test_acc /= test_samples
    max_test_acc = max(max_test_acc, test_acc)

    print(f'Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, '
          f'test_acc={test_acc:.4f}, max_test_acc={max_test_acc:.4f}')
```

---

## 3. snnTorch Framework

### 3.1 Version Requirements

| Dependency | Version |
|---|---|
| Python | >= 3.9 |
| PyTorch | Any modern version (no explicit pin; tested with 1.x and 2.x) |
| numpy | Required |
| pandas | Required |
| matplotlib | Optional (for visualization) |
| nir / nirtorch | Optional (for NIR exchange format) |

**Source:** https://github.com/jeshraghian/snntorch/blob/master/setup.py

### 3.2 Installation

```bash
pip install snntorch

# For neuromorphic dataset support with Tonic
pip install tonic

# Full install with all optional dependencies
pip install snntorch[full]
```

### 3.3 Neuron Models

Available in `snntorch`:

| Neuron Model | Description |
|---|---|
| `snn.Leaky` | 1st-order Leaky Integrate-and-Fire (LIF). Primary model. |
| `snn.RLeaky` | Recurrent LIF with output spike feedback connections |
| `snn.Synaptic` | 2nd-order LIF with synaptic conductance (alpha + beta decay) |
| `snn.RSynaptic` | Recurrent 2nd-order LIF |
| `snn.Lapicque` | Lapicque's RC circuit model (equivalent to Leaky, parameterized differently) |
| `snn.Alpha` | Alpha membrane model (recursive Spike Response Model) |
| `snn.LeakyParallel` | Parallelized 1st-order LIF (faster for long sequences) |
| `snn.SLSTM` | Spiking LSTM with state-thresholding |
| `snn.SConv2dLSTM` | Spiking 2D convolutional LSTM |

Key parameters for `snn.Leaky`:
- `beta`: Membrane potential decay rate (0 to 1). Higher = more memory.
- `spike_grad`: Surrogate gradient function (e.g., `surrogate.atan()`)
- `threshold`: Firing threshold (default: 1.0)
- `init_hidden`: Initialize hidden states internally (simplifies forward pass)
- `output`: Set True for the final layer to return both spikes and membrane potential

### 3.4 Complete DVS128 Gesture Training with snnTorch + Tonic

```python
import torch
import torch.nn as nn
import tonic
import tonic.transforms as transforms
from tonic import DiskCachedDataset
from torch.utils.data import DataLoader
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
import torchvision

# === Device ===
device = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))

# === Dataset with Tonic ===
sensor_size = tonic.datasets.DVSGesture.sensor_size  # (128, 128, 2)

frame_transform = transforms.Compose([
    transforms.Denoise(filter_time=10000),            # Remove noise
    transforms.ToFrame(sensor_size=sensor_size,
                       n_time_bins=16)                 # Convert to 16 frames
])

trainset = tonic.datasets.DVSGesture(save_to='./data', transform=frame_transform, train=True)
testset = tonic.datasets.DVSGesture(save_to='./data', transform=frame_transform, train=False)

# Cache to disk for faster loading on subsequent runs
cached_trainset = DiskCachedDataset(trainset,
    transform=tonic.transforms.Compose([torch.from_numpy,
                                        torchvision.transforms.RandomRotation([-10, 10])]),
    cache_path='./cache/dvs_gesture/train')
cached_testset = DiskCachedDataset(testset,
    transform=torch.from_numpy,
    cache_path='./cache/dvs_gesture/test')

trainloader = DataLoader(cached_trainset, batch_size=16, shuffle=True,
                         collate_fn=tonic.collation.PadTensors(batch_first=False))
testloader = DataLoader(cached_testset, batch_size=16,
                        collate_fn=tonic.collation.PadTensors(batch_first=False))

# === Network Architecture ===
spike_grad = surrogate.atan()
beta = 0.5

# Input: [T, N, 2, 128, 128]
# After downsampling in transforms or via adaptive pooling
net = nn.Sequential(
    nn.Conv2d(2, 12, 5),
    nn.MaxPool2d(2),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    nn.Conv2d(12, 32, 5),
    nn.MaxPool2d(2),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    nn.Flatten(),
    nn.Linear(32 * 29 * 29, 11),  # Adjust based on actual spatial dims
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
).to(device)

# === Forward Pass (iterates over timesteps) ===
def forward_pass(net, data):
    spk_rec = []
    utils.reset(net)  # Reset all neuron states

    for step in range(data.size(0)):  # Iterate over T timesteps
        spk_out, mem_out = net(data[step])
        spk_rec.append(spk_out)

    return torch.stack(spk_rec)

# === Training ===
optimizer = torch.optim.Adam(net.parameters(), lr=2e-3, betas=(0.9, 0.999))
loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

num_epochs = 50
for epoch in range(num_epochs):
    net.train()
    train_loss = 0
    train_acc = 0
    train_samples = 0

    for data, targets in trainloader:
        data = data.to(device).float()
        targets = targets.to(device)

        spk_rec = forward_pass(net, data)
        loss_val = loss_fn(spk_rec, targets)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        train_loss += loss_val.item()
        train_acc += SF.accuracy_rate(spk_rec, targets) * targets.size(0)
        train_samples += targets.size(0)

    # Evaluation
    net.eval()
    test_acc = 0
    test_samples = 0
    with torch.no_grad():
        for data, targets in testloader:
            data = data.to(device).float()
            targets = targets.to(device)
            spk_rec = forward_pass(net, data)
            test_acc += SF.accuracy_rate(spk_rec, targets) * targets.size(0)
            test_samples += targets.size(0)

    print(f"Epoch {epoch}: Train Loss={train_loss/len(trainloader):.4f}, "
          f"Train Acc={train_acc/train_samples:.4f}, "
          f"Test Acc={test_acc/test_samples:.4f}")
```

---

## 4. DVS128 Data Pipeline

### 4.1 Dataset Overview

| Property | Value |
|---|---|
| Sensor | DVS128 (Dynamic Vision Sensor) |
| Resolution | 128 x 128 pixels |
| Channels | 2 (ON polarity, OFF polarity) |
| Classes | 11 hand/arm gestures |
| Training samples | 1,176 |
| Test samples | 288 |
| Total subjects | 29 |
| Lighting conditions | 3 (natural, fluorescent, LED) |
| File format | AEDAT 3.1 |

**Original source:** IBM Research - https://research.ibm.com/interactive/dvsgesture/

### 4.2 The 11 Gesture Classes

0. Hand Clapping
1. Right Hand Wave
2. Left Hand Wave
3. Right Arm CW (Clockwise)
