# Technology Stack for SNN DVS128 Gesture Recognition

Researching what i'd actually need to build an SNN-based DVS128 gesture recognition system from scratch. Going through the frameworks, data pipeline, training setup, etc.

The two main SNN frameworks are **SpikingJelly** (Peking University group, published in Science Advances) and **snnTorch** (UC Santa Cruz, Jason Eshraghian). Both sit on top of PyTorch and support surrogate gradient backprop for training deep SNNs.

SpikingJelly has a more complete built-in pipeline for DVS128 Gesture -- its own dataset loader, a pre-built DVSGestureNet model, full training scripts. snnTorch is more modular and tutorial-driven, and integrates with Tonic for neuromorphic data loading. For a thesis, SpikingJelly gets you to a working baseline faster, but snnTorch has better educational resources and more flexible visualization.

The DVS128 Gesture dataset has 1176 training and 288 test samples across 11 gesture classes, recorded with a Dynamic Vision Sensor at 128x128 resolution. Raw data is AEDAT 3.1 format -- polarity events (x, y, timestamp, polarity). Events need to be converted to frame tensors for batch training, and both frameworks handle this.

Training on a single GPU (RTX 2080 Ti, 12GB VRAM) takes about 18-28 seconds per epoch, with 64-256 epochs to reach ~96-97% accuracy. Apple M-series Macs work via PyTorch MPS backend, but you lose the CuPy/Triton acceleration. Energy can be estimated without neuromorphic hardware using the **syops** library (computes SynOps, estimates energy at 45nm technology costs).

---

## SpikingJelly Framework

### Version Requirements

| Dependency | Version |
|---|---|
| Python | >= 3.6 (recommended: 3.9 - 3.11) |
| PyTorch | >= 2.2.0 (tested on 2.7.1) |
| torchvision | Required (version matching PyTorch) |
| torchaudio | Required (version matching PyTorch) |
| numpy | Required |
| scipy | Required |
| tensorboard | Required |
| einops | Required |
| tqdm | Required |
| h5py | Required |
| matplotlib | Required |

Source: https://github.com/fangwei123456/spikingjelly/blob/master/setup.py

### Installation

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

### Neuron Models

SpikingJelly has these in `spikingjelly.activation_based.neuron`:

| Neuron Model | Description |
|---|---|
| `IFNode` | Integrate-and-Fire. Simplest: V += X; fire if V > threshold |
| `LIFNode` | Leaky Integrate-and-Fire. V += (X - (V - V_reset)) / tau. Most commonly used |
| `ParametricLIFNode` | PLIF: learnable membrane time constant tau per layer, from ICCV 2021 |
| `QIFNode` | Quadratic Integrate-and-Fire |
| `EIFNode` | Exponential Integrate-and-Fire |
| `IzhikevichNode` | Izhikevich neuron model |

All neurons support:
- Single-step mode (`'s'`): process one timestep at a time
- Multi-step mode (`'m'`): process all timesteps in parallel (faster)
- Three backends: `torch` (CPU/GPU), `cupy` (NVIDIA GPU only), `triton` (NVIDIA GPU only, fastest)

### Surrogate Gradient Functions

Available in `spikingjelly.activation_based.surrogate`:

- `ATan` - Arctangent surrogate (most commonly used)
- `Sigmoid` - Sigmoid surrogate
- `PiecewiseQuadratic` - Piecewise quadratic
- `PiecewiseExponential` - Piecewise exponential
- `SoftSign` - Soft sign function
- `Erf` - Gaussian error function
- `NonzeroSignLogAbs` - Log-abs surrogate
- `PiecewiseLeakyReLU` - Piecewise leaky ReLU

### DVS128 Gesture Dataset Loader

SpikingJelly has a built-in loader at `spikingjelly.datasets.dvs128_gesture.DVS128Gesture`.

Loading raw events:
```python
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

# Load raw events
event_set = DVS128Gesture(root='./data/DVS128Gesture', train=True, data_type='event')
event, label = event_set[0]
# event is a dict with keys: 't', 'x', 'y', 'p'
# label is an integer 0-10
```

Loading as frames (fixed number):
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

Loading as frames (fixed duration):
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

Handling variable-length sequences:
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

### Pre-built DVSGestureNet Architecture

SpikingJelly has a model specifically for this:

```python
from spikingjelly.activation_based.model import parametric_lif_net
from spikingjelly.activation_based import neuron, surrogate, functional

# Architecture: {Conv128-BN-LIF-MaxPool}x5 -> Dropout -> FC512 -> LIF -> Dropout -> FC11 -> LIF -> AvgPool
net = parametric_lif_net.DVSGestureNet(
    channels=128,
    spiking_neuron=neuron.LIFNode,
    surrogate_function=surrogate.ATan(),
    detach_reset=True
)

# Enable multi-step mode (processes all timesteps in parallel)
functional.set_step_mode(net, 'm')

# Optional: Use CuPy backend for speedup (NVIDIA only)
# functional.set_backend(net, 'cupy', instance=neuron.LIFNode)
```

### Complete Training Loop

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

## snnTorch Framework

### Version Requirements

| Dependency | Version |
|---|---|
| Python | >= 3.9 |
| PyTorch | Any modern version (no explicit pin; tested with 1.x and 2.x) |
| numpy | Required |
| pandas | Required |
| matplotlib | Optional (for visualization) |
| nir / nirtorch | Optional (for NIR exchange format) |

Source: https://github.com/jeshraghian/snntorch/blob/master/setup.py

### Installation

```bash
pip install snntorch

# For neuromorphic dataset support with Tonic
pip install tonic

# Full install with all optional dependencies
pip install snntorch[full]
```

### Neuron Models

Available in `snntorch`:

| Neuron Model | Description |
|---|---|
| `snn.Leaky` | 1st-order LIF. The main one. |
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

### Complete DVS128 Gesture Training with snnTorch + Tonic

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

## DVS128 Data Pipeline

### Dataset Overview

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

Original source: IBM Research - https://research.ibm.com/interactive/dvsgesture/

### The 11 Gesture Classes

0. Hand Clapping
1. Right Hand Wave
2. Left Hand Wave
3. Right Arm CW (Clockwise)
4. Right Arm CCW (Counter-clockwise)
5. Left Arm CW
6. Left Arm CCW
7. Arm Roll
8. Air Drums
9. Air Guitar
10. Other Gestures

### Raw Data Format (AEDAT 3.1)

Each recording is stored as a binary AEDAT 3.1 file with polarity events.

File structure: [header][events][header][events]...

Header (28 bytes):
```
uint16_t eventType
uint16_t eventSource
uint32_t eventSize
uint32_t eventTSOffset
uint32_t eventTSOverflow
uint32_t eventCapacity
uint32_t eventNumber
uint32_t eventValid
```

Each event (8 bytes):
```
uint32_t data       # Contains x, y, polarity packed
uint32_t timestamp  # Microsecond timestamp
```

Extracting x, y, polarity from data field:
```python
x = (data >> 17) & 0x00001FFF
y = (data >> 2) & 0x00001FFF
polarity = (data >> 1) & 0x00000001
```

### Event-to-Frame Conversion Methods

Events need to be binned into dense frame tensors for batch training. There's basically three strategies:

**Method 1: Fixed number of frames (most common)**
- Divide all events in a sample into N equal-sized groups
- Accumulate events in each group into a frame
- Result: [N, 2, 128, 128] tensor with consistent shape across samples
- SpikingJelly: `split_by='number', frames_number=16`

**Method 2: Fixed time duration**
- Divide events into windows of fixed duration (e.g., 1 second)
- Variable number of frames per sample
- Requires padding for batching
- SpikingJelly: `split_by='time', duration=1000000`

**Method 3: Fixed event count per frame**
- Each frame accumulates exactly K events
- Variable number of frames
- Tonic: `ToFrame(event_count=3000)`

### Preprocessing Steps

1. Download and extract raw AEDAT files
2. Parse AEDAT 3.1 binary format into structured events (t, x, y, p)
3. Segment recordings - each file may contain multiple gesture labels; split by label timestamps
4. Denoise - remove isolated, spurious events (Tonic: `Denoise(filter_time=10000)`)
5. Bin into frames - convert event stream to [T, C, H, W] tensors
6. Optional: spatial transforms - random rotation, random crop, horizontal flip
7. Cache to disk - first-time processing is slow; cache frames for reuse

---

## Tonic Library

### Overview

Tonic is a PyTorch-compatible library for neuromorphic/event-based datasets and transforms. Basically "torchvision for neuromorphic data."

```bash
pip install tonic
```

### Key Features

- Downloads and loads 20+ neuromorphic datasets including DVSGesture
- Event-specific transforms (denoise, spatial jitter, temporal jitter, etc.)
- Frame conversion transforms (ToFrame, ToVoxelGrid, ToTimesurface)
- DiskCachedDataset for fast reloading
- Compatible with PyTorch DataLoader
- Works with both PyTorch and JAX

### DVSGesture Dataset in Tonic

```python
import tonic

# Load dataset (downloads automatically on first use)
dataset = tonic.datasets.DVSGesture(save_to='./data', train=True)

# Access sensor size
sensor_size = tonic.datasets.DVSGesture.sensor_size  # (128, 128, 2)

# Get a sample
events, label = dataset[0]
# events: structured numpy array with fields (x, y, t, p)
# label: integer class index
```

### Available Transforms

```python
import tonic.transforms as transforms

# === Event-level transforms ===
transforms.Denoise(filter_time=10000)               # Remove isolated events
transforms.SpatialJitter(sensor_size, variance_x=2, variance_y=2)
transforms.TimeJitter(std=100)               # Add temporal noise
transforms.RefractoryPeriod(delta=1000)      # Enforce refractory period
transforms.DropEvent(p=0.1)                  # Random event dropout

# === Frame conversion transforms ===
transforms.ToFrame(sensor_size, time_window=10000)    # Fixed time bins
transforms.ToFrame(sensor_size, event_count=5000)     # Fixed event count
transforms.ToFrame(sensor_size, n_time_bins=16)       # Fixed number of bins
transforms.ToVoxelGrid(sensor_size, n_time_bins=16)   # Voxel grid (similar)
transforms.ToTimesurface(sensor_size, tau=50000)       # Time surface

# === Compose transforms ===
transform = transforms.Compose([
    transforms.Denoise(filter_time=10000),
    transforms.ToFrame(sensor_size=sensor_size, n_time_bins=16)
])
```

### Caching for Performance

```python
from tonic import DiskCachedDataset

# First epoch: processes and caches all samples to disk
# Subsequent epochs: loads from cache (much faster)
cached_dataset = DiskCachedDataset(
    dataset,
    transform=torch.from_numpy,  # Post-cache transform (applied each time)
    cache_path='./cache/dvs_gesture/train'
)
```

### Collation for Variable-Length Sequences

```python
# When using time-based binning, sequences have different lengths
# PadTensors pads all sequences in a batch to the same length
loader = DataLoader(
    dataset,
    batch_size=16,
    collate_fn=tonic.collation.PadTensors(batch_first=False)
    # batch_first=False: output shape [T, N, C, H, W]
    # batch_first=True:  output shape [N, T, C, H, W]
)
```

---

## Training Infrastructure

### GPU Requirements

| Configuration | VRAM | Notes |
|---|---|---|
| RTX 2080 Ti (12GB) | 12 GB | T=16 timesteps, batch_size=16. Reference GPU in SpikingJelly benchmarks |
| RTX 3090 (24GB) | 24 GB | Can use T=20 (original paper setting) with larger batches |
| RTX 4090 (24GB) | 24 GB | Faster clock; comfortable for training |
| A100 (40/80GB) | 40-80 GB | Large batch training; multi-run experiments |
| MacBook M1/M2/M3 | 8-36 GB unified | Works via PyTorch MPS backend. Caveats below. |

### Apple M-Series Mac Compatibility

Training does work on Apple Silicon via PyTorch MPS (Metal Performance Shaders).

```python
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
```

Caveats:
- MPS support requires macOS Monterey 12.3+ and PyTorch >= 1.12
- MPS is still marked experimental -- some PyTorch ops aren't implemented yet
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` to auto fall back to CPU for unsupported ops
- SpikingJelly CuPy and Triton backends are not available on Mac (NVIDIA-only). Have to use `torch` backend.
- snnTorch works natively on MPS since it uses standard PyTorch operations
- For small models and datasets like DVS128 (1464 samples), MPS gives meaningful speedup over CPU
- For big experiments, really want an NVIDIA GPU

One nice thing about Apple Silicon: unified memory means no separate GPU VRAM limit -- the GPU shares system RAM. An M3 Pro with 36GB can handle larger batches than a discrete GPU with 12GB VRAM.

### Training Duration Estimates

Based on SpikingJelly benchmarks on RTX 2080 Ti:

| Configuration | Time per Epoch | Notes |
|---|---|---|
| torch backend, T=16, batch=16 | ~28 sec | Standard Python implementation |
| cupy backend, T=16, batch=16 | ~18 sec | CuPy-accelerated (34% faster) |
| 64 epochs total | ~19-30 min | Minimal training run |
| 256 epochs total | ~77-120 min | Full training run to convergence |

First-time dataset processing: converting AEDAT files to frames takes 10-30 minutes on first run (cached after that).

### Memory Requirements

- DVS128 Gesture dataset raw download: ~5.8 GB
- Processed frame data (cached): ~2-4 GB (depends on T and split strategy)
- Model parameters (DVSGestureNet, channels=128): ~11 million parameters
- GPU memory during training: ~4-8 GB (depends on T and batch size)
- Disk space for logs/checkpoints: ~200 MB

---

## Evaluation and Visualization

### Energy Efficiency Estimation (Without Neuromorphic Hardware)

#### The SynOps Metric

Synaptic Operations (SynOps) count accumulate (AC) and multiply-accumulate (MAC) operations during inference:

- ANN computation: uses MAC operations (multiply + add). Energy: ~4.6 pJ per MAC (45nm technology)
- SNN computation: uses AC operations (add only, because binary spikes). Energy: ~0.9 pJ per AC (45nm technology)
- Energy ratio: E_AC is roughly 32x lower than E_MAC

#### Using the syops Library

```bash
pip install syops
```

```python
import torch
from spikingjelly.activation_based import surrogate, neuron, functional
from spikingjelly.activation_based.model import spiking_resnet
from syops import get_model_complexity_info

net = parametric_lif_net.DVSGestureNet(
    channels=128,
    spiking_neuron=neuron.IFNode,
    surrogate_function=surrogate.ATan(),
    detach_reset=True
)

# Compute SynOps with a data sample
ops, params = get_model_complexity_info(
    net,
    (2, 128, 128),         # Input shape (C, H, W)
    dataloader,             # Provide actual data for spike rate estimation
    as_strings=True,
    print_per_layer_stat=True,
    verbose=True
)
# Outputs: ACs, MACs, parameter count, and estimated energy consumption
```

#### Manual SynOps Calculation

For a convolutional layer with N input channels, M output channels, input size I x I, kernel size k x k, output size O x O, and Spiking Activity (SA) = average fraction of neurons that fire:

```
SynOps_SNN = SA * N * M * k^2 * O^2  (per timestep, per layer)
FLOPs_ANN  = N * M * k^2 * O^2       (per layer)
```

Total energy estimation:
```
Energy_SNN = sum(SynOps_layer * E_AC) + sum(non_spike_ops * E_MAC)
Energy_ANN = sum(FLOPs_layer * E_MAC)
```

#### Adding SynOps Loss for Energy Optimization

```python
# Add a regularization term to push model toward fewer spikes
synops_loss_weight = 1e-3
total_loss = classification_loss + synops_loss_weight * synops_count
```

### Spike Raster Plots (snnTorch)

```python
import snntorch.spikeplot as splt
import matplotlib.pyplot as plt

# spk_data shape: [num_steps, num_neurons]
fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
splt.raster(spk_data, ax, s=1.5, c="black")
plt.title("Spike Raster Plot")
plt.xlabel("Time Step")
plt.ylabel("Neuron Index")
plt.show()
```

### Membrane Potential Traces (snnTorch)

```python
# mem_data shape: [num_steps, num_neurons]
# spk_data shape: [num_steps, num_neurons] (optional spike overlay)
splt.traces(mem_data, spk=spk_data, dim=(3, 3), spk_height=5)
plt.show()
```

### Spike Count Animation (snnTorch)

```python
# spk_rec shape: [num_steps, num_classes]
labels = ['Clap', 'RWave', 'LWave', 'RCW', 'RCCW',
          'LCW', 'LCCW', 'Roll', 'Drums', 'Guitar', 'Other']

fig, ax = plt.subplots(facecolor='w', figsize=(12, 7))
anim = splt.spike_count(spk_rec[:, sample_idx].detach().cpu(),
                        fig, ax, labels=labels,
                        animate=True, interpolate=1)
# Save as GIF
from IPython.display import HTML
HTML(anim.to_html5_video())
```

### Input Event Visualization (snnTorch)

```python
# Animate DVS frames
frame_data = train_set[0][0]  # [T, 2, 128, 128]
summed = frame_data[:, 0] + frame_data[:, 1]  # Sum on/off channels

fig, ax = plt.subplots()
anim = splt.animator(summed, fig, ax, interval=40, cmap='plasma')
anim.save("dvs_gesture_input.gif", writer='ffmpeg', fps=25)
```

### Visualization with Tonic

```python
import tonic

# Plot events as a grid of frames
dataset = tonic.datasets.DVSGesture(save_to='./data', train=True)
events, label = dataset[0]
tonic.utils.plot_event_grid(events)
```

### TensorBoard Integration (SpikingJelly)

SpikingJelly's training script logs to TensorBoard:
```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(out_dir)
writer.add_scalar('train_loss', train_loss, epoch)
writer.add_scalar('train_acc', train_acc, epoch)
writer.add_scalar('test_loss', test_loss, epoch)
writer.add_scalar('test_acc', test_acc, epoch)
```

View with:
```bash
tensorboard --logdir=./logs
```

---

## End-to-End Examples

### SpikingJelly Built-in Example (fastest way to a baseline)

```bash
# Download DVS128 Gesture dataset first (manual download from IBM)
# Place in ./data/DVS128Gesture/

# Run the built-in training script
python -m spikingjelly.activation_based.examples.classify_dvsg \
    -T 16 \
    -device cuda:0 \
    -b 16 \
    -epochs 64 \
    -data-dir ./data/DVS128Gesture \
    -out-dir ./logs \
    -amp \
    -opt adam \
    -lr 0.001 \
    -channels 128
```

Expected results:
- Epoch 0: ~40% train accuracy, ~62% test accuracy
- After 64 epochs: ~93-95% test accuracy
- After 256 epochs: ~96% peak test accuracy
- Training speed: ~28 sec/epoch (torch backend) or ~18 sec/epoch (cupy backend)

Source code: `spikingjelly/activation_based/examples/classify_dvsg.py` in the SpikingJelly repo.

### Full Pipeline: Data Download to Trained Model

**Step 1: Environment Setup**
```bash
conda create -n snn python=3.10
conda activate snn
pip install torch torchvision torchaudio  # From pytorch.org
pip install spikingjelly
pip install tonic
pip install syops
pip install matplotlib tensorboard
```

**Step 2: Dataset Download**
- Visit https://research.ibm.com/interactive/dvsgesture/ (requires IBM account)
- Alternative: Use Tonic which downloads automatically:
```python
import tonic
dataset = tonic.datasets.DVSGesture(save_to='./data', train=True)
# Downloads preprocessed version (~5.8 GB) automatically
```

**Step 3: First-Time Processing**
```python
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

# First call processes AEDAT files into numpy frames (takes 10-30 min)
# Results are cached to disk for subsequent loads
train_set = DVS128Gesture(
    root='./data/DVS128Gesture',
    train=True,
    data_type='frame',
    frames_number=16,
    split_by='number'
)
print(f"Training samples: {len(train_set)}")  # 1176
print(f"Sample shape: {train_set[0][0].shape}")  # [16, 2, 128, 128]
```

**Step 4: Train** (see complete training loop in SpikingJelly section above)

**Step 5: Evaluate and Compute Energy Metrics** (see energy estimation section above)

### Third-Party Implementation

GitHub repo: https://github.com/DerrickL25/SNN_Gesture_Classification
- Uses snnTorch with DVS128 Gesture dataset
- Data dimensions: [num_steps, 2, 128, 128]
- Complete implementation from data loading to evaluation

---

## ANN Comparison and Frame Integration

### Why Compare SNN with ANN?

A thesis on SNN-based gesture recognition should show:
1. SNNs can achieve competitive accuracy to ANNs
2. SNNs are more energy-efficient (fewer operations per inference)
3. The trade-offs between accuracy, latency, and energy

### Frame Integration for ANN Baseline

To train a standard CNN on DVS128 data, first need to convert events to frames:

```python
import tonic
import tonic.transforms as transforms
import torch
from torch.utils.data import DataLoader

sensor_size = tonic.datasets.DVSGesture.sensor_size

# === Method 1: Accumulate all events into a single frame ===
# Simple but loses all temporal information
transform_single = transforms.Compose([
    transforms.Denoise(filter_time=10000),
    transforms.ToFrame(sensor_size=sensor_size, n_time_bins=1)  # Single frame
])

# === Method 2: Create fixed number of frames and stack as channels ===
# Preserves some temporal info; treat T frames as T*2 channels
transform_multi = transforms.Compose([
    transforms.Denoise(filter_time=10000),
    transforms.ToFrame(sensor_size=sensor_size, n_time_bins=10)
])

# Load dataset
trainset = tonic.datasets.DVSGesture(save_to='./data', transform=transform_multi, train=True)
testset = tonic.datasets.DVSGesture(save_to='./data', transform=transform_multi, train=False)
```

### ANN Baseline with Standard CNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DVSGestureCNN(nn.Module):
    """Standard CNN baseline for DVS128 Gesture classification.
    Input: frames integrated from events [N, T*2, 128, 128]
    where T is number of time bins and 2 is on/off polarity channels.
    """
    def __init__(self, num_time_bins=10, num_classes=11):
        super().__init__()
        in_channels = num_time_bins * 2  # T frames * 2 polarities

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),  # 4x4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x shape: [N, T, 2, 128, 128]
        N, T, C, H, W = x.shape
        x = x.reshape(N, T * C, H, W)  # Stack time bins as channels
        x = self.features(x)
        x = self.classifier(x)
        return x

# Training the ANN baseline
model = DVSGestureCNN(num_time_bins=10, num_classes=11).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(50):
    model.train()
    for frames, labels in trainloader:
        frames = frames.to(device).float()  # [T, N, 2, 128, 128]
        frames = frames.permute(1, 0, 2, 3, 4)  # [N, T, 2, 128, 128]
        labels = labels.to(device)

        outputs = model(frames)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Fair Comparison Metrics

| Metric | ANN | SNN |
|---|---|---|
| Accuracy (%) | Report test accuracy | Report test accuracy |
| Parameters | Count all weights | Count all weights (same) |
| FLOPs / SynOps | Compute FLOPs (all MACs) | Compute SynOps (mostly ACs) |
| Energy (estimated) | FLOPs * E_MAC (4.6 pJ) | ACs * E_AC (0.9 pJ) + MACs * E_MAC (4.6 pJ) |
| Latency | Single forward pass | T timesteps of forward pass |

---

## State-of-the-Art Benchmarks

### DVS128 Gesture Recognition Leaderboard (as of early 2026)

| Model | Type | Accuracy (%) | Year |
|---|---|---|---|
| TENNs-PLEIADES | SNN | Current SOTA | 2024 |
| STREAM | SNN | 100.0 | 2024 |
| EventMix (SNN-based) | SNN | 99.3 | 2024 |
| SpikePoint | SNN | 98.74 | 2023 |
| TA-SNN (Temporal Attention) | SNN | 98.6 | 2021 |
| TCN (2-stage) | ANN | 97.7 | -- |
| PLIF (SpikingJelly DVSGestureNet) | SNN | 97.6 | 2021 |
| SpikingJelly baseline (LIF, T=16) | SNN | 96.18 | Reference |
| MS-ResNet | SNN | 94.44 | 2023 |
| SCRNN (Spiking Conv RNN) | SNN | 96.59 (10-class) | 2020 |
| IBM TrueNorth (original paper) | SNN | 96.5 | 2017 |

Source: https://paperswithcode.com/sota/gesture-recognition-on-dvs128-gesture

---

## Recommended Stack Configuration

### For a Thesis Project

```
Python:          3.10 or 3.11
PyTorch:         2.2+ (latest stable recommended)
Primary SNN Lib: SpikingJelly (spikingjelly >= 0.0.0.0.15)
Dataset Loader:  SpikingJelly built-in (DVS128Gesture) OR Tonic
Transforms:      Tonic (for advanced transforms) or SpikingJelly built-in
Visualization:   snnTorch spikeplot + matplotlib + TensorBoard
Energy Metrics:  syops library
ANN Baseline:    Standard PyTorch CNN
```

### Minimal Installation

```bash
# Create environment
conda create -n snn python=3.10
conda activate snn

# Core (choose platform-specific PyTorch from pytorch.org)
pip install torch torchvision torchaudio

# SNN frameworks
pip install spikingjelly
pip install snntorch

# Data loading
pip install tonic

# Energy metrics
pip install syops

# Visualization and logging
pip install matplotlib tensorboard

# Optional
pip install einops tqdm h5py scipy
```

### Project Directory Structure

```
thesisproject/
|-- data/
|   |-- DVS128Gesture/         # Raw dataset
|   |-- cache/                  # Cached frame data
|-- src/
|   |-- models/
|   |   |-- snn_model.py       # SNN architecture (SpikingJelly)
|   |   |-- cnn_baseline.py    # ANN baseline
|   |-- data/
|   |   |-- dataset.py         # Dataset loading and transforms
|   |-- train_snn.py           # SNN training script
|   |-- train_cnn.py           # ANN baseline training
|   |-- evaluate.py            # Evaluation metrics + energy estimation
|   |-- visualize.py           # Spike raster, membrane potential plots
|-- logs/                       # TensorBoard logs and checkpoints
|-- notebooks/                  # Jupyter notebooks for analysis
|-- requirements.txt
|-- README.md
```

---

## Sources

### Framework Documentation
- SpikingJelly GitHub: https://github.com/fangwei123456/spikingjelly
- SpikingJelly PyPI: https://pypi.org/project/spikingjelly/
- SpikingJelly Documentation: https://spikingjelly.readthedocs.io/
- snnTorch GitHub: https://github.com/jeshraghian/snntorch
- snnTorch Documentation: https://snntorch.readthedocs.io/
- snnTorch Tutorial 7 (Neuromorphic Datasets): https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_7.html
- Tonic GitHub: https://github.com/neuromorphs/tonic
- Tonic Documentation: https://tonic.readthedocs.io/
- Tonic DVSGesture: https://tonic.readthedocs.io/en/latest/generated/tonic.datasets.DVSGesture.html

### Dataset
- DVS128 Gesture (IBM Research): https://research.ibm.com/interactive/dvsgesture/
- DVS128 Gesture (Papers with Code): https://paperswithcode.com/dataset/dvs128-gesture-dataset
- DVS128 Gesture PyTorch loader: https://github.com/wponghiran/dvs128_gesture_pytorch

### Academic Papers
- SpikingJelly paper (Science Advances): https://www.science.org/doi/10.1126/sciadv.adi1480
- PLIF paper (ICCV 2021): https://github.com/fangwei123456/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron
- Original DVS Gesture paper (CVPR 2017): https://openaccess.thecvf.com/content_cvpr_2017/papers/Amir_A_Low_Power_CVPR_2017_paper.pdf
- Energy Efficiency of SNNs: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2020.00662/full
- Analytical Energy Estimation: https://arxiv.org/pdf/2210.13107
- Are SNNs Really More Energy-Efficient?: https://cea.hal.science/cea-03852141/file/Are_SNNs_Really_More_Energy_Efficient_Than_ANNs__An_In_Depth_Hardware_Aware_Study_versionacceptee.pdf

### Tools
- syops (SynOps counter): https://github.com/iCGY96/syops-counter
- syops PyPI: https://pypi.org/project/syops/
- snnTorch spikeplot: https://snntorch.readthedocs.io/en/latest/snntorch.spikeplot.html

### Tutorials and Examples
- SpikingJelly DVS Gesture Tutorial: https://github.com/fangwei123456/spikingjelly/blob/master/docs/source/activation_based_en/classify_dvsg.rst
- snnTorch Tutorial 7 Colab: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_7_neuromorphic_datasets.ipynb
- DerrickL25 SNN Gesture Classification: https://github.com/DerrickL25/SNN_Gesture_Classification
- Open Neuromorphic SpikingJelly: https://open-neuromorphic.org/neuromorphic-computing/software/snn-frameworks/spikingjelly/

### Hardware
- PyTorch MPS (Apple Silicon): https://developer.apple.com/metal/pytorch/
- PyTorch MPS blog post: https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/
- SNN Framework Benchmarks: https://open-neuromorphic.org/blog/spiking-neural-network-framework-benchmarking/

---

## Things i couldn't fully verify

1. SpikingJelly neuron.py full class list -- couldn't directly access the source to enumerate every neuron class. QIFNode, EIFNode, and IzhikevichNode are mentioned in docs but i couldn't confirm they're actually in the latest version.

2. snnTorch PyTorch version pinning -- snnTorch doesn't explicitly pin a PyTorch version in setup.py. It previously required `torch>=1.1.0` but that got commented out. In practice, just use PyTorch 2.x.

3. DVS128 Gesture download -- the original IBM source may require an account. Tonic provides an automated download of a preprocessed version. The SpikingJelly loader expects the original AEDAT format.

4. MPS operator coverage -- not all PyTorch operators work in Apple MPS. Some SpikingJelly operations may silently fall back to CPU. Need to monitor performance on Mac.

5. Training duration on Mac -- no published benchmarks for DVS128 Gesture training on Apple Silicon. Guessing roughly 2-4x slower than an RTX 2080 Ti based on general MPS benchmarks.

6. Energy estimation accuracy -- SynOps-based energy estimation (45nm technology) is just a proxy. Real energy on actual neuromorphic hardware (Loihi, TrueNorth, etc.) will differ. But the 45nm estimates are standard in the literature for fair comparison, so it's fine.
