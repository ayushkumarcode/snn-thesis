# SNN for Environmental Sound Classification on ESC-50: Component Verification

checked on 2026-02-28, updated 2026-03-03. running on macOS Darwin 24.5.0 (Apple Silicon). went through every pipeline component to make sure this project is actually doable.

---

## the short version

**all 12 components verified -- the whole thing is feasible.** the entire pipeline from ESC-50 data loading through mel-spectrogram extraction, spike encoding, convolutional SNN training, ANN baseline comparison, and energy measurement works using freely available, open-source tools. three components have risk flags: (1) macOS MPS compatibility with snnTorch, (2) no ready-made audio SNN tutorial exists, and (3) open bugs in snnTorch's neuron reset logic and memory management. none are hard blockers, but item 3 is worth being careful about.

the most interesting find: a March 2025 arXiv paper directly addresses SNN-based environmental sound classification on ESC-10 using snnTorch -- confirming this is a viable and currently active research direction.

### CRITICAL BUGS DISCOVERED (2026-03-03 update)

1. **Reset mechanism bug (Issue #401, OPEN):** The `reset_mechanism` parameter in `snn.Leaky()` may have inverted behavior -- "zero" and "subtract" modes may be switched or produce incorrect outputs. A contributor (DoubleGio) identified missing beta coefficients in soft reset and wrong operation ordering in hard reset when `reset_delay=True`. **Mitigation:** Use default reset_mechanism="subtract" (the most common choice) and validate neuron dynamics manually before relying on results.

2. **GPU memory leak (Issue #328, OPEN):** `SpikingNeuron.instances` class variable grows without being cleared, causing cumulative memory retention. **Mitigation:** Not relevant for this project since ESC-50 is small enough for CPU training. If using GPU, restart kernel between experiments.

3. **CUDA OOM after many epochs (Issue #394, OPEN):** Memory fragmentation during BPTT with surrogate gradients. **Mitigation:** Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, reduce batch size, or use TBPTT.

4. **NeuroBench dependency conflicts (Issue #238):** NumPy version conflicts between NeuroBench and other packages. **Mitigation:** Install NeuroBench in a separate venv or install packages in specific order.

---

## COMPONENT 1: ESC-50 DATASET

### EXISTS: YES
### POTENTIAL BLOCKER: NO

**Official Repository:** https://github.com/karolpiczak/ESC-50

| Property | Value |
|----------|-------|
| Direct download URL | https://github.com/karoldvl/ESC-50/archive/master.zip |
| Registration required | NO -- freely downloadable, no account needed |
| File format | WAV (5 seconds, 44.1 kHz, mono) |
| Total size | ~600 MB (WAV zip) or ~200 MB (OGG alternative) |
| Total files | 2,000 audio recordings |
| Classes | 50 semantic classes, 40 examples per class |
| Major categories | 5 (Animals, Natural soundscapes, Human non-speech, Interior/domestic, Exterior/urban) |
| License | Creative Commons Attribution-NonCommercial |
| Paper | Piczak, 2015, ACM Multimedia |
| Also available on | HuggingFace (`ashraq/esc50`), Kaggle |

**File naming convention:** `{FOLD}-{CLIP_ID}-{TAKE}-{TARGET}.wav`
- `{FOLD}`: Cross-validation fold index (1-5)
- `{CLIP_ID}`: Original Freesound clip ID
- `{TAKE}`: Letter disambiguating fragments from same source
- `{TARGET}`: Numeric class label [0-49]

**Metadata file:** `meta/esc50.csv` with columns: `filename, fold, target, category, esc10, src_file, take`

**Verification method:** Fetched and confirmed from official GitHub README. Repository is actively maintained with 3.3k+ stars.

---

## COMPONENT 2: AUDIO LOADING

### EXISTS: YES
### POTENTIAL BLOCKER: NO

Both librosa and torchaudio can load ESC-50 WAV files without issues.

### librosa (recommended for preprocessing)

| Property | Value |
|----------|-------|
| Current version | 0.11.0 |
| Install | `pip install librosa` |
| Python support | 3.8+ |
| macOS Apple Silicon | YES (requires numba/llvmlite, both work on ARM64) |
| WAV support | Native, no extra codecs needed |
| Documentation | https://librosa.org/doc/main/generated/librosa.load.html |

```python
import librosa

# Load ESC-50 clip preserving original sample rate
audio, sr = librosa.load('1-100032-A-0.wav', sr=44100, mono=True)
# audio shape: (220500,)  -- 5 seconds * 44100 Hz
# sr: 44100

# WARNING: librosa defaults to sr=22050 if you omit sr parameter
# Always specify sr=44100 or sr=None to preserve original rate
```

### torchaudio (recommended for GPU pipeline integration)

| Property | Value |
|----------|-------|
| Current version | 2.10.0 (matches PyTorch version) |
| Install | `pip install torchaudio` |
| macOS Apple Silicon | YES (arm64 wheels available) |
| WAV support | Native via FFmpeg or sox backends |
| Documentation | https://docs.pytorch.org/audio/stable/generated/torchaudio.load.html |

```python
import torchaudio

waveform, sample_rate = torchaudio.load('1-100032-A-0.wav')
# waveform shape: torch.Size([1, 220500])  -- [channels, samples]
# sample_rate: 44100
```

**NOTE:** torchaudio entered maintenance phase as of v2.8 (Aug 2025). Deprecation warnings appeared in v2.8, APIs were removed in v2.9, and v2.10 (Jan 2026) completed the transition. However, the following APIs we need are **explicitly preserved and still work:**
- `torchaudio.load()` / `torchaudio.save()` -- now backed by TorchCodec internally
- `torchaudio.transforms.MelSpectrogram` -- preserved in the transforms module
- All transforms, functional, and compliance.kaldi APIs remain
- See GitHub issue pytorch/audio#3902 for the full migration plan

**For this project, all needed functionality remains available.** If you encounter deprecation warnings with torchaudio >= 2.8, they can be safely ignored for load() and MelSpectrogram().

**Verification method:** Confirmed via official documentation for both libraries. Both have native WAV support and macOS ARM64 builds.

---

## COMPONENT 3: MEL-SPECTROGRAM CONVERSION

### EXISTS: YES
### POTENTIAL BLOCKER: NO

### librosa approach

```python
import librosa
import numpy as np

# Standard ESC-50 parameters (from literature)
audio, sr = librosa.load('audio.wav', sr=44100, mono=True)

S = librosa.feature.melspectrogram(
    y=audio,
    sr=44100,
    n_fft=1024,       # FFT window size
    hop_length=256,    # Step between windows
    n_mels=128,        # Number of mel bands
    fmin=20,           # Min frequency
    fmax=20000         # Max frequency (Nyquist for 44.1kHz)
)

# Convert to log scale (decibels)
S_dB = librosa.power_to_db(S, ref=np.max)
# Shape: (128, ~862) -- 128 mel bands x time frames
```

Documentation: https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html

### torchaudio approach (GPU-compatible)

```python
import torchaudio
import torchaudio.transforms as T

mel_transform = T.MelSpectrogram(
    sample_rate=44100,
    n_fft=1024,
    hop_length=256,
    n_mels=128,
    f_min=20,
    f_max=20000
)

waveform, sr = torchaudio.load('audio.wav')
mel_spec = mel_transform(waveform)
# Shape: torch.Size([1, 128, time_frames])

# Convert to log scale
log_mel = T.AmplitudeToDB()(mel_spec)
```

Documentation: https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.MelSpectrogram.html

### Parameters from recent SNN+ESC literature (arXiv:2503.11206)

The March 2025 paper "Spike Encoding for Environmental Sound: A Comparative Benchmark" used exactly these parameters for their SNN pipeline on ESC-10:
- **n_mels:** 128
- **n_fft:** 1024
- **hop_length:** 256
- **frequency range:** 20-20,000 Hz
- **sample_rate:** 44,100 Hz (original, not downsampled)

**Verification method:** Confirmed via librosa 0.11.0 docs, torchaudio 2.10.0 docs, and validated against parameters used in the only existing SNN+environmental-sound paper.

---

## COMPONENT 4: SPIKE ENCODING IN snnTorch

### EXISTS: YES
### POTENTIAL BLOCKER: NO

**Module:** `snntorch.spikegen`
**Documentation:** https://snntorch.readthedocs.io/en/latest/snntorch.spikegen.html

### Available encoding methods:

| Function | Type | Description |
|----------|------|-------------|
| `spikegen.rate(data, num_steps)` | Rate coding | Poisson spike trains; features = mean of binomial distribution |
| `spikegen.rate_conv(data)` | Rate coding | Simplified version, input clipped to [0,1] |
| `spikegen.latency(data, num_steps, tau)` | Temporal coding | Time-to-first-spike using LIF model; high values fire early |
| `spikegen.latency_code(data, num_steps, tau)` | Temporal coding | Returns spike times without temporal expansion |
| `spikegen.delta(data, threshold)` | Event-driven | Spikes when consecutive timestep difference exceeds threshold |

### Code for converting mel-spectrogram to spike train:

```python
import snntorch.spikegen as spikegen
import torch

# Assume mel_spec shape: [batch, 1, 128, time_frames] normalized to [0, 1]
mel_spec_normalized = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min())

# Rate encoding: repeat spectrogram across num_steps, generate Poisson spikes
num_steps = 25
spike_data = spikegen.rate(mel_spec_normalized, num_steps=num_steps)
# Output shape: [num_steps, batch, 1, 128, time_frames]

# Latency encoding: high-intensity pixels fire earlier
spike_data_latency = spikegen.latency(
    mel_spec_normalized,
    num_steps=num_steps,
    tau=5.0,
    threshold=0.01,
    normalize=True
)
```

**Key insight:** `spikegen.rate()` works on tensors of arbitrary shape. A 2D mel-spectrogram (128 x T) is treated identically to a 2D image (28 x 28 for MNIST). The function simply treats each value as a firing probability. No special audio handling is needed.

**Verification method:** Confirmed every function signature from official snnTorch 0.9.4 API documentation. The `rate()` function operates on generic tensors, confirmed via Tutorial 1 and Tutorial 5.

---

## COMPONENT 5: CONVOLUTIONAL SNN IN snnTorch

### EXISTS: YES
### POTENTIAL BLOCKER: NO

### Available neuron layers:

| Layer | Class | Description |
|-------|-------|-------------|
| LIF (1st order) | `snn.Leaky(beta=0.9)` | Leaky integrate-and-fire, single decay constant |
| Synaptic (2nd order) | `snn.Synaptic(alpha, beta)` | Includes synaptic current dynamics |
| Lapicque | `snn.Lapicque(beta)` | RC neuron model |
| Alpha | `snn.Alpha(alpha, beta)` | Alpha membrane model |
| Recurrent LIF | `snn.RLeaky(beta, V)` | LIF with recurrent output connections |
| Recurrent Synaptic | `snn.RSynaptic(alpha, beta, V)` | Synaptic with recurrent connections |
| Parallel LIF | `snn.LeakyParallel(beta)` | Optimized for parallel data processing |
| Spiking LSTM | `snn.SLSTM` | Spiking long short-term memory |
| Spiking Conv2d LSTM | `snn.SConv2dLSTM` | Spiking convolutional LSTM |

Documentation: https://snntorch.readthedocs.io/en/latest/snntorch.html

### Convolutional SNN architecture for spectrogram input (adapted from Tutorial 6):

```python
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

beta = 0.9
spike_grad = surrogate.fast_sigmoid()

class SpectrogramCSNN(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        # Input: [batch, 1, 128, T] mel-spectrogram
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.pool2 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(32 * 32 * (T//4), 256)  # Adjust for spectrogram dims
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc2 = nn.Linear(256, num_classes)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad, output=True)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        spk_rec = []
        mem_rec = []

        for step in range(num_steps):
            cur1 = self.pool1(self.conv1(x[step]))
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.pool2(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc1(spk2.flatten(1))
            spk3, mem3 = self.lif3(cur3, mem3)

            cur4 = self.fc2(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)

            spk_rec.append(spk4)
            mem_rec.append(mem4)

        return torch.stack(spk_rec), torch.stack(mem_rec)
```

**Tutorial reference:** Tutorial 6 -- Surrogate Gradient Descent in a Convolutional SNN
(https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html)

**Key point:** The architecture above is a straightforward adaptation of Tutorial 6's MNIST CSNN. The only changes are: (a) input dimensions match mel-spectrogram shape instead of 28x28, (b) output classes = 50 instead of 10, (c) potentially more filters since spectrograms are higher resolution than MNIST.

**Verification method:** Tutorial 6 code confirmed from official documentation. Architecture adaptation is mechanical -- Conv2d + snn.Leaky pattern is identical regardless of input modality.

---

## COMPONENT 6: ANN BASELINE

### EXISTS: YES
### POTENTIAL BLOCKER: NO

### Approach 1: Direct swap (simplest)

The snnTorch tutorial explicitly demonstrates this approach. Replace `snn.Leaky` layers with `nn.ReLU()` and remove the temporal loop:

```python
class SpectrogramCNN_ANN(nn.Module):
    """ANN equivalent of the SNN for direct comparison"""
    def __init__(self, num_classes=50):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(32 * 32 * (T//4), 256)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.fc1(x.flatten(1)))
        return self.fc2(x)
```

**Key differences in ANN baseline:**
1. No temporal loop (single forward pass per sample)
2. `nn.ReLU()` replaces `snn.Leaky()`
3. Input is the mel-spectrogram directly (no spike encoding)
4. Standard `nn.CrossEntropyLoss()` instead of spike count loss

**From snnTorch Tutorial 5:** "You can train and test the same network architecture on the same data, except now using ReLU as the activation function instead of LIF."

### Approach 2: More sophisticated ANN baseline

For stronger comparison, use a published ESC-50 CNN architecture:
- Piczak's original CNN baseline: 64.5% accuracy
- With data augmentation: ~84.9% (EnvNet-based)
- With feature fusion (MFCC+Mel+Tonnetz): ~85.6%

### Expected accuracy range for from-scratch CNN on ESC-50:

| Architecture | Accuracy |
|-------------|----------|
| Simple CNN (Piczak 2015) | 64.5% |
| Optimized CNN from scratch | 66-75% |
| CNN with data augmentation | 80-85% |
| CNN with feature stacking | 85-92% |
| Pretrained (AST, BEATs, HTSAT) | 95-98% |
| Human baseline | 81.3% |

**Verification method:** ANN swap approach confirmed from snnTorch Tutorial 5 documentation. Baseline accuracies confirmed from ESC-50 GitHub README leaderboard.

---

## COMPONENT 7: ENERGY MEASUREMENT / SYNOPS

### EXISTS: YES (with caveats)
### POTENTIAL BLOCKER: NO (but requires extra integration work)

### Option A: NeuroBench framework (recommended)

| Property | Value |
|----------|-------|
| Package | `neurobench` |
| Version | 2.2.0 |
| Install | `pip install neurobench` |
| Python | >=3.10, <4.0 |
| snnTorch integration | YES -- `SNNTorchModel` wrapper provided |
| Documentation | https://neurobench.readthedocs.io/en/latest/ |

NeuroBench provides three SynOps metrics:
- **Dense**: Total synops counting all connections (zero and nonzero)
- **Effective_MACs**: Non-zero multiply-accumulate operations (non-binary activations)
- **Effective_ACs**: Non-zero accumulate operations (binary/spike activations) -- **this is the one relevant for SNNs**

```python
from neurobench.models import SNNTorchModel
from neurobench.benchmarks import Benchmark
from neurobench.metrics.workload import (
    ClassificationAccuracy, ActivationSparsity, SynapticOperations
)
from neurobench.metrics.static import Footprint, ConnectionSparsity

model = SNNTorchModel(net)
static_metrics = [Footprint, ConnectionSparsity]
workload_metrics = [ClassificationAccuracy, ActivationSparsity, SynapticOperations]

benchmark = Benchmark(model, test_loader, preprocessors, postprocessors,
                      [static_metrics, workload_metrics])
results = benchmark.run()
# results['SynapticOperations']['Effective_ACs'] = actual energy proxy
```

**NeuroBench tutorial confirms working snnTorch integration** using Google Speech Commands task with an SNN achieving 85.6% accuracy and measuring SynapticOperations automatically.

### Option B: Manual spike counting (simpler, always works)

```python
def count_synops(model, spike_recordings):
    """Manual synaptic operations counter.
    SynOps = sum of spikes per layer * fanout of that layer"""
    total_synops = 0
    for layer_name, spk_tensor in spike_recordings.items():
        num_spikes = spk_tensor.sum().item()
        # fanout = number of output connections per neuron
        if isinstance(model.layers[layer_name], nn.Linear):
            fanout = model.layers[layer_name].out_features
        elif isinstance(model.layers[layer_name], nn.Conv2d):
            fanout = (model.layers[layer_name].out_channels *
                     model.layers[layer_name].kernel_size[0] *
                     model.layers[layer_name].kernel_size[1])
        total_synops += num_spikes * fanout
    return total_synops
```

### Option C: Simple spike count ratio (minimal viable metric)

```python
# During forward pass, record total spikes per layer
total_spikes_layer1 = spk1_rec.sum().item()
total_spikes_layer2 = spk2_rec.sum().item()
total_possible_spikes = num_steps * batch_size * num_neurons
spike_sparsity = 1.0 - (total_spikes / total_possible_spikes)
# Higher sparsity = lower energy on neuromorphic hardware
```

**Verification method:** NeuroBench 2.2.0 confirmed on PyPI with SNNTorchModel wrapper. Tutorial code verified from NeuroBench docs. Manual counting approach validated against NeuroBench's mathematical formulation.

---

## COMPONENT 8: 5-FOLD CROSS-VALIDATION

### EXISTS: YES
### POTENTIAL BLOCKER: NO

ESC-50 has **built-in, predefined** 5-fold cross-validation. This is not something you need to create -- it is part of the dataset design.

### Structure:

| Fold | Clips | Usage |
|------|-------|-------|
| 1 | 400 clips (8 per class) | Train or test |
| 2 | 400 clips (8 per class) | Train or test |
| 3 | 400 clips (8 per class) | Train or test |
| 4 | 400 clips (8 per class) | Train or test |
| 5 | 400 clips (8 per class) | Train or test |

**Protocol:** For each fold k=1..5, train on folds != k, test on fold k. Report mean accuracy across all 5 runs.

**Important constraint:** Fragments from the same original Freesound source file are always in the same fold, preventing data leakage.

### Implementation:

```python
import pandas as pd
from torch.utils.data import Subset

metadata = pd.read_csv('ESC-50-master/meta/esc50.csv')

for test_fold in range(1, 6):
    train_indices = metadata[metadata['fold'] != test_fold].index.tolist()
    test_indices = metadata[metadata['fold'] == test_fold].index.tolist()

    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)
    # Train and evaluate...
```

**Verification method:** Confirmed from ESC-50 README and original paper (Piczak 2015). Fold assignment is encoded in both the filename (first character) and the metadata CSV (`fold` column).

---

## COMPONENT 9: GPU / macOS REQUIREMENTS

### EXISTS: WORKABLE
### POTENTIAL BLOCKER: LOW RISK -- see details

### Dataset size analysis:

| Property | Value |
|----------|-------|
| Total samples | 2,000 |
| Training set (4 folds) | 1,600 |
| Test set (1 fold) | 400 |
| Mel-spectrogram per sample | ~128 x 862 floats = ~443 KB |
| Full dataset in memory (spectrograms) | ~886 MB |
| Batch of 32 spectrograms | ~14 MB |

This is a **small dataset** by deep learning standards. Training is feasible on CPU alone.

### macOS MPS (Apple Silicon GPU) status:

| Item | Status |
|------|--------|
| PyTorch MPS backend | Stable in PyTorch 2.5+ |
| snnTorch MPS support | Partially supported (PR #247 merged Oct 2023, PR #415 in progress Feb 2026) |
| Known issue | Backward pass errors reported on MPS with some snnTorch operations |
| Workaround | Use `PYTORCH_ENABLE_MPS_FALLBACK=1` environment variable |
| Alternative | Train on CPU (feasible for this dataset size) |

### CPU-only training feasibility:

**snnTorch documentation explicitly states:** "The lean requirements of snnTorch enable small and large networks to be viably trained on CPU, where needed."

With 2,000 samples, a convolutional SNN with 25 time steps, and 100 epochs:
- Estimated training time on CPU: 30-90 minutes per fold (rough estimate)
- Total for 5-fold CV: 2.5-7.5 hours
- This is entirely manageable for a thesis project

### Recommended approach for macOS:

```python
# Try MPS first, fall back to CPU
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

if torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using Apple Silicon GPU (MPS)")
else:
    device = torch.device('cpu')
    print("Using CPU")
```

### Google Colab as backup:

If local training is too slow, Google Colab provides free GPU access (T4, ~16GB VRAM). The entire ESC-50 pipeline would run comfortably on Colab.

**Verification method:** MPS status confirmed from snnTorch GitHub issues (#247, #415, #200, #160). CPU feasibility confirmed from official snnTorch documentation. Dataset size analysis computed from ESC-50 specifications.

---

## COMPONENT 10: snnTorch + PyTorch COMPATIBILITY

### EXISTS: YES
### POTENTIAL BLOCKER: LOW RISK

### Current versions:

| Package | Version | Release Date | Python |
|---------|---------|-------------|--------|
| snnTorch | 0.9.4 | Feb 16, 2025 | >=3.9 (3.9, 3.10, 3.11) |
| PyTorch | 2.10.0 | Jan 21, 2026 | >=3.9 |
| torchaudio | 2.10.0 | Jan 21, 2026 | >=3.9 |
| NeuroBench | 2.2.0 | Dec 30, 2025 | >=3.10, <4.0 |
| librosa | 0.11.0 | 2024 | >=3.8 |

### Compatibility matrix:

| Combination | Status |
|-------------|--------|
| snnTorch 0.9.4 + PyTorch 2.5.x | Expected to work (snnTorch depends on torch without version pin) |
| snnTorch 0.9.4 + PyTorch 2.6+ | Likely works but less tested; use 2.5 if issues arise |
| NeuroBench 2.2.0 + snnTorch 0.9.4 | Should work (NeuroBench provides SNNTorchModel wrapper) |

### snnTorch dependencies (from setup.py):

```
install_requires = ['numpy', 'pandas']
extras_require = {
    'full': ['torch', 'matplotlib', 'nir', 'nirtorch']
}
python_requires = '>=3.9'
```

**Key finding:** snnTorch does NOT pin a specific PyTorch version. It requires `torch` but does not specify `torch>=X.Y.Z` or `torch<X.Y.Z`. This means it should work with any reasonably recent PyTorch version, but also means edge-case incompatibilities are possible.

### Recommended safe installation:

```bash
# Create clean environment
python3.11 -m venv snn_env
source snn_env/bin/activate

# Install PyTorch first (for macOS)
pip install torch torchvision torchaudio

# Then snnTorch
pip install snntorch

# Then NeuroBench for energy measurement
pip install neurobench

# Audio processing
pip install librosa
```

### Known deprecation issues:

1. **snnTorch v0.6.0 broke backward compatibility:** `backprop.py` was deprecated, rest function scaling changed. Current v0.9.4 is stable.
2. **torchaudio v2.8+ maintenance mode:** Deprecation warnings in v2.8, removals in v2.9, completed in v2.10. Our needed APIs (load, MelSpectrogram) are explicitly preserved.
3. **Python 3.12+ may have issues:** snnTorch officially supports 3.9-3.11. NeuroBench supports 3.10+. Stick with **Python 3.10 or 3.11** for maximum compatibility.

### Known open bugs in snnTorch 0.9.4 (48 open issues as of March 2026):

| Issue | Severity | Description | Impact on Our Project |
|-------|----------|-------------|----------------------|
| #401 (OPEN) | **HIGH** | Reset mechanism "subtract"/"zero" may be switched; missing beta in soft reset | Use default "subtract", validate neuron traces manually |
| #328 (OPEN) | MODERATE | GPU memory leak from `SpikingNeuron.instances` class variable not cleared | CPU training avoids this; restart kernel between runs if GPU |
| #394 (OPEN) | MODERATE | CUDA OOM after many epochs due to BPTT graph fragmentation | Small dataset reduces risk; use `expandable_segments:True` |
| #341 (OPEN) | MODERATE | Reset of snn.Leaky uses S(t)*U_thr instead of beta*S(t)*U_thr | Related to #401; same mitigation |
| #377 (OPEN) | LOW | Training loss doesn't reduce for moderately sized networks | Architecture-specific; our small CSNN unlikely affected |
| #339 (OPEN) | LOW | AttributeError: 'Leaky' object has no attribute 'reset' | API usage error; use `utils.reset(net)` instead |

### snnTorch release cadence (from PyPI):

| Version | Date | Gap |
|---------|------|-----|
| 0.9.4 | Feb 16, 2025 | ~10 months after 0.9.1 |
| 0.9.1 | Apr 24, 2024 | 6 days after 0.9.0 |
| 0.9.0 | Apr 18, 2024 | 1 month after 0.8.1 |
| 0.8.1 | Mar 17, 2024 | -- |

The project has an active maintainer (Jason Eshraghian, UCSC) but release cadence slowed significantly between 0.9.1 and 0.9.4. This means bugs may take months to be fixed. **Plan to work around known issues rather than waiting for patches.**

### NeuroBench dependency details (from pyproject.toml):

NeuroBench 2.2.0 has snnTorch as a **required** (not optional) dependency:
```
python: ^3.10
torch: >=2.0.1
torchaudio: >=2.0.2
snntorch: >=0.7.0
numpy: >=1.24.3 (Python <3.12) | >=1.25.0 (Python >=3.12)
tonic: ^1.4.0
numba: >=0.57.1
```

**Known NeuroBench issue (GH #238):** NumPy version conflicts when installed alongside PyTorch Lightning or packages requiring numpy>=2.0. Workaround: install NeuroBench first, then other packages.

**Verification method:** Version info confirmed from PyPI pages for all packages. setup.py confirmed from snnTorch GitHub. NeuroBench pyproject.toml confirmed from NeuroBench GitHub. Bug assessment from snnTorch GitHub issues (48 open issues reviewed). Compatibility assessment based on full dependency chain analysis.

---

## COMPONENT 11: EXISTING AUDIO SNN TUTORIALS

### EXISTS: PARTIALLY
### POTENTIAL BLOCKER: LOW RISK (requires adaptation work, not missing tools)

### What exists:

| Resource | Type | Relevance | URL |
|----------|------|-----------|-----|
| snnTorch Tutorial 6 (CSNN) | Official tutorial | HIGH -- same Conv2d+Leaky architecture, just different input | https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html |
| snnTorch Tutorial 1 (Spike Encoding) | Official tutorial | HIGH -- covers rate/latency/delta encoding | https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html |
| NeuroBench GSC Tutorial | Official tutorial | MEDIUM -- audio SNN with snnTorch, but uses Speech Commands not ESC-50 | https://neurobench.readthedocs.io/en/latest/tutorial/index.html |
| arXiv:2503.11206 (2025) | Research paper | HIGHEST -- SNN on ESC-10 with snnTorch 0.9.1, mel-spectrogram, LIF neurons | https://arxiv.org/html/2503.11206v1 |
| hasithsura/Environmental-Sound-Classification | GitHub repo | MEDIUM -- ESC-50 CNN in PyTorch (ANN baseline code) | https://github.com/hasithsura/Environmental-Sound-Classification |
| kamalesh0406/Audio-Classification | GitHub repo | MEDIUM -- ESC-50 CNN with spectrograms | https://github.com/kamalesh0406/Audio-Classification |

### What does NOT exist:

- No snnTorch tutorial specifically for audio/spectrogram classification
- No end-to-end ESC-50 SNN code repository
- No snnTorch tutorial covering energy measurement

### Gap analysis:

The pipeline you need to build can be constructed by combining:

1. **Audio loading + mel-spectrogram:** Use existing PyTorch ESC-50 repos (hasithsura, kamalesh0406)
2. **Spike encoding:** snnTorch Tutorial 1 (`spikegen.rate()`)
3. **Convolutional SNN architecture:** snnTorch Tutorial 6 (change input dims and num_classes)
4. **Training loop:** snnTorch Tutorial 5/6
5. **Energy measurement:** NeuroBench tutorial
6. **ANN baseline:** Same architecture with ReLU swap (Tutorial 5)

**The March 2025 arXiv paper (2503.11206) is the closest existing work.** It used snnTorch 0.9.1 with ESC-10, mel-spectrograms (128 bands, 1024 FFT, 256 hop), 4 FC layers with 128 LIF neurons, and achieved 69% on ESC-10. However:
- They used FC layers, not convolutional (opportunity for you to improve)
- They used ESC-10 (10 classes), not ESC-50 (50 classes)
- They did not report energy metrics
- No code was released

**This gap is actually beneficial for a thesis** -- it means your convolutional SNN approach with energy comparison on full ESC-50 would be genuinely novel.

**Verification method:** Thorough search of snnTorch tutorials index, NeuroBench documentation, arXiv, GitHub, and Google Scholar. The arxiv paper was confirmed to be the closest existing work.

---

## COMPONENT 12: ESC-50 ANN BASELINES

### EXISTS: YES
### POTENTIAL BLOCKER: NO

### Published baselines from ESC-50 README leaderboard:

**From scratch (no pretraining):**

| Method | Year | Accuracy | Notes |
|--------|------|----------|-------|
| Piczak CNN | 2015 | 64.5% | Original baseline, simple ConvNet |
| Human performance | 2015 | 81.3% | Crowdsourced evaluation |
| SB-CNN | 2016 | 73.1% | Shallow CNN with data augmentation |
| EnvNet-v2 | 2017 | 84.9% | Strong augmentation |
| CNN + PSO optimization | 2022 | 93.7% | Architecture search |

**With pretraining / transfer learning:**

| Method | Year | Accuracy |
|--------|------|----------|
| AST (Audio Spectrogram Transformer) | 2021 | 95.7% |
| CLAP | 2022 | 96.7% |
| HTS-AT | 2022 | 97.0% |
| BEATs | 2023 | 98.1% |
| HTSAT-22 | 2023 | 98.25% |

### SNN baselines on ESC-10 (subset of ESC-50):

| Method | Year | Accuracy | Framework |
|--------|------|----------|-----------|
| FC-SNN + TAE encoding | 2025 | 69.0% | snnTorch 0.9.1 |
| FC-SNN + Step Forward | 2025 | ~41% | snnTorch 0.9.1 |
| FC-SNN + Moving Window | 2025 | ~35% | snnTorch 0.9.1 |

### What these baselines mean for your project:

1. **Your ANN baseline** (simple CNN from scratch on mel-spectrograms): Expect **64-75% on ESC-50**
2. **Your SNN** (convolutional SNN with rate encoding): The only existing reference achieved 69% on ESC-10 with FC layers. A CSNN on ESC-50 might achieve **40-65%** (rough estimate)
3. **The accuracy gap itself IS the interesting finding** -- your thesis analyzes WHY and quantifies the energy tradeoff
4. **If your CSNN beats 69% on ESC-10, you have a result better than the only published SNN benchmark**

**Verification method:** Baselines confirmed from ESC-50 GitHub README results table. SNN baselines from arXiv:2503.11206 (March 2025).

---

## master verification table

| # | Component | EXISTS | VERIFIED VIA | BLOCKER | RISK | 2026-03-03 UPDATE |
|---|-----------|--------|-------------|---------|------|-------------------|
| 1 | ESC-50 Dataset | YES | GitHub repo, direct download URL confirmed | NO | NONE | No change -- stable |
| 2 | Audio Loading (librosa/torchaudio) | YES | Official docs, macOS ARM64 builds | NO | NONE | torchaudio maintenance mode confirmed; our APIs preserved |
| 3 | Mel-spectrogram Conversion | YES | librosa + torchaudio docs, params from literature | NO | NONE | No change |
| 4 | Spike Encoding (spikegen) | YES | snnTorch 0.9.4 API docs, 15+ functions confirmed | NO | NONE | Full API signatures verified |
| 5 | Convolutional SNN | YES | Tutorial 6 code, 9 neuron models available | NO | **MODERATE** | **Reset mechanism bug #401 affects neuron dynamics** |
| 6 | ANN Baseline | YES | Tutorial 5, GitHub ANN-to-SNN repo confirms approach | NO | NONE | No change |
| 7 | Energy/SynOps Measurement | YES | NeuroBench 2.2.0 pyproject.toml confirmed | NO | **MODERATE** | **NeuroBench has numpy dependency conflicts (GH #238)** |
| 8 | 5-Fold Cross-validation | YES | Built into ESC-50 design, CSV structure confirmed | NO | NONE | No change |
| 9 | GPU/macOS Compatibility | WORKABLE | MPS partial, CPU feasible, Colab backup | NO | LOW-MODERATE | GPU memory leak #328 confirmed; CPU route is safer |
| 10 | snnTorch+PyTorch Compat | YES | PyPI, setup.py, no version pin issues | NO | **MODERATE** | **48 open issues, slow release cadence, reset bug unfixed** |
| 11 | Audio SNN Tutorials | PARTIAL | No direct tutorial; all pieces exist; 2025 paper validates | NO | LOW | No change |
| 12 | ESC-50 ANN Baselines | YES | GitHub README leaderboard, PapersWithCode | NO | NONE | Updated with top scores: HTSAT-22 98.25%, BEATs 98.10% |

---

## risk assessment

### No Hard Blockers

Every component exists and is accessible. The project is technically feasible.

### Moderate-Risk Items (require attention and workarounds):

1. **snnTorch reset mechanism bug (#401, OPEN since Oct 2025):** The `reset_mechanism` parameter in `snn.Leaky()` may produce incorrect neuron dynamics. The "subtract" and "zero" modes may be switched, and the soft reset is missing a beta coefficient.
   - **Impact:** Could silently produce wrong results if you rely on specific reset behavior.
   - **Mitigation:** (a) Use the default `reset_mechanism="subtract"` which is the most tested path, (b) visually validate membrane potential traces before running full experiments, (c) document which reset mechanism you use and note the known bug in your thesis.
   - **Severity for this project:** MODERATE -- the default behavior works for most use cases and Tutorial 6 uses defaults.

2. **NeuroBench dependency conflicts (#238):** NumPy version conflicts can cause installation failures.
   - **Impact:** May not be able to install NeuroBench alongside other packages without careful version management.
   - **Mitigation:** (a) Install packages in this exact order: PyTorch first, then snnTorch, then NeuroBench last, (b) use a dedicated venv, (c) if NeuroBench fails, fall back to manual SynOps counting.

3. **snnTorch GPU memory issues (#328, #394):** Memory leak from `SpikingNeuron.instances` and BPTT graph fragmentation.
   - **Impact:** GPU training may crash after many epochs or when creating/destroying networks.
   - **Mitigation:** (a) Train on CPU -- ESC-50 is small enough, (b) if using GPU, set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and restart kernel between experiments.

### Low-Risk Items (manageable with workarounds):

4. **macOS MPS + snnTorch:** Open PR (#415) for MPS compatibility. Workaround: use CPU (fast enough for 2,000 samples) or Google Colab.

5. **No ready-made ESC-50 SNN tutorial:** You must combine pieces from 3-4 different tutorials/repos. This is normal for thesis-level work and actually strengthens your contribution.

6. **Python version sensitivity:** Use Python 3.10 or 3.11 to avoid compatibility issues between snnTorch, NeuroBench, and PyTorch.

7. **snnTorch slow release cadence:** 10-month gap between 0.9.1 and 0.9.4. Open bugs may not be fixed during your thesis timeline. Plan to work around issues rather than waiting for patches.

### Things That Will Just Work:

- ESC-50 download (no registration, ~600MB zip)
- librosa/torchaudio audio loading (WAV, 44.1kHz, mono)
- Mel-spectrogram extraction (both librosa and torchaudio APIs confirmed)
- spikegen rate/latency/delta encoding of spectrograms (15+ functions available)
- Conv2d + snn.Leaky architecture (Tutorial 6 pattern, works identically for spectrograms)
- ANN baseline with ReLU swap (explicitly documented approach)
- 5-fold cross-validation using metadata CSV (built into dataset design)
- Training on CPU (snnTorch docs confirm this is viable)

---

## recommended technology stack

```
Python 3.11 (required for snnTorch 3.9-3.11 AND NeuroBench >=3.10 overlap)
PyTorch 2.5.x (stable, well-tested with snnTorch; torch >=2.0.1 for NeuroBench)
torchaudio 2.5.x (matched to PyTorch; torchaudio >=2.0.2 for NeuroBench)
snnTorch 0.9.4 (latest; snntorch >=0.7.0 for NeuroBench)
librosa 0.11.0
neurobench 2.2.0
numpy >=1.24.3, <2.0 (to avoid NeuroBench conflicts)
pandas, matplotlib, scikit-learn
```

### Installation script for macOS (DEPENDENCY ORDER MATTERS):

```bash
# Create clean environment with Python 3.11
python3.11 -m venv ~/snn_esc50_env
source ~/snn_esc50_env/bin/activate
pip install --upgrade pip

# Step 1: Install PyTorch ecosystem FIRST
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
