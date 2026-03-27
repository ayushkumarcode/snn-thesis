# SNN for Environmental Sound Classification on ESC-50: Component Verification Report

> Research compiled: 2026-02-28 | Updated: 2026-03-03
> Platform: macOS Darwin 24.5.0 (Apple Silicon)
> Purpose: Exhaustive technical verification of every pipeline component
> Update note: Second-pass deep verification with bug investigation and dependency audit

---

## EXECUTIVE SUMMARY

**Overall Feasibility: YES -- all 12 components verified as existing and accessible.** The entire pipeline from ESC-50 data loading through mel-spectrogram extraction, spike encoding, convolutional SNN training, ANN baseline comparison, and energy measurement is technically feasible using freely available, open-source tools. Three components carry risk flags that require attention: (1) macOS MPS compatibility with snnTorch, (2) absence of a ready-made audio SNN tutorial, and (3) **open bugs in snnTorch's neuron reset logic and memory management**. None are hard blockers, but item 3 is a newly discovered concern that warrants caution.

The most significant finding is a March 2025 arXiv paper that directly addresses SNN-based environmental sound classification on ESC-10 using snnTorch -- confirming this is a viable and currently active research direction.

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
