# SNN for ECG Classification on PTB-XL: Complete Technical Component Verification

**Date:** 2026-02-27
**Purpose:** Exhaustive verification that every component needed for an undergraduate thesis "SNN for ECG Classification on PTB-XL" exists, works, and is accessible.

---

## EXECUTIVE SUMMARY

**14 out of 14 components exist and are accessible.** However, there are **2 significant concerns** and **1 architectural design decision** that require careful attention:

1. **GPU Memory with 1000 timesteps** -- BPTT over 1000 SNN timesteps will blow up GPU memory. Truncated BPTT (TBPTT) or the 100 Hz version of PTB-XL (reducing to 1000 samples) are mandatory mitigations.
2. **No existing SNN-on-PTB-XL code** -- Zero prior implementations of SNN classification on PTB-XL exist in the literature. All prior SNN-ECG work uses MIT-BIH or smaller datasets. This means you are doing genuinely novel work, which is good for a thesis but means you cannot copy an existing pipeline.
3. **Conv1d in snnTorch** -- Technically supported (neurons are shape-agnostic) but zero official examples exist. You will need to adapt the Conv2d tutorial pattern yourself.

**Overall feasibility verdict: YES, the pipeline is buildable end-to-end, but requires careful engineering on the SNN timestep/memory dimension.**

---

## COMPONENT-BY-COMPONENT VERIFICATION

---

### 1. PTB-XL Dataset

| Field | Detail |
|---|---|
| **EXISTS** | YES |
| **VERIFIED HOW** | PhysioNet official page: https://physionet.org/content/ptb-xl/1.0.3/ |
| **POTENTIAL BLOCKER** | NO |

**Details:**
- **Host:** PhysioNet, version 1.0.3 (latest)
- **Format:** WFDB (WaveForm DataBase) with 16-bit precision at 1 uV/LSB resolution
- **Size:** 3.0 GB uncompressed, 1.7 GB ZIP download
- **Records:** 21,799 clinical 12-lead ECGs from 18,869 patients, each 10 seconds long
- **Sampling rates:** 500 Hz (records500/) and 100 Hz downsampled (records100/)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0) -- fully open access
- **Data Use Agreement:** NONE required. Open access -- "Anyone can access the files, as long as they conform to the terms of the specified license."
- **No credentialed access needed** (unlike MIMIC-III which requires training)

**Source:** [PTB-XL v1.0.3 on PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/)

---

### 2. WFDB Python Library

| Field | Detail |
|---|---|
| **EXISTS** | YES |
| **VERIFIED HOW** | PyPI: https://pypi.org/project/wfdb/ , Docs: https://wfdb.readthedocs.io/ |
| **POTENTIAL BLOCKER** | NO |

**Details:**
- **Install:** `pip install wfdb`
- **Latest version:** 4.3.1 (released February 2026)
- **Python requirement:** >= 3.9
- **Can load PTB-XL directly:** YES

**Exact loading code (from PhysioNet's official example_physionet.py):**

```python
import pandas as pd
import numpy as np
import wfdb
import ast

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

path = '/path/to/ptb-xl/'
sampling_rate = 100

# Load and convert annotation data
Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data
X = load_raw_data(Y, sampling_rate, path)
# X shape at 100 Hz: (21799, 1000, 12)  -- records x samples x leads
# X shape at 500 Hz: (21799, 5000, 12)  -- records x samples x leads
```

**Source:** [PhysioNet example_physionet.py](https://www.physionet.org/content/ptb-xl/1.0.2/example_physionet.py), [WFDB Python GitHub](https://github.com/MIT-LCP/wfdb-python)

---

### 3. Signal Preprocessing

| Field | Detail |
|---|---|
| **EXISTS** | YES |
| **VERIFIED HOW** | scipy docs, neurokit2 docs, published research |
| **POTENTIAL BLOCKER** | NO |

**Standard preprocessing for PTB-XL:**

1. **Bandpass filter:** 0.5--40 Hz (removes baseline wander below 0.5 Hz and high-frequency noise above 40 Hz)
2. **Normalization:** Z-score normalization per lead (zero mean, unit variance)
3. **Important finding:** Research shows that for deep learning on PTB-XL, preprocessing makes minimal difference. The Strodthoff benchmark uses minimal/no preprocessing.

**Exact scipy code for bandpass filtering:**

```python
from scipy.signal import butter, sosfiltfilt
import numpy as np

def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=100, order=4):
    sos = butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
    filtered = sosfiltfilt(sos, signal, axis=0)
    return filtered

# Apply per-record: filtered_ecg = bandpass_filter(X[i], fs=100)
```

**NeuroKit2 alternative:**

```python
import neurokit2 as nk

# pip install neurokit2  (latest version, actively maintained)
cleaned = nk.ecg_clean(ecg_signal, sampling_rate=100, method='neurokit')
```

**Key finding from literature:** "Band-passing makes no measurable difference in performance" for deep learning on PTB-XL (arxiv 2311.04229). Recommendation: apply minimal preprocessing -- just Z-score normalization.

**Sources:** [SciPy butter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html), [SciPy sosfiltfilt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html), [NeuroKit2](https://pypi.org/project/neurokit2/), [ECG Pre-Processing Best Practices](https://arxiv.org/pdf/2311.04229)

---

### 4. Delta Encoding in snnTorch

| Field | Detail |
|---|---|
| **EXISTS** | YES |
| **VERIFIED HOW** | Official docs: https://snntorch.readthedocs.io/en/latest/snntorch.spikegen.html |
| **POTENTIAL BLOCKER** | NO |

**Exact function signature:**

```python
snntorch.spikegen.delta(
    data,           # torch.Tensor, shape [num_steps x batch x input_size]
    threshold=0.1,  # float: change magnitude to trigger spike
    padding=False,  # bool: how first timestep is handled
    off_spike=False  # bool: enable negative spikes for decreases
)
# Returns: torch.Tensor of spike values (0, 1, or -1 if off_spike=True)
```

**ECG-specific usage:**

```python
import snntorch.spikegen as spikegen
import torch

# ECG data shape: [1000, batch_size, 12]  (timesteps x batch x leads)
ecg_tensor = torch.tensor(ecg_data).permute(1, 0, 2)  # from (batch, time, leads)
spike_data = spikegen.delta(ecg_tensor, threshold=0.1, off_spike=True)
# off_spike=True recommended for ECG: captures both rising and falling edges
```

**The delta function is explicitly designed for time-series data.** The documentation states it "accepts a time-series tensor as input and takes the difference between each subsequent feature across all time steps." This is exactly what ECG needs.

**Threshold selection:** Smaller threshold = more spikes (better resolution, more computation). Larger threshold = fewer spikes (sparser, faster). This is a hyperparameter to tune.

**Other available encoding methods:** rate(), latency(), latency_code(), rate_conv() -- delta is the best fit for ECG because it captures signal changes (morphology), which is clinically meaningful.

**Source:** [snntorch.spikegen documentation](https://snntorch.readthedocs.io/en/latest/snntorch.spikegen.html), [Tutorial 1 - Spike Encoding](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html)

---

### 5. 1D Convolutional SNN (CRITICAL COMPONENT)

| Field | Detail |
|---|---|
| **EXISTS** | YES (with caveats) |
| **VERIFIED HOW** | Source code analysis, documentation, PyTorch Forums |
| **POTENTIAL BLOCKER** | LOW RISK -- requires manual adaptation |

**Can snnTorch do Conv1d-based SNNs?**

**YES, but there are no official Conv1d examples.** Here is the evidence chain:

1. **Official documentation states:** "Each layer of spiking neurons are therefore agnostic to fully-connected layers, convolutional layers, residual connections, etc."

2. **Source code confirms:** The Leaky neuron's forward() method uses `torch.zeros_like(input_)` to initialize membrane potential, meaning it dynamically matches ANY input tensor shape -- 2D (Linear), 3D (Conv1d), or 4D (Conv2d).

3. **Conv2d is proven to work** in Tutorial 6 with 4D tensors (batch x channels x height x width) feeding directly into snn.Leaky(). Conv1d produces 3D tensors (batch x channels x length), which follow the same pattern.

4. **BatchNormTT1d exists** in snnTorch specifically for 1D temporal batch normalization, confirming 1D data is a supported use case.

5. **One PyTorch Forum user attempted Conv1d + snnTorch** (https://discuss.pytorch.org/t/157693) and encountered shape errors, but these were caused by incorrect channel/feature dimensions in their architecture, NOT a fundamental incompatibility.

**Recommended Conv1d SNN architecture for ECG:**

```python
import torch.nn as nn
import snntorch as snn

class ECG_SNN(nn.Module):
    def __init__(self, beta=0.9):
        super().__init__()
        # Input: (batch, 12, 1000) -- 12 leads, 1000 timesteps at 100Hz
        self.conv1 = nn.Conv1d(12, 32, kernel_size=7, padding=3)
        self.lif1 = snn.Leaky(beta=beta, init_hidden=True)
        self.pool1 = nn.MaxPool1d(2)
