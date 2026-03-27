# SNN for ECG Classification on PTB-XL: Component Verification

checked on 2026-02-27 whether every piece needed for an undergrad thesis "SNN for ECG Classification on PTB-XL" actually exists and works.

---

## the short version

all 14 components exist and are accessible. but there are 2 real concerns and 1 architecture decision to think about:

1. **GPU memory with 1000 timesteps** -- BPTT over 1000 SNN timesteps will absolutely blow up GPU memory. truncated BPTT (TBPTT) or the 100 Hz version of PTB-XL (reducing to 1000 samples) are mandatory.
2. **no existing SNN-on-PTB-XL code** -- nobody has ever implemented SNN classification on PTB-XL. all prior SNN-ECG work uses MIT-BIH or smaller datasets. this means i'd be doing genuinely novel work, which is good for a thesis but means i can't just copy an existing pipeline.
3. **Conv1d in snnTorch** -- technically supported (neurons are shape-agnostic) but zero official examples exist. i'd need to adapt the Conv2d tutorial pattern myself.

overall: yes, the pipeline is buildable end-to-end, but need to be careful about the SNN timestep/memory dimension.

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
