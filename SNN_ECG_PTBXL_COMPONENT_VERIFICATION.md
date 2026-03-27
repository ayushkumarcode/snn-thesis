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
