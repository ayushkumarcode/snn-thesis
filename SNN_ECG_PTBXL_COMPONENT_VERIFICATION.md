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
