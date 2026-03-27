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
