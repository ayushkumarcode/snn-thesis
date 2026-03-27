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
