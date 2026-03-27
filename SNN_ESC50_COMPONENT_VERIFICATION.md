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
