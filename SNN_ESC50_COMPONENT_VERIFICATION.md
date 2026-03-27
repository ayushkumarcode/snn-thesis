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
