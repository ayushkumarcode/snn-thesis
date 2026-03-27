# Final Verdict: Publishable Novelty Assessment for ICONS 2026

**Reviewer role:** Neutral senior researcher
**Date of assessment:** 9 March 2026
**Subject:** UoM COMP30040 undergraduate thesis — Convolutional SNN for ESC-50 Environmental Sound Classification
**Target venue:** ICONS 2026 (ACM International Conference on Neuromorphic Systems), deadline April 1 2026

---

## Part 1: What Is Genuinely Novel (Line by Line)

### Claim 1: First convolutional SNN on the full ESC-50 benchmark (50 classes, 5-fold CV)

**The claim.** No prior publication reports SNN accuracy on the full 50-class ESC-50 dataset. The only comparable work, Larroza et al. (arXiv:2503.11206, submitted March 2025), explicitly states "no state-of-the-art solution has yet encoded environmental sound datasets using spike-based methods" — and themselves evaluate only ESC-10 (10 classes) with fully-connected networks. Two independent 2025 survey papers (Basu et al. arXiv:2502.15056; Baek & Lee, Biomedical Engineering Letters 14:5) confirm no ESC-50 SNN results in the literature.

**Why this is novel, not just first.** First results are not automatically interesting; what matters is whether the benchmark is meaningful. ESC-50 is: it has predefined 5-fold CV making results directly reproducible and comparable, it has been a standard ANN benchmark since Piczak 2015, and it is substantially harder than ESC-10 (5× more classes, more acoustic diversity). Filling this gap is a real service to the field: it establishes a baseline from which all future SNN audio work can measure progress. The use of a convolutional architecture (rather than FC-only) also represents a step up in architectural ambition that matches what the ANN field has known to be necessary for this task since 2015.

**Novelty strength: Strong.**

---

### Claim 2: Systematic comparison of 7 spike encoding methods on the same architecture and dataset

**The claim.** This thesis compares 7 encodings (direct, rate, phase, population, latency, delta, burst) on a single fixed architecture and fixed dataset. The next most comprehensive study is Larroza et al. (2025) with 3 encodings on ESC-10. Yarga et al. (ICONS 2022) compared 4 on speech digits. No prior paper has compared 7 encodings for any audio SNN task.

**Why this is novel.** The encoding hierarchy found — direct (47.15%) >> rate (24.00%) ≈ phase (24.15%) > population (19.15%) > latency (16.30%) >> delta (7.25%) ≈ burst (6.50%) — is not something the field could have predicted in advance. The result that phase coding matches rate coding with exactly 1 deterministic spike per neuron vs approximately 7 stochastic spikes is a genuine insight about the information content of temporal vs rate representations in this setting. The finding that burst coding catastrophically fails (6.5%, near random) and the mechanistic explanation (front-loaded spikes create temporal window mismatch with LIF integration) are both interpretable and useful to the field. This is not a "horse race" contribution; it is a controlled experiment with a clear causal story for each outcome. The ICONS 2022 Yarga paper is the direct precursor, and this work meaningfully extends it in scope, dataset difficulty, and depth of analysis.

**Novelty strength: Strong.**

---

### Claim 3: The PANNs + SNN head transfer learning experiment and the gap-collapse finding

**The claim.** No prior paper combines a frozen AudioSet-pretrained audio model (PANNs CNN14) with an SNN classifier head. After exhaustive search, zero prior works were found using any audio foundation model (PANNs, VGGish, wav2vec, HuBERT, CLAP, BEATs, AST) as frozen features for an SNN classifier. The result — 92.50% SNN vs 93.45% ANN on the same features, a 0.95 percentage point gap — is the first quantified demonstration of the feature-learning bottleneck hypothesis in audio.

**Why this is novel.** The scientific value here is not the 92.50% number. It is the 17.6× collapse ratio: scratch gap is 16.70 pp, features-equal gap is 0.95 pp. This ratio exceeds what has been reported in vision (Spikformer V2 achieves roughly 7× collapse going from scratch to SSL pretraining on ImageNet). The interpretation — that spiking computation itself is not the bottleneck, that LIF neurons can classify just as well as ReLU neurons when both get equal-quality representations — is a claim with implications for how the entire field should think about SNN accuracy gaps. It reframes 47.15% vs 63.85% as a data-scale and feature-learning problem, not a fundamental limitation of the spiking formalism. The closest parallel in the literature is SAFE (ICLR 2025 submission, later withdrawn), which uses CNN features plus an SNN classifier for fake audio detection, but SAFE does not measure or discuss the gap collapse. This thesis is the first to frame it explicitly as a scientific finding and quantify the ratio.

One honest caveat: the gap-collapse finding is consistent with a pattern already visible in vision (ANN-to-SNN conversion literature, Spikformer V2), so it does not shock the theory. What it adds is the first audio-domain empirical evidence, the clearest quantification, and the cleanest experimental design (same architecture, only activation type differs). That is enough.

**Novelty strength: Strong.**

---

### Claim 4: First neuromorphic hardware deployment for environmental sound classification beyond pure tones

**The claim.** The only prior SNN-on-SpiNNaker audio paper is Dominguez-Morales et al. (ICANN 2016), which classifies 8 pure synthetic tones (130–1397 Hz) — a task that is trivially simple compared to 50-class environmental sound. This thesis deploys an SNN on SpiNNaker for ESC-50, achieves 33.1% ± 6.9% across 5 folds, characterises the hardware-software accuracy gap (12.8 ± 4.1 pp), and documents the root cause of that gap (FC1 input activation distribution incompatibility with integer weights and binary-threshold SpiNNaker neurons).

**Why this is novel.** Hardware deployment papers at ICONS are first-class contributions, not ancillary ones. The community values honest characterisation of what works and what does not on real hardware. The fact that the result is 33.1% rather than matching the 47.15% software result is not a weakness — it is the scientific content. The FC1 cancellation root-cause analysis (near-zero-mean weights across 1398 simultaneous binary inputs producing net negative current) and the FC2-only hybrid solution are exactly the kind of insight that accelerates subsequent work by others attempting SpiNNaker deployment. The Option A and Option C experiments further document what does and does not transfer from software to hardware in a systematic way. There is nothing comparable in the literature for environmental sound on any neuromorphic chip.

**Novelty strength: Strong at ICONS specifically.** For a general ML venue, hardware deployment at 33% accuracy would be a weakness. At ICONS, which explicitly includes systematic hardware characterisation as a category of valued contribution, this is well within the normal profile of accepted hardware papers.
