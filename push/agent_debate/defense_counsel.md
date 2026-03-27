# Defense Counsel: The Case for Publishing This Work at ICONS 2026

**Role:** Senior neuromorphic computing researcher and conference paper author
**Date:** 9 March 2026
**Task:** Provide the strongest honest case for each of the six contributions; address the supervisor's concern directly

---

## Prefatory Note on Standards

ICONS is the premier dedicated venue for neuromorphic systems research. It is not NeurIPS. It is not ICLR. Its own call for papers explicitly welcomes "benchmark tasks for neuromorphic computing," "hardware deployment," and "algorithms and training" work. The historical acceptance rate has been ~59% (ICONS 2018 data), and both ICONS 2024 and 2025 accepted multiple papers whose scientific value rested on first-ever demonstrations and systematic methodology rather than competitive accuracy numbers. The ICONS 2022 most directly comparable paper (Yarga et al.) benchmarked 4 encoding schemes on speech digit recognition and was accepted. The ICONS 2025 best paper was about turbulence modeling using neuron random walks — no classification accuracy metric whatsoever.

The question is not whether this paper is publishable at a top ML venue. It is not. The question is whether it is publishable at ICONS, which explicitly serves the community this work is designed for. The answer, argued contribution by contribution below, is yes.

---

## C1: First Convolutional SNN Evaluation on Full ESC-50

### Core Novelty Claim

This is the first evaluation of any spiking neural network architecture on the ESC-50 benchmark in its full 50-class, 5-fold cross-validation configuration.

### Why It Is Genuinely Novel

The claim is not contested by any paper in existence. Larroza et al. (arXiv:2503.11206, March 2025) — the closest competitor in the world, posted simultaneously with this thesis — explicitly state in their own abstract: "no state-of-the-art solution has yet encoded environmental sound datasets using spike-based methods." They then evaluate only ESC-10 (10 classes, a curated subset), using a fully-connected 3-layer architecture, testing 3 encoding schemes, with no hardware deployment. The full ESC-50 benchmark — 50 classes, 2,000 recordings, the standard benchmark used by every ANN paper in the field since Piczak (2015) — has never been addressed by any SNN paper. Research agents confirmed this across arXiv, IEEE Xplore, ACM DL, Semantic Scholar, and Google Scholar. The survey paper by Basu et al. (arXiv:2502.15056, February 2025), a 24-page dedicated survey of neuromorphic audio classification, reaches the same conclusion without finding a single full-ESC-50 SNN result.

### Why It Has Scientific Significance Beyond Just "First"

The novelty of a "first" result is only as valuable as the benchmark itself. ESC-50 is not an obscure or trivial dataset. It is the standard 50-class benchmark in environmental sound classification, with human performance established at 81.3% (Piczak 2015) and ANN SOTA at 99.1% (OmniVec2, CVPR 2024). Every serious ANN paper in environmental audio reports on ESC-50. Establishing the first SNN results creates the reference point — the zero-line — against which all future SNN audio work will be measured. Without this, the field literally cannot quantify progress. The matched-architecture comparison (SNN vs ANN under identical training protocol) is methodologically clean: the gap of 16.7 percentage points is attributable to the spiking mechanism and surrogate gradient training alone, not to architectural differences. This is precisely the kind of controlled characterization that the neuromorphic field needs.

### Strongest Honest Framing for ICONS Reviewers

"We establish the first SNN benchmark on ESC-50, providing a reproducible, architecturally controlled baseline (47.15% SNN vs 63.85% ANN, matched architecture) that serves as the reference point for future SNN audio research. Our 5-fold cross-validation protocol follows the ESC-50 predefined folds exactly, ensuring comparability with all prior ANN work."

---

## C2: Systematic Comparison of 7 Spike Encoding Methods

### Core Novelty Claim

This is the most comprehensive spike encoding comparison ever conducted on a standard audio benchmark: 7 encoding schemes on ESC-50, yielding a complete ranking with mechanistic explanations for each result.

### Why It Is Genuinely Novel

No prior work compares more than 4 encoding schemes on any audio benchmark. Larroza et al. compare 3 schemes (TAE, Step Forward, Moving Window) on ESC-10 with an FC-only architecture. Yarga et al. (ICONS 2022) compare 4 schemes on speech digit recognition. Bian et al. (arXiv:2407.09260, 2024) compare 8 variants on an IMU sensor dataset, not audio, and not on any standard benchmark. The thesis compares 7 schemes — rate, delta, latency, direct, burst, phase, population — on the same architecture, same dataset, same 5-fold protocol. The ordering that emerges (direct >> rate ≈ phase > population > latency >> delta ≈ burst) is internally consistent and mechanistically explicable. Each failure mode is documented with a different root cause: delta fails because static spectrograms have no temporal variation to encode; burst fails because front-loading spikes in the first 5 of 25 timesteps creates temporal window mismatch; latency fails because spectrogram features are not naturally compatible with first-spike timing; population underperforms because MSE count loss is harder to optimise than cross-entropy rate loss.

The most important finding within C2 is the rate-phase tie: phase coding achieves 24.15% using exactly 1 spike per neuron, while rate coding achieves 24.00% using approximately 7 spikes per neuron. They are statistically indistinguishable despite a 7-fold difference in spike count. This confirms the information preservation principle: temporal window coverage is what matters, not the spike count. This finding is consistent with Guo et al. (Frontiers in Neuroscience, 2021) and has direct implications for energy-efficient inference, since phase coding's 7x spike reduction translates directly to 7x fewer AC operations on neuromorphic hardware.

### Why It Has Scientific Significance Beyond Just "First"

The literature has reached no consensus on which encoding is best for audio. Guo et al. showed phase coding is most noise-robust for MNIST-like tasks. Kim et al. (ICASSP 2022) showed direct coding beats rate at low timesteps for image classification. Larroza et al. found threshold-adaptive encoding best for ESC-10. The thesis provides the first test of 7 schemes simultaneously on a complex, real-world audio benchmark, under controlled conditions. The full ordering is an empirical contribution of lasting value: researchers designing SNN audio systems will be able to consult this work to select an encoding. The mechanistic explanations for failures (not just rankings) constitute an additional contribution — the burst coding failure analysis (front-loading → temporal window mismatch → severe overfitting at 45-62% train, 5-9% test) is a distinct finding that does not appear anywhere in the literature.

### Strongest Honest Framing for ICONS Reviewers

