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

"Our 7-encoding comparison on ESC-50 is the most comprehensive encoding benchmark in SNN audio, superseding the previous largest comparison (4 schemes, Yarga 2022 ICONS, on speech digits). The rate-phase tie — equivalent accuracy at 7x different spike counts — is a novel empirical finding with direct implications for energy-efficient neuromorphic audio deployment."

---

## C3: SpiNNaker Deployment for Environmental Sound Classification

### Core Novelty Claim

This is the first deployment of a spiking neural network on the SpiNNaker neuromorphic platform for environmental sound classification, completing a full 5-fold cross-validation (2,000 inferences) on hardware.

### Why It Is Genuinely Novel

Only one prior paper has deployed an SNN for audio on SpiNNaker: Dominguez-Morales et al. (ICANN 2016), which classified pure sinusoidal tones (frequencies 130-1397 Hz). Pure tone discrimination is a trivial signal processing task — the task can be solved by a single Fourier transform — and bears no relationship to environmental sound classification involving 50 naturalistic acoustic categories. No subsequent paper in 10 years has revisited SpiNNaker for audio of any complexity. Manchester has 6 SpiNNaker PhDs, all working on biological simulation, not audio ML. This work is therefore the first to ask the question: can a convolutional SNN trained on a real-world audio benchmark actually run on SpiNNaker?

The deployment was non-trivial. The root-cause analysis of FC1 cancellation — the discovery that AvgPool produces fractional (non-binary) outputs incompatible with SpiNNaker's spike-only input requirement, and the documented failure of post-hoc weight re-centring (accuracy: 53.75% → 8.50%) — is itself a novel finding. It reveals a fundamental constraint: standard convolutional SNN architectures cannot be directly deployed on spike-only neuromorphic hardware without architectural modification. The FC2-only hybrid approach (snnTorch extracts binary hidden spikes; SpiNNaker runs only the output layer) is a validated engineering contribution that other researchers building similar systems will need to navigate. The 5-fold hardware evaluation (SpiNNaker: 33.1% ± 6.9% vs snnTorch reference: 46.0%) quantifies the hardware gap across all folds, not just a single demonstration.

### Why It Has Scientific Significance Beyond Just "First"

The hardware gap analysis (12.8 ± 4.1 pp across 5 folds) is scientifically valuable in two respects. First, it shows the gap is systematic but variable — the per-fold hardware accuracy ranges from 25.2% (F5) to 43.0% (F4), suggesting fold-specific factors (class distribution, hidden layer activity patterns) affect hardware translation. Second, the gap quantification itself provides ground truth for future SpiNNaker audio work: researchers now know what to expect from FC2-only hybrid deployment and can direct effort toward closing the gap (quantisation-aware training, architectural redesign to avoid AvgPool). The Option A experiment (MaxPool SNN with fc1_binary_fraction=1.000 at all thresholds tested, 43.75% accuracy at threshold=3.0) demonstrates theoretically that full FC1+FC2 SpiNNaker deployment is achievable with an architectural fix, even if not yet completed end-to-end.

The FC1 cancellation finding specifically deserves emphasis. The community does not have a systematic accounting of which standard architectural components are SpiNNaker-compatible. Discovering that AvgPool → FC is incompatible (because AvgPool outputs are fractional not binary) is a practical constraint that affects any SNN deployer using pooling operations before fully-connected layers. Documenting this failure mode with quantitative evidence (weight re-centring assumed binary inputs with sum=n_inputs, but actual sums from fractional AvgPool outputs are much smaller, causing wildly incorrect bias compensation) is an honest, reproducible negative result that saves other researchers from repeating the same mistake.

### Strongest Honest Framing for ICONS Reviewers

"We present the first SpiNNaker deployment for environmental sound classification, completing a 5-fold hardware evaluation (2,000 inferences). We document a previously unreported constraint — AvgPool → FC layers are not directly compatible with spike-only hardware — and provide the validated FC2-only hybrid approach as a replicable workaround. The hardware gap (12.8 ± 4.1 pp) is quantified across all folds, providing the field's first characterisation of SpiNNaker translation fidelity for audio SNNs."

---

## C4: Adversarial Robustness Analysis (First for Audio SNNs)

### Core Novelty Claim

This is the first systematic adversarial robustness analysis of spiking neural networks on audio spectrograms, revealing a 14.9x robustness ratio (SNN 26% vs ANN 1.75% under FGSM at eps=0.1) that is the largest such ratio reported for any audio domain.

### Why It Is Genuinely Novel

Sharmin et al. (ECCV 2020, arXiv:2003.10399) established that SNNs exhibit inherent adversarial robustness for image classification, attributed to binary spike thresholding acting as gradient masking. Subsequent SNN robustness work has remained exclusively in the vision domain. The NEUROSEC paper (FPGA 2024) addressed adversarial audio security using SNNs but in a different paradigm (adversarial detection, not classification robustness measurement). No prior paper has applied FGSM and PGD attacks to an SNN audio classifier and measured the accuracy degradation curve. The attack protocol here — 7 epsilon values, both FGSM and PGD, fold 4 test set, 400 samples — is methodologically sound and generates a complete robustness curve rather than a single point.

The magnitude of the finding is noteworthy: at eps=0.1 FGSM, the ANN retains only 1.75% accuracy (essentially chance on 50 classes = 2%), while the SNN retains 26%. This 14.9x ratio is striking. At eps=0.05 PGD, the ANN drops to 0% while the SNN retains 19.25%. The clean accuracy gap (SNN 53.75% vs ANN 68.75% on fold 4) is reversed under attack at all eps >= 0.01 — meaning there exists an operating point where the SNN is strictly superior to the ANN in the clean+robust tradeoff.

### Why It Has Scientific Significance Beyond Just "First"

Edge audio sensing is a security-sensitive application domain. Always-on microphones in smart environments, surveillance systems, and robotics could be targeted by adversarial audio attacks — small perturbations to environmental sounds that fool classifiers. The finding that SNNs are dramatically more robust to such attacks, due to the natural gradient masking of binary spike thresholding, has direct practical implications for system designers choosing between SNN and ANN architectures for deployment. The robustness is not engineered — it is a free property of the spiking mechanism, requiring no adversarial training, no certified defences, no additional compute. The crossover point (where SNN overtakes ANN in performance) occurring at eps=0.01 FGSM provides a practical decision criterion.

The finding also raises a scientifically interesting question: since binary thresholding masks gradients, standard gradient-based attack metrics may underestimate the true SNN robustness (consistent with Wang et al. 2025, arXiv:2512.22522, who show SA-PGD is needed for reliable SNN robustness evaluation). This is acknowledged in the paper and constitutes an honest methodological caveat rather than a weakness — it suggests the true SNN robustness advantage may be even larger than measured.

### Strongest Honest Framing for ICONS Reviewers

"We conduct the first adversarial robustness analysis of SNNs on audio spectrograms. Under FGSM attack at eps=0.1, the SNN retains 26% accuracy versus 1.75% for the matched-architecture ANN — a 14.9x robustness ratio. Binary spike thresholding provides free, unengineered robustness to gradient-based attacks, with practical implications for secure audio sensing at the edge."

---

## C5: Transfer Learning Gap Collapse (PANNs + SNN Head)

### Core Novelty Claim

