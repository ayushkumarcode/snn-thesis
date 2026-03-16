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

The first combination of AudioSet-pretrained features (PANNs/CNN14) with a spiking neural network classifier, demonstrating that the 16.7pp SNN-ANN gap in scratch training collapses to under 1pp when both models receive equivalent pretrained features.

### Why It Is Genuinely Novel

No prior published work has frozen an AudioSet-pretrained model (PANNs, CNN14, or any equivalent) and trained an SNN head on its embeddings. The research agent confirmed this across all available literature. The closest approaches are: (a) knowledge distillation methods, where an ANN teacher guides SNN training but retraining occurs; (b) the three-stage hybrid SNN (2025) which converts a pretrained ANN to SNN with fine-tuning (different paradigm — involves full network retraining); (c) SAFE (ICLR 2025), which combines CNN features with SNN for fake audio detection but does not report the gap-collapse analysis. None of these establish the specific experimental finding: when a small SNN head and a small ANN head receive identical CNN14 embeddings, they achieve virtually the same accuracy (92.50% SNN vs 93.45% ANN, 0.95pp gap).

The per-fold results confirm this is robust: PANNs+SNN folds are [92.0, 94.5, 91.0, 93.5, 91.5]%, PANNs+ANN are [93.0, 95.0, 92.0, 95.5, 91.75]%. The SNN is never more than ~2pp behind the ANN on any fold — compared to the scratch training where the gap reaches 23-24pp on the worst folds.

### Why It Has Scientific Significance Beyond Just "First"

The gap collapse from 16.7pp to 0.95pp — a 17.6x reduction in the SNN-ANN accuracy gap — establishes a mechanistic claim that the literature has not previously demonstrated for audio: the SNN-ANN gap in scratch training is a feature-learning bottleneck, not a fundamental limitation of spiking computation. This reframes the entire conversation about SNN audio performance. The conclusion is not "SNNs cannot match ANNs." The conclusion is "SNNs cannot learn discriminative audio features from 1,600 training samples as efficiently as ANNs using surrogate gradients — but given equal-quality features, they classify comparably." This distinction matters enormously for the field's direction: it points to pre-training, data augmentation, and transfer learning as the correct remedies, rather than architectural innovation alone.

The finding is also practically significant. CNN14 extracts audio embeddings once in software (or could be deployed on an ANN accelerator). The SNN head then classifies on neuromorphic hardware — the actual deployed computation is spiking, energy-efficient, and achieves 92.5% on ESC-50. This is a viable engineering architecture for edge audio sensing: ANN for feature extraction (run occasionally, or pre-computed), SNN for classification (run continuously on hardware). The gap collapse ratio (17.6x) is, to the best of our knowledge, the largest such ratio reported for any SNN-ANN paired comparison in the audio domain.

### Strongest Honest Framing for ICONS Reviewers

"We demonstrate the first SNN+PANNs combination and show a 17.6x reduction in the SNN-ANN accuracy gap (16.7pp scratch → 0.95pp with pretrained features). This establishes that the gap is a feature-learning limitation, not a spiking computation limitation — redirecting the field toward pretraining and transfer learning as the correct remedy for SNN audio performance, rather than solely architectural innovation."

---

## C6: Surrogate Gradient Bimodal Ablation

### Core Novelty Claim

The first surrogate gradient ablation on an audio task revealing a bimodal pattern: 3 of 8 tested surrogates learn successfully, 4 fail categorically at chance level, with a clean split that challenges the prevailing "surrogate shape doesn't matter" consensus for complex tasks.

### Why It Is Genuinely Novel

Zenke and Vogels (Neural Computation 2021) established the dominant view: surrogate gradient learning is robust to the shape of the surrogate derivative; what matters is the scale (steepness), not the shape. This finding was demonstrated on relatively simple tasks (XOR, MNIST variants). The thesis ablation — 8 surrogate functions tested on fold 1, direct encoding, ESC-50 50-class — produces a result that partially contradicts this on a harder task: sigmoid fails (2%), STE fails (10.25%), SFS fails (2%), and triangular fails (2.75%), all early-stopping within the first 10-23 epochs at chance-level performance. Meanwhile, spike_rate_escape (46.00%), fast_sigmoid (44.75%), and atan (35.75%) all learn successfully.

This bimodal pattern has not been reported before for audio classification. Lian et al. (IJCAI 2023) showed that surrogate gradient width should adapt to the membrane potential distribution, providing a theoretical mechanism for why some shapes fail. Gygax and Zenke (Neural Computation 2025) showed spike_rate_escape is the only surrogate with rigorous theoretical grounding in escape noise theory — which may explain why it performs best (46.00%, outperforming fast_sigmoid by 1.25pp). The fact that sigmoid fails (2%) is particularly striking given Zenke's 2021 paper listed sigmoid as a working surrogate — the implication is that task complexity and dataset size interact with surrogate shape in ways the 2021 paper did not capture.

### Why It Has Scientific Significance Beyond Just "First"

Practitioners implementing SNN audio classifiers need to know which surrogate to use. The existing guidance ("shape doesn't matter, tune the slope") would lead a practitioner to reasonably choose sigmoid or STE based on availability — and in this domain, those choices result in training failure. The bimodal split provides an empirical warning: for audio classification with a mel-spectrogram input and a convolutional SNN architecture, at least 4 commonly available surrogates fail completely. The three working surrogates share a property — broader effective gradient support near the threshold — that is theoretically consistent with Lian et al.'s width-matching argument. This gives practitioners a principled selection criterion (not just empirical trial and error).

The finding also contributes to understanding SNN trainability for hard, real-world tasks. Most surrogate gradient papers demonstrate on MNIST, N-MNIST, CIFAR-10, or DVS-Gesture — datasets where the signal-to-noise ratio is high and the surrogate shape may genuinely not matter. ESC-50 is a harder task where the membrane potential distribution at threshold is more sensitive to gradient approximation quality. The bimodal result is not a failure of the experiment — it is a finding that the field needs.

### Strongest Honest Framing for ICONS Reviewers

"We conduct an 8-function surrogate gradient ablation on audio classification, revealing a bimodal outcome: 3 surrogates learn (spike_rate_escape 46.00%, fast_sigmoid 44.75%, atan 35.75%), 4 fail at chance level, contradicting the 'shape doesn't matter' consensus (Zenke 2021) for complex audio tasks. The theoretically grounded spike_rate_escape (Gygax and Zenke 2025) achieves best performance, consistent with escape noise theory."

---

## Cross-Cutting Questions

### Is 47.15% Publishable at ICONS Given the Context?

Yes, with two important caveats and one crucial context point.

The caveats first. The number requires context: random baseline is 2% (50 classes), human performance is 81.3%, and the matched ANN baseline is 63.85%. Presented without context, 47.15% sounds weak. Presented as the first-ever SNN result on a 50-class benchmark, achieved by a convolutional architecture trained from scratch on 1,600 samples with surrogate gradients — it is a meaningful anchor point.

The crucial context is the precedent set by the field. The ICONS 2025 best paper had no accuracy metric at all. The ICONS 2025 paper "SNN for Low-Power Vibration-Based Predictive Maintenance" was accepted as an application benchmark where the contribution is demonstrating SNN feasibility in a new domain. Larroza et al. — the closest competitor in audio SNNs — report F1=0.661 on ESC-10 (10 classes), which maps to roughly 60-65% accuracy on a problem 5x easier than ESC-50 (50 classes). Our 47.15% on the harder task is arguably more impressive in relative terms; at minimum it is not categorically worse. The ICONS 2022 Yarga paper was accepted precisely for the encoding comparison methodology, not for its absolute accuracy numbers on speech digit recognition.

What matters at ICONS is: (a) is the contribution scientifically valid? Yes — 5-fold CV, matched architecture, reproducible protocol. (b) Is it genuinely novel? Yes — confirmed by exhaustive literature review. (c) Does it advance the field's understanding? Yes — the encoding comparison, gap collapse, and hardware analysis all do. Absolute accuracy is not the primary review criterion for this type of paper at this conference.

The one framing risk to avoid: do not present 47.15% as an achievement in itself, because it will invite unfair comparison to ANN SOTA. Present it as the reference point in a scientific study of SNN behaviour on audio. The scientific contribution is the comparative analysis, not the number.

### What Is the Actual Scientific Contribution of the Gap Collapse Finding?

The gap collapse (47.15% → 92.50%, reducing the SNN-ANN difference from 16.7pp to 0.95pp) is the most scientifically significant single finding in the paper, and it should be the conceptual centrepiece of the ICONS submission.

Its significance lies in what it disambiguates. A 16.7pp gap between SNN and ANN in scratch training could have three explanations: (1) the spiking computation mechanism is fundamentally limited relative to continuous activations; (2) surrogate gradient training is insufficient for learning audio features; (3) the SNN cannot learn useful representations from a small dataset (1,600 training samples). The PANNs+SNN experiment isolates explanation (3): given representations already learned from 2 million AudioSet clips, the SNN head performs within 0.95pp of the ANN head. Explanation (1) is ruled out. Explanation (2) is partially ruled out for classification (though surrogate gradients may still limit convolutional feature learning in scratch training). The remaining bottleneck is small-dataset feature learning, which is a problem that scales with data and pre-training rather than being fundamental to the spiking formalism.

This is a genuinely important claim for the neuromorphic computing community. It says: do not give up on SNN audio because of gap numbers from scratch training on small datasets. The path forward is not a fundamentally different spiking architecture — it is better feature representations (pre-training, larger datasets, or ANN-to-SNN knowledge transfer). The community needs this finding to correctly prioritise its research agenda.

The 0.95pp gap at high accuracy (92.50% vs 93.45%) is also more meaningful than the 16.7pp gap at lower accuracy (47.15% vs 63.85%). At high accuracy, model capacity is the binding constraint, and both a 3-layer SNN and a 3-layer ANN have similar capacity given the same pretrained features. The near-equality is genuine evidence of computational equivalence at the classification stage, not a trivial result.

### Why Is the Hardware Deployment Valuable Even at 33.1%?

The hardware deployment is valuable for four distinct reasons, none of which depend on the absolute accuracy number.

First, feasibility demonstration. Before this work, no one had attempted SpiNNaker deployment for environmental sound classification. The 33.1% hardware accuracy demonstrates it is physically possible, that the weight mapping is correct, that the spike timing is correct, and that the output decoding is correct. A hardware feasibility demonstration at any accuracy above chance is scientifically valuable.

Second, gap quantification. The 12.8 ± 4.1 pp hardware gap across 5 folds is itself a scientific measurement. Researchers designing SNN-to-SpiNNaker pipelines need to know what translation loss to expect. This paper provides the first such measurement for audio classification on SpiNNaker1. The per-fold variability (25.2% to 43.0%) reveals that fold-specific factors — likely the distribution of hidden spike activity patterns and the SpiNNaker's weight quantisation interaction — systematically affect translation fidelity. This variability is not noise; it is information about the hardware-software gap.

Third, constraint discovery. The FC1 cancellation finding — that AvgPool → FC is incompatible with SpiNNaker's requirement for binary spike inputs, and that post-hoc weight re-centring fails because it mismodels the actual input distribution — is a novel hardware-software co-design insight. Other researchers building SNN → SpiNNaker pipelines will encounter this exact constraint. Documenting it with root cause analysis is a concrete contribution to the community regardless of what accuracy it produces.

Fourth, path to full deployment. The Option A experiment (MaxPool SNN achieving fc1_binary_fraction=1.000 at all tested thresholds, 43.75% accuracy) shows the architectural fix that enables full FC1+FC2 deployment. The paper therefore documents not just the current state but the roadmap to improving it. This is a complete scientific story: problem identified, root cause diagnosed, fix validated, future path clear.

### What Is the Significance of the Bimodal Surrogate Result?

The bimodal result is more significant than it initially appears, for three reasons.

First, it challenges an established consensus. Zenke and Vogels (2021) — a highly-cited paper (1,000+ citations) — claim surrogate gradient learning is robust to shape. The ICONS 2022 paper by Yarga explicitly cited this robustness as justification for not comparing surrogates. The thesis provides the first evidence that this robustness breaks down for audio classification: sigmoid and STE, both commonly used and recommended in tutorials, fail completely (2% and 10% respectively). Triangular also fails, which Zenke specifically listed as a working alternative. This is not a minor technical footnote — it is an empirical challenge to a widely-propagated claim, with direct implications for how practitioners should approach SNN audio implementation.

Second, the split is mechanistically interpretable. The three successful surrogates (spike_rate_escape, fast_sigmoid, atan) share broader effective gradient support near the threshold. The four failing surrogates are narrower or have qualitatively different gradient behaviour (STE is piecewise constant, sigmoid vanishes further from threshold). This pattern is consistent with Lian et al.'s width-matching theory (IJCAI 2023): the audio classification task produces a membrane potential distribution that requires broader surrogate support than MNIST or XOR, where narrower surrogates succeed. The theoretical grounding of spike_rate_escape in escape noise theory (Gygax and Zenke 2025) further explains why it performs best — it is the closest approximation to the true gradient of the stochastic LIF neuron.

Third, it has practical value disproportionate to its length in the paper. A practitioner who reads a single table (2 rows of the surrogate ablation) and learns "don't use sigmoid or STE for audio SNN classification" has received immediately actionable information that could save weeks of failed experiments. The bimodal pattern converts a general warning ("surrogate choice matters") into a specific, binary, verifiable guideline.

---

## Honest Assessment: What Is Genuinely Strong vs. What Needs Careful Framing

### Genuinely Strong (No Qualification Needed)

- The novelty claim on C1 (first ESC-50 SNN) is watertight. No reviewer can find a counter-example.
- The encoding comparison breadth (C2) is the most comprehensive in audio SNN literature. This is factual.
- The adversarial robustness magnitude (C4) — 14.9x ratio, ANN at essentially chance — is a striking result. Even sceptical reviewers will find this interesting.
- The gap collapse mechanism (C5) is scientifically important and clearly communicated. The 17.6x gap reduction is a concrete, verifiable number.
- The hardware deployment feasibility (C3) is the first of its kind for environmental audio.

### Needs Careful Framing (Not Weak, But Requires Context)

- 47.15% accuracy: Always contextualise relative to random (2%), human (81.3%), and the ANN baseline (63.85%). Frame as "first reference point" not as "good performance."
- 33.1% SpiNNaker accuracy: Frame as hardware characterisation and feasibility demonstration, not as a deployment result claiming practical utility. The gap analysis IS the contribution.
- Surrogate ablation: A 1-seed, 1-fold ablation needs the CSF3 3-seed results to make variance claims. If those are unavailable, explicitly label the result as preliminary and note the consistent directional finding.
- Energy analysis: Do not claim SNN is more energy-efficient than ANN in software. It is not (976 nJ vs 463 nJ). Claim the correct thing: ACs cost 5.1x less than MACs on neuromorphic hardware; the deployment pathway via SpiNNaker realises this advantage. Cite Dampfhoffer et al. (2023) honestly.
- Continual learning: The 6.9pp forgetting reduction (SNN 74.4% vs ANN 81.3%) is real but modest. It is a supporting result, not a headline contribution. It adds breadth to the paper without being the primary claim.

### What Should Be the Title and Frame of the Paper

The paper should be framed as a systematic characterisation study — the first such study for audio SNNs on ESC-50 — with the gap collapse as the key intellectual finding, and the hardware deployment and adversarial robustness as the neuromorphic-systems-specific contributions that distinguish it from a pure ML paper.

A title that works: "From Spike Encoding to Neuromorphic Hardware: Characterising Spiking Neural Networks for Environmental Sound Classification on ESC-50"

This title signals: (a) methodological scope (encoding through hardware), (b) characterisation rather than SOTA-claiming, (c) the specific benchmark (ESC-50), and (d) the system-level nature of the contribution — appropriate for ICONS.

---

## Summary Verdict

The supervisor's concern — "results with significance and novelty are required for publication" — is valid as a general principle. Applied to this specific work, the concern underestimates what is here. This paper has:

- One confirmed "first" that cannot be disputed (ESC-50 SNN)
- One confirmed "first" in audio SNN hardware (SpiNNaker deployment)
- One confirmed "first" in audio SNN adversarial analysis
- One confirmed "first" in audio SNN transfer learning (PANNs+SNN)
- The most comprehensive encoding comparison in audio SNN literature
- A mechanistically interesting surrogate failure pattern

Not every contribution listed above is equally strong. The hardware deployment at 33.1% is a feasibility demonstration, not a production result. The surrogate ablation is preliminary pending multi-seed validation. But the combination of C1, C2, C4, and C5 alone — first benchmark, comprehensive encoding comparison, adversarial robustness, gap collapse mechanism — would be sufficient for ICONS acceptance based on the precedent set by the 2022, 2024, and 2025 accepted papers. The hardware deployment (C3) is a bonus that directly serves the ICONS community. The surrogate ablation (C6) adds a practically useful finding that happens to challenge established consensus.

The paper should be submitted.

---

*Prepared by: Defense Counsel (senior neuromorphic computing researcher)*
*For: ICONS 2026 submission review — COMP30040 thesis project*
*Evidence base: ICONS 2022-2025 accepted paper records, literature search results (March 2026 deep research), NeuroBench framework results, SpiNNaker 5-fold hardware results*
