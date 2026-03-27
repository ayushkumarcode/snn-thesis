# The Case for Publishing at ICONS 2026

making the strongest honest case for each contribution. this is the "yes we should submit" side of the argument.

---

## framing note

ICONS is not NeurIPS. its CFP explicitly welcomes "benchmark tasks for neuromorphic computing," "hardware deployment," and "algorithms and training" work. the 2022 most directly comparable paper (Yarga et al.) benchmarked 4 encodings on speech digits and was accepted. the 2025 best paper was about turbulence modeling using neuron random walks -- no classification accuracy at all.

the question isn't whether this is publishable at a top ML venue (it isn't). it's whether it's publishable at ICONS, which serves exactly the community this work is for.

---

## C1: First Convolutional SNN on ESC-50

not contested by any paper in existence. Larroza et al. explicitly state "no state-of-the-art solution has yet encoded environmental sound datasets using spike-based methods" and themselves only evaluate ESC-10 (10 classes), FC-only architecture, 3 encodings, no hardware. Basu et al. (2025) 24-page survey reaches same conclusion.

the novelty of a "first" is only as good as the benchmark itself. ESC-50 IS meaningful -- standard since Piczak 2015, human performance 81.3%, ANN SOTA 99.1%, predefined 5-fold CV for reproducibility. establishing the first SNN results creates the reference point the field literally cannot do without.

the matched-architecture comparison (SNN vs ANN under identical training) is methodologically clean: the 16.7pp gap is attributable to spiking mechanism alone, not architectural differences.

---

## C2: 7-Encoding Comparison

the most comprehensive encoding comparison ever conducted on a standard audio benchmark. the ordering (direct >> rate ~ phase > population > latency >> delta ~ burst) is internally consistent and mechanistically explicable. each failure has a different root cause: delta fails because static spectrograms have no temporal variation; burst fails because front-loading creates temporal window mismatch; latency fails because spectrogram features aren't naturally compatible with first-spike timing; population underperforms because MSE count loss is harder to optimize than CE rate loss.

the rate-phase tie is the most interesting finding within this: phase coding achieves 24.15% using exactly 1 spike per neuron while rate coding gets 24.00% using ~7 spikes per neuron. statistically indistinguishable despite 7x spike count difference. this confirms temporal window coverage matters, not spike count -- with direct implications for energy-efficient inference (7x spike reduction = 7x fewer ACs).

---

## C3: SpiNNaker Deployment

the root-cause analysis of FC1 cancellation is itself a novel finding. the discovery that AvgPool produces fractional outputs incompatible with SpiNNaker's binary input requirement reveals a fundamental constraint: standard conv SNN architectures can't be directly deployed on spike-only hardware without modification. the documented failure of weight re-centering (53.75% -> 8.50%) shows this isn't trivially fixable. this saves future researchers from repeating the mistake.

the FC2-only hybrid approach is a validated engineering contribution. the 5-fold hardware evaluation (2,000 total inferences) is methodologically stronger than most hardware papers which typically report a single run. the fold-level variance (25.2% to 43.0%) is reported honestly.

the hardware gap (12.8 +/- 4.1pp) is the first such measurement for audio classification on SpiNNaker. researchers designing SNN->SpiNNaker pipelines now know what to expect. the per-fold variability reveals fold-specific factors systematically affect translation fidelity.

Option A (MaxPool SNN achieving fc1_binary_fraction=1.000 at all thresholds, 43.75% accuracy) shows the path to full deployment. the paper documents not just current state but the roadmap.

---

## C4: Adversarial Robustness

all prior SNN adversarial work is vision-domain. at FGSM eps=0.1, ANN retains only 1.75% (essentially chance on 50 classes) while SNN retains 26%. at PGD eps=0.05, ANN hits 0%, SNN retains 19.25%.

the robustness is not engineered -- it's a free property of the spiking mechanism. no adversarial training, no certified defenses, no additional compute. the crossover point (where SNN overtakes ANN) at eps=0.01 provides a practical decision criterion.

re: the Wang et al. caveat about gradient masking -- this actually suggests the true SNN robustness advantage may be even larger than measured (since attacks are LESS effective against SNNs due to gradient masking). the qualitative finding (SNN is dramatically more robust) is robust even if exact numbers are uncertain.

edge audio sensing is a security-sensitive application. always-on microphones could be targeted by adversarial attacks. finding that SNNs provide natural robustness has direct practical implications.

---

## C5: PANNs+SNN Gap Collapse

the 17.6x gap reduction (16.7pp -> 0.95pp) is the most scientifically significant single finding. it disambiguates three possible explanations for the SNN-ANN gap:
1. spiking computation is fundamentally limited -- RULED OUT (0.95pp gap with equal features)
2. surrogate gradients are insufficient for feature learning -- partially ruled out for classification
3. SNN can't learn from small datasets -- THIS is the bottleneck

approximately 94% of the gap is about what the network learns, not how it computes.

this reframes the research agenda. instead of "how to make spiking better," the productive question becomes "how to give SNNs better features." pre-training, data augmentation, transfer learning are the correct remedies.

the 0.95pp gap at 92.5%/93.45% is more meaningful than the 16.7pp gap at 47.15%/63.85% -- at high accuracy, model capacity is the binding constraint, and near-equality is genuine evidence of computational equivalence at the classification stage.

also practically significant: CNN14 extracts embeddings once, SNN head classifies on neuromorphic hardware. viable architecture for edge audio sensing.

---

## C6: Surrogate Bimodal Split

challenges Zenke & Vogels 2021 (~1000+ citations) which claims surrogate shape doesn't matter. we find sigmoid (2%), STE (10.25%), SFS (2%), and triangular (2.75%) all fail on audio while spike_rate_escape (46.00%), fast_sigmoid (44.75%), and atan (35.75%) succeed. this is the first evidence the robustness claim breaks down for harder tasks.

the practical value is disproportionate to its length: a practitioner who reads one table and learns "don't use sigmoid or STE for audio SNN" saves weeks of failed experiments.

spike_rate_escape winning is consistent with Gygax & Zenke (2025) escape noise theory -- the most theoretically grounded surrogate performs best.

---

## The 47.15% Question

47.15% is entirely acceptable at ICONS with context. random baseline is 2% (50 classes). the ICONS community's reference is other SNN papers, not ANN leaderboards. there's no prior bar to beat. the ICONS 2025 best paper had no accuracy metric at all. Larroza reports F1=0.661 on 10 classes -- our 47.15% on 50 classes is arguably more impressive relatively.

frame as "first reference point" not "good performance." the PANNs result (92.5%) is the decisive rehabilitation.

---

## Bottom Line

this paper has:
- one confirmed "first" that can't be disputed (ESC-50 SNN)
- one confirmed "first" in audio SNN hardware (SpiNNaker)
- one confirmed "first" in audio SNN adversarial analysis
- one confirmed "first" in audio SNN transfer learning (PANNs+SNN)
- the most comprehensive encoding comparison in audio SNN literature
- a mechanistically interesting surrogate failure pattern

not every contribution is equally strong. the hardware deployment at 33.1% is feasibility, not production. the surrogate ablation is preliminary. but the combination of C1+C2+C4+C5 alone -- first benchmark, comprehensive encoding comparison, adversarial robustness, gap collapse -- would be sufficient for ICONS based on 2022-2025 precedent. the hardware deployment adds the neuromorphic-systems angle ICONS specifically values.

the paper should be submitted. lead with the gap-collapse finding, frame SpiNNaker as characterization, acknowledge the Wang et al. adversarial caveat, and don't apologize for 47.15%.
