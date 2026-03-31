# ICONS 2026 Rebuttal Preparation

Pre-prepared responses to anticipated reviewer criticisms. Each response follows the structure: restate the concern, acknowledge its validity, provide the factual rebuttal, and offer a concrete revision if appropriate.

---

## Criticism 1: "60% accuracy is too low for practical use"

**Anticipated phrasing:** "The best SNN accuracy of 61.10% on ESC-50 is far below human performance (81.3%) and ANN SOTA (99%+). This limits practical applicability."

**Response:**

We agree that 61.10% is insufficient for standalone deployment in safety-critical applications. However, we respectfully note three points:

1. **This is the first SNN evaluated on ESC-50.** No prior work exists for comparison. The closest prior work (Larroza et al. 2025) achieves 69% on ESC-10 (10 classes) with a fully-connected architecture. Our 61.10% on ESC-50 (50 classes, 5x harder) with a convolutional SNN is a reasonable first result that establishes a benchmark for future work.

2. **The accuracy gap is a feature-learning problem, not a spiking limitation.** We demonstrate this directly: with pretrained CNN14 features, the SNN achieves 92.50%---only 0.95 pp below the ANN (Table 6). The 16.7 pp scratch-training gap reflects difficulty in learning convolutional filters from 1,600 training samples with surrogate gradients, consistent with Deng & Gu (2020). This diagnosis is itself a contribution.

3. **The paper's central contribution is the pruning-as-regularization finding for hardware deployment, not absolute accuracy.** The +3.1 pp improvement from 50% pruning on SpiNNaker, the 0.6 pp hardware gap at 85% sparsity, and the negative gaps at 60%/80% sparsity are the novel findings. These results hold regardless of absolute accuracy level.

**Revision offered:** We can add a sentence in the Discussion explicitly contextualizing the accuracy relative to the dataset difficulty and positioning the PANNs hybrid result (92.50%) as the practical deployment path.

---

## Criticism 2: "Only FC2 deployed on SpiNNaker, not the full network"

**Anticipated phrasing:** "The hybrid deployment only runs the final FC layer (256 to 50) on SpiNNaker. The convolutional feature extraction still runs on a GPU. This undermines the neuromorphic deployment claim."

**Response:**

This is a fair criticism that we address transparently in the paper. We document the full-deployment failure (Section 3.5) with root-cause analysis: AvgPool produces fractional outputs that cause excitatory-inhibitory cancellation in FC1, resulting in zero hidden spikes. We believe documenting this failure and its root cause is more valuable to the community than omitting the deployment attempt.

Key mitigations:

1. **The hybrid approach is the emerging standard.** Seekings et al. (ICONS 2024) deploy a similar hybrid SNN on Loihi and Jetson, with heavy layers in software and the spiking classifier on neuromorphic hardware. Our approach follows this established paradigm.

2. **We verified the fix.** Replacing AvgPool with MaxPool preserves binary outputs and enables full deployment (verified on fold 4: 43.75% accuracy with threshold=3.0). Full MaxPool-retrained deployment is listed as immediate future work.

3. **FC2 is where the pruning and quantization effects manifest.** The pruning-as-regularization finding, which is the paper's central contribution, concerns FC2 weight quantization on SpiNNaker. Deploying additional layers would not change this finding.

4. **22,000 hardware inferences is substantial.** The deployment scale (5 folds x 11 pruning levels x 400 samples) demonstrates a systematic, reproducible pipeline, not a single proof-of-concept.

**Revision offered:** We can strengthen the Discussion by explicitly quantifying what fraction of total computation runs on SpiNNaker (FC2 = 12,800 of 621,906 parameters, ~2%) and framing this as a validated pipeline that scales to full deployment with the MaxPool fix.

---

## Criticism 3: "Energy numbers are NeuroBench estimates, not measured"

**Anticipated phrasing:** "The energy analysis uses NeuroBench operation counting with Horowitz (2014) energy constants. These are theoretical estimates, not measured power consumption on SpiNNaker hardware."

**Response:**

This is correct, and we state this limitation explicitly (Section 5, Limitation 2). Actual SpiNNaker power measurement requires on-board metering hardware that is not available through remote spalloc access to the Manchester million-core machine.

However:

1. **NeuroBench is the community standard.** Published in Nature Communications (2025) and endorsed by 60+ institutions, NeuroBench defines the methodology we follow. ICONS papers routinely use operation-counting energy estimates rather than board-level measurements.

2. **The relative comparisons are robust.** Even if absolute energy numbers shift with measured values, the relative ordering (pruning reduces energy monotonically) and the pruning ratios (3.5x at 85% sparsity) are determined by operation counts, which are exact.

3. **The Horowitz constants are conservative for neuromorphic hardware.** The 0.9 pJ/AC constant from ISSCC 2014 (45nm CMOS) overestimates SpiNNaker's actual per-AC cost because SpiNNaker 1 uses 130nm technology with different energy characteristics. Our energy estimates are thus conservative in favor of the ANN baseline.

**Revision offered:** We can add a sentence noting that the energy ratios (which are the paper's focus) are invariant to the absolute energy constants, and that measured SpiNNaker power is listed as future work.

---

## Criticism 4: "ESC-50 is a small dataset"

**Anticipated phrasing:** "ESC-50 contains only 2,000 recordings. Results may not generalize to larger, more realistic datasets like UrbanSound8K (8,732 clips) or FSD50K (51,197 clips)."

**Response:**

ESC-50's size is both a limitation and a deliberate choice:

1. **ESC-50 is the standard benchmark for environmental sound classification.** Piczak (2015) has 4,000+ citations. Its predefined 5-fold CV protocol ensures reproducibility and direct comparability with the 200+ papers that use it. UrbanSound8K and FSD50K use different evaluation protocols.

2. **The small size makes it a harder test for SNNs.** With only 1,600 training samples (400 per fold held out), learning good convolutional features with surrogate gradients is genuinely difficult. Larger datasets would likely improve SNN accuracy by providing more training signal---the gap-collapse result with CNN14 features (trained on AudioSet, 2M clips) confirms this.

3. **The hardware deployment findings are architecture-agnostic.** Pruning-as-regularization for SpiNNaker deployment is a property of integer weight quantization interacting with sparse weight matrices. This phenomenon should generalize to any dataset deployed through the same pipeline.

4. **Cross-dataset validation is listed as future work.** We explicitly note (Section 6) that UrbanSound8K and FSD50K testing would strengthen generalization claims.

**Revision offered:** None needed; this limitation is already acknowledged.

---

## Criticism 5: "No comparison with SOTA (99% OmniVec2)"

**Anticipated phrasing:** "The paper compares SNN accuracy to a small ANN baseline (63.85%) rather than to state-of-the-art ESC-50 results (99%+ with large pretrained models). The comparison is misleading."

**Response:**

We deliberately compare against a matched ANN baseline---same architecture (622K params), same training protocol, same data---to isolate the effect of spiking computation. This is the standard methodology in SNN research (Eshraghian et al. 2023, Deng & Gu 2020).

1. **Comparing to OmniVec2 (99%) would be scientifically invalid.** OmniVec2 uses a 300M+ parameter transformer pretrained on ImageNet-21k and fine-tuned with extensive data augmentation. Comparing our 622K-parameter SNN to this would conflate architecture scale, pretraining data, and compute budget with the ANN-vs-SNN question.

2. **We do reference SOTA.** The background section cites OmniVec2 (99%+) and human performance (81.3%) to contextualize our results. The PANNs experiment (92.50% SNN accuracy) shows that with SOTA features, the SNN nearly matches the ANN---closing the gap to 0.95 pp.

3. **The controlled comparison is the paper's methodological strength.** By holding everything constant except the neuron model (LIF vs ReLU), we can make causal claims about the effect of spiking. An uncontrolled comparison against OmniVec2 would not support any conclusions about SNNs.

**Revision offered:** We can add a brief contextualizing sentence in the Results noting the absolute accuracy gap to large pretrained models, while emphasizing that the controlled comparison is the appropriate scientific methodology.

---

## Criticism 6: "The Rhythm SNN is not well motivated theoretically"

**Anticipated phrasing:** "The oscillatory threshold modulation (Eq. 1) is presented as inspired by biological neural oscillations, but the connection is loose. Why would theta-range oscillations help with spectrogram classification? The improvement could be from the reduced timestep count (T=3 vs T=25) rather than the oscillation itself."

**Response:**

This is a perceptive observation that we partially address in the paper but can strengthen.

1. **The biological motivation is suggestive, not causal.** We cite Schuman et al. (2022) for the connection between neural oscillations and auditory processing, but we do not claim a direct mechanistic correspondence. The oscillatory modulation creates learnable windows of increased/decreased excitability, which is functionally useful regardless of biological plausibility.

2. **The learned parameters are consistent with biology.** Frequencies of 2.1--2.9 correspond to theta-range oscillations (4--8 Hz), which are associated with auditory attention and temporal binding in neuroscience. This emerged from training, not from initialization.

3. **Disentangling the contributions is important.** The improvement comes from both the oscillatory modulation and the reduced timestep count. A proper ablation would compare: (a) standard SNN at T=3, (b) Rhythm SNN at T=3, (c) standard SNN at T=25, (d) Rhythm SNN at T=25. Our thesis report includes (a), (c), and (b), showing that T=3 alone does not explain the full improvement (standard SNN at T=3 achieves ~52%, well below Rhythm's 61.10%).

**Revision offered:** We can add the T=3 standard SNN baseline number (from our thesis data) to explicitly separate the contributions of timestep reduction and oscillatory modulation.

---

## Criticism 7: "Pruning improving accuracy is well-known (lottery ticket hypothesis)"

**Anticipated phrasing:** "The observation that pruning improves accuracy is not novel. The lottery ticket hypothesis (Frankle & Carlin, 2019) established that sparse subnetworks can match or exceed dense network performance. Eshraghian et al. (2023) discuss this for SNNs."

**Response:**

We cite the lottery ticket hypothesis connection (Section 5) and acknowledge it. However, our finding is distinct in two important ways:

1. **We observe pruning improving hardware deployment accuracy, not software accuracy.** The lottery ticket hypothesis concerns finding subnetworks that match the original network's accuracy during training. Our finding is that pruning improves accuracy specifically when deploying to integer-quantized neuromorphic hardware. The unpruned model achieves 59.50% in software but only 57.35% on SpiNNaker; 50% pruning improves the SpiNNaker accuracy to 60.45% while software accuracy also improves to 61.85%. The key insight is that the hardware gap shrinks with pruning.

2. **The mechanism is different.** Lottery ticket hypothesis identifies winning tickets via iterative magnitude pruning during training. Our pruning is applied post-training before hardware deployment. The improvement arises from removing small-magnitude weights that are most susceptible to integer quantization error, reducing the quantization noise floor. This is a hardware co-design insight, not a training insight.

3. **The negative hardware gaps are novel.** At 60% and 80% pruning, SpiNNaker exceeds the software simulation. This is not predicted by the lottery ticket hypothesis and suggests a beneficial interaction between integer rounding noise and sparse weight matrices at specific sparsity levels.

**Revision offered:** We can add a sentence explicitly distinguishing our finding from the lottery ticket hypothesis: "Unlike the lottery ticket hypothesis, which concerns software training dynamics, our finding concerns the interaction between pruning and integer weight quantization during hardware deployment."

---

## Criticism 8: "The negative hardware gaps might be noise, not real"

**Anticipated phrasing:** "The negative gaps at 60% and 80% pruning (-0.45 pp and -0.30 pp) are within the standard deviation of the measurements (std = 3.8% and 2.6% respectively). These could be statistical noise rather than a genuine phenomenon."

**Response:**

This is a legitimate concern, and we are careful not to overclaim. The negative gaps are small relative to the fold-level variance. However:

1. **They occur at two separate pruning levels (60% and 80%), not just one.** A single negative gap could be noise; two gaps at different sparsity levels suggest a pattern, even if each individual gap is not statistically significant.

2. **The overall trend is robust.** The monotonic decrease in hardware gap from 2.15 pp (unpruned) to 0.6 pp (85%) across 8 pruning levels, with 22,000 hardware inferences, is statistically robust. The negative gaps are consistent with the smoothly decreasing trend.

3. **We provide a mechanistic hypothesis.** Integer weight rounding introduces noise that, at specific sparsity levels where surviving weights have favorable magnitude distributions, acts as beneficial stochastic quantization. This is consistent with the stochastic resonance effect we independently verified (+0.25 pp with sigma=0.02 noise).

4. **We do not claim statistical significance for the negative gaps specifically.** The paper says "we hypothesize" rather than "we demonstrate." The claim is that the trend exists, not that each individual negative gap is significant.

**Revision offered:** We can add confidence intervals for the gap measurements and explicitly note that the individual negative gaps are not statistically significant, while the overall trend of gap reduction with pruning is robust.

---

## Criticism 9: "Why not use Loihi or other modern neuromorphic hardware?"

**Anticipated phrasing:** "SpiNNaker 1 is a decade-old platform. Why not deploy on Loihi 2 or SpiNNaker 2, which have better precision and on-chip learning?"

**Response:**

1. **Access constraints.** SpiNNaker 1 is the neuromorphic platform available to us through the University of Manchester's APT group, which built and maintains the million-core SpiNNaker machine. Loihi 2 requires an Intel INRC membership that our institution does not hold for this project. SpiNNaker 2 (Hoppner et al. 2024) is not yet available for general research use.

2. **SpiNNaker 1's constraints make our results stronger.** SpiNNaker 1 uses integer-quantized weights and fixed-point membrane potentials. If pruning-as-regularization improves accuracy on this constrained platform, the effect should persist or strengthen on more precise hardware (Loihi 2's 8-bit weights, SpiNNaker 2's ARM Cortex-M4F with FPU).

3. **SpiNNaker deployment for audio is novel.** The only prior SNN audio work on SpiNNaker is Dominguez-Morales et al. (2016) for 8 pure tones. Our 50-class environmental sound deployment is a significant extension regardless of the hardware generation.

4. **Future work explicitly targets SpiNNaker 2.** Section 6 identifies SpiNNaker 2 deployment with MaxPool-retrained architectures as the immediate next step, enabling full on-chip inference.

**Revision offered:** None needed; SpiNNaker 2 is already listed as future work.

---

## Criticism 10: "The 7-encoding comparison is incremental over Yarga et al."

**Anticipated phrasing:** "Yarga et al. (ICONS 2022) compared 4 encodings for speech digits. Adding 3 more encodings (delta, burst, population) on a different dataset is an incremental contribution."

**Response:**

We respectfully disagree that this is merely incremental. The extension is qualitative, not just quantitative:

1. **50-class vs. single-digit classification.** ESC-50 has 50 diverse environmental sound classes across 5 super-categories (animals, natural soundscapes, human non-speech, domestic, urban). This is fundamentally more challenging than speech digit classification, which has ~10 highly structured classes with shared phonemic properties.

2. **Convolutional architecture vs. fully-connected.** Yarga et al. use FC networks; we use a convolutional SNN. The interaction between encoding methods and convolutional feature extraction is unexplored territory---for example, our finding that delta encoding fails on static spectrograms (7.25%) reveals an encoding-architecture interaction not visible in FC networks.

3. **Novel analyses not present in Yarga et al.** We contribute: (a) cross-encoding transfer analysis quantifying encoding specificity (transfer ratio = 0.255, the first such measurement for any SNN), (b) the phase-rate equivalence finding (24.15% vs 24.00%, p=0.93, 7x fewer spikes), and (c) the information preservation principle explaining why direct encoding dominates.

4. **Hardware deployment with multiple encodings.** Yarga et al. do not deploy on hardware. We deploy the direct-encoding trained model on SpiNNaker across 10 pruning levels.

**Revision offered:** We can strengthen the framing by explicitly noting the qualitative differences (task complexity, architecture type, novel analyses) rather than just the numerical extension (7 vs 4 encodings).

---

## Criticism 11: "The adversarial robustness analysis is limited (no SA-PGD)"

**Anticipated phrasing:** "Wang et al. (2025) showed that standard PGD underestimates SNN vulnerability because it does not account for the temporal dynamics. Without SNN-Adapted PGD (SA-PGD), the robustness claims are unreliable."

**Response:**

We cite Wang et al. (2025) explicitly and state: "We acknowledge that standard PGD may underestimate SNN vulnerability; our results represent a conservative lower bound." This was a deliberate design choice:

1. **We use FGSM and PGD as standard baselines.** These are the attacks used by Sharmin et al. (2020, ECCV) in their SNN adversarial robustness analysis on images, which is the most cited prior work in this area. Using the same attack methods enables direct comparison with established results.

2. **SA-PGD was published after our experimental design was finalized.** Wang et al. appeared on arXiv in late 2025. Implementing SA-PGD requires adapting the attack to temporal dynamics (generating adversarial examples that account for membrane potential accumulation across timesteps), which is non-trivial for audio spectrograms.

3. **The 6.0x robustness advantage is a relative comparison.** Both SNN and ANN are attacked with the same FGSM/PGD methods. Even if SA-PGD reduces the SNN's absolute robustness, the relative SNN-vs-ANN comparison under standard attacks is valid and informative.

4. **We frame this as a first analysis, not a definitive claim.** The paper explicitly states this is "the first such analysis on audio spectrograms." The contribution is establishing the direction, not exhaustively characterizing the attack surface.

**Revision offered:** We can add SA-PGD evaluation as explicit future work in the Discussion, and soften the robustness language from "6.0x more robust" to "6.0x more robust under standard FGSM attack."

---

## Criticism 12: "What about structured pruning?"

**Anticipated phrasing:** "Unstructured pruning does not translate to real speedup on most hardware. Structured pruning (removing entire neurons or channels) would be more hardware-relevant for SpiNNaker's core-level parallelism."

**Response:**

Correct. Unstructured pruning zeroes individual weights, which on conventional hardware does not reduce computation unless sparse matrix operations are supported. However:

1. **Neuromorphic hardware natively exploits unstructured sparsity.** On SpiNNaker, each synapse is processed independently. A zero-weight synapse generates no synaptic event, directly saving energy and computation. This is fundamentally different from GPU/CPU execution where unstructured zeros in dense matrices still require memory accesses. Unstructured pruning is the appropriate pruning strategy for spike-driven neuromorphic hardware.

2. **SpiNNaker's core allocation already provides structured sparsity benefits.** With 32 neurons per core, reducing the number of active neurons (via neuron-level pruning) would reduce core count. However, our FC2 layer has only 50 output neurons---already a single core in most configurations. Structured pruning of FC2 would mean reducing the number of output classes, which is not meaningful.

3. **Structured pruning of convolutional layers is future work.** For a full SpiNNaker deployment (after MaxPool retraining), structured pruning of Conv layers (removing entire filters) would reduce the number of cores needed and is a natural extension.

**Revision offered:** We can add structured pruning to the future work section explicitly, noting its relevance for full-network deployment on SpiNNaker.

---

## Criticism 13: "The PANNs result shows SNNs aren't needed"

**Anticipated phrasing:** "The PANNs+Linear baseline achieves 93.80%, outperforming PANNs+SNN (92.50%). If a linear classifier suffices with good features, what is the point of the SNN classifier head?"

**Response:**

This is an important observation that we believe strengthens rather than weakens the paper's message:

1. **The PANNs experiment is a diagnostic tool, not a deployment recommendation.** Its purpose is to isolate where the SNN-ANN gap originates (feature learning vs. spiking classification). The answer---that the gap is almost entirely in feature learning---is a scientific contribution that informs future SNN research directions.

2. **The SNN head enables neuromorphic deployment.** A linear classifier achieves 93.80% but requires conventional hardware for inference. The SNN head (92.50%) can run on SpiNNaker with only 0.95 pp loss, at 3.5x lower energy with 85% pruning. In an always-on edge device, the total system energy matters more than the classifier head's accuracy.

3. **The practical path forward combines both insights.** CNN14 features in software + pruned SNN head on SpiNNaker = 92.50% accuracy + neuromorphic energy efficiency. This hybrid paradigm is explicitly what Seekings et al. (ICONS 2024) advocate.

4. **SNNs offer additional benefits beyond accuracy.** Our adversarial robustness analysis shows 6.0x greater robustness under FGSM attack and 4.8 pp less catastrophic forgetting---properties that a linear classifier does not provide.

**Revision offered:** We can add a sentence framing the PANNs result as motivating the specific hybrid deployment architecture rather than as evidence against SNNs.

---

## Criticism 14: "Statistical significance with n=5 folds is weak"

**Anticipated phrasing:** "ESC-50's 5-fold CV gives only 5 data points per condition. With n=5, statistical tests have very low power. Many of the reported p-values may not survive correction for multiple comparisons."

**Response:**

This is a valid statistical concern inherent to the ESC-50 evaluation protocol:

1. **The 5-fold protocol is mandatory for ESC-50 comparability.** Piczak (2015) defined the 5 predefined folds; all ESC-50 papers use this protocol. Using a different evaluation scheme would prevent comparison with the 200+ papers on this benchmark.

2. **We use appropriate statistical tests.** Paired t-tests exploit the within-fold pairing to maximize power. The SNN-vs-ANN comparison yields p=0.001 (Table 3), which survives even aggressive Bonferroni correction for the number of comparisons we report.

3. **We complement p-values with effect sizes and standard deviations.** Every result table reports mean +/- std across folds. The pruning-as-regularization finding (+3.1 pp at 50% sparsity) is observed consistently across all 5 folds, not driven by a single outlier.

4. **22,000 hardware inferences provide substantial sample-level evidence.** While the 5-fold aggregation gives n=5 means, each mean is computed from 400 individual hardware inferences. The within-fold accuracy estimates are thus precise; the fold-level variance reflects genuine data partitioning effects.

5. **We acknowledge where significance is lacking.** The phase-rate equivalence (p=0.93) is reported as non-significant, and the negative hardware gaps are presented as hypotheses, not claims.

**Revision offered:** We can add a brief note on statistical power limitations and report confidence intervals alongside p-values.

---

## General Rebuttal Strategy

### Tone
- Respectful and constructive throughout
- Acknowledge valid concerns before responding
- Distinguish between limitations we can address in revision and fundamental constraints (ESC-50 fold count, SpiNNaker access)
- Offer concrete revisions where appropriate

### Key Talking Points to Emphasize
1. **First SNN on ESC-50** --- verified novelty, no prior work
2. **Pruning-as-regularization for hardware deployment** --- distinct from lottery ticket hypothesis, specific to neuromorphic integer quantization
3. **22,000 hardware inferences** --- comprehensive and reproducible, not a single demo
4. **Honest about limitations** --- FC2-only deployment, NeuroBench energy estimates, SA-PGD absence all acknowledged in the paper
5. **ICONS community fit** --- builds directly on Yarga et al. (ICONS 2022) and Seekings et al. (ICONS 2024), targets audio applications + hardware deployment + benchmarking

### Common Reviewer Psychology
- Reviewers respect transparency about limitations more than attempts to hide them
- Providing specific numbers in rebuttals (not vague qualifiers) builds credibility
- Offering to add ablations or additional baselines shows willingness to improve
- Connecting to prior ICONS papers shows community awareness
