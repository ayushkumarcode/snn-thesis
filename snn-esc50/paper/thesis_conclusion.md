# chapter 8: conclusion and future work

wrapping it all up. need to restate contributions clearly, answer each RQ directly, and give future directions.

---

## 8.1 contributions

Six original contributions:

**C1: First conv SNN evaluation on ESC-50.** SpikingCNN (2 conv + 2 FC, 622K params) gets 47.15% +/- 4.50% on ESC-50 (50 classes, 5-fold). First published conv SNN result on the full benchmark. Prior work (Larroza et al. 2025) used FC-only on 10/50 classes.

**C2: Systematic encoding comparison for audio.** Seven methods under identical conditions: direct (47.15%), rate (24.00%), phase (24.15%), population (19.15%), latency (16.30%), delta (7.25%), burst (6.50%). Ordering explained by information preservation. Rate/phase near-equality (0.15 pp) is novel: deterministic 1-spike timing ~ stochastic ~7-spike counting at T=25. Population underperforms despite 10x output neurons due to MSE loss difficulty. Most comprehensive audio SNN encoding comparison in published literature.

**C3: First SNN deployment on SpiNNaker for environmental sound.** FC2-only hybrid on SpiNN-5. 40% pilot (Run 5, n=20); 400-sample: 43.0% SpiNNaker vs 51.25% snnTorch (8.25 pp gap). 5-fold: 33.1% +/- 6.9% vs 46.0% (gap 12.8 +/- 4.1 pp) -- first 5-fold cross-validated SpiNNaker result for any audio task. FC1 cancellation (AvgPool producing fractional outputs) documented as new failure mode.

**C4: First adversarial robustness analysis of SNNs on audio.** FGSM and PGD across 7 eps values. At eps=0.1 FGSM: SNN 26.00% vs ANN 1.75%. Largest reported SNN adversarial advantage in any audio domain.

**C5: First PANNs + SNN combination.** SNN head on frozen CNN14 embeddings: 92.50% +/- 1.30%, exceeding human (81.3%), gap collapses from 16.70 pp to 0.95 pp. Gap is feature-learning, not spiking computation.

**C6: NeuroBench energy analysis.** 5-fold validated: SNN 968 +/- 37 nJ/sample (1.08M ACs), ANN 454 +/- 11 nJ (101K MACs). SNN 2.1x more expensive in software due to T=25. On neuromorphic hardware ACs cost 5.1x less than MACs. PANNs + SpiNNaker FC2 is Pareto-optimal: 92.50%, ~86 nJ for classification.

---

## 8.2 answers to research questions

**RQ1: Can conv SNNs classify environmental sounds competitively with matched ANNs?**

Partially. From-scratch: 47.15% vs 63.85% -- 16.70 pp gap, p=0.001. SNN learns meaningful audio structure way above chance (2%) but can't match ANN on small data. **With PANNs pretrained features, gap collapses to 0.95 pp (92.50% vs 93.45%).** Answer: not competitively from scratch, but competitively with pretrained features.

**RQ2: Which encoding performs best, and why?**

Direct at 47.15%. Ordering: direct > rate ~ phase > population > latency >> delta ~ burst. Winner determined by information preservation -- how much spectrogram magnitude structure reaches the conv layers in a usable form. Direct preserves everything; rate adds Bernoulli noise; phase uses deterministic single-spike timing across full window (= rate accuracy); population limited by MSE loss; latency clusters spikes early; delta and burst discard most info. The general principle -- information preservation predicts performance -- is the novel theoretical contribution.

**RQ3: Can a trained SNN be deployed on SpiNNaker, what's the cost?**

Yes, with limitations. Full SpikingCNN can't deploy natively due to AvgPool-FC1 cancellation. Hybrid: software features + SpiNNaker FC2 = 40% pilot, 43.0% full validation (8.25 pp gap, 64.5% agreement). 5-fold: 33.1% +/- 6.9%. Gap fluctuates (n=208: 0.0 pp; n=400: 8.25 pp) due to sample variability. Full native deployment needs Option A retraining (MaxPool), which is theoretically unblocked (fold 4 validated).

**RQ4: Do SNNs exhibit natural adversarial robustness?**

Dramatically yes. eps=0.1 FGSM: SNN 26.00% vs ANN 1.75%. eps=0.05 PGD: ANN = 0%, SNN = 19.25%. Spike threshold provides natural filtering by requiring perturbations to cross a hard nonlinearity. First confirmation for environmental sound classification.


**RQ4: Do SNNs exhibit natural adversarial robustness compared to matched ANNs on audio inputs?**

Dramatically yes. At ε=0.1 (FGSM), the SNN retains 26.00% accuracy while the ANN collapses to 1.75%. At ε=0.05 (PGD), the ANN reaches 0% while the SNN maintains 19.25%. The spike threshold provides natural adversarial filtering by requiring perturbations to cross a hard non-linearity to affect the output. This is the first confirmation of this effect for environmental sound classification.

---

## 8.3 Limitations and Reflections

This work has three principal limitations:

1. **Small training set:** 1,600 samples per fold is insufficient for from-scratch deep learning. The SNN accuracy (47.15%) reflects this data limitation as much as any property of spiking computation. Future work with larger datasets (FSD50k, UrbanSound8K) would give a clearer picture.

2. **Partial neuromorphic deployment:** The FC2-only hybrid delivers only the final classification layer on SpiNNaker. True energy efficiency on neuromorphic hardware requires the full forward pass to be executed on-chip. Option A retraining (MaxPool SNN, fold 4 threshold sweep) confirms the architectural fix: fc1_binary_fraction=1.000 for all thresholds, and threshold=3.0 achieves 43.75% accuracy with 956 FC1 active inputs/step. The full FC1+FC2 SpiNNaker deployment with Option A weights is theoretically unblocked; hardware testing is the remaining step.

3. **Single-seed surrogate gradient ablation:** Surrogate gradient ablation (7 testable surrogates, fold 1) is complete with 1 seed (seed=42); results are fully documented in §4.3. LSO crashed due to a Python 3.14/snnTorch 0.9.4 incompatibility. A 3-seed CSF3 run is pending retrieval and would provide tighter variance estimates, but the bimodal learning vs. failure split is already clear from the single seed.

---

## 8.4 Future Work

**Immediate extensions (within the scope of this project):**

1. **SpiNNaker-aware retraining (Option A) — partially completed:** A fold 4 threshold sweep with MaxPool SNN confirms fc1_binary_fraction=1.000 for all thresholds (1.0–3.0). Threshold=3.0 achieves 43.75% test accuracy with 956 FC1 active inputs/step (sparsity 58.5%). Full 5-fold retraining and SpiNNaker hardware testing with the threshold=3.0 model would complete this contribution.

2. **CSF3 results retrieval:** Augmented training (100 epochs, SpecAugment) and surrogate gradient ablation results are pending on CSF3. Retrieving these would complete §4.3 and §4.4.

**Longer-term research directions:**

3. **SpiNNaker2** (Hoppner et al. 2024): The successor platform features 22nm FDSOI process, 10× better energy than SpiNNaker, and native convolutional support. Deployment on SpiNNaker2 would resolve the AvgPool barrier and provide meaningful energy comparisons.

4. **STDP unsupervised pre-training:** Spike-timing-dependent plasticity pre-training on unlabelled audio could provide richer initialisation for the convolutional layers, reducing the data-limitation gap without requiring external pretrained models.

5. **Temporal coding losses:** Training with a first-spike loss (target: correct class fires earliest) rather than the rate loss used here could create SNNs that exploit temporal coding, potentially achieving competitive accuracy with fewer timesteps (T=5 instead of T=25, further reducing energy).

6. **Learnable LIF parameters:** Making β (decay rate) a learnable per-neuron parameter (`nn.Parameter`) allows the network to adapt its temporal integration window to the task. This is supported by snnTorch and could improve accuracy without architectural changes.

7. **Spiking transformers (SpikFormer):** Emerging spiking attention mechanisms (Zhou et al. 2023, NeurIPS) have achieved near-ANN performance on image tasks. Extension to audio with mel spectrograms is a natural next step.

8. **Online streaming audio:** Converting the classification pipeline to process 16ms audio frames in real time, feeding live event streams to SpiNNaker via UDP, would demonstrate the practical utility of the deployment in a realistic scenario.

9. **SA-PGD adversarial evaluation:** Applying Stable Adaptive PGD (Wang et al. 2025) would provide rigorous adversarial robustness numbers, replacing the conservative upper bounds reported in §6.1.

10. **Generalisation to larger benchmarks:** FSD50k (51,000 clips, Fonseca et al. 2022) and UrbanSound8K (8,732 clips, Salamon et al. 2014) would test whether the encoding hierarchy (direct > rate ≈ phase > latency >> delta ≈ burst) generalises beyond ESC-50.

---

## 8.5 Final Statement

Spiking neural networks for environmental sound classification are not yet competitive with ANNs when trained from scratch on small datasets. This thesis establishes the baseline numbers, documents the encoding methods that work and those that do not, deploys the first SNN for environmental sound on neuromorphic hardware, and — perhaps most importantly — shows that the SNN-ANN gap collapses when good features are available.

The energy argument for SNNs remains nuanced: SNNs are cheaper per operation on neuromorphic hardware, but require more total operations than equivalent ANNs in current implementations. The adversarial robustness of SNNs is not nuanced: it is dramatic, reproducible, and potentially decisive for secure edge audio applications.

This work provides the first complete picture of SNN capability on a standard audio benchmark, from feature encoding to neuromorphic hardware deployment. Future work building on these results has clear directions: better features, hardware-native architectures, and temporal coding objectives that exploit the temporal nature of spiking computation that current rate-coded approaches leave untapped.

---

*Appendices follow: A (Full results tables), B (Confusion matrices), C (SpiNNaker parameter tables), D (Reproducibility statement and GitHub link)*
