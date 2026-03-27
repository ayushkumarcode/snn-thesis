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

---

## 8.3 limitations and reflections

Three main limitations:

1. **Small training set.** 1600 samples/fold isn't enough for deep learning from scratch. SNN accuracy reflects this data limitation as much as any spiking property. Larger datasets (FSD50k, UrbanSound8K) would clarify.

2. **Partial neuromorphic deployment.** FC2-only hybrid delivers only the final layer on SpiNNaker. True energy efficiency needs full forward pass on-chip. Option A retraining confirms the fix (fc1_binary_fraction=1.000, threshold=3.0 gets 43.75% with 956 active/step). Full FC1+FC2 SpiNNaker deployment is the remaining step.

3. **Single-seed surrogate ablation.** 7 testable surrogates, fold 1, 1 seed. LSO crashed (Python 3.14 thing). 3-seed CSF3 run pending. Bimodal learning/failure split is clear from single seed though.

---

## 8.4 future work

**Near-term:**

1. **Option A full deployment:** fold 4 validated, threshold=3.0, fc1_binary=1.000. Need 5-fold retraining and SpiNNaker hardware test.

2. **CSF3 results retrieval:** augmented training and surrogate ablation 3-seed results still on CSF3.

**Longer-term:**

3. **SpiNNaker2** (Hoppner et al. 2024): 22nm FDSOI, 10x better energy, native conv support. Would resolve AvgPool barrier.

4. **STDP unsupervised pre-training** on unlabelled audio for richer conv initialisation without external pretrained models.

5. **Temporal coding losses:** first-spike loss (correct class fires earliest) instead of rate loss. Could get competitive accuracy with fewer timesteps (T=5 instead of 25, major energy saving).

6. **Learnable LIF params:** make beta a learnable per-neuron parameter. Supported by snnTorch, could improve accuracy without arch changes.

7. **Spiking transformers (SpikFormer):** emerging spiking attention (Zhou et al. 2023, NeurIPS) near-ANN on images. Natural extension to audio.

8. **Online streaming audio:** 16ms frames in real time, live event streams to SpiNNaker via UDP. Would demonstrate practical deployment.

9. **SA-PGD adversarial evaluation** (Wang et al. 2025) for rigorous robustness numbers replacing our upper bounds.

10. **Larger benchmarks:** FSD50k (51k clips), UrbanSound8K (8.7k clips) to test whether encoding hierarchy generalises beyond ESC-50.

---


10. **Generalisation to larger benchmarks:** FSD50k (51,000 clips, Fonseca et al. 2022) and UrbanSound8K (8,732 clips, Salamon et al. 2014) would test whether the encoding hierarchy (direct > rate ≈ phase > latency >> delta ≈ burst) generalises beyond ESC-50.

---

## 8.5 Final Statement

Spiking neural networks for environmental sound classification are not yet competitive with ANNs when trained from scratch on small datasets. This thesis establishes the baseline numbers, documents the encoding methods that work and those that do not, deploys the first SNN for environmental sound on neuromorphic hardware, and — perhaps most importantly — shows that the SNN-ANN gap collapses when good features are available.

The energy argument for SNNs remains nuanced: SNNs are cheaper per operation on neuromorphic hardware, but require more total operations than equivalent ANNs in current implementations. The adversarial robustness of SNNs is not nuanced: it is dramatic, reproducible, and potentially decisive for secure edge audio applications.

This work provides the first complete picture of SNN capability on a standard audio benchmark, from feature encoding to neuromorphic hardware deployment. Future work building on these results has clear directions: better features, hardware-native architectures, and temporal coding objectives that exploit the temporal nature of spiking computation that current rate-coded approaches leave untapped.

---

*Appendices follow: A (Full results tables), B (Confusion matrices), C (SpiNNaker parameter tables), D (Reproducibility statement and GitHub link)*
