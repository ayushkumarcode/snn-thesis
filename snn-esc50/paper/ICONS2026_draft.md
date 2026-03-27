# DRAFT: Convolutional Spiking Neural Networks for Environmental Sound Classification: A Comprehensive Evaluation on ESC-50

**Target:** ICONS 2026 (ACM International Conference on Neuromorphic Systems)
**Deadline:** April 1, 2026
**Format:** 8 pages, ACM proceedings
**Status:** DRAFT -- core results final as of 4 March. Still need to tidy up the surrogate ablation section once CSF3 3-seed results come back. writing style needs another pass before submission.

---

## Abstract

Environmental sound classification (ESC) is a challenging audio task with applications in smart environments, surveillance, and robotics. SNNs offer a biologically-plausible, energy-efficient alternative to conventional deep learning, but their effectiveness on complex audio benchmarks remains basically unexplored. We present the first systematic evaluation of convolutional SNNs on ESC-50 (50 classes, 5-fold CV), comparing seven spike encoding methods and deploying the trained network on SpiNNaker neuromorphic hardware. Our scratch-trained conv SNN achieves 47.15% +/- 4.50% with direct encoding -- a 16.7 pp gap vs the matched ANN baseline (63.85% +/- 3.07%). This gap collapses to under 1 pp (92.50% vs 93.45%) when frozen AudioSet-pretrained features (PANNs) replace raw spectrograms, showing the bottleneck is feature learning, not spiking computation. Despite lower clean accuracy, SNNs show dramatically superior adversarial robustness: at FGSM eps=0.1, SNN retains 26.00% vs ANN's 1.75%. NeuroBench energy analysis (5-fold) shows SNN performs 1.08M ACs per sample (968 +/- 37 nJ in simulation) vs ANN's 101K MACs (454 +/- 11 nJ); SNN overhead from T=25 timesteps. On neuromorphic hardware ACs cost 5.1x less per op than MACs. We deploy on SpiNNaker via a validated hybrid approach; 20-sample pilot = 40%, 400-sample validation = **43.0% SpiNNaker vs 51.25% snnTorch -- 8.25 pp gap, 64.5% agreement**. These constitute the first SNN evaluation on ESC-50, the first SNN hardware deployment for environmental sound, and the first adversarial robustness analysis of SNNs on audio.

todo: abstract is too long for 8-page format. need to cut ~30%

---

## 1. Introduction

The proliferation of always-on audio sensing at the edge demands both accuracy and efficiency. CNNs achieve near-human accuracy on environmental sound benchmarks [1] (human 81.3%, SOTA 98.25%), but their energy cost prohibits deployment on battery-constrained neuromorphic devices. SNNs communicate via binary spikes rather than continuous activations; on dedicated hardware like Loihi [15] or SpiNNaker [2], SNNs execute as accumulate-only (AC) ops rather than MACs, yielding 5-10x energy reductions when spike rates are low enough [8].

Despite extensive SNN research on images [3], **no prior work has evaluated conv SNNs on the full ESC-50 benchmark**. Closest is [9] which uses ESC-10 (10 classes) with FC layers only, getting ~60% with direct encoding. Dominguez-Morales et al. [20] put SNNs on SpiNNaker for audio but only evaluated pure tones.

Contributions:
1. First conv SNN evaluation on ESC-50 (50 classes, 2000 recordings, 5-fold)
2. Systematic comparison of 7 spike encoding methods
3. First SpiNNaker deployment for environmental sound, with documented root-cause analysis of FC1 cancellation
4. First FGSM/PGD adversarial analysis of SNNs on audio spectrograms
5. PANNs + SNN head showing gap narrows to <1% with pretraining
6. NeuroBench-compliant energy analysis [7]

Key finding: **the SNN-ANN accuracy gap isn't fundamental** -- it collapses from 16.7 pp to <1 pp with pretrained features, suggesting the bottleneck is representation learning not the spiking formalism. Meanwhile SNNs exhibit natural properties (adversarial robustness, hardware compatibility) that make them compelling for edge audio.

---

## 2. Background

### 2.1 ESC-50

ESC-50 [1]: 2000 five-second recordings, 50 classes, 5-fold CV. Human 81.3%, ANN SOTA 98.25% [14]. We use log-mel spectrograms (64 bins, n_fft=1024, hop=512, sr=22050), normalised to [0,1], giving 64x216 inputs.

### 2.2 Spiking neural networks

LIF neurons accumulate weighted input, emit binary spikes at threshold, reset. Surrogate gradients [4] approximate the spike derivative for backprop; we use fast sigmoid (slope=25). Training via snnTorch [3], CE on summed membrane over T=25 timesteps.

### 2.3 Encoding methods

| Encoding | Description | Key property |
|----------|-------------|--------------|
| Rate | spike prob proportional to intensity | stochastic |
| Latency | high intensity = earlier spike | temporal precision |
| Delta | spikes on intensity changes | sparse, change-sensitive |
| Direct | continuous values repeated T times | no conversion |
| Burst | n_spikes proportional to intensity, front-loaded | dense early |
| Phase | spike time within oscillation cycle | deterministic, 1 spike/neuron |
| Population | 10 output neurons/class (500 total) | multi-neuron vote |

### 2.4 SpiNNaker

SpiNNaker [2]: massively parallel neuromorphic platform, IF_curr_exp neurons, integer weights, spike-driven (AC only). Calibrated: tau_syn=5ms, v_thresh=1.0, tau_m=20ms.

---

## 3. Architecture and training

**SpikingCNN:**
```
Conv2d(1->32, k=3) -> BN -> MaxPool(2) -> LIF1
Conv2d(32->64, k=3) -> BN -> MaxPool(2) -> LIF2
AvgPool(4x6)
Linear(2304->256) -> LIF3 -> Linear(256->50) -> LIF4
```
~622K params. ANN mirror with ReLU.

**Training:** Adam (lr=1e-3, wd=1e-4), ReduceLROnPlateau, early stopping (patience=10), 50 epochs, batch=32, standard 5-fold CV.

**Augmentation (negative result):** SpecAugment [19] + TimeShift on 100 epochs: SNN 40.75% +/- 16.03% (-6.40 pp vs baseline), ANN 61.70% (-2.15 pp). Hurts both models on this small dataset. Baseline results used throughout. Mechanistic explanation in thesis ch4: early stopping too aggressive + LIF threshold interacts badly with mean-value masks.

---

## 4. Results

### 4.1 Encoding comparison

| Encoding | Mean Acc | Std | SNN/ANN ratio |
|----------|----------|-----|---------------|
| **ANN baseline** | **63.85%** | 3.07% | -- |
| Direct | **47.15%** | 4.50% | 73.8% |
| Phase | 24.15% | 1.66% | 37.8% |
| Rate | 24.00% | 1.90% | 37.6% |
| Population | 19.15% | 2.79% | 30.0% |
| Latency | 16.30% | 1.62% | 25.5% |
| Delta | 7.25% | 0.94% | 11.4% |
| Burst | 6.50% | 1.54% | 10.2% |

All 7 complete. Direct consistently best. The 16.7 pp gap is significant (paired t: p=0.001; Wilcoxon: p=0.0625, minimum with n=5).

**Key finding: phase (24.15%) ties with rate (24.00%).** Phase = exactly 1 spike/neuron deterministically; rate = ~7 spikes/neuron stochastically. Same accuracy with 6-7x fewer spikes. Confirms the **information preservation principle** -- temporal window coverage matters more than spike count.

Delta fails (7.25%): static spectrograms have no temporal variation. Burst fails (6.50%): front-loading spikes into 5/25 timesteps creates window mismatch with LIF integration. Severe overfitting (train 45-62%, test 5-9%).

### 4.2 Surrogate gradient ablation

fold 1, seed=42, 8 surrogates from snnTorch 0.9.4. 3-seed CSF3 run pending.

| Surrogate | Best Acc | Best Epoch |
|-----------|----------|------------|
| fast_sigmoid | 44.75% | 50 |
| atan | 35.75% | 49 |
| sigmoid | 2.00% (early stop) | 1 |
| ste | 10.25% (early stop) | 1 |
| triangular | 2.75% (early stop) | 13 |
| spike_rate_escape | **46.00%** | 50 |
| lso | CRASHED | -- |
| sfs | 2.00% (early stop) | 1 |

**Bimodal split:** {sre, fast_sigmoid, atan} learn (35-46%) vs {STE, sigmoid, sfs, triangular} fail (2-10%). Stronger shape effect than Zenke & Vogels [4] predicted.

### 4.3 PANNs + SNN head

Frozen CNN14 [6] embeddings (2048-d), 3-layer SNN head.

| Model | Mean | Std |
|-------|------|-----|
| PANNs + SNN | **92.50%** | 1.30% |
| PANNs + ANN | 93.45% | 1.54% |
| PANNs + Linear | 93.80% | 1.69% |

Gap collapses from 16.7 pp to <1 pp. SNN can classify competitively when given good features. The scratch-training gap is feature learning, not spiking computation.

---

## 5. Adversarial robustness

FGSM [11] and PGD [12] (torchattacks, fold 4, 400 samples), 7 epsilon values. First such analysis on audio SNNs.

| eps | FGSM SNN | FGSM ANN | PGD SNN | PGD ANN |
|-----|----------|----------|---------|---------|
| 0.00 | 53.75% | 68.75% | 53.75% | 68.75% |
| 0.01 | 37.50% | 22.50% | 23.50% | 14.75% |
| 0.02 | 32.00% | 8.75% | 20.50% | 2.00% |
| 0.05 | 29.00% | 2.50% | 19.25% | 0.00% |
| 0.10 | **26.00%** | **1.75%** | 6.25% | 0.00% |
| 0.20 | 21.50% | 1.25% | 1.25% | 0.00% |
| 0.30 | 20.75% | 0.75% | 1.25% | 0.00% |

At eps=0.1 FGSM: SNN keeps 26.00% vs ANN's 1.75% (14.9x more robust). Binary spike thresholding = natural gradient masking. Note: SA-PGD [21] recommended for future work as standard PGD may underestimate SNN vulnerability.

---

## 6. SpiNNaker deployment

### 6.1 FC1 cancellation problem

AvgPool between LIF2 and FC1 produces fractional outputs, breaking SpiNNaker's binary spike requirement. Weight re-centering (Option C) failed: 53.75% -> 8.50% because compensation assumes binary inputs.

### 6.2 FC2-only hybrid

Conv + FC1 + LIF3 in software -> binary hidden spikes (256-d, 21.7% active). Only FC2 (256->50) runs on SpiNNaker.

**Hardware:** SpiNN-5, IF_curr_exp, tau_syn=5ms, v_thresh=1.0, tau_m=20ms, weight_scale=1.0.
**Pilot (Run 5, n=20):** 40%.
**Full validation (Run 6, n=400):** SpiNNaker 43.0% vs snnTorch 51.25% (8.25 pp gap, 64.5% agreement).
**5-fold (2000 inferences):** SpiNNaker 33.1% +/- 6.9% vs snnTorch 46.0% (gap 12.8 +/- 4.1 pp). Per fold: F1=29.0%, F2=32.0%, F3=36.5%, F4=43.0%, F5=25.2%. Confirms hybrid approach generalises across all 5 folds.

### 6.3 Energy analysis (NeuroBench [7])

| Model | Ops/sample | Energy/sample | Type |
|-------|-----------|---------------|------|
| SNN | 1.08M ACs | 968 +/- 37 nJ (5-fold) | AC x 0.9 pJ |
| ANN | 101K MACs | 454 +/- 11 nJ (5-fold) | MAC x 4.6 pJ |

In software ANN is 2.1x cheaper (T=25 overhead). On neuromorphic hardware each AC is 5.1x cheaper than MAC but SNN has more total ops so still costs more at our 25.8% spike rate. Dampfhoffer et al. [8]: need <6.4% spike rate to beat quantised ANNs. We're at 25.8%.

todo: should probably add a sentence about how this motivates the energy reduction work we're doing

---

## 7. Discussion

**Why direct wins:** feeds continuous values every timestep; LIF does its own rate computation. Rate discards magnitude by Poisson sampling; latency loses temporal richness; delta amplifies noise.

**Gap is feature learning, not SNN limitation.** PANNs confirms: given AudioSet features, SNN ~ ANN (92.50% vs 93.45%). The 16.7 pp scratch gap is from the SNN's difficulty learning discriminative conv filters from 1600 samples. Consistent with Deng & Gu [13].

**FC1 cancellation as co-design insight.** Standard conv SNN architectures (conv -> pool -> FC) aren't directly deployable on spike-only hardware without AvgPool removal. Novel, practically important constraint. Option A (MaxPool retraining) confirms fix: fc1_binary_fraction=1.000, threshold=3.0 gets 43.75% with 956 active/step.

**Adversarial robustness.** The 14.9x advantage at FGSM eps=0.1 suggests binary thresholding provides free robustness. Implications for audio security at the edge.

---

## 8. Conclusions

First conv SNN evaluation on ESC-50: 47.15% vs 63.85% ANN (16.7 pp gap), collapsing to <1 pp with AudioSet pretraining (92.50% vs 93.45%). SNNs show 14.9x greater adversarial robustness under FGSM. Deployed on SpiNNaker via FC2-only hybrid: 43.0% SpiNNaker vs 51.25% snnTorch (8.25 pp, 64.5% agreement) -- first neuromorphic deployment for environmental sound. NeuroBench: 968 +/- 37 nJ (SNN, 1.08M ACs) vs 454 +/- 11 nJ (ANN, 101K MACs) in simulation. Systematic encoding comparison establishes direct as the winner. Future work: SpiNNaker2, STDP pre-training, larger benchmarks.

---

## key numbers (for my reference, don't include in submission)

| Metric | Value | Source |
|--------|-------|--------|
| ANN 5-fold | 63.85% +/- 3.07% | results/ann/none/summary.json |
| SNN direct 5-fold | 47.15% +/- 4.50% | results/snn/direct/summary.json |
| SNN rate | 24.00% +/- 1.90% | results/snn/rate/summary.json |
| SNN latency | 16.30% +/- 1.62% | results/snn/latency/summary.json |
| SNN delta | 7.25% +/- 0.94% | results/snn/delta/summary.json |
| SNN burst | 6.50% +/- 1.54% | results/snn/burst/ |
| SNN phase | 24.15% +/- 1.66% | results/snn/phase/summary.json |
| SNN population | 19.15% +/- 2.79% | results/snn/population/summary.json |
| PANNs+SNN | 92.50% +/- 1.30% | results/panns/ |
| PANNs+ANN | 93.45% +/- 1.54% | same |
| FGSM eps=0.1 SNN | 26.00% | results/adversarial/ |
| FGSM eps=0.1 ANN | 1.75% | same |
| PGD eps=0.05 SNN | 19.25% | same |
| PGD eps=0.05 ANN | 0.00% | same |
| SNN energy | 968 +/- 37 nJ (5-fold) | results/neurobench/ |
| ANN energy | 454 +/- 11 nJ (5-fold) | same |
| SpiNNaker pilot | 40% (8/20) | spinnaker_results/ |
| SpiNNaker Run 6 | 43.0% SpiNN, 51.25% snnTorch, 8.25 pp gap | same |
| Human | 81.3% | Piczak 2015 |
| SOTA | 98.25% | leaderboard |

---

## References

1. **Piczak, K.J. (2015).** ESC: Dataset for Environmental Sound Classification. ACM MM'15, 1015-1018.

2. **Furber, S.B., Galluppi, F., Temple, S., & Plana, L.A. (2014).** The SpiNNaker Project. Proc. IEEE, 102(5), 652-665.

3. **Eshraghian, J.K., Ward, M., Neftci, E.O., Wang, X., Liang, G., Linares-Barranco, B., & Lu, W.D. (2023).** Training Spiking Neural Networks Using Lessons From Deep Learning. Proc. IEEE, 111(9), 1016-1054.

4. **Zenke, F., & Vogels, T.P. (2021).** The Remarkable Robustness of Surrogate Gradient Learning for Instilling Complex Function in Spiking Neural Networks. Neural Computation, 33(4), 899-925.

5. **Neftci, E.O., Mostafa, H., & Zenke, F. (2019).** Surrogate Gradient Learning in Spiking Neural Networks. IEEE SPM, 36(6), 51-63.

6. **Kong, Q., Cao, Y., Iqbal, T., Wang, Y., Wang, W., & Plumbley, M.D. (2020).** PANNs: Large-Scale Pretrained Audio Neural Networks. IEEE/ACM TASLP, 28, 2880-2894.

7. **Yik, J., Ahmed, S., Ahmed, Z., ..., & Bengio, Y. (2025).** Neurobench: A Framework for Benchmarking Neuromorphic Computing. Nature Communications, 16, 1589.

8. **Dampfhoffer, M., Mesquida, T., Valentian, A., & Anghel, L. (2023).** Are SNNs Really More Energy-Efficient than ANNs? IEEE TETCI, 7(3), 731-741.

9. **Larroza, A., Reyes, L., Maudes-Raedo, J., & Rodriguez, G. (2025).** Evaluation of Spiking Neural Networks for Audio Classification. arXiv:2503.11206.

10. **Sharmin, S., Rathi, N., Panda, P., & Roy, K. (2020).** Inherent Adversarial Robustness of Deep Spiking Neural Networks. ECCV 2020, 768-784.

11. **Goodfellow, I.J., Shlens, J., & Szegedy, C. (2015).** Explaining and Harnessing Adversarial Examples. ICLR 2015.

12. **Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018).** Towards Deep Learning Models Resistant to Adversarial Attacks. ICLR 2018.

13. **Deng, S., & Gu, S. (2020).** Rethinking the Performance Comparison Between SNNs and ANNs. Neural Networks, 121, 294-307.

14. **Gong, Y., Chung, Y.A., & Glass, J. (2021).** AST: Audio Spectrogram Transformer. Interspeech 2021, 571-575.

15. **Davies, M., Srinivasa, N., Lin, T.H., ..., & Taba, B. (2018).** Loihi: A Neuromorphic Manycore Processor. IEEE Micro, 38(1), 82-99.

16. **Golden, R., Delanois, J.E., Sanda, P., & Bhatt, D.L. (2022).** Sleep Prevents Catastrophic Forgetting in Spiking Neural Networks. PLoS Comp Bio, 18(11), e1010628.

17. **Yousefzadeh, A., Stromatias, E., Soto, M., Serrano-Gotarredona, T., & Linares-Barranco, B. (2019).** On Practical Issues for Stochastic STDP Hardware. Front. Neurosci., 13, 760.

18. **Hoppner, S., Yan, Y., Garbers, C., ..., & Furber, S. (2024).** SpiNNaker 2: A Large-Scale Neuromorphic System. arXiv:2401.04491.

19. **Park, D.S., Chan, W., Zhang, Y., Chiu, C.C., Zoph, B., Cubuk, E.D., & Le, Q.V. (2019).** SpecAugment. Interspeech 2019, 2613-2617.

20. **Dominguez-Morales, J.P., et al. (2016).** Multilayer Spiking Neural Network for Audio Samples Classification Using SpiNNaker. ICANN 2016, LNCS 9886, pp. 45-53.

21. **Wang, J., Zhao, D., Chen, R., Zhang, Q., & Zeng, Y. (2025).** Towards Reliable Evaluation of Adversarial Robustness for Spiking Neural Networks. arXiv:2512.22522.

18. **Hoppner, S., Yan, Y., Garbers, C., ..., & Furber, S. (2024).** SpiNNaker 2: A Large-Scale Neuromorphic System for Event-Based and Asynchronous Machine Learning. *arXiv:2401.04491*.

19. **Park, D.S., Chan, W., Zhang, Y., Chiu, C.C., Zoph, B., Cubuk, E.D., & Le, Q.V. (2019).** SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition. *Interspeech 2019*, 2613–2617.

20. **Dominguez-Morales, J.P., Jiménez-Fernandez, Á., Rios-Navarro, A., Cerezuela-Escudero, E., Gutierrez-Galan, D., Domínguez-Morales, M.J., & Jiménez-Moreno, G. (2016).** Multilayer Spiking Neural Network for Audio Samples Classification Using SpiNNaker. *Artificial Neural Networks and Machine Learning – ICANN 2016*, LNCS Vol. 9886, pp. 45–53. Springer. DOI: 10.1007/978-3-319-44778-0_6.

21. **Wang, J., Zhao, D., Chen, R., Zhang, Q., & Zeng, Y. (2025).** Towards Reliable Evaluation of Adversarial Robustness for Spiking Neural Networks. *arXiv:2512.22522*.

---

*ICONS 2026 deadline: April 1, 2026. ACM format, 8 pages.*
*Target: submit by March 25 to allow revision time.*
