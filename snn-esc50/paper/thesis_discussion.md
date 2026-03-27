# chapter 7: discussion

this is where i tie everything together and actually answer the research questions. need to be careful to connect the findings rather than just restating results.

---

## 7.1 the accuracy-efficiency trade-off

Core finding: **conv SNN from scratch on ESC-50 gets 47.15% (direct), ANN gets 63.85% -- 16.70 pp gap, p=0.001.** On SpiNNaker via FC2 hybrid: 40% pilot (Run 5), 400-sample validation = 43.0% vs 51.25% snnTorch (8.25 pp hardware gap, 5.1x per-op energy reduction at hardware level).

This gap isn't a failure of spiking computation -- its a well-understood consequence of current SNN training. Deng & Gu (2020) explicitly argue researchers should "rethink the performance comparison between SNNs and ANNs" by isolating confounders. That's exactly what the matched-architecture design does. The 16.70 pp gap under controlled conditions is the genuine cost of spiking computation with surrogate gradients on small-data audio.

**PANNs reframes the narrative.** When feature-learning burden is removed (AudioSet CNN14 embeddings), gap collapses from 16.70 pp to 0.95 pp (92.50% vs 93.45%). Strongest evidence that the gap is a **feature-learning problem, not a spiking computation problem.** With equal-quality features, spiking and non-spiking classifiers achieve statistically indistinguishable accuracy on 50-class audio.


---

## 7.2 The Encoding Hierarchy and Its Explanation

The encoding ordering **direct > rate ≈ phase > population > latency >> delta ≈ burst** (all 7 methods complete) is consistent and instructive. It can be fully explained by the **information preservation principle**:

**Direct encoding dominates** because it preserves full continuous magnitude information across all 25 timesteps. Every LIF neuron receives the same current at every timestep, allowing the temporal integration to accumulate sufficient evidence for a reliable rate-code decision. The CNN convolutional layers process continuous-valued inputs that exactly match their training distribution (the preprocessing produces values in [0,1] at all timesteps), and the LIF threshold converts these to sparse binary spikes at the first-spike time.

**Rate encoding underperforms direct** because Bernoulli stochasticity introduces noise that increases the effective variance of the training signal. At T=25, a pixel with intensity 0.3 fires 7.5 times on average, but with standard deviation $\sqrt{25 \times 0.3 \times 0.7} = 2.29$ — sufficient to cause different fold images of the same sound to generate very different spike patterns. The CNN must learn to be robust to this noise, reducing its effective capacity for discrimination.

**Latency encoding further underperforms** because the logarithmic mapping concentrates spikes in the first few timesteps for high-intensity pixels, reducing the temporal diversity that the LIF integration can exploit. Effectively, latency coding at τ=5.0 collapses much of the T=25 window into T≈5 usable timesteps.

**Delta and burst perform near chance** for fundamentally different reasons:
- *Delta:* Static spectrograms have no temporal variation → no spikes generated → network cannot learn
- *Burst:* All information in first 5 of 25 timesteps → severe train/test distribution mismatch (network trains on the first-5-timestep-rich pattern but the remaining 20 timesteps contribute noise to test-time decisions). The result (6.50% ± 1.54% mean accuracy across 5 folds, with per-fold range 5.00%–9.25%, and train accuracy reaching 45–62%) demonstrates that the LIF integration window T=25 is not matched to the burst window N_max=5.

**What would make burst work?** Either (a) reducing T to match N_max (T≈5, sacrificing temporal resolution), (b) using a different network architecture that explicitly reads out only the first N_max timesteps, or (c) using an attention mechanism over timesteps. None of these are explored in this work — burst is documented as a negative result with full mechanistic explanation.

**Phase encoding achieves 24.15% ± 1.66%**, essentially tied with rate coding (24.00% ± 1.90%). Phase fires exactly once per neuron but distributes spike times uniformly over the full T=25 window (unlike latency's τ-parameterised clustering into early timesteps). The near-equality with rate coding is the most surprising result in the encoding comparison. Despite using only 1 spike per neuron vs rate's ~T×p ≈ 6–7 spikes, phase achieves identical accuracy. The explanation is information efficiency: phase deterministically maps intensity to spike time without stochastic noise, preserving the full magnitude ordering in the timing domain. Rate coding spreads information redundantly across multiple noisy spikes. At T=25, the information content of deterministic single-spike timing exactly matches stochastic multi-spike counting for this task. This finding has implications for energy efficiency: if 1 spike per neuron (phase) achieves the same accuracy as ~7 spikes per neuron (rate), phase is dramatically more energy-efficient at test time while achieving identical classification accuracy.

**Population coding achieves 19.15% ± 2.79%**, falling between phase (24.15%) and latency (16.30%). Population coding expands the output layer to 500 neurons (10 per class) and uses MSE count loss. The hypothesis was that a multi-neuron class representation would reduce sensitivity to individual neuron noise. This hypothesis is rejected: population coding underperforms rate coding despite 10× more output neurons. The explanation is training, not capacity. MSE count loss produces shallower gradients than cross-entropy rate loss — training accuracy at termination reaches only 18–24% vs rate coding's ~50% at comparable epoch count. The 500-neuron output layer adds parameters but the bottleneck is the 256-dimensional FC1 representation, not the output layer width. The higher variance (std=2.79% vs 1.90% for rate) confirms that the MSE loss landscape has multiple local minima with variable quality. Population coding as implemented here is not recommended for audio SNN classification.

---

## 7.3 SpiNNaker Deployment: Hardware-Software Co-Design Insights

The SpiNNaker deployment produced the most technically instructive failures of this thesis. The root cause of the FC1 cancellation problem (§5.1) reveals a fundamental tension between common SNN training practices and neuromorphic hardware requirements:

**The AvgPool problem:** Modern spiking CNNs routinely use pooling between convolutional layers to reduce spatial dimensions. For ANNs and software SNNs, AvgPool is a straightforward spatial aggregation. For neuromorphic deployment, AvgPool applied to binary spike tensors produces fractional outputs in [0, 1], violating the spike-only communication model of hardware like SpiNNaker. This violation propagates to the subsequent FC layer, where near-zero-mean weights cannot distinguish the fractional inputs from the training distribution (binary), producing systematic under-activation.

**Weight re-centering failure (Option C):** The mathematical analysis (§5.1.2) shows that weight re-centering with bias compensation is only valid for binary inputs. For fractional inputs, the compensation factor $n / \sum_j x_j$ can be substantially larger than 1, causing wild over-correction. This failure is a useful data point for the field: post-hoc weight adjustments for neuromorphic deployment cannot fix architectural incompatibilities.

**The lesson for SNN design:** Networks intended for full neuromorphic deployment should replace AvgPool between LIF layers with MaxPool (which preserves binary spike semantics — a max of binary values is binary) or with stride in the convolutional layer itself. This should be a design constraint applied at training time, not a post-hoc fix.

**Option A validation (4 March 2026):** A fold-4 threshold sweep with the MaxPool SNN confirms the prediction. FC1 binary fraction = 1.000 for all thresholds {1.0, 1.5, 2.0, 3.0} — MaxPool on binary LIF spikes guarantees binary FC1 inputs, as expected. Threshold=3.0 achieves 43.75% accuracy (fold 4) with 956 FC1 active inputs/step (sparsity 58.5%). This is 10.25 pp below the original AvgPool model's fold 4 result (54.0%), reflecting the information difference between max and average pooling for binary inputs. However, the FC1 cancellation problem is now absent: the model trains and evaluates correctly in software, and full SpiNNaker deployment (FC1+FC2 on-chip) is theoretically unblocked pending hardware router capacity testing.

**Hybrid deployment as a pragmatic solution:** The FC2-only hybrid (§5.2) demonstrates that partial neuromorphic deployment is viable and achieves meaningful results. Run 6 (fold 4, n=400) achieves **43.0% SpiNNaker accuracy** vs 51.25% snnTorch reference — an **8.25 pp hardware gap**. Five-fold cross-validation (400 samples × 5 folds = 2,000 total inferences) yields **33.1% ± 6.9% SpiNNaker mean accuracy** vs 46.0% snnTorch reference — a mean hardware gap of **12.8 ± 4.1 pp**. The gap is fold-dependent (varying across folds), suggesting class distribution difficulty varies meaningfully across folds. The gap fluctuated across the run (trajectory: n=108: 1.9 pp → n=208: 0.0 pp → n=400: 8.25 pp), reflecting sample-batch variability: samples in the final portion of the run included disproportionately hard classes for SpiNNaker (insects, helicopter, engine all 0% SpiNNaker, glass_breaking 0% SpiNNaker vs 50% snnTorch — a 50 pp class-level gap). The agreement rate drops from ~81% mid-run to 64.5% at n=400, indicating later samples were systematically harder for the IF_curr_exp model. SpiNNaker does outperform snnTorch on specific sound types: airplane (+37.5 pp), mouse_click (+25 pp) — sounds with simple, consistent spectrotemporal patterns that IF_curr_exp integrates reliably. The final 8.25 pp hardware gap is modest for a hybrid deployment without hardware-specific calibration, and comparable to the 10 pp seen in Run 5 (n=20).

---

## 7.4 Adversarial Robustness: A Surprising Finding

The adversarial robustness results (§6.1) are the most striking finding in this thesis and deserve extended discussion. At ε=0.1 (FGSM), the SNN retains 26.00% accuracy while the ANN collapses to 1.75% — a 24.25 pp advantage for the SNN, and 14.9× more robust by ratio, despite starting from a 15 pp accuracy deficit on clean inputs (fold 4 local: SNN 53.75% vs ANN 68.75%).

**This finding has a clear mechanistic explanation (§6.1.4):** the spike threshold acts as a non-linear input filter. Adversarial perturbations of magnitude ε add or subtract at most ε from each spectrogram pixel. For pixels near the threshold, this pushes them across; for pixels far from the threshold, it does not. The binary spike output is unchanged unless the perturbation is sufficient to change the spike pattern — and the ANN, without this hard threshold, is more sensitive to small gradient-aligned perturbations.

**Why is this practically important?** Adversarial robustness matters for audio intelligence in two scenarios:
1. **Adversarial attacks on intelligent audio monitors** — an attacker could add imperceptible perturbations to audio to prevent a smart building system from detecting an intrusion (glass breaking, alarm). The SNN is substantially harder to fool.
2. **Natural noise and distribution shift** — real audio recordings contain electrical noise, acoustic reflections, and bandwidth limitations. The spike threshold's filtering effect may generalise to natural robustness, not just adversarial perturbations.

**Caveat (standard PGD limitation):** The PGD results should be interpreted cautiously. Standard PGD with vanishing surrogate gradients may underestimate SNN vulnerability by computing inaccurate gradient directions (Wang et al. 2025). Future work should apply Stable Adaptive PGD for rigorous evaluation. The FGSM results (single-step attack) are less affected by this issue and are more reliable.

**Comparison with Sharmin et al. (2020):** Our finding is consistent in direction with Sharmin et al., who reported higher SNN adversarial accuracy with Poisson (rate) encoding in black-box settings. We extend this to white-box attacks on audio spectrograms, confirming that the robustness advantage generalises beyond images to audio and beyond rate encoding to direct encoding.

---

## 7.5 PANNs + SNN: What it Reveals About Feature Learning

The collapse of the SNN-ANN gap from 16.70 pp to 0.95 pp with pre-trained features (§4.6) is the most practically actionable finding in this thesis. It establishes that:

1. **SNN computation is not the bottleneck.** A 3-layer SNN trained on 2048-d CNN14 embeddings achieves 92.50% — approaching human performance (81.3%) and significantly above the from-scratch SNN (47.15%) and ANN (63.85%).

2. **The bottleneck is the feature learning regime.** A ~622K parameter network trained on 1,600 samples cannot learn the rich spectro-temporal features that distinguish 50 sound classes. The CNN convolutional layers are simply too small and too data-limited for from-scratch feature extraction.

3. **Linear classifier (93.80%) barely outperforms SNN head (92.50%).** This implies that the CNN14 embeddings are already near-linearly separable for ESC-50 classes. The SNN head's additional non-linearity provides minimal benefit over a linear probe, suggesting that the main task at the classification stage is already trivial with good features.

**Practical deployment pathway:** A realistic neuromorphic audio intelligence system would:
1. Extract features using a frozen pretrained ANN (CNN14 or similar) — this runs once, efficiently, on a general CPU or edge NPU
2. Classify using an SNN trained on the embeddings — this runs on SpiNNaker or Loihi, consuming ~86 nJ/sample (FC2 layer, estimated)
3. The overall pipeline achieves 92.50% accuracy with hardware-compatible SNN inference

This is a more promising direction for near-term deployment than end-to-end spiking computation.

---

## 7.6 Continual Learning: What the Experiment Reveals

The continual learning experiment (§6.2) confirms the expected finding — both SNN and ANN suffer severe catastrophic forgetting — but provides a quantitative comparison and an unexpected finding: **the SNN forgets 4.7 pp less than the ANN** (69.9% ± 4.3% vs 74.7% ± 2.4% mean forgetting, 5-fold validated).

**Why the SNN forgets less:** The mechanism is the same as for adversarial robustness — the spike threshold acts as a computational gate. During backward pass, only weights connected to neurons that actually fired receive gradient updates. With 74.16% activation sparsity, approximately 74% of weights receive zero gradient per forward pass. When fine-tuning on a new task, fewer weights are modified, preserving more of the earlier task representation. The ANN's dense activations produce non-zero gradients for all weights on all forward passes, enabling more complete overwriting. The 5-fold validation confirms this advantage is consistent across folds (SNN std=4.3%, ANN std=2.4%).

**Why both forget aggressively:** The experimental setting is extreme — no replay, no regularisation, 320 training samples per task out of 1,600 total. The gradient pressure on each task is proportionally large relative to the task size. Mean forgetting of 69.9% and 74.7% (5-fold validated) significantly exceeds the 30–50% typical in regularisation-equipped continual learning. This is baseline forgetting, not regularised forgetting, and should be interpreted as such.

**Relationship to the adversarial robustness finding (§6.1):** Both findings point to the same underlying property of SNNs: the binary spike threshold creates selective information flow. For adversarial robustness, this makes the SNN harder to fool by small perturbations. For continual learning, this makes the SNN harder to fully overwrite by new task gradients. The spike threshold is not just a training approximation — it is a computational primitive that shapes information processing at inference and training time.

**Implications for future work:** The baseline forgetting rate (69.9% ± 4.3% for SNN, 5-fold validated) quantifies the improvement achievable by SNN-specific consolidation mechanisms. Golden et al. (2022, PLoS Comp Bio) demonstrated sleep-based consolidation reducing forgetting by >50% in simpler spike-rate networks. Applied to this architecture, that would project to ~35% forgetting with consolidation, approaching the 30–40% ANN forgetting with EWC regularisation (Kirkpatrick et al. 2017).

**Validation:** The experiment is now 5-fold validated (20 epochs per task, no hyperparameter tuning specific to the continual setting, lr=5e-4 chosen to be smaller than training lr=1e-3 to reduce catastrophic overwriting). The 4.7 pp SNN advantage is consistent across folds.

---

## 7.7 Threats to Validity

**Internal validity:**
- **Model selection bias:** Reporting best validation accuracy per fold (not final epoch) means results represent optimistic performance. The chosen model may not generalise to new data beyond the fold. Standard practice in the field justifies this.
- **Fold sample size:** Each fold test set contains 400 samples (8 per class). Per-class analyses are statistically limited (n=8 per class per fold). Confidence intervals for class-level effects are wide.
- **Adversarial attack reliability:** Standard PGD may underestimate SNN vulnerability. SNN robustness numbers are upper bounds.

**External validity:**
- **Dataset generalisability:** ESC-50 is a standard benchmark but contains only 5-second isolated clips. Real audio is continuous, overlapping, and variable-length. Performance may differ substantially in streaming deployment.
- **Architecture generalisation:** Results are specific to the SpikingCNN architecture with T=25, β=0.95, threshold=1.0. Different LIF parameters or timestep counts would produce different accuracy/efficiency trade-offs.
- **Hardware generalisability:** SpiNNaker results (SpiNN-5 board) may not generalise to SpiNNaker2, Loihi 2, or TrueNorth due to different neuron models, precision, and communication latency.

**Construct validity:**
- **Energy measurement:** NeuroBench SynapticOperations are software-based proxies for energy consumption at 45nm CMOS. Actual chip measurements would be needed for rigorous energy claims. This work treats SynOps as an upper bound / comparative metric.
- **Neuromorphic deployment accuracy:** FC2-only hybrid with 20-sample validation (Run 5) has high uncertainty. The 40% accuracy estimate has wide confidence intervals (95% CI: 18.5%–61.5% for n=20).

---

## 7.8 Broader Implications

**For the SNN research community:** This work provides the first convolutional SNN results on ESC-50 — a standardised reference that future SNN audio work can compare against. The systematic encoding comparison and negative results (delta, burst) are informative: the field should not assume that biologically-motivated encodings outperform simpler alternatives.

**For neuromorphic computing:** The AvgPool-FC1 cancellation problem is a previously undocumented failure mode for neuromorphic deployment of standard spiking CNN architectures. The documented failure and hybrid solution provide practical guidance for future SpiNNaker deployments.

**For edge audio intelligence:** The adversarial robustness finding suggests that SNNs have a practical advantage for secure edge audio monitoring beyond their energy efficiency. A sensor that is both energy-efficient and adversarially robust is more deployable than one that is only energy-efficient.

**Per-class pattern and the threshold energy hypothesis:** The per-class analysis reveals an unexpected but mechanistically clear pattern: the SNN outperforms the ANN on 6/50 classes, all of which are high-energy, spectrally distinctive sounds (crying_baby, door_wood_knock, coughing, pouring_water, crackling_fire, footsteps). The SNN fails most severely on low-energy, quiet sounds (engine 8%, door_wood_creaks 10%, clock_tick 23%). This suggests the LIF threshold acts as an energy-gated filter: high-energy sounds reliably cross the membrane threshold, producing consistent spike patterns that train well. Low-energy sounds hover near the threshold, producing stochastic and unreliable spike patterns that resist generalisation. For applications where robustness to quiet sounds is important (e.g., detecting leaks, monitoring breathing), the ANN remains strongly preferred. For detecting loud alarms, impacts, and vocal events, the SNN may achieve competitive or superior accuracy.

**For the accuracy debate:** The PANNs result definitively shows that the SNN-ANN accuracy gap is not fundamental. With pre-trained features, SNNs and ANNs achieve near-identical accuracy. The gap is a data and training regime problem, not a computational paradigm problem. This is an optimistic finding for the long-term trajectory of SNN research.

---

## 7.9 Data Augmentation: When Standard Methods Fail for SNNs

The augmentation experiment (§4.4) produced a clear and instructive negative result: SpecAugment + TimeShift harms the SNN (47.15% → 40.75%, −6.40 pp; std 4.50% → 16.03%) and marginally harms the ANN (63.85% → 61.70%, −2.15 pp). This is worth discussing explicitly, because data augmentation is a default recommendation in deep learning that does not transfer cleanly to SNNs on small datasets.

**The mechanism is specific, not general.** Two interacting failure modes explain the SNN degradation:

1. **Early stopping mismatch.** SpecAugment slows SNN convergence by randomising the input signal each epoch — the model cannot memorise patterns, so it must generalise. This is the intended effect. However, with early stopping patience=10 and augmentation pushing loss landscapes to be flatter and noisier, the stopping criterion fires prematurely on some folds (folds 3 and 5 stopped at epochs 39 and 33 respectively, when training accuracy was only 26.4% and 21.3%). The model never converges to a meaningful representation. ANN convergence is faster (smaller per-epoch loss drop from the stronger gradient signal), so ANN early stopping fires only slightly early.

2. **LIF threshold interacts poorly with mean-value infill.** SpecAugment masks spectrogram regions with the mean value, creating a constant-valued band across those frequency or time bins. For the ANN, mean-value input is simply processed continuously. For the SNN, mean-value inputs produce a constant membrane current at every timestep — not "absent" (zero current), but undifferentiated. Across T=25 timesteps, this constant current accumulates uniformly, potentially reaching threshold and firing regardless of the actual sound content. The masked region thus actively introduces spurious spikes that compete with the discriminative signal. This is not a failure of the augmentation concept — it is a failure of the specific mask infill strategy for LIF neurons.

**The fold 4 exception is informative.** Fold 4's augmented SNN reached 63.75% — the highest SNN fold accuracy recorded in this study (+9.75 pp over baseline). Fold 4 ran for the full 100 epochs (training acc 80.2% at ep 100, best at ep 90). This demonstrates that augmentation *can* substantially benefit SNNs when convergence is permitted. The bimodal outcome across folds (two folds degraded dramatically, one improved substantially, two were neutral) reflects the sensitivity of SNN training to early stopping parametrisation under augmentation.

**Practical guidance.** For SNN training with SpecAugment on small datasets: (1) use patience ≥ 20 to allow the augmentation-slowed convergence to complete; (2) reduce mask sizes (F=4, T=10 rather than F=8, T=20); (3) consider zero-filling rather than mean-filling masked regions to avoid spurious LIF threshold crossings. These modifications are left as direct future work — the 9.75 pp improvement seen on fold 4 justifies the experimental cost of a properly configured augmented run.
