# chapter 7: discussion

this is where i tie everything together and actually answer the research questions. need to be careful to connect the findings rather than just restating results.

---

## 7.1 the accuracy-efficiency trade-off

Core finding: **conv SNN from scratch on ESC-50 gets 47.15% (direct), ANN gets 63.85% -- 16.70 pp gap, p=0.001.** On SpiNNaker via FC2 hybrid: 40% pilot (Run 5), 400-sample validation = 43.0% vs 51.25% snnTorch (8.25 pp hardware gap, 5.1x per-op energy reduction at hardware level).

This gap isn't a failure of spiking computation -- its a well-understood consequence of current SNN training. Deng & Gu (2020) explicitly argue researchers should "rethink the performance comparison between SNNs and ANNs" by isolating confounders. That's exactly what the matched-architecture design does. The 16.70 pp gap under controlled conditions is the genuine cost of spiking computation with surrogate gradients on small-data audio.

**PANNs reframes the narrative.** When feature-learning burden is removed (AudioSet CNN14 embeddings), gap collapses from 16.70 pp to 0.95 pp (92.50% vs 93.45%). Strongest evidence that the gap is a **feature-learning problem, not a spiking computation problem.** With equal-quality features, spiking and non-spiking classifiers achieve statistically indistinguishable accuracy on 50-class audio.

Practical implication: optimal pipeline isn't fully spiking end-to-end, but a **hybrid** -- efficient feature extractor (ANN or bio-inspired) provides representations to lightweight SNN classifier. Exactly what we validate on SpiNNaker.

---

## 7.2 the encoding hierarchy and its explanation

Ordering: **direct > rate ~ phase > population > latency >> delta ~ burst** (all 7 complete). Fully explained by information preservation.

**Direct dominates** because it preserves full continuous magnitude across all 25 timesteps. CNN layers get inputs matching their training distribution at every step.

**Rate < direct** because Bernoulli stochasticity introduces noise. At T=25, pixel at 0.3 fires 7.5 times avg but std=2.29 -- enough to make nearby pixels indistinguishable. CNN has to learn robustness to this noise, reducing effective capacity.

**Latency underperforms further** because log mapping concentrates spikes into first ~5 timesteps for high-intensity pixels. Effectively reduces T=25 to T~5.

**Delta and burst near chance** for different reasons:
- Delta: static spectrograms = no temporal variation = no spikes = can't learn
- Burst: all info in first 5/25 steps = train/test distribution mismatch. LIF integration window doesn't match burst window. 6.50% +/- 1.54%, train acc 45-62% but test 5-9%.

What would fix burst? T=5 (sacrificing temporal resolution), readout from first N_max steps only, or timestep attention. None explored -- documented as negative result.

**Phase at 24.15% ties with rate at 24.00%.** Most surprising result. Despite 1 spike/neuron vs ~7, identical accuracy. Phase deterministically maps intensity to time without stochastic noise, preserving full magnitude ordering. Rate spreads info redundantly across noisy spikes. At T=25 these representations have equivalent discriminative content. Implication: phase is dramatically more energy-efficient at test time for the same accuracy.

**Population at 19.15%** falls between phase and latency. 500 output neurons (10/class), MSE count loss. Hypothesis that multi-neuron class representation would help is rejected. MSE produces shallower gradients -- training acc only 18-24% vs rate's ~50%. Bottleneck is 256-d FC1, not output width. Higher variance (std=2.79%) confirms MSE landscape has multiple local minima.

---

## 7.3 SpiNNaker deployment: co-design insights

The deployment failures are honestly some of the most instructive parts of this thesis.

**The AvgPool problem:** modern spiking CNNs routinely use pooling between conv layers. For software SNNs, AvgPool is fine. For neuromorphic deployment, AvgPool on binary spikes produces fractional outputs violating spike-only communication. This propagates to FC where near-zero-mean weights can't distinguish fractional inputs from the training distribution, causing systematic under-activation.

**Weight re-centering failure (Option C):** math shows compensation is only valid for binary inputs. For fractional, the factor n/sum(x_j) can be much larger than 1, causing over-correction. Useful data point: post-hoc weight adjustments can't fix architectural incompatibilities.

**The lesson:** networks for full neuromorphic deployment should use MaxPool between LIF layers (preserves binary semantics) or stride in conv. Design constraint at training time, not a post-hoc fix.

**Option A validation:** MaxPool SNN fold 4, fc1_binary_fraction = 1.000 at all thresholds. threshold=3.0: 43.75% accuracy, 956 active inputs/step. 10.25 pp below original AvgPool model (54.0%) -- info difference between max and avg pooling for binary inputs. But FC1 cancellation is now absent. Full SpiNNaker deployment (FC1+FC2 on-chip) theoretically unblocked.

**Hybrid deployment works:** FC2-only (section 5.2) achieves meaningful results. Run 6 fold 4: 43.0% SpiNNaker, 51.25% snnTorch, 8.25 pp gap. 5-fold: 33.1% +/- 6.9% vs 46.0% snnTorch, mean gap 12.8 +/- 4.1 pp. Gap is fold-dependent, fluctuates within runs. SpiNNaker actually beats snnTorch on specific sounds: airplane (+37.5 pp), mouse_click (+25 pp) -- simple consistent patterns that IF_curr_exp handles well. Final 8.25 pp gap is modest for hybrid deployment without hardware-specific calibration.

---

## 7.4 adversarial robustness: a surprising finding

The adversarial results are honestly the most striking finding. At eps=0.1 FGSM, SNN retains 26.00% while ANN collapses to 1.75% -- 24.25 pp advantage, 14.9x ratio, despite starting from 15 pp accuracy deficit on clean inputs.

**Mechanism:** spike threshold is a non-linear filter. Adversarial perturbations of magnitude eps shift pixels by at most eps. For pixels near threshold, this pushes them across; for pixels far from it, nothing happens. Binary output unchanged unless perturbation changes the spike pattern. ANN without this hard threshold is more sensitive.

**Why it matters practically:**
1. Adversarial attacks on audio monitors -- attacker adds imperceptible perturbations to prevent detection (glass breaking, alarm). SNN substantially harder to fool.
2. Natural noise and distribution shift -- threshold filtering may generalise beyond adversarial to natural robustness.

**Caveat:** PGD results should be interpreted cautiously. Standard PGD + vanishing surrogate gradients may underestimate vulnerability (Wang et al. 2025). FGSM results more reliable. Future work: SA-PGD for rigorous evaluation.

Consistent with Sharmin et al. (2020) who reported similar effects with rate encoding in black-box settings. We extend to white-box attacks on audio, confirming robustness generalises beyond images to audio and beyond rate to direct encoding.

---

## 7.5 PANNs + SNN: what it reveals about feature learning

The gap collapse from 16.70 pp to 0.95 pp with pretrained features is the most practically actionable finding.

1. **SNN computation isn't the bottleneck.** 3-layer SNN on CNN14 embeddings = 92.50%, approaching human (81.3%) and well above from-scratch SNN (47.15%) and ANN (63.85%).

2. **Bottleneck is feature learning.** ~622K params on 1600 samples can't learn rich spectro-temporal features for 50 classes. Conv layers too small and too data-limited.

3. **Linear (93.80%) barely beats SNN head (92.50%).** CNN14 embeddings are already nearly linearly separable for ESC-50. The SNN head's extra nonlinearity provides minimal benefit -- classification is basically trivial with good features.

**Practical deployment:**
1. Frozen pretrained ANN (CNN14 or similar) extracts features -- runs once on CPU/NPU
2. SNN classifier runs on SpiNNaker/Loihi, ~86 nJ/sample (FC2 layer)
3. 92.50% accuracy with hardware-compatible inference

More promising than end-to-end spiking for near-term deployment.

---

## 7.6 continual learning: what the experiment reveals

CL experiment (section 6.2) confirms the expected -- both suffer severe forgetting -- but with a quantitative surprise: **SNN forgets 4.7 pp less** (69.9% +/- 4.3% vs 74.7% +/- 2.4%, 5-fold validated).

**Why:** same mechanism as adversarial robustness. Spike threshold = computational gate. During backward pass, only weights connected to neurons that fired get updates. With 74.16% sparsity, ~74% of weights get zero gradient per pass. Fine-tuning on new task modifies fewer weights, preserving earlier representation. ANN's dense activations = non-zero gradients for all weights = more complete overwriting.

**Why both forget aggressively:** extreme setting -- no replay, no regularisation, 320 samples/task out of 1600. 69.9% and 74.7% significantly exceeds the 30-50% typical with regularisation. This is baseline forgetting, not regularised.

**Relationship to adversarial finding:** both point to same underlying property. Binary spike threshold creates selective information flow. For adversarial: harder to fool. For CL: harder to fully overwrite. The threshold isn't just a training approximation -- its a computational primitive shaping information processing.

**Future:** baseline 69.9% quantifies improvement achievable by SNN consolidation. Golden et al. (2022) showed >50% reduction with sleep-based consolidation. Applied here, that'd project ~35% forgetting, approaching ANN forgetting with EWC.

---

## 7.7 threats to validity

**Internal:**
- Model selection bias: reporting best val acc per fold = optimistic. Standard practice though.
- Fold sample size: 400 samples (8/class), per-class analyses statistically limited. Wide CIs at class level.
- Adversarial attack reliability: standard PGD may underestimate vulnerability. SNN numbers are upper bounds.

**External:**
- ESC-50 is isolated 5-second clips. Real audio is continuous, overlapping, variable-length. Performance could differ substantially in streaming.
- Results specific to SpikingCNN with T=25, beta=0.95, threshold=1.0. Different params = different tradeoffs.
- SpiNNaker results (SpiNN-5) may not generalise to SpiNNaker2, Loihi 2, TrueNorth.

**Construct:**
- NeuroBench SynOps are software proxies for energy at 45nm. Actual chip measurements needed for rigorous energy claims.
- FC2-only hybrid with n=20 validation (Run 5) has wide CIs (95% CI: 18.5-61.5% for n=20). check this CI calculation

---

## 7.8 broader implications

**For SNN community:** first conv SNN results on ESC-50 -- standardised reference. Encoding comparison and negative results (delta, burst) informative: don't assume bio-motivated encodings beat simpler alternatives.

**For neuromorphic computing:** AvgPool-FC1 cancellation is a previously undocumented failure mode. Documented failure and hybrid solution = practical guidance.

**For edge audio:** adversarial robustness advantage goes beyond energy efficiency. Sensor that's both energy-efficient and adversarially robust is more deployable than one thats only efficient.

**Per-class pattern:** SNN outperforms ANN on 6/50 classes, all high-energy spectrally distinctive (crying_baby, door_knock, coughing, pouring_water, crackling_fire, footsteps). Fails on low-energy quiet sounds (engine 8%, door_creaks 10%, clock_tick 23%). LIF threshold = energy-gated filter. For apps needing robustness to quiet sounds (leak detection, breathing monitoring), ANN strongly preferred. For loud alarms, impacts, vocal events -- SNN may match or beat ANN.

**For the accuracy debate:** PANNs result shows the gap isn't fundamental. With pretrained features, SNN ~ ANN. Its a data/training regime problem, not a computational paradigm problem. Optimistic for long-term SNN trajectory.

---

## 7.9 data augmentation: when standard methods fail for SNNs

The augmentation experiment produced a clear negative result: SpecAugment + TimeShift hurts SNN (-6.40 pp, variance tripled) and marginally hurts ANN (-2.15 pp). Worth discussing because augmentation is a default recommendation in deep learning that doesn't transfer cleanly to SNNs on small datasets.

**Two interacting failure modes:**

1. **Early stopping mismatch.** SpecAugment slows convergence by randomising input each epoch. Patience=10 fires prematurely on some folds (3 and 5 stopped at ep39 and 33, training acc 26.4% and 21.3%). Model never converges. ANN converges faster (stronger gradient signal) so early stopping fires only slightly early.

2. **LIF threshold + mean-value infill.** SpecAugment masks to mean value, creating constant-valued bands. For ANN, just processed continuously. For SNN, constant current at every timestep isn't "absent" (zero) -- its undifferentiated current that accumulates uniformly, potentially reaching threshold regardless of actual content. Masked region actively introduces spurious spikes. Not a failure of the augmentation concept -- failure of the specific infill for LIF neurons.

**Fold 4 exception:** reached 63.75% (highest SNN fold ever, +9.75 pp). Ran full 100 epochs with train acc 80.2%. Shows augmentation *can* substantially help when convergence is permitted. Bimodal outcome across folds reflects sensitivity to early stopping under augmentation.

**Practical guidance:** for SNN + SpecAugment on small datasets: patience >= 20, reduce mask sizes (F=4, T=10), consider zero-filling instead of mean-filling to avoid spurious threshold crossings. Left as future work -- the fold 4 result justifes the effort.
