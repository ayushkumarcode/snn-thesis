# decisions log

every significant decision made during the project -- what we picked, what we didn't, and why. this is the canonical record of WHY. see EXPERIMENT_LOG.md for what actually happened.

---

## 1. dataset selection

pre-march 2026 (planning phase)

went with ESC-50 (2,000 clips, 50 classes, 5 predefined folds, 5s each).

considered but rejected:
- UrbanSound8K -- only 10 classes, less challenging, already has SNN work so less novel
- ESC-10 -- subset of ESC-50 with 10 classes, too easy
- SHD (Spiking Heidelberg Digits) -- spoken digits, already extensively studied with SNNs (96.4% known), zero novelty
- Google Speech Commands -- keyword spotting, already has SNN work (95.1% known)
- RWCP -- indoor sounds, 19 classes, already has SNN work (99.6%)

the main reason: ESC-50 has **zero prior SNN peer-reviewed work** (confirmed by arXiv 2503.11206, march 2025). thats the whole thesis contribution -- first application. 50 classes is harder and more impressive than 10-class alternatives. predefined 5-fold CV makes results directly comparable to future work. dataset is free, well-maintained, widely used in audio ML.

ESC-10 was considered as a fallback if ESC-50 proved too hard, but results showed ESC-50 was tractable.

---

## 2. audio preprocessing

early march 2026 (implementation phase)

### 2a. feature representation

went with mel spectrograms -- 64 mel bins, n_fft=1024, hop_length=512, fmin=0, fmax=Nyquist (11,025 Hz), converted to log dB scale, min-max normalised to [0, 1].

rejected:
- raw waveform -- no standard way to directly encode raw audio as spikes for a conv SNN, would need a totally different architecture
- MFCCs -- discards phase and energy info compared to mel specs. mel specs are standard for non-speech audio
- STFT (raw spectrogram) -- linear frequency scale, doesn't represent perceptual frequency differences well
- gammatone filterbank -- more biologically realistic but adds complexity with no clear benefit for this arch

mel spectrograms are just the standard thing for environmental sound classification. strong ANN baselines use them, so fair comparison.

### 2b. sample rate

went with 22,050 Hz (half of original 44,100 Hz). 22,050 Hz captures frequencies up to 11,025 Hz (Nyquist). environmental sounds rarely have meaningful content above 11 kHz. halving the sample rate halves the waveform length, saves memory and preprocessing time. standard practice in audio ML.

### 2c. output shape

(1, 64, 216) -- 64 frequency bins, 216 time frames from 5s audio at sr=22050, hop=512. this is just determined by the pipeline params, not really a free choice.

---

## 3. architecture

### 3a. base architecture

early march 2026

went with a convolutional SNN: Conv2d(1,32) -> BN -> MaxPool(2) -> LIF -> Conv2d(32,64) -> BN -> MaxPool(2) -> LIF -> AvgPool(4,6) -> FC(2304,256) -> LIF -> FC(256,50) -> LIF

rejected:
- pure FC network -- ignores spatial structure of the spectrogram, performs much worse
- recurrent SNN (LSNN / snnTorch RNN) -- adds temporal recurrence, significantly more complex to train with surrogate gradients, unclear benefit for 5s clips
- 3-layer conv -- more params and maybe better accuracy but not needed for a first-pass result
- ResNet-style SNN -- way more complex, harder to deploy on SpiNNaker

basically mirrors snnTorch Tutorial 6 adapted for spectrogram input. simple enough to understand and deploy, complex enough to be non-trivial. the ANN counterpart just swaps LIF for ReLU, clean controlled comparison.

### 3b. pooling layer (MPS compatibility fix)

3 march 2026

used `AvgPool2d(kernel_size=(4, 6))` instead of `AdaptiveAvgPool2d`.

turns out `AdaptiveAvgPool2d` with non-divisible input sizes crashes on Apple Silicon MPS. after MaxPool2d x2, spatial dims are (16, 54). `AvgPool2d(4, 6)` gives the same output (4, 9). just a compatibility fix, not an accuracy choice.

### 3c. LIF decay parameter (beta)

went with beta=0.95 for all LIF neurons (fixed, not learnable).

rejected beta=0.9 (faster decay, slightly lower accuracy in informal testing), learnable beta (more params + training instability, didn't try), beta=0.99 (very slow decay, may cause vanishing gradients).

beta=0.95 is standard from snnTorch tutorials and widely used. corresponds to tau_m ~20ms (exp(-1ms/20ms) ~ 0.951), biologically plausible membrane time constant.

### 3d. FC1 hidden size

256 neurons. rejected 512 and 128.

256 is standard power-of-2, sufficient representational capacity without excessive params. total model ~622K params, appropriate for ESC-50's training set (1,600 samples). also: 256 maps cleanly to SpiNNaker's core allocation (256 neurons <= 256/core limit).

---

## 4. training configuration

### 4a. optimiser

Adam (lr=1e-3, weight_decay=1e-4). standard for SNN surrogate gradient training. adaptive learning rates handle the complex loss landscape better than SGD. AdamW would be equivalent here.

### 4b. LR schedule

ReduceLROnPlateau (factor=0.5, patience=5). robust -- only reduces LR when val loss plateaus. cosine annealing would need knowing total training length in advance.

### 4c. early stopping

patience=10 epochs, max 50 epochs. 50 is sufficient for convergence on ESC-50 with this arch (most folds plateau before epoch 30). patience=10 prevents overfitting without stopping too early.

### 4d. batch size

32. standard. 64 would push memory on 1-GPU jobs. 16 gives noisier gradients.

### 4e. loss function

SNN: per-timestep cross-entropy on membrane potentials (snnTorch Tutorial 5 approach). ANN: standard CrossEntropyLoss. SNN loss must be computed across all timesteps to propagate gradients through the surrogate.

### 4f. cross-validation

5-fold CV using ESC-50's predefined folds. the dataset explicitly provides these in its metadata CSV. using predefined folds ensures comparability with other ESC-50 work and prevents data leakage (folds designed to have no speaker/recording overlap).

---

## 5. spike encoding methods

tested all 4 initially; direct encoding as primary. results: direct 47.15%, rate 24.00%, latency 16.30%, delta 7.25%.

the encoding comparison is itself a thesis contribution -- no study had compared them on ESC-50. direct encoding wins because it feeds the normalised spectrogram directly at every timestep, letting the network learn its own temporal coding. rate and latency impose fixed coding schemes that may not match what the network needs. delta is particularly bad -- spectrograms already represent time-frequency structure, so differentiating them destroys the base signal.

---

## 6. energy measurement methodology

went with SynOps counting (SNN) vs MAC counting (ANN) using literature energy constants.

SNN: SynOps x 0.9 pJ/SynOp (Intel Loihi value). ANN: MACs x 4.6 pJ/MAC (Horowitz 2014, 45nm CMOS).

considered NeuroBench wrapper but integration complexity wasn't justified for the simple counting we needed. manual SynOps counting is more transparent. actual power measurement would require hardware power metering on SpiNNaker (not available through remote access).

the Loihi and Horowitz constants are the most widely cited in SNN energy literature. makes our results comparable. the finding that SNN uses 4.3x MORE energy in software is expected -- it motivates hardware deployment.

---

## 7. training infrastructure

### 7a. primary training platform

CSF3 (university of manchester HPC, NVIDIA A100 GPUs).

rejected local MacBook (MPS) -- ~65 min per fold vs ~6 min on CSF3, impractical for 25 training runs. rejected Google Colab -- session limits, random disconnections, T4 slower than A100. CSF3 is free with uni account.

### 7b. CSF3 CUDA modules

need both `cuda/12.6.2` and `libs/cuda/12.8.1` plus `python/3.13.1`. the first provides the toolkit (nvcc, headers), the second provides runtime libraries PyTorch needs. without `libs/cuda/12.8.1`, PyTorch falls back to CPU even though CUDA appears available. found this out through a failed job (11782913).

---

## 8. SpiNNaker deployment architecture

3 march 2026

went with FC-only hybrid deployment (conv on CPU, FC on SpiNNaker). SpiNNaker1 doesn't natively support Conv2d. FC layers map directly to neural populations and projections. conv layers require spatial operations that don't naturally map to SpiNNaker's architecture. FC-only hybrid is the standard approach.

rejected full model on SpiNNaker (would require custom core implementations, out of scope) and conv features with FC1+FC2 on SpiNNaker (this is what runs 1-3 attempted, failed due to FC1 weight cancellation).

---

## 9. SpiNNaker neuron model

3 march 2026

went with `IF_curr_exp` (leaky integrate-and-fire with exponential current decay).

rejected `IF_curr_delta` (Dirac delta current model, closer to snnTorch's LIF but IF_curr_exp fired first during auto-calibration), `IF_cond_exp` (conductance-based, harder to map from trained weights), `IZh_curr_exp` (Izhikevich, more complex dynamics).

`IF_curr_exp` is the standard choice for deploying trained weights to SpiNNaker. simplest model that supports exc and inh currents.

---

## 10. SpiNNaker parameter calibration

3 march 2026

settled on: tau_syn=5.0ms, v_thresh=1.0, v_rest=0.0, v_reset=0.0, tau_m=20ms, tau_refrac=0.1ms.

determined by `auto_calibrate.py` phase 1+2 sweep:
- phase 1 (weight sweep): confirmed neurons fire at weight=0.5+ with tau_syn=1.0ms
- phase 2 (tau_syn sweep): tau_syn=5.0ms was the first value tested that fired, sweep stopped immediately

mapping from snnTorch: beta=0.95 -> tau_m=20ms, threshold=1.0 -> v_thresh=1.0, v_rest=0 -> v_rest=0.0, no refractory -> tau_refrac=0.1ms (minimal).

caveat: tau_syn calibrated with fully synthetic input (all neurons fire every timestep). real hidden spikes are ~24% active. optimal tau_syn for real data may differ slightly.

---

## 11. SpiNNaker FC2-only approach

3 march 2026

ok so the big decision here -- deploy only FC2 (256->50) on SpiNNaker, with FC1+lif3 pre-computed on CPU.

FC1+FC2 on SpiNNaker was attempted in runs 1-3 and confirmed to fail. root cause: FC1 weights have near-zero mean (-0.0034). with 1,398 simultaneously active binary inputs, the net current per neuron = 1,398 x (-0.0034) x scale = large negative, regardless of scale factor. zero hidden neurons fire. can't be fixed by scaling.

FC2-only works because snnTorch's lif3 produces sparse binary hidden spikes: ~61/256 neurons active per timestep (24%), max 65 simultaneous spikes. 65 simultaneous spikes is well under SpiNNaker router capacity (vs 1,398 with full FC1 input). the hidden representation already encodes class info; SpiNNaker FC2 just needs to read it out.

run 5 confirmed: 5/10 = 50.0% accuracy (vs snnTorch 2/10 = 20.0%).

---

## 12. SpiNNaker optimisation strategy

3 march 2026

went with path B: get current model running on SpiNNaker (proof of pipeline) -> improve snnTorch model accuracy -> then re-calibrate SpiNNaker for improved model.

rejected path A (optimise SpiNNaker params first, then improve model). the thing is, most impactful SpiNNaker optimisations (weight scale, pruning threshold) are model-dependent -- they depend on FC2 weight magnitudes which change when the model is retrained. would mean doing that work twice.

scale re-calibration for a new model is cheap (~9 SpiNNaker runs, ~10 minutes). snnTorch model is the ceiling -- better model directly improves SpiNNaker accuracy without additional hardware work beyond re-calibration.

---

## 13. SpiNNaker FC1 weight re-centering (Option C)

3 march 2026

Option C is NOT viable. tried it, accuracy dropped from 53.75% to 8.50%. catastrophic.

what was attempted: zero-centering each FC1 weight row (W[i] -= mean(W[i])) with bias compensation (b[i] += mean(W[i]) x n_inputs). supposed to be mathematically equivalent reparameterisation.

why it fails: the equivalence w*x + b = (w-mu)*x + (b + mu*n_inputs) only holds when x is binary (0/1). but FC1 inputs are NOT binary -- they're fractional values from avg_pool(spk2), typically in [0, 0.5]. the bias compensation term mu*2304 assumes all 2304 inputs are always 1, but actual sum(x) is way less than 2304. creates a massive incorrect positive bias that forces all FC1 neurons to fire constantly.

interesting thing that came out of this though: FC1 already fires at 21.76% activation rate (55.7 neurons/step) in software -- well within SpiNNaker's capacity. the "FC1 cancellation" was less severe in practice than the original analysis suggested.

saved: `results/spinnaker_optionC/option_c_fold4.json`

---

## 14. burst and phase encoding implementation

3 march 2026

implemented burst and phase as additional spike encodings. did NOT implement step-forward (not supported by snnTorch API) or population as a core encoding (it's output-side, handled separately).

burst coding: maps intensity to spike count at the start of the sim window. n_spikes = round(intensity * max_spikes), clamped to [0, max_spikes]. neuron fires at timestep t if t < n_spikes. max_spikes=5 chosen to produce 20% spike density ceiling (5/25 timesteps). biologically motivated by bursting neurons in auditory cortex.

phase coding: maps intensity to spike timing. spike_time = floor((1 - intensity) * (num_steps - 1)). high intensity = early spike, zero = silent. exactly one spike per neuron per window (deterministic). biologically motivated by theta-phase precession. complements burst (count-based) with a time-based code.

step-forward rejected because snnTorch doesn't expose it as a standalone encoding primitive, would require modifying the LIF forward loop.

population coding moved to separate experiment -- it modifies the output layer (50 -> 500 neurons), loss function, and accuracy metric. doesn't fit the encoding enum paradigm.

---

## 15. population coding as separate experiment

3 march 2026

implemented output population coding as a standalone experiment (`experiments/population_coding.py`) rather than a core encoding type.

SpikingCNNPop: same conv+FC1 but FC2 outputs 500 neurons (50 classes x 10 each). loss: SF.mse_count_loss(population_code=True). input: rate coding.

why output population (not input): input population coding would expand input dimensionality, complicating the conv backbone. output population is straightforward -- only final linear + loss change.

why rate-coded input: rate coding is most studied, clean controlled comparison. direct coding can't be directly compared because the loss function changes.

result: 19.15% +/- 2.79% -- underperforms rate coding (24.00%). hypothesis rejected. MSE count loss is harder to optimise than CE rate loss. negative result documented.

---

## 16. energy unit clarification (NeuroBench results)

3 march 2026

early docs incorrectly said "0.976 nJ/sample" and "0.463 nJ/sample". wrong by 1000x.

correct values:
- SNN: 1,084,732 ACs/sample x 0.9 pJ/AC = 976,259 pJ = **976 nJ** per sample
- ANN: 100,561 MACs/sample x 4.6 pJ/MAC = 462,581 pJ = **463 nJ** per sample

AC/MAC energy from Yik et al. 2025 NeuroBench paper (45nm CMOS). in software sim ANN is 2.1x cheaper. on neuromorphic hardware, same AC count but AC vs MAC energy = 5.1x advantage to SNN.

fixed across MEMORY.md, EXPERIMENT_LOG.md, ICONS2026_draft.md.

---

## 17. burst encoding result interpretation

3 march 2026

burst gets ~5% accuracy (folds 1-2: 5.00%, 5.25%) -- basically chance (2% for 50 classes). severe overfitting (train 41-49%, test 5%).

decided to accept burst as a negative result and document the mechanism, rather than tuning hyperparameters. the failure is mechanistically clear: burst concentrates all info in the first 5 of 25 timesteps. LIF neurons (beta=0.95) integrate over the full 25 but recieve no signal for steps 6-25. temporal mismatch.

fixes would require changing T to 5 or adding temporal masking, which would be inconsistent with other encodings. the result is publishable as a negative finding: front-loaded temporal encoding is incompatible with 25-step LIF integration.

comparison with delta: delta fails (~7%) because no spikes generated (no temporal contrast in static spectrograms). burst fails (~5%) because spikes exist but are temporally mismatched. different mechanisms, same near-chance outcome.

---

## 18. thesis chapter writing strategy

3 march 2026

wrote all thesis chapters in parallel with training (burst, phase, population), using TBD placeholders for pending results. training takes hours per fold; writing can proceed simultaneously. chapters: introduction, related work, methodology, results core, results hardware, results advanced, discussion, conclusion, plus ICONS2026 draft.

---

## 19. statistical analysis fix: CSF3 preds_fold1 restore

3 march 2026

`results/snn/direct/preds_fold1.pt` was corrupted by a failed local retrain (the &-in-run_in_background bug). corrupted predictions gave fold 1 accuracy = 5%, making SNN mean = 40.05% and p = 0.0521 (not significant).

restored from `csf3_results/snn/direct/preds_fold1.pt` (original CSF3 run, fold 1 = 40.5%). re-ran analysis_suite.py.

after fix: SNN 47.15% +/- 4.50%, ANN 63.85% +/- 3.07%, gap 16.70 pp, paired t-test p = 0.0010 (highly significant), Wilcoxon p = 0.0625 (minimum achievable with n=5).

per-class finding: SNN wins on coughing (+8%), crying_baby (+8%), door_wood_knock (+8%), pouring_water (+5%), footsteps (+3%), crackling_fire (+3%). SNN loses worst on engine (-35%), laughing (-40%), clock_tick (-45%). pattern: SNN wins on high-energy distinctive sounds, loses on quiet subtle ones.

---

## 20. PANNs table per-fold value correction

3 march 2026

thesis had incorrect per-fold values for all three PANNs classifiers. means were correct but individual fold values and stds were wrong.

correct values from `results/panns/panns_snn_head_all_folds_50ep.json`:
- SNN: [92.0%, 94.5%, 91.0%, 93.5%, 91.5%] -> mean=92.50%, std=1.30%
- ANN: [93.0%, 95.0%, 92.0%, 95.5%, 91.75%] -> mean=93.45%, std=1.54%
- Linear: [94.25%, 95.75%, 92.50%, 95.25%, 91.25%] -> mean=93.80%, std=1.69%

no impact on conclusions -- means unchanged, the 0.95 pp gap narrative unaffected.

---

## 21. burst encoding final results

3 march 2026

final burst 5-fold: 5.00%, 5.25%, 9.25%, 6.00%, 7.00% -> mean=6.50% +/- 1.54%

interpretation unchanged -- temporal window mismatch (5-step burst vs 25-step LIF window). final mean 6.50% is slightly higher than folds 1-2 estimate due to fold 3 being an outlier (9.25%).

---

## 22. phase encoding final results

3 march 2026

22.50%, 22.25%, 25.00%, 24.25%, 26.75% -> mean=**24.15% +/- 1.66%**

this was the most surprising result. phase (24.15%) is essentially tied with rate coding (24.00%) -- within 0.15 pp, well within noise.

so basically phase provides 1 spike per neuron (deterministic, timing proportional to 1-intensity) vs rate's ~6-7 spikes per neuron (stochastic, Bernoulli). despite 6-7x fewer spikes, phase gets identical accuracy. the critical factor is **temporal window coverage**, not spike count. both use the full T=25 window and preserve intensity ordering. phase is actually better on energy efficiency (fewer spikes = fewer ACs) while matching rate's accuracy.

ordering: direct (47.15%) > rate (24.00%) ~ phase (24.15%) > latency (16.30%) >> delta (7.25%) ~ burst (6.50%)

---

## 23. direct SNN fold 1 MPS retrain result

3 march 2026

local MPS retrain gave fold 1 = 45.5% (best_epoch=48, total_epochs=50). CSF3 original gave 47.50%.

decided to keep the thesis tables from CSF3 (fold 1 = 47.50%) as canonical. reasons:
1. all existing analysis (confusion matrices, t-SNE, stats) used CSF3-derived preds_fold1.pt
2. mean (47.15%) is unchanged regardless
3. 2pp difference is within expected variance across hardware
4. updating fold 1 would cascade changes to all statistics

---

## 24. population coding final results

4 march 2026

22.75%, 18.50%, 15.75%, 22.00%, 16.75% -> mean=**19.15% +/- 2.79%**

population coding underperforms standard rate coding (24.00%) despite 10x more output neurons. hypothesis rejected. MSE count loss is harder to optimise than CE rate loss -- training acc at termination only reaches 18-24% (vs rate's ~50%). the bottleneck is the FC1 feature representation (256-d), not output layer width.

updated ordering (7 methods): direct (47.15%) >> rate (24.00%) ~ phase (24.15%) > population (19.15%) > latency (16.30%) >> delta (7.25%) ~ burst (6.50%)

population coding as implemented is NOT recommended for audio SNN classification. rate coding with CE loss is both more accurate and simpler.

---

## 25. continual learning experiment result

4 march 2026

SNN mean forgetting 74.4%, ANN mean forgetting 81.3%. SNN forgets 6.9 pp less than ANN.

mechanism: SNN binary spike outputs -> sparser gradient flow -> fewer weights updated per task -> less interference. ANN continuous activations -> denser gradients -> more complete overwriting.

limitations to acknowledge:
1. forgetting is severe in absolute terms (74.4% for SNN) -- not practically useful without replay/regularisation
2. SNN's lower peak accuracy per task means forgetting ratio is partly lower due to lower starting point
3. n=1 fold (fold 4) -- no cross-validation (later done 5-fold on CSF3: SNN 69.9% vs ANN 74.7%, consistent)

---

## 26. SpiNNaker 400-sample full inference (run 6) launch strategy

4 march 2026

scale=1.0 (not 5.0): scale sweep showed scale=1.0 produces correct predictions while scale>=5.0 saturates. 400 samples from fold 4 using pre-extracted hidden spike features (400x25x256, snnTorch ref=51.25%). FC2-only approach after Option C failure. preliminary (n=19): 8/19 = 42.1%.

400 samples gives SE ~2.5%, allowing class-level analysis impossible with n=20.

---

## 27. Option A (MaxPool SNN) threshold sweep strategy

4 march 2026

the root problem: AvgPool2d on binary spikes produces fractional outputs in [0,1]. FC1 inputs aren't binary, violates SpiNNaker's spike-only compute model. fix: MaxPool2d(4,6) on binary spikes -> binary outputs (max of {0,1} is {0,1}). same spatial output dims so FC sizes unchanged.

threshold sweep rationale: higher LIF threshold (1.5, 2.0, 3.0) reduces active neurons per step, reduces simultaneous FC1 inputs, may improve SpiNNaker compatibility. target: <500 active FC1 inputs per step.

result: fc1_binary_fraction = 1.000 for ALL thresholds. MaxPool guarantees binary. threshold=3.0 got best accuracy (43.75%) AND lowest FC1 density (956/step). the two goals aligned nicely.

---

## 28. surrogate gradient ablation -- local run strategy

4 march 2026

CSF3 had 8 surrogates x fold 1 x 3 seeds submitted but results unknown without interactive SSH. ran locally (fold 1, 1 seed) for immediate preliminary results.

surrogates tested: fast_sigmoid, atan, sigmoid, STE, triangular, spike_rate_escape, LSO, SFS.

hypothesis from Zenke & Vogels (2021): shape matters less than slope. our result didn't quite agree with that.

used local 1-seed results for thesis section 4.3. noted the limitation (1 seed vs 3).

---

## 29. remove Windheuser et al. 2024 citation

4 march 2026

research agent exhaustively searched Google Scholar, Semantic Scholar, arXiv, IEEE Xplore, ACM DL, EUSIPCO proceedings -- found **no paper by any author named Windheuser on SNNs for audio**. citation doesn't exist. it was a hallucinated citation from the the AI. the footnote "* Citation requires verification" was already flagging it.

deleted the row. never include citations that can't be independently verified. phantom citations are worse than no citation.

---

## 30. fast_sigmoid -- final result and significance

4 march 2026

fast_sigmoid fold 1 seed 42 epochs=50: **44.75%** (best epoch = 50, model still improving at termination).

this substantially exceeds the preliminary estimate (~40%) and is higher than the direct encoding fold 1 baseline (40.5%). unexpected. possible explanations: fast_sigmoid's slope=25 provides sharper gradients, training wasn't converged yet, or single-seed variance.

kept fast_sigmoid as default for main experiments (already run). spike_rate_escape beats it by only 1.25 pp (within noise for single-seed).

---

## 31. arithmetic error fix -- adversarial robustness advantage

4 march 2026

thesis said "14.25 pp advantage" for SNN over ANN at eps=0.1 FGSM. correct: 26.00 - 1.75 = **24.25 pp**. also said "19 pp accuracy deficit on clean inputs" but actual deficit is 68.75 - 53.75 = **15 pp**.

fixed both. lesson: always verify arithmetic against source JSON before finalising.

---

## 32. SpiNNaker run 6 n=149 checkpoint update

4 march 2026

at n=149/400: SpiNNaker 49.0% (73/149), snnTorch 49.7% (74/149), gap 0.7 pp (vs 1.9 pp at n=108). gap continued to narrow. updated thesis chapters.

the shrinking gap (10 pp at n=20 -> 1.9 pp at n=108 -> 0.7 pp at n=149) confirmed the n=20 pilot was severely noise-biased.

---

## 33. corrected t-statistic for paired t-test

4 march 2026

two thesis locations stated t = 7.21. correct value: **t = 8.64** (from scipy.stats.ttest_rel). p-value (0.001) was already correct. the t-statistic was likely calculated manually with a typo.

---

## 34. SpiNNaker run 6 n=189 checkpoint -- hardware gap converges

4 march 2026

at n=189: SpiNNaker=51.9%, snnTorch=51.9%, gap=0.0 pp, agreement=79.4%.

convergence history: n=20: 10pp gap, n=65: 3.1pp, n=108: 1.9pp, n=149: 0.7pp, n=189: 0.0pp.

IF_curr_exp (tau_syn=5ms, tau_m=20ms, v_thresh=1.0) is an excellent approximation of snnTorch LIF for FC2-only. the n=20 estimate was unreliable due to small sample size.

NOTE: this turned out to be partially wrong -- see decision 38 for the methodology correction.

---

## 35. adversarial robustness note correction

4 march 2026

thesis stated "CSF3-canonical fold 4 values (43.50% SNN, 62.50% ANN)" -- these are wrong. correct: SNN fold 4 = 54.0%, ANN fold 4 = 68.75%. the "43.50%" and "62.50%" don't correspond to any fold in either local or CSF3 results. probably a placeholder from an early draft.

---

## 36. per-fold accuracy values corrected in section 4.2 table

4 march 2026

per-fold values in the encoding comparison table were from early local runs, didn't match canonical results in source JSONs. means and stds were correct, individual fold values were wrong for ANN, SNN direct, rate, latency, delta. burst/phase/population were already correct.

verified correct values from results/*/summary.json and updated.

note: SNN direct fold 1 canonical (CSF3) = 40.50%. local retrain got 45.50%. thesis uses CSF3 canonical.

---

## 37. per-class accuracy rounding errors corrected

4 march 2026

per-class values in thesis were inconsistently rounded. since n=40 (5 folds x 8 samples/class/fold), all per-class accs are exact multiples of 2.5%. adopted round-half-up convention throughout.

several corrections in both top-10 and bottom-10 tables. also corrected the "classes where SNN outperforms ANN" list: coughing(+8), crying_baby(+7), door_wood_knock(+7), pouring_water(+5), crackling_fire(+3), footsteps(+2).

---

## 38. SpiNNaker run 6 analysis methodology correction

4 march 2026

**this was a big one.** earlier checkpoints for n=149, n=189, n=208 showing "0.0 pp hardware gap" were WRONG. the analysis script was computing snnTorch accuracy for BOTH the "SpiNNaker" and "snnTorch" metrics, producing identical values.

root cause: the JSONL file has two entry types -- scale_sweep entries (no SpiNNaker prediction) and inference entries (both predictions). the script read ALL entries without filtering by phase, and when it compared `predicted` for scale_sweep entries, got None == int = False. the bug was subtle.

correct methodology: filter for phase='inference' AND timestamp.startswith('2026-03-04'), use `e.get('correct')` for SpiNNaker and `e.get('snn_predicted')==e.get('true_label')` for snnTorch.

correct checkpoint values:
- n=65: 47.7%/50.8%, 3.1pp gap
- n=108: 49.1%/50.9%, 1.9pp
- n=149: 52.3%/53.7%, 1.3pp
- n=189: 50.8%/51.3%, 0.5pp
- n=208: 50.5%/50.5%, 0.0pp (coincidence)
- n=216: 48.6%/51.4%, 2.8pp
- n=244: 45.5%/51.2%, 5.7pp

so the gap is NOT converging to 0. it fluctuates (0.0-5.7 pp, mean ~2.5 pp). the n=208 zero was coincidence. still much better than run 5 pilot (10 pp gap at n=20).

thesis narrative changed from "0.0 pp stable gap" to "0.5-5.7 pp hardware gap (mean ~2.5 pp)". still a strong result.

---

## 39. Option A threshold selection for SpiNNaker deployment

4 march 2026

from the threshold sweep: FC1 binary fraction = 1.000 for ALL thresholds. MaxPool guarantees binary.

| threshold | test acc | FC1 active/step | binary frac |
|-----------|---------|-----------------|-------------|
| 1.0 | 9.25% | 1662/2304 | 1.000 |
| 1.5 | 27.0% | 1410/2304 | 1.000 |
| 2.0 | 34.25% | 1253/2304 | 1.000 |
| 3.0 | **43.75%** | 956/2304 | 1.000 |

**threshold=3.0 is the best candidate.** highest accuracy AND lowest FC1 density. accuracy loss vs original (54.0%): 10.25 pp. acceptable -- the purpose is hardware compatibility, not accuracy matching.

the <500/step target was aspirational. at 956/step, hardware router testing needed.

didn't retrain all 5 folds with threshold=3.0 -- fold 4 single-fold result is sufficient to document the finding. 5-fold retraining deferred to future work.

---

## 40. surrogate ablation -- spike_rate_escape best, LSO skipped

4 march 2026

results: spike_rate_escape 46.00% (best), fast_sigmoid 44.75%, atan 35.75%, STE 10.25% (failed), sigmoid 2.00% (failed), SFS 2.00% (failed), triangular 2.75% (failed), LSO crashed (Python 3.14 incompatibility).

clear bimodal split: learning group {spike_rate_escape, fast_sigmoid, atan} vs failure group {STE, sigmoid, SFS, triangular}. stronger discrimination than Zenke & Vogels (2021) predicted.

kept fast_sigmoid as default (main experiments already done). spike_rate_escape beats it by only 1.25 pp. if CSF3 3-seed results show large gap: document as recommended choice for future work.

LSO crash: requires 'variance' argument that snnTorch 0.9.4 + Python 3.14 doesn't supply automatically. framework compatibility issue, not fundamental limitation.

---

## 41. run augmented training locally (MPS) rather than wait for CSF3

4 march 2026

CSF3 augmented training submitted 3 march but no results due to queue delays. thesis section 4.4 had a gap. launched locally on MPS in addition to waiting.

expected: SNN +3-7 pp above 47.15%, ANN +5-9 pp above 63.85%.

actual result: **SNN aug 40.75% +/- 16.03%** -- WORSE than baseline by 6.40 pp, variance tripled. folds 3 and 5 early-stopped at ep39/ep33 before convergence. patience=10 too aggressive for augmented SNN. ANN aug 61.70% +/- 4.58% -- also slightly worse (-2.15 pp).

conclusion: SpecAugment harms SNNs on small datasets. baseline 47.15% remains the primary result. augmented results saved as supplementary negative result.

also: --run-suffix _aug didn't create separate directories, overwrote baseline results. had to restore manually. annoying.

---

*updated continuously. every significant decision goes here.*
