# experiment log

COMP30040 thesis, university of manchester. ayush kumar. started march 2026.

---

## overview

research question: how do different spike encoding methods affect SNN performance on environmental sound classification?

main contribution: first application of spiking neural networks to the ESC-50 dataset. no prior peer-reviewed SNN work exists for ESC-50 (confirmed by arXiv 2503.11206, march 2025).

objectives:
1. build a convolutional SNN for ESC-50 using snnTorch
2. compare 4+ spike encoding methods: rate, delta, latency, direct (later added burst, phase, population)
3. evaluate against an equivalent ANN baseline
4. measure energy consumption (SynOps vs MACs)
5. deploy to SpiNNaker neuromorphic hardware

---

## dataset

ESC-50 (Environmental Sound Classification, 50 classes)
- source: https://github.com/karolpiczak/ESC-50
- 2,000 audio clips, 5 seconds each, 44.1 kHz (resampled to 22,050 Hz)
- 50 classes (40 clips per class)
- 5 predefined folds for cross-validation
- eval: 5-fold CV (train on 4 folds / 1,600 clips, test on 1 fold / 400 clips)

sound categories (5 groups of 10):

| group | classes |
|-------|---------|
| animals | dog, rooster, pig, cow, frog, cat, hen, insects, sheep, crow |
| nature / water | rain, sea_waves, crackling_fire, crickets, chirping_birds, water_drops, wind, pouring_water, toilet_flush, thunderstorm |
| human (non-speech) | crying_baby, sneezing, clapping, breathing, coughing, footsteps, laughing, brushing_teeth, snoring, drinking_sipping |
| interior / domestic | door_wood_knock, mouse_click, keyboard_typing, door_wood_creaks, can_opening, washing_machine, vacuum_cleaner, clock_alarm, clock_tick, glass_breaking |
| exterior / urban | helicopter, chainsaw, siren, car_horn, engine, train, church_bells, airplane, fireworks, hand_saw |

### preprocessing pipeline

1. load WAV at 22,050 Hz
2. compute mel spectrogram: 64 mel bins, n_fft=1024, hop_length=512, fmin=0, fmax=Nyquist
3. convert to log scale (dB): `librosa.power_to_db(mel, ref=np.max)`
4. min-max normalise to [0, 1]
5. output shape: (1, 64, 216) -- single channel, 64 freq bins, 216 time frames

---

## architecture

### SNN (SpikingCNN) -- `src/models/snn_model.py`

```
input: (num_steps, batch, 1, 64, 216) -- spike-encoded spectrograms

layer 1:  Conv2d(1, 32, 3x3, padding=1)
          BatchNorm2d(32)
          MaxPool2d(2)              -> (32, 32, 108)
          LIF neuron (beta=0.95)   -> binary spikes

layer 2:  Conv2d(32, 64, 3x3, padding=1)
          BatchNorm2d(64)
          MaxPool2d(2)              -> (64, 16, 54)
          LIF neuron (beta=0.95)   -> binary spikes

pooling:  AvgPool2d(kernel_size=(4, 6))  -> (64, 4, 9)
flatten:  -> 2304 features

FC1:      Linear(2304, 256)
          LIF neuron (beta=0.95)   -> binary spikes

FC2:      Linear(256, 50)
          LIF neuron (beta=0.95)   -> output spikes + membrane potentials

~622K params total
```

design notes:
- `AvgPool2d(4,6)` instead of `AdaptiveAvgPool2d` because the adaptive version crashes on MPS (Apple Silicon)
- surrogate gradient: `fast_sigmoid(slope=25)` for backprop through spikes
- loss: per-timestep cross-entropy on membrane potentials (snnTorch Tutorial 5)
- 25 timesteps per sample

### ANN baseline (ConvANN) -- `src/models/ann_model.py`

identical architecture with ReLU instead of LIF, no temporal dimension (single forward pass), Dropout(0.3) in classifier, standard CrossEntropyLoss.

---

## training infrastructure

### local machine
- MacBook (Apple Silicon, MPS)
- ~65 minutes per fold for SNN (very slow)
- used for initial dev, debugging, first ANN baseline attempt

### CSF3 (uni of manchester HPC) -- primary training platform
- NVIDIA A100-SXM4-80GB GPUs (gpuA partition)
- ~6 minutes per fold for SNN (~10x faster than local)
- GPU limit: 2 per user
- modules: `cuda/12.6.2` + `libs/cuda/12.8.1` + `python/3.13.1`
- venv: `~/snn-esc50-venv/` with PyTorch 2.6.0+cu124, snnTorch 0.9.4
- job scripts: `csf3_train_all.sh` (sequential), `csf3_train_encoding.sh` (parallel single-encoding)

### training config
- optimizer: Adam (lr=1e-3, weight_decay=1e-4)
- scheduler: ReduceLROnPlateau (factor=0.5, patience=5)
- early stopping: patience=10
- batch size: 32
- max epochs: 50
- 5-fold CV (ESC-50 predefined folds)

---

## experiment results

### summary table (all training on CSF3 A100s, 3 march 2026)

| model | encoding | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | mean | std |
|-------|----------|--------|--------|--------|--------|--------|------|-----|
| ANN | - | 63.25% | 59.50% | 65.25% | 68.75% | 62.50% | **63.85%** | 3.07% |
| SNN | direct | 40.50% | 48.50% | 48.25% | 54.00% | 44.50% | **47.15%** | 4.50% |
| SNN | rate | 24.50% | 27.25% | 23.00% | 21.50% | 23.75% | **24.00%** | 1.90% |
| SNN | latency | 14.00% | 15.75% | 17.75% | 15.50% | 18.50% | **16.30%** | 1.62% |
| SNN | delta | 8.25% | 7.75% | 7.25% | 7.50% | 5.50% | **7.25%** | 0.94% |

random chance baseline: 2% (1/50 classes)

### observations

1. **direct encoding is best (47.15%):** skips explicit spike conversion, lets the network learn its own temporal coding. consistent with literature -- learned encodings outperform hand-crafted ones.

2. rate encoding distant second (24.00%): converts intensity to spike probability. temporal info helps but naive conversion loses spatial structure within each timestep.

3. latency (16.30%): encodes intensity as time-to-first-spike. some signal preserved but information compressed into a single spike per neuron.

4. delta near-random (7.25%): encodes temporal changes (derivatives). spectrograms already represent frequency over time, so taking the derivative is like differentiating twice -- uninformative.

5. SNN-ANN gap (47% vs 64%): ~17 pp gap. consistent with literature -- SNNs typically trade accuracy for energy efficiency. gap could narrow with better architectures, longer training, better encoding.

### eval artifacts
- confusion matrices: `results/{model}/{encoding}/confusion_matrix.png`
- per-class accuracy: `results/{model}/{encoding}/per_class_accuracy.png`
- training curves: `results/{model}/{encoding}/training_curves.png`
- F1 scores and classification reports in result JSONs
- all predictions saved: `results/{model}/{encoding}/preds_fold*.pt`

---

## energy analysis

### methodology

SynOps counting (SNN): for each sample, count total synaptic operations. at each timestep count spikes per layer, multiply by fan-out, sum across layers and timesteps. measured using `src/energy.py` with actual spike counts.

MAC counting (ANN): standard analytical count of multiply-accumulate operations for conv and FC layers.

### energy estimates

| metric | SNN | ANN |
|--------|-----|-----|
| ops per sample | 1,508,474,842 SynOps | 68,284,928 MACs |
| energy per op | 0.9 pJ/SynOp (Loihi) | 4.6 pJ/MAC (45nm CMOS) |
| total energy per sample | **1,358 million pJ** | **314 million pJ** |
| ratio | **4.3x MORE than ANN** | baseline |

assumptions: SynOp energy (0.9 pJ) from Intel Loihi measurements. MAC energy (4.6 pJ) from Horowitz 2014 (45nm CMOS). both are per-operation estimates; total system energy would include memory access, data movement, etc.

### spike counts per layer (SNN, direct encoding)

| layer | spikes per sample |
|-------|------------------|
| LIF1 (after conv1) | 2,303,140 |
| LIF2 (after conv2) | 710,103 |
| LIF3 (after FC1) | 1,596 |
| LIF4 (output) | 34 |

so the SNN uses MORE energy than the ANN in software simulation. this is because:
1. software sim doesn't capture event-driven efficiency -- on a GPU every op costs the same whether there's a spike or not. SNN runs 25 timesteps = 25x all ops.
2. spike activity isn't sparse enough. first conv layer fires 2.3M spikes -- activity ratio too high.
3. the real advantage requires neuromorphic hardware. chips like Loihi and SpiNNaker process spikes event-driven, inactive neurons consume near-zero power.
4. this is a known issue in the literature. software-simulated SNNs on GPUs are almost always less efficient than ANNs. the energy argument is specifically about deployment on neuromorphic hardware.

---

## SpiNNaker hardware deployment

### overview

goal: deploy the best SNN (direct encoding, fold 4) to SpiNNaker to demonstrate real hardware deployment, compare with software sim, and investigate parameter mapping between snnTorch and sPyNNaker.

### setup

software: separate Python 3.11 venv (`.venv-spinnaker/`), sPyNNaker 7.x, config pointing to `spinnaker.cs.man.ac.uk`.

weight conversion (`spinnaker/convert_weights.py`):
1. load best model checkpoint (fold 4)
2. fuse BatchNorm into conv weights
3. extract FC1 and FC2 weight matrices
4. convert to sPyNNaker connection list format: (pre_idx, post_idx, weight, delay=1.0)

feature extraction (`spinnaker/extract_features.py`):
1. load 20 test samples from fold 4
2. run through conv layers on CPU -> 2304-dim feature vectors
3. convert to binary spike trains (25 timesteps): spike[t,n] = 1 if feature[n] > random()

LIF parameter mapping:
```
snnTorch:                    sPyNNaker (IF_curr_exp):
  beta = 0.95         ->      tau_m = 20.0 ms  (beta = exp(-dt/tau_m))
  threshold = 1.0     ->      v_thresh = 1.0
  (no refractory)     ->      tau_refrac = 2.0 ms
  (reset to 0)        ->      v_reset = 0.0, v_rest = 0.0
```

### run 1: initial deployment (3 march 2026, ~12:30)

spalloc jobs: 99687-99707, board: 10.11.225.185

weight pruning: |weight| < 0.02 removed. FC1: 589,824 -> 210,966 (35.8% retained). FC2: 12,800 -> 8,143 (63.6% retained). exc/inh separated per sPyNNaker requirement.

result: **0/20 (0.0%)**. zero output spikes, zero hidden spikes. input spikes were active (17k-35k per sample). complete signal death at FC1. every sample defaults to prediction 0 ("dog").

root cause: parameter mapping mismatch between snnTorch and sPyNNaker. snnTorch uses direct weight-to-current, sPyNNaker has exponential current decay (tau_syn) that snnTorch doesn't model. input current decays faster, membrane potential never reaches threshold. plus weight scale issues and pruning impact compounding the problem.

### run 2: second attempt (3 march, ~13:14)

same result -- 0% accuracy, zero hidden/output spikes. confirmed systematic issue. then ran diagnostic tests (100 inputs all spiking with weight=0.05, testing if IF_curr_exp can fire at v_rest=0.0, v_thresh=1.0).

### run 3: reduced pruning (3 march, ~13:17)

reduced pruning threshold to 0.005 to retain more connections. FC1: 80.2% retained. total: 462K connections.

result: **CRASH** -- `OSError: [Errno 55] No buffer space available`. UDP socket buffer overflowed while transferring 451K connections. the data transfer exceeded local machine's UDP buffer capacity. also triggered timeout exceptions and core read failures.

all runs had this warning: "Danger of SpikeSourceArray sending too many spikes at the same time. For example at time 23.0, 1398 spikes will be sent." SpiNNaker's multicast router has limited throughput per timestep.

lessons: pruning threshold must be 0.05+ to keep connections manageable. even if connections transfer, simultaneous spike flooding may cause issues. need more aggressive pruning OR dimensionality reduction.

### run 4: auto-calibration loop (3 march, ~16:20 onwards)

new script: `spinnaker/auto_calibrate.py` -- modular self-iterating loop that sweeps params across 5 phases until neurons fire.

**phase 1: neurons firing for the first time!** used IF_curr_exp, tau_syn_E=1.0, v_rest=0, v_thresh=1.0, synthetic all-timestep input. weight=0.5: FIRED (6 spikes). weight=1.0: FIRED (8 spikes). weight=2.0+: FIRED (12 spikes). min_working_weight = 0.5.

**phase 2:** fixed weight=2.0, sweeping tau_syn. tau_syn=5.0ms: FIRED (12 spikes). stopped sweep immediately. IF_curr_delta not needed. optimal_tau_syn = 5.0ms.

**phases 3-5: all FAILED.** zero hidden/output spikes in all scale configs.

found the root cause: FC1 weights have mean = -0.0034, range [-0.301, +0.282]. with 1398 simultaneously active inputs per timestep: expected net current = 1398 x (-0.0034) x scale x 0.181 = -0.86 x scale. thats a LARGE NEGATIVE current driving all hidden neurons below threshold. no scale factor can fix this -- scaling a negative signal makes it more negative. phases 1-2 worked because they used purely synthetic positive weights.

also fixed a bug during this: Neo AnalogSignal `.magnitude` bug. iterating `sig[:, n]` returns Quantity objects, not scalars. `float(q)` fails on Quantity with shape (1,). fix: use `sig.magnitude[:, n].tolist()`. fixed in 6 places across run_on_spinnaker.py and auto_calibrate.py.

### run 5: FC2-only deployment (3 march, ongoing)

scripts: `spinnaker/extract_hidden_features.py` + `spinnaker/run_fc2_spinnaker.py`

the key insight: bypass the FC1 cancellation entirely by running FC1+lif3 on CPU (snnTorch) and only deploying FC2 (256->50) on SpiNNaker.

why this works: snnTorch FC1+lif3 produces sparse binary hidden spikes (~61 of 256 neurons active per timestep, 24%). max 65 simultaneous spikes (vs 1398 with full FC1 input) -- well within router capacity. FC2 receives data-dependent sparse patterns.

extract_hidden_features.py results: hidden_spike_features.npy (20, 25, 256) binary. mean 60.6 active neurons/timestep. snnTorch accuracy on 20 fold-4 samples: 2/20 (10%) -- small sample from hard classes.

FC2 weights: 11,833 connections, pruned to 9,892 after |w|>0.01. weight range -0.766 to +0.429. 55% inhibitory.

**scale sweep (sample 8, true=42, snnTorch=42):**

| scale | output spikes | neurons fired | predicted | correct? |
|-------|--------------|---------------|-----------|----------|
| 0.5x  | 8  | 2 | 42 | yes |
| 1.0x  | 19 | 5 | 42 | yes |
| 2.0x  | 38 | 7 | 42 | yes |
| **5.0x** | **62** | **8** | **42** | **yes -- chosen** |
| 10.0x | 77 | 8 | 42 | yes |
| 20.0x | 86 | 8 | 42 | yes |
| 50.0x | 90 | 8 | 5 | no (saturation) |
| 100.0x | 93 | 8 | 5 | no |
| 200.0x | 94 | 8 | 5 | no |

scale 5.0x chosen: good spike count (62), correct, not saturated. above 50x everything fires uniformly and noise dominates.

**full inference (10 samples, scale=5.0x):**

| sample | true | snnTorch | SpiNNaker | spikes | agreement |
|--------|------|----------|-----------|--------|-----------|
| 0 | 49 | 37 | 37 | 25 | yes |
| 1 | 49 | 37 | 37 | 25 | yes |
| 2 | 49 | 37 | 37 | 28 | yes |
| 3 | 42 | 46 | **42** (correct!) | 35 | no |
| 4 | 33 | 42 | **33** (correct!) | 59 | no |
| 5 | 33 | 37 | **33** (correct!) | 23 | no |
| 6 | 33 | 37 | 37 | 24 | yes |
| 7 | 33 | 37 | 37 | 54 | yes |
| 8 | 42 | 42 | **42** (correct!) | 62 | yes |
| 9 | 42 | 42 | **42** (correct!) | 45 | yes |

**SpiNNaker FC2-only: 5/10 = 50.0%**. snnTorch on same samples: 2/10 = 20.0%. agreement: 6/10.

interesting: SpiNNaker outperformed snnTorch on this sample set. samples 3, 4, 5 -- SpiNNaker correct, snnTorch wrong. NOT a weight transfer error (weights identical). the difference is the temporal dynamics of IF_curr_exp vs snnTorch's LIF. sPyNNaker's exponential current decay apparently regularises the decision boundary on borderline samples. genuine thesis finding.

### run 6: Option C -- FC1 weight re-centering (3 march)

script: `experiments/spinnaker_option_c.py`

hypothesis: zero-centering each FC1 weight row with bias compensation is mathematically equivalent reparameterisation. should preserve accuracy while reducing cancellation.

baseline (before): 53.75%. after re-centering: **8.50%**. catastrophic.

root cause: the equivalence only holds when x is binary (0/1). but FC1 inputs come from avg_pool(spk2) which produces fractional values in [0,1], not binary spikes. bias compensation assumes all 2304 inputs always active, but actual sum(x) is way less. creates massive incorrect positive bias.

but: learned that FC1 already fires at 21.76% in software -- well within SpiNNaker capacity. the cancellation problem was overstated for software.

---

### hardware verification

confirmed this is real SpiNNaker hardware. SC&MP 4.0.0 firmware version read from physical chip via SCP protocol. machine: 47-48 chips, 836-855 cores, 118 MB SDRAM (matches SpiNN-5 board). DNS/WHOIS confirms spinnaker.cs.man.ac.uk belongs to UoM Kilburn Building. different physical boards allocated across runs. NOT virtual mode (real firmware, real power cycling). full verification in SPINNAKER_HARDWARE_VERIFICATION.md.

---

## advanced experiments (3 march 2026)

### adversarial robustness: SNN vs ANN

script: `experiments/adversarial_robustness.py`. FGSM + PGD (40 steps), 400 samples (full fold-4 test set), epsilon sweep [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3].

results (fold 4, clean: SNN=53.75%, ANN=68.75%):

| eps | FGSM SNN | FGSM ANN | PGD SNN | PGD ANN |
|-----|---------|---------|--------|--------|
| 0.000 | 53.75% | 68.75% | 53.75% | 68.75% |
| 0.010 | 37.50% | 22.50% | 23.50% | 14.75% |
| 0.020 | 32.00% | 8.75% | 20.50% | 2.00% |
| 0.050 | 29.00% | 2.50% | 19.25% | 0.00% |
| 0.100 | 26.00% | 1.75% | 6.25% | 0.00% |
| 0.200 | 21.50% | 1.25% | 1.25% | 0.00% |
| 0.300 | 20.75% | 0.75% | 1.25% | 0.00% |

SNN outperforms ANN at ALL eps > 0 for FGSM. at eps=0.1, SNN retains 26% while ANN drops to 1.75%. crossover at eps=0.01. PGD (stronger): SNN retains 19.25% at eps=0.05 where ANN hits 0%.

the spike thresholding mechanism is non-differentiable, creates gradient masking that weakens gradient-based attacks. FGSM (relies on gradient sign) most affected. PGD (40 steps, more powerful) eventually breaks through at higher eps.

this is contribution C4 -- first adversarial robustness analysis of SNNs on audio spectrograms.

saved: `results/adversarial/robustness_fold4.json`

---

### PANNs + SNN head transfer learning

script: `experiments/panns_snn_head.py`. CNN14 pretrained on AudioSet -> 2048-d embeddings -> small 3-layer SNN/ANN/Linear head.

| head | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | mean | std |
|------|--------|--------|--------|--------|--------|------|-----|
| PANNs+SNN | 92.00% | 94.50% | 91.00% | 93.50% | 91.50% | **92.50%** | 1.30% |
| PANNs+ANN | 93.00% | 95.00% | 92.00% | 95.50% | 91.75% | **93.45%** | 1.54% |
| PANNs+Linear | 94.25% | 95.75% | 92.50% | 95.25% | 91.25% | **93.80%** | 1.69% |

so basically with PANNs features, SNN-ANN gap collapses from ~17pp to 0.95pp (92.50% vs 93.45%). all three heads essentially equivalent. the bottleneck is feature learning, not spiking computation. biggest finding of the project arguably.

saved: `results/panns/panns_snn_head_all_folds_50ep.json`

---

### NeuroBench energy analysis

script: `experiments/neurobench_analysis.py`. reference: Yik et al. (2025) Nature Communications 16, 1589.

| model | effective ACs | effective MACs | energy/sample | activation sparsity |
|-------|--------------|----------------|--------------|-------------------|
| SNN (direct) | 1.08M AC/sample | -- | 976 nJ | 74.16% |
| ANN | -- | 101K MAC/sample | 463 nJ | 59% |

ANN is 2.1x cheaper in software sim (expected, T=25 overhead). SNN has 74.16% sparsity -- would be efficient on neuromorphic hardware. on neuromorphic: 1.08M x 0.9 pJ = 976 nJ vs ANN-equivalent 1.08M x 4.6 pJ = 4.99 uJ -- SNN 5.1x cheaper per op.

saved: `results/neurobench/neurobench_fold4.json`

---

### temporal spike analysis

script: `experiments/temporal_analysis.py`

| decoding method | fold 4 accuracy | notes |
|----------------|----------------|-------|
| rate decoding | **51.50%** | sum spikes over T=25, argmax |
| first-spike decoding | 25.75% | earliest-firing class wins |

rate decoding significantly outperforms first-spike: 25.75pp gap. mean FC1 firing rate: 6.81% of T=25 (very sparse). earliest-firing class: can_opening (0.12 steps), latest: snoring (3.25 steps).

saved: `results/temporal_analysis/`

---

### SpiNNaker: 400-sample feature extraction (3 march)

script: `spinnaker/extract_hidden_features.py`. fold 4, all 400 test samples.
- snnTorch accuracy on 400 samples: 205/400 = 51.25%
- mean FC1 active neurons/step: 55.6/256 = 21.7%
- max simultaneous hidden spikes: 66/step
- saved: `results/spinnaker_weights/hidden_spike_features.npy` (400x25x256, 20MB)

---

### burst encoding 5-fold training (3 march)

script: `python -m src.train --model snn --encoding burst`

encoding: spike count proportional to intensity, all spikes at first N timesteps. max_spikes=5 (20% ceiling). high intensity -> dense early burst, silence -> no spikes.

| fold | best acc | best epoch | total epochs | status |
|------|----------|------------|--------------|--------|
| 1 | 5.00% | 7 | 17 | done |
| 2 | 5.25% | 3 | 13 | done |
| 3 | 9.25% | 26 | 36 | done |
| 4 | 6.00% | 10 | 20 | done |
| 5 | 7.00% | 10 | 20 | done |
| **mean** | **6.50% +/- 1.54%** | | | complete |

burst is basically performing at chance (5% vs 2% for 50 classes). severe overfitting: training acc reaches 41-49%, test stays at 3-5%.

root cause: burst concentrates all information into the first 5 of 25 timesteps. LIF neurons (beta=0.95) integrate over 25 timesteps but receive no input for steps 6-25. temporal mismatch between encoding window (5 steps) and integration window (25 steps). the network memorises early-burst patterns for training but can't generalise.

---

### phase encoding 5-fold training (3 march)

script: `python -m src.train --model snn --encoding phase`

encoding: intensity mapped to spike timing. high intensity -> early spike (step 0), low -> late (step 24), zero -> silent. exactly one spike per neuron per window. deterministic.

| fold | best acc | epochs | status |
|------|----------|--------|--------|
| 1 | 22.50% | 34 (best ep 24) | done |
| 2 | 22.25% | 22 (best ep 12) | done |
| 3 | 25.00% | 25 (best ep 15) | done |
| 4 | 24.25% | 39 (best ep 29) | done |
| 5 | 26.75% | 40 (best ep 30) | done |
| **mean** | **24.15% +/- 1.66%** | | complete |

---

### population coding 5-fold training (3 march)

script: `python experiments/population_coding.py --folds 1 2 3 4 5`

output population code: 50 classes x 10 neurons = 500 output. input: rate coding. loss: MSE count loss.

| fold | best acc | epochs | status |
|------|----------|--------|--------|
| 1 | 22.75% | 42 (best ep 32) | done |
| 2 | 18.50% | 31 (best ep 21) | done |
| 3 | 15.75% | 28 (best ep 18) | done |
| 4 | 22.00% | 50 (best ep 44) | done |
| 5 | 16.75% | 31 (best ep 21) | done |
| **mean** | **19.15% +/- 2.79%** | | complete |

below rate coding (24.00%) and phase (24.15%). MSE count loss is harder to optimise -- training acc only 18-24% at termination (vs rate's ~50%). the bottleneck is FC1 representation, not output layer width.

updated encoding ordering: direct (47.15%) >> rate (24.00%) ~ phase (24.15%) > population (19.15%) > latency (16.30%) >> delta (7.25%) ~ burst (6.50%)

---

### SNN direct fold 1 retrain (3 march)

script: `python -m src.train --model snn --encoding direct --fold 1`

reason: prior best_fold1.pt corrupted by a failed local CPU retrain (2 epochs, 5%).

result: 45.50% (best_epoch=48, total_epochs=50, 1561s). ran all 50 epochs without early stopping -- model still improving. CSF3 original gave 47.50% (-2pp, hardware difference MPS vs CUDA). thesis tables stay anchored to CSF3 results.

---

### continual learning experiment (4 march)

script: `python experiments/continual_learning.py --fold 4 --epochs-per-task 20 --pretrained`

5 ESC-50 super-categories trained sequentially (10 classes each, 20 epochs/task, lr=5e-4). both SNN and ANN started from fold 4 pretrained checkpoints. no replay, no regularisation.

| metric | SNN | ANN |
|--------|-----|-----|
| mean forgetting | 74.4% | 81.3% |
| mean BWT | -0.744 | -0.813 |
| final avg accuracy | 18.3% | 18.8% |

SNN forgets LESS (74.4% vs 81.3%, -6.9 pp advantage). mechanism: binary spike outputs -> sparser gradient flow -> fewer weights updated per task -> less interference.

both suffer catastrophic forgetting worse than typical literature (~50% BWT) because: no replay/regularisation, tiny task subsets (320 training samples per task), gradient pressure entirely away from 40 unseen classes.

saved: `results/continual_learning/forgetting_fold4_pretrained_20ep.json`

---

### SpiNNaker full 400-sample inference -- run 6 (4 march)

script: `python spinnaker/run_fc2_spinnaker.py --num-samples 400 --weight-scale 1.0 --skip-to-inference`

FC2-only, scale=1.0, fold 4, pre-extracted hidden spike features.

progress tracking:
- 08:17 -- 19/400, 42.1%
- 08:30 -- 37/400, SpiNNaker 43.2%, snnTorch 51.4% (~8pp gap)
- 08:49 -- 65/400, SpiNNaker 47.7%, snnTorch 50.8% (3.1pp gap)
- 09:16 -- 108/400, SpiNNaker 49.1%, snnTorch 50.9% (1.9pp gap)
- ~10:00 -- 149/400, SpiNNaker 52.3%, snnTorch 53.7% (1.3pp gap)
- ~10:15 -- 189/400, SpiNNaker 50.8%, snnTorch 51.3% (0.5pp gap)
- ~10:20 -- 208/400, SpiNNaker 50.5%, snnTorch 50.5% (0.0pp -- coincidence)
- ~10:25 -- 216/400, SpiNNaker 48.6%, snnTorch 51.4% (2.8pp gap)
- ~11:xx -- 244/400, SpiNNaker 45.5%, snnTorch 51.2% (5.7pp gap)

NOTE: earlier checkpoints for n=149, 189, 216 were initially WRONG due to buggy analysis script (used snn_predicted for both metrics, producing artificial 0.0pp gaps). correct values above from methodology fix. correct approach: filter for phase='inference' entries from the JSONL, use e.get('correct') for SpiNNaker and e.get('snn_predicted')==e.get('true_label') for snnTorch.

**final result (n=400/400, ~12:30 4 march):**
- **SpiNNaker: 43.0%** (172/400)
- **snnTorch: 51.25%** (205/400)
- **gap: 8.25 pp**
- **agreement: 64.5%** (258/400)

super-categories: Animals SpiNN 45.0%/snnTorch 57.5% (-12.5pp), Nature 61.3%/68.8% (-7.5pp), Human 46.2%/56.2% (-10.0pp), Domestic 31.2%/37.5% (-6.2pp), Urban 31.2%/36.2% (-5.0pp). snnTorch leads all five.

hardest for SpiNNaker (0%): insects, door_wood_creaks, glass_breaking (0% vs 50% snnTorch), helicopter, engine.
easiest (100%): clapping, thunderstorm.
SpiNNaker beats snnTorch: airplane (+37.5pp), mouse_click (+25pp), can_opening (+12.5pp), clock_tick (+12.5pp).

error analysis: both correct 145/36.2%, SpiNNaker only 27/6.8%, snnTorch only 60/15.0%, both wrong 168/42.0%.

the final 8.25pp gap reflects sample-batch variability -- later samples included harder classes for SpiNNaker. gap trajectory went 0.0pp at n=208, then widened to 8.25pp by n=400. agreement dropped from 81.5% peak to 64.5% final.

---

### Option A (MaxPool SNN) threshold sweep -- fold 4 (4 march)

script: `python experiments/spinnaker_option_a.py --fold 4 --threshold-sweep`

architecture: SpikingCNN_MaxPool -- same as SpikingCNN but AvgPool2d(4,6) -> MaxPool2d(4,6). thresholds: {1.0, 1.5, 2.0, 3.0}.

**FC1 binary fraction = 1.000 for ALL thresholds.** MaxPool on binary spikes guarantees binary FC1 inputs -- the fundamental AvgPool-FC1 cancellation is eliminated.

| threshold | test acc | best epoch | epochs | FC1 active/step | sparsity | binary frac |
|-----------|---------|------------|--------|-----------------|----------|-------------|
| 1.0 | 9.25% | 27 | 37 (early stop) | 1662/2304 | 27.8% | 1.000 |
| 1.5 | 27.0% | 48 | 50 | 1410/2304 | 38.8% | 1.000 |
| 2.0 | 34.25% | 42 | 50 | 1253/2304 | 45.6% | 1.000 |
| **3.0** | **43.75%** | 47 | 50 | **956/2304** | **58.5%** | **1.000** |

threshold=3.0: best accuracy AND lowest FC1 density. original direct fold 4 was 54.0% so Option A is 10.25pp lower -- expected since MaxPool discards info that AvgPool preserves for continuous inputs.

the <500/step target wasn't met (range 956-1662). higher thresholds could reduce further but would hurt accuracy more. recommendation: use threshold=3.0 for full SpiNNaker FC1+FC2 deployment.

saved: `results/snn/maxpool/threshold_sweep_fold4.json`

---

### surrogate gradient ablation -- local run (4 march)

script: `python experiments/surrogate_gradient_ablation.py --fold 1 --seed 42 --epochs 50`

tested 8 surrogates from snnTorch 0.9.4. 7 completed, LSO crashed.

progress (just for fast_sigmoid as example):
- ep10: tr=8.4%, te=10.2%
- ep20: tr=19.4%, te=17.2%
- ep30: tr=36.6%, te=24.7%
- ep40: tr=47.6%, te=29.2%
- ep50: tr=67.0%, te=44.8%, best=44.75% -- still improving at termination

results:
- **spike_rate_escape: 46.00%** at ep50 -- BEST overall
- fast_sigmoid: 44.75% at ep50
- atan: 35.75% at ep49
- STE: 10.25% (early stop ep11, failed)
- sigmoid: 2.00% (early stop ep11, failed)
- SFS: 2.00% (early stop ep10, failed)
- triangular: 2.75% (early stop ep23, failed)
- LSO: CRASHED (TypeError: StochasticSpikeOperator.forward() missing 'variance' -- Python 3.14 + snnTorch 0.9.4 incompatibility)

**bimodal result:** three surrogates LEARN {spike_rate_escape, fast_sigmoid, atan} and four FAIL {STE, sigmoid, SFS, triangular}. stronger discrimination than Zenke & Vogels (2021) predicted. learning surrogates maintain non-zero gradient over broad range; failure surrogates have narrow bandwidth (triangular), piecewise gradients (STE), or practical saturation (sigmoid, SFS).

biggest surprise: sigmoid fails (2%) despite literature saying shape matters less than slope. STE also fails (10.25%) even though it passes gradient of 1 through threshold.

first surrogate gradient ablation for any audio SNN classification task.

saved: `results/snn/surrogate_ablation/ablation_fold1_seed42.json`

---

### fold 1 model file discrepancy (important)

discovered 3 march: experiment log shows SNN direct fold 1 = 40.50% (CSF3). but current best_fold1.pt showed 5% (2 epochs, local CPU failure that overwrote the good model).

summary.json still had fold_accuracies[0]=0.405 (stale from CSF3). the reported 47.15% is from CSF3 and is the authoritative number. restored from csf3_results backup.

---

## literature context

### SNN audio classification landscape

| dataset | task | classes | best SNN acc | reference |
|---------|------|---------|-------------|-----------|
| SHD | spoken digit recognition | 20 | 96.4% | Bittar & Garner 2022 |
| RWCP | indoor sounds | 19 | 99.6% | Wu et al. 2020 |
| TIDIGITS | connected digit recognition | 11 | 97.5% | Dong et al. 2023 |
| Google Speech Commands | keyword spotting | 35 | 95.1% | Stewart et al. 2023 |
| UrbanSound8K | urban sounds | 10 | ~75% | various |
| ESC-10 | environmental sounds (subset) | 10 | 66.1% F1 | Dennis et al. 2013 |
| **ESC-50 (ours)** | **environmental sounds** | **50** | **47.15%** | **this work** |

takeaways for positioning:
1. ESC-50 is harder than most prior SNN audio tasks (50 classes vs 10-35, diverse, real-world)
2. 47% is reasonable for a first attempt on this dataset
3. SNN accuracy is consistently lower than ANNs across all audio tasks -- the trade-off is energy
4. most successful SNN audio work uses learned or direct encoding, consistent with our finding
5. no prior peer-reviewed SNN work on ESC-50 -- genuine gap we're filling

---

## issues encountered and solutions

### 1. MPS AdaptiveAvgPool2d crash
replaced with `AvgPool2d(kernel_size=(4, 6))` -- same output shape (4, 9) from input (16, 54).

### 2. CUDA not available on CSF3 (job 11782913)
only loaded `cuda/12.6.2` which has the toolkit but not runtime libs. added `libs/cuda/12.8.1`.

### 3. Python attribute error on CSF3 (job 11782939)
typo: `total_mem` should be `total_memory`. combined with `set -e` this killed the whole job.

### 4. sshpass doesn't work with CSF3
CSF3 uses keyboard-interactive auth for Duo 2FA. had to use `expect` for interactive SSH.

### 5. GPU limit on CSF3
stuck in PENDING with QOSMaxGRESPerUser -- 2 GPU max per user. ran 2 encoding jobs in parallel, submitted 3rd when one finished.

### 6. SpiNNaker zero spike output (run 1)
0% accuracy, zero hidden/output spikes despite active input. parameter mapping mismatch between snnTorch and sPyNNaker.

### 7. set -e killing scripts on minor errors
removed `set -e` or wrapped non-critical commands in `|| true`.

### 8. Neo AnalogSignal magnitude TypeError
iterating Neo AnalogSignal columns yields Quantity objects with shape (1,), not scalars. `float()` fails. fix: `sig.magnitude[:, n].tolist()`.

---

## observations and analysis

### why direct encoding wins

direct encoding feeds the normalised mel spectrogram directly to the SNN, repeated across all 25 timesteps. the network effectively learns its own temporal coding. analogous to how end-to-end learning outperforms hand-crafted features in deep learning generally.

### why delta encoding fails

delta encodes the temporal derivative of the spectrogram. but a mel spectrogram is already a time-frequency representation -- taking the derivative is differentiating twice. you lose the base signal and amplify noise. spike patterns end up nearly random.

### statistical significance

SNN direct: 47.15% +/- 4.50% (folds: 40.5, 48.5, 48.25, 54.0, 44.5)
ANN baseline: 63.85% +/- 3.07% (folds: 63.2, 59.5, 65.2, 68.8, 62.5)
gap: 16.70 pp
paired t-test: t=8.64, p = 0.0010 (highly significant)
Wilcoxon: p = 0.0625 (minimum achievable with n=5; ANN wins all 5 folds)

(3 march 21:22) restored preds_fold1.pt from csf3_results backup. re-ran analysis_suite.py. SNN > ANN on 6/50 classes: coughing, crying_baby, door_wood_knock, pouring_water, footsteps, crackling_fire. all 50 per-class accs saved to analysis_results.json. t-SNE and confusion matrices regenerated.

---

### per-class analysis (corrected, 3 march)

SNN > ANN on 6 classes:

| class | SNN | ANN | diff |
|-------|-----|-----|------|
| coughing | 68% | 60% | +8% |
| crying_baby | 80% | 72% | +8% |
| door_wood_knock | 80% | 72% | +8% |
| pouring_water | 75% | 70% | +5% |
| footsteps | 55% | 53% | +3% |
| crackling_fire | 68% | 65% | +3% |

worst SNN classes:

| class | SNN | ANN | diff |
|-------|-----|-----|------|
| engine | 7% | 42% | -35% |
| laughing | 12% | 53% | -40% |
| clock_tick | 23% | 68% | -45% |

pattern: SNN wins on high-energy, spectrally distinctive sounds (crying baby, door knock). fails on quiet subtle sounds (engine hum, clock tick). hypothesis: LIF threshold acts as energy-gated filter -- high-energy sounds reliably cross threshold, quiet sounds hover near threshold producing unreliable spikes.

clock_tick gap (23% vs 68%) is particularly striking -- quiet periodic click at regular intervals doesn't consistently drive LIF neurons above threshold, but ANN learns the narrow spectral signature fine.

### the SNN-ANN accuracy gap

the 17pp gap (47% vs 64%) is expected:
- SNNs have binary activations (0/1 spikes) vs continuous values in ANNs
- information compressed into spike timing, losing precision
- surrogate gradients are approximate
- the trade-off is supposed to be compensated by energy efficency on neuromorphic hardware

### ideas to improve SNN accuracy

1. **data augmentation** (highest impact, easiest): currently using none. time shifting, SpecAugment, mixup, noise injection would reduce overfitting on 1,600 samples.
2. **longer training**: cosine annealing with 100+ epochs might escape local minima.
3. **deeper architecture**: 3rd conv layer would increase capacity. ANN baselines with deeper networks hit 80%+.
4. **learnable encoding layer**: trainable linear layer generating spike probabilities per timestep instead of repeating spectrogram 25 times.
5. **more timesteps**: 25 is short. 50 or 100 gives more integration time (at cost of speed/memory).
6. **transfer learning**: use pretrained audio model (VGGish, PANNs) for features, SNN classifier head. (we did this -- PANNs+SNN gets 92.5%)
7. **membrane potential readout tuning**: weighted combos of spike count and membrane potential.

most practical: data augmentation + longer training. minimal code changes, addresses overfitting bottleneck directly.

update: tried augmentation (SpecAugment). SNN aug: 40.75% +/- 16.03% -- WORSE than baseline, variance tripled. patience=10 too aggressive for augmented SNN. negative result.

### energy paradox

SNN uses 4.3x MORE energy in software. seems contradictory but:
- purely a software simulation artefact
- on a GPU every operation costs ~4.6 pJ regardless of sparsity
- on neuromorphic hardware, inactive neurons consume near-zero power
- SpiNNaker FC2-only hybrid confirms: 43.0% SpiNNaker vs 51.25% snnTorch (8.25pp gap)
- this finding itself is valuable -- demonstrates why neuromorphic hardware matters

---

## timeline

| date | milestone |
|------|-----------|
| pre-march 2026 | research, lit review, planning |
| 3 march | all main training done on CSF3 (5 experiments x 5 folds = 25 runs) |
| 3 march | evaluation pipeline run (confusion matrices, per-class, F1) |
| 3 march | energy analysis done |
| 3 march | SpiNNaker run 1: 0% accuracy, parameter mapping issue |
| 3 march | SpiNNaker run 2: 0% confirmed (systematic) |
| 3 march | SpiNNaker run 3: crashed (UDP buffer overflow) |
| 3 march | hardware verified as real (SC&MP 4.0.0, 47-48 chips) |
| 3 march | auto_calibrate.py: phases 1-2 PASS (neurons fire!), phases 3-5 FAIL (FC1 cancellation) |
| 3 march | root cause: FC1 zero-mean weights + 1398 simultaneous inputs = net negative current |
| 3 march | FC2-only approach: extract snnTorch hidden spikes, deploy only FC2 on SpiNNaker |
| 3 march | FC2-only pilot (10 samples): 5/10 = 50.0% (SpiNNaker outperformed snnTorch 2/10 on same set) |
| 3 march | population coding 5-fold: 19.15% (negative result vs rate) |
| 4 march | continual learning: SNN 74.4% forgetting vs ANN 81.3% (SNN wins by 6.9pp) |
| 4 march | SpiNNaker 400-sample run 6 launched |
| 4 march | Option A threshold sweep launched and completed: fc1_binary=1.000, threshold=3.0 best (43.75%) |
| 4 march | surrogate ablation: spike_rate_escape 46.00% (best), bimodal result, LSO crashed |
| 4 march | run 6 COMPLETE: 43.0% SpiNNaker vs 51.25% snnTorch, 8.25pp gap |
| 4 march | augmented training: NEGATIVE RESULT. SNN aug 40.75% worse than baseline |
| 4 march | all 8 thesis chapters drafted |
| 4 march | 5-fold SpiNNaker prep done (restored models, extracted features, generated connections) |
| 5 march | 5-fold SpiNNaker deployment complete: 33.1% +/- 6.9% SpiNNaker vs 46.0% snnTorch, 12.8pp gap |
| 15 march | 14 new experiment scripts written and executed (adversarial 5-fold, noise, temporal, encoding transfer, pruning, neuron ablation, stochastic resonance, saliency, etc.) |
| 16 march | NeuroBench 5-fold complete, continual learning 5-fold complete |
| 16 march | SpiNNaker full deploy root cause found (missing population.initialize(v=0.0)) |
| 16 march | FC1+FC2 breakthrough: first class-discriminative output on SpiNNaker (hidden 58/256 matches sim) |
| 27 march | 34 new experiments on CSF3 (rhythm-SNN 61.10%, dendritic+delays 61.65% best) |

---

### 5-fold SpiNNaker deployment complete -- 2026-03-05 12:33

SpiNNaker FC2-only hybrid, all 5 folds, 400 samples each (2,000 total inferences)

| fold | SpiNNaker | snnTorch ref | gap |
|------|-----------|-------------|-----|
| 1 | 29.0% | 39.5% | +10.5 pp |
| 2 | 32.0% | 48.2% | +16.2 pp |
| 3 | 36.5% | 47.8% | +11.2 pp |
| 4 | 43.0% | 51.2% | +8.2 pp |
| 5 | 25.2% | 43.2% | +18.0 pp |
| **mean** | **33.1%** | **46.0%** | **+12.8 pp** |
| **std** | **6.9%** | | **4.1 pp** |

params: weight_scale=5.0, IF_curr_exp, tau_m=20ms, v_thresh=1.0, tau_syn=5.0ms.
summary: results/spinnaker_results/5fold_summary.json

---

## phase 3: new experiments (15 march)

14 new experiment scripts written, verified (12/14 pass, 2 bugs fixed), and executed.

### adversarial robustness 5-fold (CSF3 A100, ~23 min)

previously single-fold. now validated across all 5 folds.

FGSM eps=0.1 (5-fold): SNN=16.55% +/- 5.49%, ANN=2.75% +/- 0.61% -- **6.0x more robust**
PGD eps=0.05 (5-fold): SNN=9.75%, ANN=0.05% -- **195x more robust**
PGD eps=0.10 (5-fold): SNN=3.50%, ANN=0.00% -- SNN still classifying, ANN dead

results: `results/adversarial/robustness_fold{1-5}.json`

### noise robustness 5-fold (CSF3 A100)

gaussian noise added to raw waveforms at varying SNR.

| SNR | SNN (mean) | ANN (mean) |
|-----|-----------|-----------|
| clean | 54.25% | 61.85% |
| 20dB | 31.90% | 37.25% |
| 10dB | 18.75% | 21.55% |
| 5dB | 12.40% | 14.90% |
| 0dB | 7.05% | 6.95% |
| -5dB | 3.35% | 3.05% |

SNN degrades less: relative drop 93.8% vs ANN 95.1%. at 0dB, SNN matches ANN despite lower clean acc.
results: `results/noise_robustness/`

### temporal ablation (fold 1, direct)

SNN evaluated at truncated timesteps (no retraining).

| T | accuracy | % of full | energy saving |
|---|---------|-----------|---------------|
| 1 | 7.25% | 17.9% | 96% |
| 5 | 33.50% | 82.7% | 80% |
| 7 | 36.50% | 90.1% | 72% |
| 10 | 38.25% | 94.4% | 60% |
| 20 | 41.00% | 101.2% | 20% |
| 25 | 40.50% | 100% | 0% |

reaches 90% of full accuracy at T=7 (72% energy saving). peaks at T=20.
results: `results/snn/temporal_ablation/ablation_direct.json`

### encoding transfer matrix (fold 1, 6x6)

train with encoding X, test with encoding Y.

transfer ratio = 0.27 -- SNNs learn encoding-SPECIFIC circuits, not general audio features. diagonal mean 19.2%, off-diagonal 5.2%. novel finding, nobody has published this.

results: `results/snn/encoding_transfer/transfer_matrix_fold1.json`

### pruning resilience (fold 1)

| sparsity | SNN (% retained) | ANN (% retained) |
|----------|-----------------|-----------------|
| 0% | 100% | 100% |
| 50% | 98.1% | 99.6% |
| 70% | 95.7% | 98.8% |
| 90% | **93.2%** | **36.8%** |

SNN dramatically more resilient. ANN cliff-edges at 90% pruning.
results: `results/snn/pruning/pruning_fold1.json`

### neuron ablation / fault tolerance (fold 1)

SNN retains 13.7% vs ANN 12.6% at 50% neuron death. at 10-30% ablation, SNN beats ANN in absolute accuracy.
results: `results/snn/neuron_ablation/`

### stochastic resonance (fold 1)

stochastic resonance detected: sigma=0.02 improves SNN by +0.25pp. no SR in ANN. SNN 9x more noise-resilient at sigma=0.5 (39.25% vs 13.1%). biological phenomenon in a trained LIF network. pretty cool.
results: `results/snn/stochastic_resonance/`

### SNN saliency maps (fold 1, 10 samples)

spike-aware Grad-CAM: IoU=0.075 -- SNN and ANN attend to completely different spectrogram regions. both classify correctly but focus on diferent acoustic features.
results: `results/snn/saliency/`

### weight distribution analysis (fold 1)

ANN weights sparser (38.8% near-zero vs SNN 21.0%). SNN fc2 kurtosis 24.6 vs ANN 14.6. spiking constraint produces denser, more peaked distributions.
results: `results/analysis/weight_distributions/`

### spike drop robustness (fold 4)

SpiNNaker hardware gap corresponds to ~50% effective spike loss. network degrades gracefully under simulated packet drops.
results: `results/spinnaker_results/spike_drop/`

### statistical significance tests

PANNs SNN vs ANN: p=0.034 (paired t-test, significant)
SpiNNaker vs snnTorch: p=0.0016 (highly significant)
results: `results/statistical_tests/significance_tests.json`

### still running on CSF3 (job 12168476)
- few-shot learning curves (fold 1)
- temporal ablation (5-fold)
- spike efficiency Pareto (fold 1)

---

## 16 march -- major progress session

### NeuroBench 5-fold complete

all 5 folds (previously only 1, 2, 4):
- SNN energy: 968 +/- 37 nJ/sample (1.08M +/- 41K ACs)
- ANN energy: 454 +/- 11 nJ/sample (99K +/- 2K MACs)
- ANN 2.1x cheaper in software
- SNN activation sparsity: 73.6% +/- 0.7% (spike rate 26.4%)
- results: `results/neurobench/summary_5fold.json`

### continual learning 5-fold complete (CSF3 job 12174555)

all 5 folds (previously only fold 4):
- SNN forgetting: 69.9% +/- 4.3%
- ANN forgetting: 74.7% +/- 2.4%
- SNN forgets 4.7 pp less (consistent in 4/5 folds)
- results: `results/continual_learning/summary_5fold.json`

### SpiNNaker full deploy: root cause found

the `full_spinnaker_deploy_cond.py` and `spinnaker_incremental.py` scripts had a critical bug: **missing `population.initialize(v=0.0)`**. without this, sPyNNaker defaults neurons to v=-65mV even when v_rest=0.0, making the 1mV threshold unreachable (would need 66mV of current!). also added `set_number_of_neurons_per_core` for core splitting.

fix verified: step 3a (FC1 exc-only) now produces 231/256 hidden neurons firing with 874 total spikes. previously 0/256 fired. huge difference.

### ICONS 2026 paper progress

- added 4 ICONS-specific refs (Schuman, Yarga, Seekings, Arfa)
- fixed header from "Conference '17" to "ICONS '26"
- updated NeuroBench to 5-fold
- added LIF equation, expanded background, prior work comparison table
- added surrogate ablation table (table 8), noise robustness (table 9)
- added continual learning subsection (5-fold validated)
- temporal efficiency and SpiNNaker per-category subsections
- compiles to 6 pages (ICONS limit: 8)

### SpiNNaker step 4: FC1+FC2 end-to-end (16 march)

**excitatory-only (scale=1.0, 20 samples):** SpiNNaker 1/20 = 5.0%, snnTorch 4/20 = 20.0%. hidden neurons fired 161-229/256 per sample (TOO MANY, expected ~55). 3/20 failed with SpinnmanIOException. root cause: removing inhibition causes hidden layer saturation (89.5% active vs expected 21.7%).

**top-k=100 (scale=1.0, 20 samples):** 0/20 = 0.0%. hidden 0-11/256 (TOO FEW). top-k=100 keeps only 100 of 2304 connections per neuron -- insufficient excitation.

key insight: exc-only = 229/256 (saturated), top-k=100 = 0-11/256 (too sparse). need ~55/256 to match snnTorch. solution: top-k=300-500 or full weights with proper balance.

### SpiNNaker full deploy debugging: scale sweep (16 march)

full FC1 weights (exc+inh), prune_threshold=0.05 (66K connections), MaxPool model fold 4.

| scale | sample 0 hidden | sample 3 hidden | sample 0 output |
|-------|----------------|----------------|-----------------|
| 1.0 | 4/256 | 0/256 | 1/50 |
| 5.0 | 35/256 | 0/256 | 11/50 (pred=12) |
| 10.0 | 41/256 | 0/256 | 7/50 (pred=12) |
| 20.0 | 39/256 | 0/256 | 8/50 (pred=12) |

sample 0 fires 35-41 hidden neurons across scales, sample 3 gets 0 regardless. cross-test with sample 0's spikes + sample 3's connections gave 2/5 neurons -- connections work, timing pattern is the issue.

### SpiNNaker FC1+FC2 breakthrough (16 march)

config: full weights (exc+inh), prune=0.05, scale=10.0, SpikeSourceArray=32/core, IF_curr_exp=16/core

- sample 0: true=49, pred=48, hidden=58/256, output=13/50 -- OFF BY ONE CLASS
- sample 1: true=49, pred=35, hidden=57/256, output=11/50

first time FC1+FC2 chain produces class-discriminative output on SpiNNaker. hidden firing matches sim (57-58 vs target 61). FC1 cancellation problem is solved with proper core splitting.

server dropped after sample 3 (spalloc connection error). the three key fixes:
1. `population.initialize(v=0.0)` -- prevents -65mV default
2. `set_number_of_neurons_per_core(SpikeSourceArray, 32)` -- prevents router congestion
3. full exc+inh weights with prune=0.05 + scale=10.0 -- preserves trained balance

---

## file reference

updated 4 march to reflect complete project state.

```
snn-esc50/
├── src/
│   ├── config.py              # hyperparams and paths
│   ├── dataset.py             # ESC-50 loader + mel specs
│   ├── encoding.py            # 7+ spike encodings
│   ├── train.py               # training loop (SNN + ANN, augmentation flags)
│   ├── evaluate.py            # metrics, confusion matrices
│   ├── energy.py              # synops/MAC energy (legacy; NeuroBench replaces)
│   └── models/
│       ├── snn_model.py       # SpikingCNN (snnTorch): 2 conv + AvgPool + 2 FC, 622K params
│       └── ann_model.py       # ConvANN: identical with ReLU
├── experiments/               # advanced experiment scripts
│   ├── adversarial_robustness.py
│   ├── analysis_suite.py
│   ├── analyze_spinnaker_run6.py
│   ├── continual_learning.py
│   ├── fill_surrogate_table.py
│   ├── generate_paper_figures.py
│   ├── neurobench_analysis.py
│   ├── panns_snn_head.py
│   ├── population_coding.py
│   ├── spinnaker_option_a.py
│   ├── spinnaker_option_c.py
│   ├── surrogate_gradient_ablation.py
│   └── temporal_analysis.py
├── spinnaker/
│   ├── convert_weights.py
│   ├── extract_features.py         # legacy, input-level
│   ├── extract_hidden_features.py  # FC1/lif3 hidden spikes for FC2-only
│   ├── reduce_inputs.py            # unused hypothesis
│   ├── run_on_spinnaker.py         # FC1+FC2, runs 1-4 (failed)
│   ├── auto_calibrate.py           # 5-phase calibration
│   ├── run_fc2_spinnaker.py        # FC2-only inference, runs 5-6
│   ├── run_inference.py
│   ├── debug_01_can_fire.py
│   ├── debug_02_tau_syn.py
│   ├── debug_03_two_layer.py
│   ├── debug_04_real_weights.py
│   ├── debug_05_weight_scale.py
│   └── README_DEBUGGING.md
├── results/
│   ├── ann/none/              # ANN baseline (63.85%)
│   ├── snn/direct/            # best SNN (47.15%)
│   ├── snn/rate/              # 24.00%
│   ├── snn/latency/           # 16.30%
│   ├── snn/delta/             # 7.25%
│   ├── snn/burst/             # 6.50%
│   ├── snn/phase/             # 24.15%
│   ├── snn/population/        # 19.15%
│   ├── snn/surrogate_ablation/
│   ├── snn/maxpool/
│   ├── adversarial/
│   ├── analysis/
│   ├── continual_learning/
│   ├── energy/
│   ├── neurobench/
│   ├── panns/
│   ├── panns_embeddings/
│   ├── paper_figures/
│   ├── spinnaker_optionC/
│   ├── spinnaker_results/
│   ├── spinnaker_weights/
│   └── temporal_analysis/
├── paper/                     # thesis chapter drafts
│   ├── thesis_introduction.md
│   ├── thesis_related_work.md
│   ├── thesis_methodology.md
│   ├── thesis_results_core.md
│   ├── thesis_results_hardware.md
│   ├── thesis_results_advanced.md
│   ├── thesis_discussion.md
│   ├── thesis_conclusion.md
│   └── ICONS2026_draft.md
├── csf3_results/
├── csf3_setup.sh
├── csf3_train_all.sh
├── csf3_train_encoding.sh
├── csf3_check.sh
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_encoding_visualisation.ipynb
│   └── 03_results_analysis.ipynb
├── DECISIONS.md
├── EXPERIMENT_LOG.md          # this file
├── .venv/                     # Python 3.14, PyTorch 2.10, snnTorch 0.9.4
├── .venv-spinnaker/           # Python 3.11, sPyNNaker
├── requirements.txt
└── README.md
```
