# chapter 3: methodology

notes on experimental design, architecture, encodings, training, SpiNNaker deployment. this is probably the most important chapter to get right since everything else depends on it.

---

## 3.1 experimental design philosophy

The central principle is **controlled comparison**: every architectural and training decision in the SNN is mirrored in the ANN baseline. This way, performance differences are attributable to the spiking formalism and encoding, not capacity or regularisation or training differences.

Specifically:
- Both use identical conv feature extraction (Conv2d, BN, MaxPool)
- Both have identical FC layers (2304 -> 256 -> 50)
- Both trained with same optimiser (Adam), same LR schedule (ReduceLROnPlateau), same early stopping (patience=10), same augmentation
- Both evaluated using 5-fold CV on ESC-50's predefined folds, held-out fold = test
- All reported accuracies are best validation accuracy per fold (not final epoch)

Follows Deng & Gu (Neural Networks 2020) recommendation for fair SNN-ANN comparison. Parameter count deliberately kept small (~622K) to avoid overfitting on 1600 training samples per fold.

---

## 3.2 dataset

**ESC-50** (Piczak, 2015): 2000 five-second environmental sound recordings, 50 classes (40 clips/class), from Freesound.org. Five broad categories: animals (0-9), natural soundscapes (10-19), human sounds (20-29), domestic (30-39), urban (40-49). Predefined 5-fold splits balance class distribution.

Human accuracy: 81.3%. ANN SOTA: 98.25%.

**Preprocessing pipeline** (all fixed, no tuning):
```
1. Load WAV at sr=22050 Hz, duration=5.0s
2. Zero-pad if shorter than 110250 samples
3. Mel spectrogram: n_mels=64, n_fft=1024, hop_length=512, f_min=0, f_max=None
   -> shape: (64, 216)
4. Convert to dB: librosa.power_to_db(mel, ref=np.max)
5. Min-max normalise per sample to [0, 1]
6. Add channel dim: (1, 64, 216)
```

The [0,1] normalisation is critical for spike encoding -- rate coding generates spikes with probability equal to pixel value, latency maps value to spike time, direct feeds values as membrane currents. All need [0,1].

All 2000 spectrograms precomputed and cached in memory (precompute=True) to avoid I/O overhead during training.

---

## 3.3 model architecture

### 3.3.1 SpikingCNN

```
Input: (T, B, 1, 64, 216) -- T=25 timesteps, B=batch

Block 1:  Conv2d(1->32, k=3, p=1) -> BN(32) -> MaxPool(2) -> LIF1
          Output: (T, B, 32, 32, 108)

Block 2:  Conv2d(32->64, k=3, p=1) -> BN(64) -> MaxPool(2) -> LIF2
          Output: (T, B, 64, 16, 54)

Pooling:  AvgPool2d(kernel=(4,6))  -- MPS-compatible substitute for AdaptiveAvgPool
          Output: (T, B, 64, 4, 9) -> flatten -> (T, B, 2304)

FC1:      Linear(2304, 256) -> LIF3
FC2:      Linear(256, 50) -> LIF4
Output:   spk_out: (T, B, 50)  mem_out: (T, B, 50)
```

**LIF params** (all layers, fixed):
- beta = 0.95 (decay per timestep)
- Threshold = 1.0
- Surrogate: fast sigmoid (slope=25), following snnTorch defaults. Systematic ablation of all 8 surrogates in 4.3.

**AdaptiveAvgPool issue:** PyTorch's AdaptiveAvgPool2d doesn't work on Apple MPS. AvgPool2d(kernel=(4,6)) achieves the same reduction from (16,54) -> (4,9) and is MPS-compatible. bit annoying but whatever

**Parameters:** ~622K total (both SNN and ANN same weight count despite different neuron types).

### 3.3.2 ConvANN (baseline)

Same architecture with LIF replaced by ReLU:
```
Conv2d(1->32) -> BN -> MaxPool(2) -> ReLU
Conv2d(32->64) -> BN -> MaxPool(2) -> ReLU
AvgPool(4x6) -> flatten
Linear(2304->256) -> ReLU -> Linear(256->50)
Output: logits (B, 50)
```

---

## 3.4 spike encoding methods

Seven methods evaluated. All transform (B, 1, 64, 216) into (T, B, 1, 64, 216), except direct which produces continuous values.

### rate coding
Each pixel spikes at each timestep with probability = its intensity.
$$\text{spk}[t, i] \sim \text{Bernoulli}(x_i), \quad t = 1, \ldots, T$$
Via `snntorch.spikegen.rate()`. Info content: O(T) bits/neuron.

### latency coding
Higher intensity = earlier spike. One spike per neuron.
$$t_{\text{fire}}(x_i) = -\tau \ln\left(\frac{x_i}{x_i - \theta}\right) \quad \text{(linearised)}$$
Via `snntorch.spikegen.latency()` with tau=5.0, theta=0.01, linear=True. Info: O(log T) bits/neuron.

### delta coding
Spikes on positive temporal changes above threshold.
- Applied to a noise-perturbed repetition of the spectrogram (since its static)
- Bio motivation: on-centre cells in auditory cortex
- Threshold = 0.1
- **Performs terribly** (7.25%): static spectrograms have no inherent temporal variation so... yeah

### direct (continuous) coding
Normalised spectrogram repeated unchanged across all T timesteps. LIF neurons get continuous current and generate their own timing through integration.
$$\text{input}[t, :] = x, \quad \forall t$$
No spike conversion. Network does implicit rate coding through LIF dynamics. **Best encoding at 47.15%**.

### burst coding
Spike count proportional to intensity, concentrated at the start:
$$n_{\text{spikes}}(x_i) = \text{round}(x_i \times N_{\max}), \quad N_{\max} = 5$$
$$\text{spk}[t, i] = \mathbf{1}[t < n_{\text{spikes}}(x_i)]$$
Bio motivation: burst-firing neurons in auditory cortex. Max density = 5/25 = 20%.

### phase coding
Intensity mapped to timing within a single oscillation cycle:
$$t_{\text{fire}}(x_i) = \lfloor (1 - x_i) \times (T-1) \rfloor, \quad x_i > 0$$
High intensity = early spike; exactly 1 spike per active neuron. Zero-intensity = silent. Bio motivation: theta-phase precession.

### population coding
Output-side: each of 50 classes represented by 10 neurons (500 total output). Input uses rate coding. Loss: MSE count loss:
$$\mathcal{L} = \text{MSE}\left(\frac{1}{T}\sum_t \text{spk}_{out,t}, y_{\text{pop}}\right)$$
where y_pop is a {0,1}^500 target vector. Classification: argmax of total spike count summed over grouped pools. Via `snntorch.functional.mse_count_loss(population_code=True, num_classes=50)`.

---

## 3.5 training protocol

**Optimiser:** Adam (lr=1e-3, wd=1e-4)
**LR schedule:** ReduceLROnPlateau (factor=0.5, patience=5, monitor val loss)
**Early stopping:** patience=10
**Max epochs:** 50
**Batch size:** 32
**Loss (SNN):** per-timestep CE on membrane potentials, summed over T:
$$\mathcal{L} = \sum_{t=1}^{T} \text{CE}(\text{mem}_t, y)$$
Follows snnTorch Tutorial 5, gives gradient flow through all timesteps.
**Loss (ANN):** standard CE on logits.
**Inference (SNN):** predicted class = argmax of summed membrane potential across T.

**Hardware:** MPS (Apple Silicon) locally; A100 on CSF3 for 5-fold CV.

### data augmentation

Two augmentations on training data only:
1. **SpecAugment** (Park et al. 2019): 2 freq masks (F=8 mel bins) + 2 time masks (T=20 frames), applied to 2D spectrogram
2. **TimeShift**: random cyclic shift +/-10% of frames (+/-21 frames), applied at audio level before spectrogram

note: augmentation turned out to be a negative result -- see chapter 4

---

## 3.6 SpiNNaker deployment

### 3.6.1 architecture constraint

The AvgPool between LIF2 and FC1 averages binary spikes into fractional values in [0, 0.5], breaking SpiNNaker's binary spike assumption. So direct deployment of full network on SpiNNaker isn't possible without architectural changes.

**Weight re-centering (Option C, failed):** tried zero-centering each FC1 weight row with bias compensation. Mathematically equivalent only for binary inputs. With fractional inputs the compensation over-corrects by n/sum(x_j), destroying FC1 selectivity. Accuracy went from 53.75% to 8.50%. Confirms the incompatibility.

### 3.6.2 FC2-only hybrid approach

Validated hybrid: conv + FC1 + LIF3 in software (snnTorch CPU), produces binary hidden spike tensors (T, N, 256). Only FC2 + LIF4 (256->50) deployed on SpiNNaker.

**Hardware:** SpiNN-5 at spinnaker.cs.man.ac.uk, sPyNNaker 1.0.0.

**Neuron model:** IF_curr_exp

**Calibrated params** (9-point scale sweep + LIF param calibration):
```
cm = 1.0 nF
tau_m = 20.0 ms
tau_refrac = 0.1 ms
tau_syn_E = 5.0 ms
v_thresh = 1.0 mV
v_rest = v_reset = 0.0 mV
weight_scale = 1.0
```

**Input:** binary hidden spikes (256-d/timestep, 21.7% active avg) as SpikeSourceArray, 1ms/timestep, 25ms total.

**Classification:** argmax of total output spike count over 25ms.

**Run 5 validation (n=20):** 8/20 = 40.0% vs snnTorch 10/20 = 50.0%.

---

## 3.7 advanced experiments

### 3.7.1 adversarial robustness

FGSM and PGD (torchattacks) at 7 eps values {0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3}, fold 4 test set (400 samples). SNN wrapped in SNNWrapper computing sum_t mem_t as differentiable logits. PGD: 40 steps per Wang et al. 2025 recommendation.

### 3.7.2 transfer learning (PANNs + SNN head)

CNN14 (Kong et al. 2020), AudioSet-pretrained, extracts 2048-d embeddings. 3-layer SNN head (2048->512->256->50, LIF, rate encoding, T=25) trained on frozen embeddings for 50 epochs. All 2000 embeddings precomputed and cached.

### 3.7.3 NeuroBench energy

NeuroBench v2.2.0 wraps model in SNNTorchModel:
- Effective_ACs: accumulate-only ops (non-zero binary activations)
- Effective_MACs: multiply-accumulate (non-zero non-binary activations)
- Energy: AC = 0.9 pJ, MAC = 4.6 pJ (45nm CMOS, Yik et al.)

### 3.7.4 continual learning


Output population code: each of 50 classes is represented by 10 output neurons (500 total). Loss: SF.mse_count_loss(correct_rate=1.0, incorrect_rate=0.0, population_code=True, num_classes=50). Classification: argmax of total spike count over grouped neuron pools.

---

## 3.8 Reproducibility

All code is available at [GitHub repository — TBD]. Fixed random seeds (torch.manual_seed(42), numpy.random.seed(42)) are used for all local experiments. CSF3 cluster jobs use seed=fold_number for deterministic per-fold results. The ESC-50 dataset is downloaded automatically from the official GitHub repository.

**Python environment:** Python 3.13 (CSF3) / 3.14 (local), PyTorch 2.10, snnTorch 0.9.4, torchattacks, neurobench 2.2.0, panns-inference, librosa 0.10.x.
