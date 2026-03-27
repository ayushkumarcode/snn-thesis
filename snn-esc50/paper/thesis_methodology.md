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


**Parameters:** ~622K total (both SNN and ANN use identical weight counts, despite different neuron types).

### 3.3.2 ConvANN (Baseline)

Architecturally identical to SpikingCNN with LIF neurons replaced by ReLU activations:
```
Conv2d(1→32) → BN → MaxPool(2) → ReLU
Conv2d(32→64) → BN → MaxPool(2) → ReLU
AvgPool(4×6) → flatten
Linear(2304→256) → ReLU → Linear(256→50)
Output: logits (B, 50)
```

---

## 3.4 Spike Encoding Methods

Seven encoding methods are evaluated. All transform a static mel spectrogram (B, 1, 64, 216) into a T-timestep spike tensor (T, B, 1, 64, 216), except direct encoding which produces continuous values.

### Rate Coding
Each pixel generates a spike at each timestep with probability equal to its intensity.
$$\text{spk}[t, i] \sim \text{Bernoulli}(x_i), \quad t = 1, \ldots, T$$
Implemented via `snntorch.spikegen.rate()`. Information content: $O(T)$ bits per neuron.

### Latency Coding
Higher intensity → earlier spike time. Each neuron fires exactly once.
$$t_{\text{fire}}(x_i) = -\tau \ln\left(\frac{x_i}{x_i - \theta}\right) \quad \text{(linearised)}$$
Implemented via `snntorch.spikegen.latency()` with τ=5.0, θ=0.01, linear=True. Information content: O(log T) bits per neuron.

### Delta Coding
Spikes generated on positive temporal changes above a threshold.
- Applied to a time-varying version of the spectrogram (noise-perturbed repetition)
- Biologically motivated by on-centre cells in auditory cortex
- Threshold = 0.1 per pixel change
- **Performs poorly** (7.25%): static spectrograms have no inherent temporal variation

### Direct (Continuous) Coding
The normalised spectrogram is repeated unchanged across all T timesteps. LIF neurons receive continuous membrane current input and generate their own spike timing through integration.
$$\text{input}[t, :] = x, \quad \forall t$$
No spike conversion: the network performs its own implicit rate coding through LIF dynamics. **Best-performing encoding (47.15%)**.

### Burst Coding
Spike count ∝ intensity, concentrated at the beginning of the simulation window:
$$n_{\text{spikes}}(x_i) = \text{round}(x_i \times N_{\max}), \quad N_{\max} = 5$$
$$\text{spk}[t, i] = \mathbf{1}[t < n_{\text{spikes}}(x_i)]$$
Biologically motivated by burst-firing neurons in auditory cortex. Maximum spike density = 5/25 = 20%.

### Phase Coding
Intensity mapped to spike timing within a single oscillation cycle:
$$t_{\text{fire}}(x_i) = \lfloor (1 - x_i) \times (T-1) \rfloor, \quad x_i > 0$$
High intensity → early spike; exactly one spike per active neuron per window. Zero-intensity neurons are silent. Biologically motivated by theta-phase precession (O'Keefe & Recce, 1993) and auditory cortex phase-of-firing codes.

### Population Coding
Output-side population representation: each of 50 classes is represented by 10 output neurons (500 total output neurons). Input encoding uses rate coding. Loss: MSE count loss targeting correct class neurons at count=1.0 and incorrect class neurons at count=0.0:
$$\mathcal{L} = \text{MSE}\left(\frac{1}{T}\sum_t \text{spk}_{out,t}, y_{\text{pop}}\right)$$
where $y_{\text{pop}} \in \{0, 1\}^{500}$ is the population target vector. Classification: argmax of total spike count summed over grouped neuron pools. Implemented via `snntorch.functional.mse_count_loss(population_code=True, num_classes=50)`.

---

## 3.5 Training Protocol

**Optimiser:** Adam (lr=1e-3, weight decay=1e-4)
**Learning rate schedule:** ReduceLROnPlateau (factor=0.5, patience=5 epochs, monitor: validation loss)
**Early stopping:** Patience=10 (stop if no validation accuracy improvement for 10 consecutive epochs)
**Maximum epochs:** 50
**Batch size:** 32
**Loss function (SNN):** Per-timestep cross-entropy on membrane potentials, summed over T:
$$\mathcal{L} = \sum_{t=1}^{T} \text{CE}(\text{mem}_t, y)$$
This follows snnTorch Tutorial 5 and provides gradient flow through all timesteps simultaneously.
**Loss function (ANN):** Standard cross-entropy on logits.
**Inference (SNN):** Predicted class = $\arg\max \sum_t \text{mem}_t$ (summed membrane potential vote).

**Hardware:** Apple M-series (MPS backend) for local runs; NVIDIA A100 on CSF3 cluster for 5-fold cross-validation.

### Data Augmentation

Two augmentations applied stochastically to training data only:
1. **SpecAugment** (Park et al. 2019): 2 frequency masks (width F=8 mel bins each) + 2 time masks (width T=20 frames each), applied to the 2D mel spectrogram tensor
2. **TimeShift**: random cyclic shift of ±10% of total time frames (±21 frames at 216 total), applied at the audio level before spectrogram computation

---

## 3.6 SpiNNaker Deployment

### 3.6.1 Architecture Constraint

The SpikingCNN architecture contains an AvgPool layer between LIF₂ and FC₁. This pool averages binary spikes into fractional values ∈ [0, 0.5], breaking the binary spike assumption required for SpiNNaker's spike-driven computation. Consequently, direct deployment of the full network (conv → FC₁ → FC₂) on SpiNNaker is not possible without architectural changes.

**Weight re-centering (Option C, failed):** Zero-centering each FC₁ weight row ($w_i \leftarrow w_i - \mu_i$) with bias compensation ($b_i \leftarrow b_i + \mu_i n$) was tested as a post-hoc fix. This reparameterisation is mathematically equivalent only for binary inputs (where the sum equals n). With fractional inputs, the compensation over-corrects by a factor of $n / \sum x_j$, destroying FC₁ selectivity. Accuracy dropped from 53.75% to 8.50% (−45.25 pp), confirming the incompatibility.

### 3.6.2 FC2-Only Hybrid Approach

Adopted a validated hybrid deployment: the feature extraction pipeline (conv blocks + FC₁ + LIF₃) runs in software (snnTorch on CPU), producing binary hidden spike tensors of shape (T, N, 256). Only the final classification layer (FC₂ + LIF₄: 256 → 50) is deployed on SpiNNaker hardware.

**SpiNNaker hardware:** SpiNN-5 board at `spinnaker.cs.man.ac.uk`, accessed via sPyNNaker 1.0.0.

**Neuron model:** IF_curr_exp (integrate-and-fire with exponential synaptic current)

**Calibrated parameters** (9-point scale sweep + LIF parameter calibration):
```
cm = 1.0 nF                  # membrane capacitance
tau_m = 20.0 ms              # membrane time constant
tau_refrac = 0.1 ms          # refractory period
tau_syn_E = 5.0 ms           # excitatory synaptic time constant
v_thresh = 1.0 mV            # spike threshold
v_rest = v_reset = 0.0 mV   # resting/reset potential
weight_scale = 1.0           # FC2 weights scaled to integer SpiNNaker format
```

**Input encoding:** Binary hidden spikes (256-d per timestep, 21.7% active on average) fed as SpikeSourceArray with 1ms per simulation timestep, 25ms total per sample.

**Classification:** Argmax of total output spike count over 25ms window.

**Validated accuracy (Run 5, n=20):** 8/20 = 40.0% vs snnTorch reference 10/20 = 50.0%.

---

## 3.7 Advanced Experiments

### 3.7.1 Adversarial Robustness

FGSM and PGD attacks (torchattacks library) at 7 ε values ∈ {0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3}, applied to fold 4 test set (400 samples). The SNN is wrapped in SNNWrapper that computes $\sum_t \text{mem}_t$ as differentiable logits. PGD uses 40 steps (following Wang et al. 2025 recommendation for reliable SNN evaluation).

### 3.7.2 Transfer Learning (PANNs + SNN Head)

CNN14 (Kong et al. 2020), pretrained on AudioSet (2M clips, 527 tags), extracts 2048-d embeddings. A 3-layer SNN classification head (2048→512→256→50, LIF neurons, rate encoding) is trained on frozen PANNs embeddings for 50 epochs. All 2,000 ESC-50 embeddings are precomputed and cached.

### 3.7.3 NeuroBench Energy Analysis

NeuroBench v2.2.0 (Yik et al. 2025, Nature Comms) wraps the model in SNNTorchModel and measures:
- **Effective_ACs**: accumulate-only operations (non-zero × binary activations)
- **Effective_MACs**: multiply-accumulate operations (non-zero × non-binary activations)
- Energy constants: AC = 0.9 pJ, MAC = 4.6 pJ (45nm CMOS, Yik et al.)

### 3.7.4 Continual Learning

Sequential training across 5 ESC-50 super-categories (Animals, Nature, Human, Domestic, Urban; 10 classes each). After training on each task, backward transfer (BWT) is measured as the average change in accuracy on previously seen tasks: BWT = $\frac{1}{T-1}\sum_{i<T}(R_{T,i} - R_{i,i})$ where $R_{k,i}$ is accuracy on task $i$ after training on task $k$. Negative BWT indicates catastrophic forgetting.

### 3.7.5 Population Coding

Output population code: each of 50 classes is represented by 10 output neurons (500 total). Loss: SF.mse_count_loss(correct_rate=1.0, incorrect_rate=0.0, population_code=True, num_classes=50). Classification: argmax of total spike count over grouped neuron pools.

---

## 3.8 Reproducibility

All code is available at [GitHub repository — TBD]. Fixed random seeds (torch.manual_seed(42), numpy.random.seed(42)) are used for all local experiments. CSF3 cluster jobs use seed=fold_number for deterministic per-fold results. The ESC-50 dataset is downloaded automatically from the official GitHub repository.

**Python environment:** Python 3.13 (CSF3) / 3.14 (local), PyTorch 2.10, snnTorch 0.9.4, torchattacks, neurobench 2.2.0, panns-inference, librosa 0.10.x.
