# Bio-Inspired Audio Representations for SNNs: Deep Research Report

**Date**: 25 March 2026
**Objective**: Identify audio representations that could give SNNs a genuine advantage over ANNs for ESC-50 sound classification
**Current baseline**: SNN 47.15% / ANN 63.85% on log-mel spectrograms (64 bins, 216 time frames)

---

## Executive Summary

The research reveals a critical and well-supported hypothesis: **mel spectrograms are an ANN-native representation that fundamentally disadvantages SNNs**. Multiple independent research lines confirm that bio-inspired cochlear representations (cochleagrams, gammatonegrams) consistently improve SNN performance over mel spectrograms, sometimes dramatically. The most promising finding is from Wall et al. (ICONS 2022): all spike encoding methods yield **higher classification accuracy using significantly fewer spikes** when encoding a cochleagram vs. a traditional STFT spectrogram. On TIDIGITS, SOD encoding with cochleagram achieves 97% accuracy -- matching the unencoded baseline.

The highest-novelty, highest-impact option is a **CARFAC cochleagram + BAE (Biologically plausible Auditory Encoding) pipeline** feeding an SNN. No one has tried this on ESC-50. The closest work (Larroza et al. 2025, arXiv:2503.11206) only covers ESC-10 with mel spectrograms and simple FC-SNN. Another high-novelty option is **Resonate-and-Fire neurons as input encoders** that perform spectral decomposition and spike encoding simultaneously from raw audio -- entirely eliminating the spectrogram stage.

The dream finding (SNN beating ANN through input representation) is theoretically achievable via noise robustness: cochleagram + spike encoding naturally filters noise, and SNNs with bio-inspired frontends show up to 2x the noise robustness of ANNs (Nature Communications, 2025).

---

## 1. Cochlear Models

### 1.1 CARFAC (Cascade of Asymmetric Resonators with Fast-Acting Compression)

**What it is**: A computational model of the cochlea developed by Richard Lyon at Google. Models the basilar membrane (CAR: cascade of asymmetric resonators) and outer hair cell gain control (FAC: fast-acting compression). Produces a "cochleagram" -- a time-frequency representation that mimics auditory nerve firing rates.

**Key details**:
- Output: Neural Activity Pattern (NAP) -- estimated average instantaneous nerve firing rates per frequency channel
- Typical configurations: 64-71 frequency channels (matching our 64 mel bins)
- Output format: 2D array [frequency_channels x time_steps], directly analogous to mel spectrogram
- Nonlinear frequency spacing (logarithmic, like the cochlea)
- Includes automatic gain control (AGC) -- naturally handles dynamic range
- Fine temporal structure preserved (instantaneous phase information retained, unlike frame-based STFT)

**v2 (April 2024)**: New Python/NumPy and JAX implementations. Fixes DC distortion anomaly, reduces neural synchrony at high frequencies. JAX version is differentiable -- could be integrated into end-to-end training.

**Implementation**:
- `pip install carfac` or `github.com/google/carfac` (Python/NumPy, JAX, Matlab, C++)
- Also: `github.com/vschaik/CARFAC` (Jupyter notebooks for education)
- Integration with Auditory Modeling Toolbox (AMT)

**Has it been tried with SNNs?**
- Yes, on FPGA: Xu et al. (2018) implemented 70-section CARFAC + LIF neurons on Cyclone V FPGA
- Xu et al. (2023, Frontiers in Neuroscience) combined CAR-FAC + LIF + FEAST for TIDIGITS classification
- CARFAC + LSTM/SVM for speech emotion recognition (2025, Biomimetics)
- **NOT tried on ESC-50 or any environmental sound dataset with SNN** -- HIGH NOVELTY

**Expected accuracy impact**: Cochleagram consistently outperforms mel spectrogram in CNN studies:
- Cochleagram 98.03% vs mel 95.07% on acoustic event recognition (Sharan et al.)
- Better noise robustness: cochleagram features outperform mel at SNR <= 15dB
- For SNN specifically: cochleagram + spike encoding yields higher accuracy with fewer spikes (Wall et al. 2022)

**Spike compatibility**: EXCELLENT. NAP output directly represents neural firing rates. Natural fit for rate coding. With LIF threshold applied to NAP channels, produces spike trains directly.

**References**:
- Lyon et al. "The CARFAC v2 Cochlear Model" arXiv:2404.17490 (2024)
- Xu et al. "A FPGA Implementation of the CAR-FAC Cochlear Model" Front. Neurosci. 12:198 (2018)
- Xu et al. "Event-driven spectrotemporal feature extraction using a silicon cochlea model" Front. Neurosci. 17:1125210 (2023)

### 1.2 Gammatone Filterbanks

**What it is**: A bank of filters whose impulse response is a gammatone function -- the product of a gamma distribution and a sinusoidal tone. Models the frequency selectivity of the basilar membrane. Produces a "gammatonegram" or when combined with hair cell model, a "cochleagram."

**Key details**:
- Standard: 64 channels, center frequencies 50 Hz - 8000 Hz (ERB scale)
- Well-established psychoacoustic model
- Computationally simpler than CARFAC
- Available in scipy: `scipy.signal.gammatone`
- PyTorch GPU: `nnAudio2` provides `Gammatonegram` on GPU
- Also: `pip install gammatone` (detly package)

**Has it been tried with SNNs?**
- Indirectly: Wu et al. (2018, Frontiers) used mel-scaled filter banks (similar concept) for SOM-SNN achieving 99.6% on RWCP environmental sounds
- Wall et al. (ICONS 2022): cochleagram (gammatone-based) + spike encoding dramatically outperforms STFT
- **Not on ESC-50 with SNN** -- NOVEL

**Expected accuracy impact**: Marginal improvement over mel spectrogram for CNNs (they use similar frequency scales), but **significant improvement for spike encoding** because gammatone's temporal response is closer to auditory nerve behavior.

**Implementation complexity**: LOW. Drop-in replacement for mel spectrogram. Same dimensionality.

### 1.3 Lyon's Cochlear Model (Original)

**What it is**: Lyon's 1982 passive longwave cochlear model. Precursor to CARFAC.

**Implementation**: `github.com/sciforce/lyon` (Python port from Auditory Toolbox)

**Verdict**: Superseded by CARFAC v2. Use CARFAC instead.

---

## 2. Event-Driven Audio

### 2.1 Dynamic Audio Sensor (DAS) / Silicon Cochlea

**What it is**: Hardware neuromorphic cochlea developed at University of Zurich / ETH Zurich. An asynchronous event-based silicon cochlea that takes stereo audio inputs and outputs a stream of address-events (spikes) across 64 frequency channels (logarithmically distributed, 50 Hz - 20 kHz).

**Key details**:
- 0.5V, 55 uW power consumption
- 64 x 2 channels (binaural)
- Output: Address-Event Representation (AER) -- timestamp + channel + polarity
- Models basilar membrane frequency selectivity, inner hair cell rectification, auditory nerve spike generation
- Processes sound asynchronously and in parallel

**Has it been tried with SNNs?**
- Yes: FEAT-based networks achieve up to 97.7% on TIDIGITS
- Yes: Sound source localization on SpiNNaker using AER from silicon cochlea
- Multiple papers on neuromorphic audio processing (Delbruck lab, UZH)

**For ESC-50?** Not practical -- requires hardware. But can be **emulated in software**.

**Software emulation**:
- CARFAC + LIF threshold = software DAS equivalent
- The `cochlea` Python package (Rudnicki et al.) provides sound-in, spikes-out models
- Spikify library (Politecnico di Torino): `pip install spikify` -- converts signals to spike trains with gammatone/Butterworth preprocessing

### 2.2 Converting Standard Audio to Events

**Multiple approaches exist**:

1. **Send-on-Delta (SOD)**: Emit spike when signal changes by more than threshold. Wall et al. showed SOD on cochleagram achieves 97% on TIDIGITS.

2. **Threshold Adaptive Encoding (TAE)**: Dynamically adjusts threshold based on signal characteristics. Larroza et al. (2025) showed TAE is best for environmental sounds: 69% on ESC-10 (but with simple FC-SNN).

3. **Level-Crossing ADC paradigm**: Neuromorphic level-crossing sampling with decaying threshold. Outperforms fixed-threshold approaches.

4. **Sigma-Delta Modulation**: LIF neurons naturally implement sigma-delta modulation. Yarga & Wood (INTERSPEECH 2024): PDM microphone directly into SNN achieves 91.54% on Google Speech Commands, **bypassing all intermediate processing**.

5. **Hilbert Transform + RZCC**: Haghighatshoar & Muir (Communications Engineering, 2025). Hilbert transform extracts robust phase signal, RZCC encodes as spike events. Achieves 0.29 degree MAE for sound localization, 25-59x more efficient than conventional approaches.

**References**:
- Wall et al. "Efficient spike encoding algorithms for neuromorphic speech recognition" ICONS 2022
- Larroza et al. "Spike Encoding for Environmental Sound: A Comparative Benchmark" arXiv:2503.11206
- Yarga & Wood "Neuromorphic Keyword Spotting with PDM MEMS Microphones" INTERSPEECH 2024
- Haghighatshoar & Muir "Low-power SNN audio source localisation using Hilbert Transform" Comms Eng. 2025

---

## 3. Auditory Nerve Models

### 3.1 cochlea Python Package (Zilany-Bruce-Carney Model)

**What it is**: Full biophysical inner ear models. Sound in, spike trains out. Implements the Zilany, Bruce, & Carney (2014) auditory periphery model.

**Key details**:
- `pip install cochlea` or `github.com/mrkrd/cochlea3` (Python 3 version)
- Models: cochlear filtering, inner hair cell transduction, synaptic vesicle release, auditory nerve fiber spike generation
- Can generate responses for the ENTIRE human auditory nerve (~30,000 ANFs)
- Output: spike times per fiber, fiber characteristic frequency, fiber type (high/medium/low spontaneous rate)
- Fast enough for thousands of fibers
- Interoperable with NEURON and Brian simulators

**Has it been tried with SNNs?**
- Used in auditory neuroscience research
- Not directly combined with SNN classification on ESC-50

**Spike compatibility**: PERFECT. Output IS spike trains. No encoding needed.

**Implementation complexity**: MODERATE. Need to:
1. Process each ESC-50 audio through the model
2. Choose number of fibers and frequency range
3. Convert spike train format to SNN input tensor
4. May need to subsample (30K fibers is too many for our 2304-input SNN)

**Expected accuracy impact**: UNKNOWN for classification. These models are designed for biological accuracy, not classification performance. Could be very good (captures biologically relevant features) or poor (too much detail, insufficient abstraction).

**NOVELTY**: VERY HIGH. No one has used a full auditory periphery model for ESC-50 SNN classification.

### 3.2 BAE (Biologically plausible Auditory Encoding) -- Pan et al. 2020

**What it is**: A complete bio-inspired encoding pipeline that emulates the human auditory system: cochlear filter bank -> inner hair cells -> auditory masking -> spike encoding.

**Key details (critical paper)**:
- Pipeline: CQT-based cochlear filtering (20 channels, 200-8000 Hz) -> log energy -> simultaneous masking (frequency domain) -> temporal masking (exponential decay) -> threshold coding (15 thresholds per channel) -> sparse spike output
- **Result on RWCP environmental sounds: 99.5% accuracy with only 245 spikes/sec**
- Compare: population coding gets 99.0% with 4,627 spikes/sec (19x more spikes)
- Compare: latency coding gets 10.1% with 1,598 spikes/sec
- Auditory masking removes ~50% of spikes while maintaining accuracy
- Released Spike-TIDIGITS and Spike-TIMIT datasets

**Has it been tried with SNNs?** YES, that is its primary purpose. Validated on RWCP (environmental sounds) and TIDIGITS.

**Has it been tried on ESC-50?** NO.

**Expected accuracy impact**: HIGH. 99.5% on RWCP environmental sounds is remarkable. The 20-channel CQT cochlear bank + masking + threshold encoding is purpose-built for environmental sounds.

**Implementation complexity**: MODERATE. The pipeline is well-defined:
1. CQT cochlear filtering (20 channels) -- can use librosa or custom
2. Log energy computation per frame
3. Psychoacoustic masking (MPEG-1 Layer III model, well-documented)
4. Temporal masking (exponential decay)
5. Threshold crossing encoding (15 levels)

**NOVELTY for ESC-50**: VERY HIGH. BAE on ESC-50 with convolutional SNN = completely novel.

**Reference**: Pan et al. "An Efficient and Perceptually Motivated Auditory Neural Encoding and Decoding Algorithm for Spiking Neural Networks" Front. Neurosci. 13:1420 (2020)

---

## 4. Time-Frequency Representations Beyond Mel

### 4.1 Constant-Q Transform (CQT)

**What it is**: A time-frequency representation where frequency bins are logarithmically spaced (constant ratio between consecutive bins). Higher frequency resolution at low frequencies, higher time resolution at high frequencies -- exactly matching the cochlea.

**Key details**:
- Available in librosa: `librosa.cqt()`
- PyTorch GPU: `nnAudio` provides GPU-accelerated CQT
- Upcoming in torchaudio (PR #3804)
- Used in the BAE pipeline as the cochlear filterbank

**Has it been tried with SNNs?** Indirectly via BAE (Pan et al. 2020).

**Expected accuracy impact**: Small to moderate improvement over mel spectrogram. CQT provides better frequency resolution at low frequencies, which matters for environmental sounds with strong low-frequency content.

**Implementation complexity**: LOW. Drop-in replacement for mel spectrogram.

### 4.2 Wavelet Scattering Transform (Anden & Mallat)

**What it is**: A mathematical framework that cascades wavelet convolutions and modulus operators to produce a locally translation-invariant, time-warping-stable representation. Extends MFCCs by computing modulation spectrum coefficients of multiple orders.

**Key details**:
- `pip install kymatio` (Python, PyTorch/TensorFlow compatible, GPU)
- 1D scattering for audio
- Provably stable to time-warping deformations
- Translation invariant
- No learned parameters (fixed wavelet filters)

**Has it been tried with SNNs?** NO evidence found. This is a gap.

**For ESC-50 with ANN**: Scattering transforms achieve competitive results on audio classification. Not ESC-50 SOTA, but strong.

**Spike compatibility**: MODERATE. Output is continuous-valued features. Would still need spike encoding. But the multi-scale, stable representation might survive spike encoding better than mel features.

**NOVELTY**: HIGH. Scattering transform + SNN is unexplored territory.

**Implementation**: `kymatio` library, PyTorch compatible.

### 4.3 Gammatonegrams / Cochleagrams

Covered in Section 1.2. Key addition: `nnAudio2` provides PyTorch GPU-accelerated `Gammatonegram` class.

### 4.4 Spectrotemporal Modulation (STM) Features

**What it is**: A signal processing method that mimics the neurophysiological representation in the human auditory cortex. Decomposes spectrograms into temporal modulation (amplitude changes) and spectral modulation (frequency changes).

**Key details (May 2025, Chang et al., arXiv:2505.23509)**:
- Input: 128 cochlear-modeled frequency bands (170-7000 Hz)
- 2D FFT of spectrogram -> modulation domain
- 2,420 features (temporal: +/-15 Hz; spectral: 0-7.09 cycles/octave)
- Achieves ROC-AUC 0.988 vs mel spectrogram 0.944
- MLP with 1.1M params comparable to pretrained models (VGGish 72.1M params)
- Code: https://doi.org/10.5281/zenodo.15521995

**Has it been tried with SNNs?** NO.

**Spike compatibility**: MODERATE. Fixed-dimension feature vector could be spike-encoded.

**NOVELTY**: VERY HIGH. STM + SNN is completely novel. But STM is designed for cortical-level processing -- may not align well with SNN temporal dynamics.

---

## 5. Spike-Native Audio Representations

### 5.1 Send-on-Delta (SOD) / Threshold Crossing

**What it is**: Emit a spike whenever the signal changes by more than a threshold delta. Naturally event-driven. ON-spike for increases, OFF-spike for decreases.

**Critical result**: Wall et al. (ICONS 2022) showed SOD on cochleagram achieves classification accuracy matching the unencoded baseline (97% on TIDIGITS) with < 7% spike density.

**For ESC-50**: Apply SOD to each frequency channel of a cochleagram. Produces sparse, event-driven representation. Compatible with our existing SNN architecture.

### 5.2 Threshold Adaptive Encoding (TAE)

**What it is**: Like SOD but with adaptive threshold that adjusts to signal characteristics. Larroza et al. (2025, arXiv:2503.11206) showed it is the best encoding for environmental sounds.

**Results**: ESC-10 69%, UrbanSound8K 53.5%, TAU-3Class 69%. But with simple 4-layer FC SNN (128 neurons per hidden layer). Our convolutional SNN should do much better.

**Spike rate**: Lowest of all tested encodings (38-50%), indicating superior efficiency.

### 5.3 Level-Crossing / Zero-Crossing Encoding

**What it is**: Emit spikes at zero crossings or level crossings. Related to sigma-delta modulation.

**Recent work**: Robust Zero-Crossing Conjugate (RZCC) method (Haghighatshoar & Muir, 2025) robustly extracts phase from wideband audio. Combined with LIF neurons for low-power processing.

### 5.4 LIF Encoding (Direct Neural Encoding)

**What it is**: Feed audio signal (or each frequency channel) directly into LIF neurons. The neurons naturally threshold, integrate, and produce spike trains. This IS our delta encoding, but applied to cochleagram channels instead of mel spectrogram pixels.

**Key insight from information theory** (Gutierrez-Galan et al., arXiv:2202.09619): LIF encoding achieves the best coding efficiency (~80%) at ~18% spike density. BSA achieves ~71% at ~13%.

### 5.5 PDM Microphone Direct-to-SNN

**What it is**: Pulse Density Modulation (PDM) microphones output a binary stream that is mathematically similar to a spike train. Yarga & Wood (INTERSPEECH 2024) showed you can feed PDM directly into an SNN.

**Result**: 91.54% on Google Speech Commands, surpassing Spiking Speech Commands SOTA.

**For ESC-50**: Would require recording ESC-50 samples through a PDM microphone or simulating PDM conversion. Not directly applicable to existing dataset.

---

## 6. Wild Cards: Temporal and Raw Waveform Approaches

### 6.1 Resonate-and-Fire (RF) Neurons as Input Encoders

**What it is**: RF neurons have oscillatory membrane dynamics that resonate at specific frequencies. When used as an input layer, they perform frequency analysis AND spike encoding simultaneously -- replacing both the spectrogram computation and spike encoding step.

**Key paper**: Auge et al. (2021) "Resonate-and-Fire Neurons as Frequency Selective Input Encoders for SNN"
- RF neurons perform spectral transform directly on raw analog audio
- Results comparable to FFT + conventional processing
- Neuromorphic implementation consumes significantly less energy
- NeurIPS 2024: RF neurons with phase-locking coding achieve SOTA sound source localization with exceptional noise robustness

**Recent**: Balanced RF neurons (BRF, 2024) show superior training convergence, gradient stability, and spike efficiency vs. adaptive LIF.

**For ESC-50**: Replace our entire mel spectrogram + encoding pipeline with a layer of RF neurons tuned to different frequencies. Each RF neuron resonates at a different frequency, fires when it detects that frequency in the input.

**NOVELTY**: EXTREMELY HIGH. RF neurons for ESC-50 classification = completely unexplored.

**Implementation complexity**: HIGH. Would need to:
1. Create an RF neuron layer in snnTorch (or custom implementation)
2. Tune resonant frequencies to cover audio spectrum
3. Feed raw audio waveform as input current
4. Train the SNN backend on RF output spikes

**Expected accuracy impact**: UNKNOWN but theoretically sound. RF neurons are biologically grounded and have been shown to work for audio tasks.

### 6.2 Raw Waveform to SNN

**What it is**: Skip the spectrogram entirely. Feed raw audio waveform into the SNN.

**Recent work**:
- Three-stage hybrid SNN (Frontiers, 2025): operates on raw waveform speech for speech enhancement
- Neuromorphic audio processing: small SNN receives raw, unprocessed audio from microphone

**For ESC-50**: Would require different architecture (1D convolutions instead of 2D). Our current architecture assumes 2D spectrogram input.

**Implementation complexity**: HIGH. Architecture redesign needed.

### 6.3 Spiking-LEAF (Learnable Auditory Front-end)

**What it is**: Song et al. (ICASSP 2024). A learnable filterbank + IHC-LIF neuron model designed specifically for SNNs.

**Key details**:
- 1D Gabor filterbank (40 filters, learnable center frequency and bandwidth)
- IHC-LIF: two-compartment neuron (dendritic + somatic) inspired by inner hair cells
- Lateral feedback + spike regularization
- PCEN dynamic range compression

**Results**:
- KWS (keyword spotting): 92.24% (vs 83.03% for fixed filterbank) -- **9.2% improvement**
- KWS with recurrent SNN: 93.95%

**For ESC-50**: Replace mel spectrogram with Spiking-LEAF front-end. The 9.2% improvement on KWS is very encouraging.

**NOVELTY**: HIGH for ESC-50 application. Spiking-LEAF was designed for speech, not environmental sounds.

**Implementation complexity**: MODERATE-HIGH. Need to implement IHC-LIF neurons and learnable filterbank.

### 6.4 Spectrotemporal Receptive Field (STRF) / Modulation Features

**What it is**: Model of auditory cortex processing. Decomposes sounds into temporal and spectral modulation rates.

**For ESC-50**: Environmental sounds have distinct modulation patterns (e.g., helicopter = low temporal modulation, dog bark = impulsive). STM features capture this naturally.

**NOVELTY**: VERY HIGH for SNN application.

---

## 7. Specific to ESC-50: Environmental Sound Characteristics

### 7.1 What Makes Environmental Sounds Special

Environmental sounds differ from speech in key ways:
- **Temporal structure varies enormously**: impulsive (gunshot, clap) vs. sustained (rain, wind) vs. periodic (helicopter, clock)
- **Wide frequency range**: from 20 Hz (engine) to 20 kHz (crickets)
- **Non-harmonic content**: many environmental sounds lack harmonic structure (unlike speech/music)
- **Onset characteristics are critical**: many sounds identifiable from onset alone (dog bark, clock tick)

### 7.2 Onset-Focused Representations

**Delta encoding** (our current encoding #5) captures onsets naturally -- it fires spikes at signal changes. But it gets only 7.25% because it loses steady-state information.

**Better approach**: Multi-scale temporal representation that captures BOTH onsets AND sustained energy. The BAE pipeline does this with simultaneous + temporal masking.

### 7.3 Multi-Resolution Representations

Recent ANN work (2024) shows that Multi-Frequency Resolution (MFR) features combining three different frequency resolutions outperform single-resolution features for environmental sound classification.

**For SNN**: Use CARFAC (which naturally provides multi-resolution temporal processing via its cascade structure) + multi-threshold encoding (different thresholds capture different temporal scales).

---

## 8. Synthesis: Ranked Recommendations

### Tier 1: Highest Impact, Feasible for Thesis

| Rank | Approach | Expected Impact | Novelty | Implementation Effort | Risk |
|------|----------|-----------------|---------|----------------------|------|
| 1 | **CARFAC cochleagram + direct encoding** | +5-15% over mel | VERY HIGH | LOW (drop-in replacement) | LOW |
| 2 | **Gammatonegram + TAE encoding** | +3-10% over mel | HIGH | LOW | LOW |
| 3 | **BAE pipeline (CQT + masking + threshold)** | +5-20% over mel | VERY HIGH | MODERATE | MODERATE |
| 4 | **Cochleagram + Send-on-Delta** | +5-15% over mel | HIGH | LOW | LOW |

### Tier 2: High Impact, Moderate Effort

| Rank | Approach | Expected Impact | Novelty | Implementation Effort | Risk |
|------|----------|-----------------|---------|----------------------|------|
| 5 | Spiking-LEAF frontend | +5-10% | HIGH | MODERATE-HIGH | MODERATE |
| 6 | CQT + rate encoding | +2-5% | MODERATE | LOW | LOW |
| 7 | Wavelet scattering + SNN | Unknown | VERY HIGH | MODERATE | HIGH |

### Tier 3: High Novelty, High Risk

| Rank | Approach | Expected Impact | Novelty | Implementation Effort | Risk |
|------|----------|-----------------|---------|----------------------|------|
| 8 | RF neurons as input encoders | Unknown | EXTREMELY HIGH | HIGH | HIGH |
| 9 | Full auditory nerve model | Unknown | EXTREMELY HIGH | HIGH | HIGH |
| 10 | Raw waveform SNN (1D conv) | Unknown | HIGH | HIGH | HIGH |

---

## 9. The Recommended Experiment

### Phase 1: Quick Win (1-2 hours)

**CARFAC cochleagram as drop-in replacement for mel spectrogram**:

```
Current:  audio -> mel spectrogram (64, 216) -> direct encoding -> SNN -> 47.15%
Proposed: audio -> CARFAC cochleagram (64, ~216) -> direct encoding -> SNN -> ???
```

Steps:
1. `pip install carfac` (or use google/carfac NumPy implementation)
2. Process each ESC-50 audio through CARFAC with 64 channels
3. Subsample/reshape output to match current input dimensions (64, 216)
4. Train existing SNN architecture on CARFAC output
5. Compare accuracy

**Why this first**: Minimal code change, directly tests the hypothesis that representation matters. If CARFAC beats mel, it validates the entire research direction.

### Phase 2: Gammatonegram Alternative (1-2 hours)

```
audio -> gammatonegram (64, 216) -> direct encoding -> SNN -> ???
```

Using scipy.signal.gammatone or nnAudio2.Gammatonegram. Same approach as Phase 1.

