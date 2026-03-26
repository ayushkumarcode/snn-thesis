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
