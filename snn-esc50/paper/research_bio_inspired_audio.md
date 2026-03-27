# bio-inspired audio representations for SNNs -- paper reading notes

25 march 2026
trying to find audio representations that could give SNNs a genuine advantage over ANNs for ESC-50.
current baseline: SNN 47.15% / ANN 63.85% on log-mel spectrograms (64 bins, 216 time frames)

---

## the main hypothesis

after reading a bunch of papers i'm pretty convinced that **mel spectrograms are an ANN-native representation that fundamentally disadvantages SNNs**. multiple independent research lines confirm that bio-inspired cochlear representations (cochleagrams, gammatonegrams) consistently improve SNN performance over mel spectrograms, sometimes dramatically.

the strongest evidence: Wall et al. (ICONS 2022) found ALL spike encoding methods yield higher classification accuracy using significantly fewer spikes when encoding a cochleagram vs traditional STFT spectrogram. on TIDIGITS, SOD encoding with cochleagram gets 97% -- matching the unencoded baseline.

the most promising option is a **CARFAC cochleagram + BAE pipeline** feeding an SNN. nobody has tried this on ESC-50. the closest work (Larroza et al. 2025) only covers ESC-10 with mel spectrograms and simple FC-SNN.

another high-novelty option: **Resonate-and-Fire neurons as input encoders** that do spectral decomposition and spike encoding simultaneously from raw audio -- entirely eliminating the spectrogram stage.

the dream finding (SNN beating ANN through input representation) is theoretically achievable via noise robustness: cochleagram + spike encoding naturally filters noise, and SNNs with bio-inspired frontends show up to 2x the noise robustness of ANNs (Nature Comms, 2025).

---

## 1. cochlear models

### 1.1 CARFAC (Cascade of Asymmetric Resonators with Fast-Acting Compression)

a computational cochlea model by Richard Lyon at Google. models the basilar membrane (CAR) and outer hair cell gain control (FAC). produces a "cochleagram" -- time-frequency representation mimicking auditory nerve firing rates.

key details:
- output: Neural Activity Pattern (NAP) -- estimated average instantaneous nerve firing rates per frequency channel
- typical: 64-71 frequency channels (matches our 64 mel bins)
- output format: 2D array [freq_channels x time_steps], directly analogous to mel spectrogram
- nonlinear frequency spacing (logarithmic, like the cochlea)
- automatic gain control (naturally handles dynamic range)
- fine temporal structure preserved (phase info retained, unlike STFT)

v2 (april 2024): new Python/NumPy and JAX implementations. fixes DC distortion, reduces neural synchrony at high frequencies. JAX version is differentiable -- could integrate into end-to-end training.

implementation: `pip install carfac` or `github.com/google/carfac` (Python/NumPy, JAX, Matlab, C++). also `github.com/vschaik/CARFAC` for jupyter notebooks.

has it been tried with SNNs?
- yes on FPGA: Xu et al. (2018) implemented 70-section CARFAC + LIF on Cyclone V
- Xu et al. (2023, Frontiers) combined CAR-FAC + LIF + FEAST for TIDIGITS
- CARFAC + LSTM/SVM for speech emotion (2025, Biomimetics)
- **NOT tried on ESC-50 or any environmental sound with SNN** -- this is our opening

expected accuracy impact: cochleagram consistently outperforms mel:
- 98.03% vs 95.07% on acoustic event recognition (Sharan et al.)
- better noise robustness at SNR <= 15dB
- for SNN specifically: higher accuracy with fewer spikes (Wall et al. 2022)

spike compatibility: excellent. NAP output directly represents neural firing rates. natural fit for rate coding. apply LIF threshold to NAP channels -> spike trains directly.

references:
- Lyon et al. "The CARFAC v2 Cochlear Model" arXiv:2404.17490 (2024)
- Xu et al. "FPGA CAR-FAC" Front. Neurosci. 12:198 (2018)
- Xu et al. "Event-driven spectrotemporal feature extraction" Front. Neurosci. 17:1125210 (2023)

### 1.2 gammatone filterbanks

bank of filters with gammatone impulse response -- product of gamma distribution and sinusoidal tone. models basilar membrane frequency selectivity.

- standard: 64 channels, 50 Hz - 8000 Hz (ERB scale)
- well-established psychoacoustic model
- simpler than CARFAC
- available in scipy: `scipy.signal.gammatone`
- PyTorch GPU: `nnAudio2` provides `Gammatonegram` on GPU
- also: `pip install gammatone` (detly package)

tried with SNNs?
- indirectly: Wu et al. (2018, Frontiers) used mel-scaled filters for SOM-SNN getting 99.6% on RWCP
- Wall et al. (ICONS 2022): cochleagram (gammatone-based) + spike encoding dramatically outperforms STFT
- not on ESC-50 with SNN

marginal improvement over mel for CNNs (similar frequency scales), but significant for spike encoding since gammatone's temporal response is closer to auditory nerve.

drop-in replacement. same dimensionality. easy.

### 1.3 Lyon's original cochlear model (1982)

superseded by CARFAC v2. `github.com/sciforce/lyon` exists but just use CARFAC instead.

---

## 2. event-driven audio

### 2.1 Dynamic Audio Sensor (DAS) / silicon cochlea

hardware neuromorphic cochlea from UZurich / ETH. asynchronous event-based, 64 frequency channels (log distributed, 50 Hz - 20 kHz), 0.5V, 55 uW. output: Address-Event Representation (timestamp + channel + polarity).

tried with SNNs? yes -- FEAT networks get 97.7% on TIDIGITS. sound source localization on SpiNNaker using AER.

not practical for ESC-50 (needs hardware). but can be emulated in software:
- CARFAC + LIF threshold = software DAS equivalent
- `cochlea` python package: sound in, spikes out
- spikify library (Politecnico di Torino): `pip install spikify`

### 2.2 converting standard audio to events

multiple approaches:

1. **Send-on-Delta (SOD)**: spike when signal changes by > threshold. Wall et al. showed SOD on cochleagram gets 97% on TIDIGITS.

2. **Threshold Adaptive Encoding (TAE)**: dynamically adjusts threshold. Larroza et al. (2025) showed TAE is best for environmental sounds: 69% on ESC-10 (but with simple FC-SNN).

3. **level-crossing ADC paradigm**: neuromorphic level-crossing with decaying threshold. outperforms fixed-threshold.

4. **sigma-delta modulation**: LIF neurons naturally implement sigma-delta. Yarga & Wood (INTERSPEECH 2024): PDM microphone directly into SNN gets 91.54% on Google Speech Commands, bypassing all intermediate processing.

5. **Hilbert Transform + RZCC**: Haghighatshoar & Muir (Comms Engineering, 2025). 0.29 degree MAE for sound localization, 25-59x more efficient.

references:
- Wall et al. "Efficient spike encoding" ICONS 2022
- Larroza et al. arXiv:2503.11206
- Yarga & Wood INTERSPEECH 2024
- Haghighatshoar & Muir Comms Eng. 2025

---

## 3. auditory nerve models

### 3.1 cochlea python package (Zilany-Bruce-Carney model)

full biophysical inner ear models. sound in, spike trains out. implements the Zilany, Bruce & Carney (2014) auditory periphery.

- `pip install cochlea` or `github.com/mrkrd/cochlea3` (python 3)
- models: cochlear filtering, inner hair cell transduction, synaptic vesicle release, auditory nerve spike generation
- can generate responses for entire human auditory nerve (~30,000 ANFs)
- output: spike times per fiber, fiber characteristic frequency, fiber type
- interoperable with NEURON and Brian simulators

spike compatibility: PERFECT. output IS spike trains. no encoding needed.

implementation: moderate. need to choose number of fibers, frequency range, convert format. 30K fibers is too many for our 2304-input SNN so would need subsampling.

accuracy: UNKNOWN for classification. designed for biological accuracy, not classification performance. could be great or could be too detailed. nobody has tried it on ESC-50.

### 3.2 BAE (Biologically plausible Auditory Encoding) -- Pan et al. 2020

this is a really important paper. complete bio-inspired encoding pipeline: cochlear filter bank -> inner hair cells -> auditory masking -> spike encoding.

pipeline: CQT-based cochlear filtering (20 channels, 200-8000 Hz) -> log energy -> simultaneous masking (frequency domain) -> temporal masking (exponential decay) -> threshold coding (15 thresholds per channel) -> sparse spike output

**result on RWCP environmental sounds: 99.5% accuracy with only 245 spikes/sec**

compare:
- population coding: 99.0% with 4,627 spikes/sec (19x more spikes!)
- latency coding: 10.1% with 1,598 spikes/sec
- auditory masking removes ~50% of spikes while maintaining accuracy

released Spike-TIDIGITS and Spike-TIMIT datasets.

has it been tried on ESC-50? NO.

expected accuracy impact: HIGH. 99.5% on RWCP is remarkable. the 20-channel CQT + masking + threshold encoding is purpose-built for environmental sounds.

implementation: moderate. the pipeline is well-defined -- CQT filtering (librosa or custom), log energy, psychoacoustic masking (MPEG-1 Layer III, well-documented), temporal masking (exponential decay), threshold crossing (15 levels).

BAE on ESC-50 with conv SNN = completley novel.

reference: Pan et al. "An Efficient and Perceptually Motivated Auditory Neural Encoding and Decoding Algorithm for SNNs" Front. Neurosci. 13:1420 (2020)

---

## 4. time-frequency representations beyond mel

### 4.1 Constant-Q Transform (CQT)

time-frequency with logarithmically spaced bins (constant ratio between consecutive). higher frequency resolution at low frequencies, higher time resolution at high -- exactly matching the cochlea.

`librosa.cqt()`, nnAudio for GPU, upcoming in torchaudio. used in BAE pipeline as cochlear filterbank.

small to moderate improvement over mel. better low-frequency resolution matters for environmental sounds.

drop-in replacement. easy.

### 4.2 wavelet scattering transform (Anden & Mallat)

cascades wavelet convolutions and modulus operators. locally translation-invariant, time-warping-stable. extends MFCCs.

`pip install kymatio` (PyTorch/TF compatible, GPU). 1D scattering for audio. provably stable. no learned parameters.

has it been tried with SNNs? NO. thats a gap.

spike compatibility: moderate. continuous output, needs encoding. but multi-scale stable representation might survive encoding better than mel.

scattering + SNN is unexplored. could be interesting.

### 4.3 gammatonegrams / cochleagrams

covered in section 1.2. `nnAudio2` provides GPU-accelerated `Gammatonegram`.

### 4.4 spectrotemporal modulation (STM) features

from Chang et al. (arXiv:2505.23509, may 2025). mimics auditory cortex. decomposes spectrograms into temporal and spectral modulation.

128 cochlear frequency bands, 2D FFT of spectrogram -> modulation domain, 2,420 features. achieves ROC-AUC 0.988 vs mel 0.944. code available.

never tried with SNNs. very novel but STM is designed for cortical-level processing -- may not align well with SNN temporal dynamics.

---

## 5. spike-native audio representations

### 5.1 Send-on-Delta (SOD) / threshold crossing

emit spike when signal changes by > threshold delta. ON-spike for increases, OFF-spike for decreases. naturally event-driven.

critical result: Wall et al. (ICONS 2022) -- SOD on cochleagram matches unencoded baseline (97% TIDIGITS) at < 7% spike density. thats insane.

for ESC-50: apply SOD to each frequency channel of cochleagram. produces sparse event-driven representation compatible with our architecture.

### 5.2 Threshold Adaptive Encoding (TAE)

like SOD but with adaptive threshold. Larroza et al. (2025): best encoding for environmental sounds. ESC-10 69%, UrbanSound8K 53.5%. but with simple 4-layer FC-SNN (128 neurons/hidden). our conv SNN should do much better.

lowest spike rate of all tested encodings (38-50%) -- most efficient.

### 5.3 level-crossing / zero-crossing

RZCC method (Haghighatshoar & Muir, 2025) robustly extracts phase from wideband audio. combined with LIF for low-power processing.

### 5.4 LIF encoding (direct neural encoding)

feed each frequency channel directly into LIF neurons. natural threshold, integration, spike trains. this IS our delta encoding but applied to cochleagram instead of mel pixels.

info-theoretic: LIF encoding achieves ~80% coding efficiency at ~18% spike density (Gutierrez-Galan et al. 2022). efficiency drops on spectrogram-structured signals because signal statistics differ.

### 5.5 PDM microphone direct-to-SNN

Yarga & Wood (INTERSPEECH 2024): PDM output (binary stream) is mathematically similar to spike train. feed directly into SNN. 91.54% on Google Speech Commands surpassing Spiking Speech Commands SOTA.

not directly applicable to existing ESC-50 dataset. would need recording through PDM mic.

---

## 6. wild cards: temporal and raw waveform approaches

### 6.1 Resonate-and-Fire (RF) neurons as input encoders

RF neurons have oscillatory membrane dynamics that resonate at specific frequencies. when used as input layer, they do frequency analysis AND spike encoding simultaneously -- replacing both spectrogram and encoding.

Auge et al. (2021): results comparable to FFT + conventional processing. significantly less energy. NeurIPS 2024: RF with phase-locking coding achieves SOTA sound source localization with exceptional noise robustness. Balanced RF neurons (BRF, 2024) show superior training convergence.

for ESC-50: replace entire mel + encoding pipeline with a layer of RF neurons tuned to different frequencies. each resonates at its frequency, fires when detected.

extremely novel. RF for ESC-50 = completley unexplored.

hard to implement though -- custom RF neuron layer in snnTorch, tune resonant frequencies, feed raw waveform.

### 6.2 raw waveform to SNN

skip spectrogram entirely. feed raw audio.

three-stage hybrid SNN (Frontiers, 2025) operates on raw waveform for speech enhancement.

would need different architecture (1D conv instead of 2D). significant redesign.

### 6.3 Spiking-LEAF (learnable auditory front-end)

Song et al. (ICASSP 2024). 1D Gabor filterbank (40 filters, learnable center freq and bandwidth) + IHC-LIF neuron + lateral feedback + PCEN.

KWS: 92.24% (vs 83.03% fixed filterbank) -- 9.2% improvement. with recurrent SNN: 93.95%.

for ESC-50: replace mel with Spiking-LEAF. the 9.2% improvement on KWS is encouraging.

designed for speech not environmental sounds, so applying to ESC-50 would be novel.

moderate-high implementation.

### 6.4 spectrotemporal receptive field (STRF) / modulation features

model of auditory cortex. decomposes sounds into temporal and spectral modulation rates. environmental sounds have distinct modulation patterns (helicopter = low temporal, dog bark = impulsive).

very novel for SNN. STM features capture this naturally.

---

## 7. what makes environmental sounds special

environmental sounds differ from speech in key ways:
- **temporal structure varies enormously**: impulsive (gunshot, clap) vs sustained (rain, wind) vs periodic (helicopter, clock)
- **wide frequency range**: 20 Hz (engine) to 20 kHz (crickets)
- **non-harmonic content**: many lack harmonic structure (unlike speech/music)
- **onset characteristics critical**: many identifiable from onset alone

### onset-focused representations

delta encoding captures onsets naturally but gets only 7.25% because it loses steady-state info.

better approach: multi-scale temporal representation that captures BOTH onsets AND sustained energy. BAE does this with simultaneous + temporal masking.

### multi-resolution representations

recent ANN work (2024): Multi-Frequency Resolution features combining three resolutions outperform single-resolution for environmental sound.

for SNN: use CARFAC (naturally multi-resolution temporal processing via cascade structure) + multi-threshold encoding (different thresholds for different temporal scales).

---

## 8. ranked recommendations

### tier 1: highest impact, feasible

| rank | approach | expected impact | novelty | effort | risk |
|------|----------|----------------|---------|--------|------|
| 1 | CARFAC cochleagram + direct encoding | +5-15% over mel | very high | low (drop-in) | low |
| 2 | gammatonegram + TAE encoding | +3-10% over mel | high | low | low |
| 3 | BAE pipeline (CQT + masking + threshold) | +5-20% over mel | very high | moderate | moderate |
| 4 | cochleagram + Send-on-Delta | +5-15% over mel | high | low | low |

### tier 2: high impact, more effort

| rank | approach | expected impact | novelty | effort | risk |
|------|----------|----------------|---------|--------|------|
| 5 | Spiking-LEAF frontend | +5-10% | high | moderate-high | moderate |
| 6 | CQT + rate encoding | +2-5% | moderate | low | low |
| 7 | wavelet scattering + SNN | unknown | very high | moderate | high |

### tier 3: high novelty, high risk

| rank | approach | expected impact | novelty | effort | risk |
|------|----------|----------------|---------|--------|------|
| 8 | RF neurons as input encoders | unknown | extremely high | high | high |
| 9 | full auditory nerve model | unknown | extremely high | high | high |
| 10 | raw waveform SNN (1D conv) | unknown | high | high | high |

---

## 9. the recommended experiment

### phase 1: quick win (1-2 hours)

CARFAC cochleagram as drop-in replacement:

```
current:  audio -> mel spectrogram (64, 216) -> direct encoding -> SNN -> 47.15%
proposed: audio -> CARFAC cochleagram (64, ~216) -> direct encoding -> SNN -> ???
```

steps:
1. `pip install carfac`
2. process each ESC-50 audio through CARFAC with 64 channels
3. subsample/reshape to match current input (64, 216)
4. train existing architecture on CARFAC output
5. compare accuracy

minimal code change. directly tests the hypothesis. if CARFAC beats mel it validates the whole direction.

we actually ran this -- cochleagram SNN got 55.35% which is +8.2pp. hypothesis confirmed!

### phase 2: gammatonegram alternative (1-2 hours)

```
audio -> gammatonegram (64, 216) -> direct encoding -> SNN -> ???
```

using scipy.signal.gammatone or nnAudio2.Gammatonegram.

### phase 3: spike encoding on bio-inspired features (2-4 hours)

take best representation and try:
1. TAE instead of direct encoding
2. SOD encoding
3. LIF encoding on cochleagram channels

### phase 4 (if time): BAE pipeline (4-8 hours)

full BAE from Pan et al.:
1. CQT cochlear filtering (20 channels)
2. psychoacoustic masking
3. temporal masking
4. threshold crossing encoding
5. train SNN on resulting spike patterns

---

## 10. evidence supporting the hypothesis

### direct evidence

1. **Wall et al. (ICONS 2022)**: "All encoding methods yield higher classification accuracy using significantly fewer spikes when encoding a bio-inspired cochleagram as opposed to a traditional STFT." SOD on cochleagram matches unencoded baseline.

2. **Pan et al. (2020)**: BAE with CQT gets 99.5% on RWCP environmental sounds with only 245 spikes/sec. population coding needs 19x more spikes for 99.0%.

3. **cochleagram noise robustness**: outperforms mel at SNR <= 15dB for speaker recognition (MDPI 2023). SNN + bio-inspired frontend should amplify this.

4. **gammatonegram**: produces best classification vs mel, smoothed spectrogram, standard spectrogram (Sharan et al., 98.03% vs 95.07%).

### indirect evidence

5. CARFAC NAP output represents neural firing rates -- naturally spike-compatible. mel represents log-power-spectral-density -- designed for Fourier analysis, not spikes.

6. CARFAC preserves fine temporal structure (phase info). mel discards phase entirely. SNNs are temporal processors -- they need temporal info.

7. the biological auditory pathway IS a cochlea-to-SNN pipeline. we're asking an SNN to process a representation designed for a fundamentally different computational paradigm. no wonder it struggles.

8. LIF encoding achieves 80% coding efficiency on biologically-structured signals (Gutierrez-Galan et al. 2022). efficiency drops on spectrogram signals because signal statistics differ.

---

## 11. what we dont know

1. nobody has compared cochleagram vs mel for conv SNN on ESC-50. thats the central gap.

2. the interaction between audio representation and spike encoding is under-studied. Larroza et al. used mel for their encoding comparison. Wall et al. showed cochleagram is better but only on speech.

3. whether conv SNNs can leverage cochleagram structure better than FC SNNs. all prior work uses FC or simple classifiers.

4. CARFAC v2 + environmental sounds specifically. validated mainly on speech. environmental sounds have very different statistics.

5. the dream: SNN > ANN on cochleagram features. theoretically possible if temporal structure aligns with SNN dynamics. nobody has shown it yet.

### risks

1. CARFAC output dimensions may not map cleanly to (64, 216)
2. CARFAC processing might be slow (sample-by-sample, not batched)
3. improvement might not be enough to justify complexity for thesis
4. may need architecture mods (different pooling for cochlear frequency spacing)

---

## 12. confidence levels

| finding | confidence |
|---------|-----------|
| cochleagram outperforms mel for spike encoding | high (multiple sources) |
| CARFAC is best available cochlear model | high (Google-maintained, v2 in 2024) |
| no prior SNN work on full ESC-50 with bio frontend | high (confirmed by survey + our search) |
| drop-in replacement is feasible | moderate (depends on output format) |
| accuracy improvement of 5-15% | moderate (extrapolating from speech) |
| SNN can beat ANN with right representation | low-moderate (no direct evidence yet) |

---

## 13. references

### cochlear models
- Lyon et al. "CARFAC v2" arXiv:2404.17490 (2024)
- Xu et al. "FPGA CAR-FAC" Front. Neurosci. 12:198 (2018)
- Xu et al. "Event-driven spectrotemporal" Front. Neurosci. 17:1125210 (2023)
- Zilany, Bruce, Carney. JASA 135(1) (2014)

### spike encoding for audio
- Wall et al. "Efficient spike encoding" ICONS 2022, arXiv:2207.07073
- Larroza et al. "Spike Encoding for Environmental Sound" arXiv:2503.11206 (2025)
- Pan et al. "BAE" Front. Neurosci. 13:1420 (2020)
- Gutierrez-Galan et al. arXiv:2202.09619 (2022)
- Forno et al. "Spike encoding for IoT" Front. Neurosci. 16:999029 (2022)
- Vasilache et al. arXiv:2504.11026 (2025)

### SNN audio classification
- Wu et al. "Robust Sound Classification" Front. Neurosci. 12:836 (2018)
- Baek & Lee. "SNN and sound review" Biomed. Eng. Lett. 14:981-991 (2024)
- Yu et al. "Sparse Key-point Encoding" IEEE TNNLS (2020), arXiv:1902.01094
- Song et al. "Spiking-LEAF" ICASSP 2024, arXiv:2309.09469

### event-driven and neuromorphic audio
- Yarga & Wood. "PDM MEMS" INTERSPEECH 2024, arXiv:2408.05156
- Haghighatshoar & Muir. "Hilbert Transform" Comms Eng. (2025)
- NeurIPS 2024: RF-PLC for sound source localization
- Nature Electronics 2023: "Neuromorphic acoustic sensing"

### audio representations
- Sharan et al. "Cochleagram" (2019) -- 98.03% vs mel 95.07%
- MDPI "Noise Robustness of Cochleogram and Mel" (2023)
- Chang et al. "STM" arXiv:2505.23509 (2025)
- Auge et al. "Resonate-and-Fire" (2021)

### SNN robustness
- Nature Comms (2025): SNN 2x robustness of ANN

### tools and libraries
- CARFAC: `github.com/google/carfac`
- cochlea: `github.com/mrkrd/cochlea3`
- spikify: `github.com/neuromorphic-polito/spikify`
- nnAudio2: `github.com/WangHelin1997/nnAudio2`
- kymatio: `github.com/kymatio/kymatio`
- speech2spikes: `pip install speech2spikes`
- Brian Hears: auditory processing in Python

---

## bottom line

the mel spectrogram is holding the SNN back. multiple independent lines of evidence show bio-inspired cochlear representations improve spike encoding quality, accuracy, and noise robustness. the most promising experiment -- CARFAC cochleagram as drop-in replacement -- requires minimal code changes and has strong support.

we already confirmed this: cochleagram SNN got 55.35% which is +8.2pp over mel baseline. the hypothesis holds.

even if the improvement is modest, the finding is scientifically significant because:
1. its the first SNN result on full ESC-50 with bio-inspired audio frontend
2. it shows input representation is a key bottleneck
3. it opens a new direction: co-optimizing audio representation and SNN architecture

and if we got really lucky? cochleagram + temporal encoding might push closer to ANN accuracy, or even beat it under noisy conditions -- which would be a pretty big deal for the neuromorphic community.
