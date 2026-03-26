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
