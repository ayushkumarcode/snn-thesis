# SNNs for speech tasks and novel applications

So i dug into where things stand with SNNs applied to speech and other non-classification tasks. The main finding is that **SNN speech processing is active and moving fast**, but still well behind conventional ANNs on most metrics. Speech-to-Text with SNNs has the most work (20+ papers), while Text-to-Speech with SNNs is basically a brand-new frontier with only 2-3 papers. Speech enhancement/denoising is an emerging sweet spot with 5-10 papers and strong practical motivation. Beyond speech, SNNs are starting to show up in object detection, segmentation, generative models (diffusion, GANs, VAEs), graph neural networks, and RL -- a lot of these could be high-novelty thesis directions.

The most interesting insight: SNN generative tasks are no longer just theoretical. SpikeVoice (ACL 2024), Spiking Vocos (2025), Spiking-Diffusion, and Spiking-GAN show that SNNs can generate continuous signals. The trick is using membrane potential (not discrete spikes) as the output, population coding, and rate-coded spike decoding.

---

## 1. Speech-to-text / ASR with SNNs

### Paper count: 20+

### Key papers

| Paper | Year | Venue | Dataset | Key Result |
|-------|------|-------|---------|------------|
| Deep Spiking Neural Networks for Large Vocabulary ASR (Wu et al.) | 2020 | Frontiers in Neuroscience | TIMIT, LibriSpeech | PER 18.7% (TIMIT test), comparable to ANN with same architecture |
| End-to-End SNN for Speech Recognition Using Resonating Input Neurons | 2021 | ICANN | Speech commands | Resonating neuron model for temporal encoding |
| Surrogate Gradient Spiking Baseline for Speech Command Recognition | 2022 | Frontiers in Neuroscience | Google Speech Commands | Baseline SNN performance on GSC |
| Speech2Spikes: Efficient Audio Encoding Pipeline | 2023 | ACM ICONS | Google Speech Commands | 88.5% accuracy on GSC, exceeded prior SNN SOTA by >10% |
| Complex Dynamic Neurons Improved Spiking Transformer for ASR | 2023 | arXiv | LibriSpeech, AISHELL | Spiking transformer for continuous ASR |
| Exploring Neural Oscillations During Speech Perception via Surrogate Gradient SNNs | 2024 | Frontiers in Neuroscience | Speech perception | Emergence of neural oscillations in trained SNN |
| ED-sKWS: Early-Decision Spiking NN for Keyword Spotting | 2024 | arXiv | Keyword spotting | Rapid energy-efficient keyword detection |
| SpikeSCR (Knowledge Distillation for Speech Commands) | 2024 | arXiv/Neural Networks | SHD, SSC, GSC | 95.01% on GSC, 93.60% on SHD, 80.25% on SSC |
| SpikCommander: High-performance Spiking Transformer | 2024 | arXiv | Google Speech Commands | Multi-view learning for speech command recognition |
| **IML-Spikeformer** (Input-aware Multi-Level Spiking Transformer) | 2025 | IEEE TNNLS | **LibriSpeech-960, AISHELL-1** | **WER 3.1%/3.4% (dev/test) -- comparable to ANN Transformer; 4.64x energy reduction** |

### Best results so far

- **Keyword/Command Recognition (GSC v2):** ~95% accuracy (SpikeSCR with knowledge distillation, 2024)
- **Large Vocabulary Continuous ASR (LibriSpeech-960):** WER 3.1%/3.4% (IML-Spikeformer, 2025) -- this is genuinely competitive with ANN transformers
- **TIMIT Phone Recognition:** PER ~18.7% (comparable to ANN baseline with same architecture)
- **Energy Efficiency:** 4-5x reduction vs ANN counterparts; as low as 0.68x total synaptic operations per frame

### How do SNNs compare with conventional models?

| Model | LibriSpeech test-clean WER | Type |
|-------|---------------------------|------|
| Whisper Large-v3 | ~2.0% | ANN (Transformer) |
| Wav2Vec 2.0 Large | ~2.3% | ANN (Self-supervised) |
| Conformer (Google) | ~2.1% | ANN (Conformer) |
| **IML-Spikeformer** | **~3.4%** | **SNN (Spiking Transformer)** |
| Earlier SNN models (2020) | ~10-15% | SNN (RNN-based) |

SNNs are now within 1-1.5% WER of SOTA ANNs on LibriSpeech, which is honestly pretty impressive. But this is with the very latest architectures (2025). Earlier SNN models have a 5-10% gap.

### Datasets used
- Google Speech Commands v1/v2 (keyword spotting, most common)
- TIMIT (phone recognition)
- LibriSpeech (large vocabulary continuous)
- AISHELL-1 (Mandarin)
- Spiking Heidelberg Digits (SHD) -- neuromorphic audio dataset
- Spiking Speech Commands (SSC) -- neuromorphic version of GSC
- TIDIGITS

### Feasibility for undergrad thesis: HIGH
- Keyword spotting on GSC v2 is very doable with snnTorch or SpikingJelly
- SpikingJelly has a built-in `speechcommands.py` example
- snnTorch has good tutorials on spike encoding and training
- Could compare multiple SNN architectures (CSNN, RSNN, Spikformer) on GSC
- Could extend to SHD/SSC neuromorphic datasets
- Could explore novel encoding schemes for audio-to-spike conversion

### Gaps i noticed
1. No SNN model matches Whisper/wav2vec2 on realistic ASR benchmarks yet (though IML-Spikeformer is getting close)
2. Most SNN work focuses on keyword spotting (small vocabulary), not open-vocabulary ASR
3. Limited work on noisy/real-world speech conditions with SNNs
4. Few studies comparing SNN inference latency vs ANN on actual neuromorphic hardware for speech
5. SNN for low-resource language ASR -- found zero papers on this

---

## 2. Text-to-speech with SNNs

### Paper count: 2-3 papers (genuinely novel frontier)

| Paper | Year | Venue | Key Result |
|-------|------|-------|------------|
| **SpikeVoice: High-Quality TTS Via Efficient SNN** | 2024 | **ACL 2024** (top NLP venue) | First SNN-based TTS; comparable quality to ANN with only 10.5% energy consumption |
| **Spiking Vocos: An Energy-Efficient Neural Vocoder** | 2025 | arXiv | First SNN-based vocoder; UTMOS 3.74, PESQ 3.45; 14.7% energy of ANN Vocos |

### What's going on here

**SpikeVoice (ACL 2024)** -- this is a landmark paper. It's explicitly stated as "the first TTS work in the SNN field." Key stuff:
- Introduced Spiking Temporal-Sequential Attention (STSA) to handle long-term dependencies
- Addressed the "partial-time dependency" problem: spiking neurons' serial nature limits capturing sequence dependencies across timesteps
- Tested on 4 datasets covering Chinese and English, single-speaker and multi-speaker
- Achieved comparable speech quality to ANN while using only 10.5% of the energy
- Published at ACL 2024, so it went through serious peer review

**Spiking Vocos (2025)** -- first SNN-based frequency-domain vocoder:
- Built on the Vocos framework
- Uses Spiking ConvNeXt module with amplitude shortcut to prevent information loss
- Self-architectural distillation + Temporal Shift Module for temporal modeling
- UTMOS: 3.74, PESQ: 3.45 (comparable to ANN Vocos)
- Only 14.7% energy consumption of ANN counterpart
- Code available: https://github.com/pymaster17/Spiking-Vocos

### How SpikeVoice actually generates speech

The key challenge: SNNs produce discrete binary spike events (0 or 1), but speech requires continuous waveforms. SpikeVoice's approach:
1. Uses membrane potential (not discrete spikes) as the output signal for waveform reconstruction
2. STSA mechanism allows capturing long-range temporal dependencies despite spike-based processing
3. The model generates spectrograms (mel-spectrograms), then a vocoder converts to waveforms
4. This two-stage approach (SNN spectrogram predictor + vocoder) bypasses the need for SNNs to directly generate audio samples

### Feasibility for undergrad: MEDIUM-HIGH (very high novelty)
- SpikeVoice code may or may not be publicly available (check ACL proceedings)
- Spiking Vocos has code on GitHub -- could be a starting point
- This area has extreme novelty -- even a modest contribution would be publishable
- Risk: complex architecture, may require significant compute
- Could focus on the vocoder component (Spiking Vocos) rather than full TTS pipeline

### Gaps (massive opportunities)
1. Only 2-3 papers exist in total -- almost any direction is novel
2. No work on SNN-based TTS for specific languages beyond Chinese/English
3. No work on SNN-based emotional/expressive TTS
4. No comparison of different SNN neuron models (LIF vs Izhikevich vs ALIF) for TTS quality
5. No work on SNN-based real-time TTS for edge devices
6. SNN-based vocoder optimized for neuromorphic hardware deployment

---

## 3. Speech-to-speech with SNNs (enhancement, conversion, translation)

### Paper count: 5-10 papers (growing fast, mostly speech enhancement)

| Paper | Year | Task | Key Result |
|-------|------|------|------------|
| **Spiking-FullSubNet** | 2024 | Speech Enhancement | Won Intel N-DNS Challenge; SOTA with large margins |
| DPSNN: Dual-Path SNN for Streaming Speech Enhancement | 2024 | Speech Enhancement | ~5ms latency; suitable for hearing aids |
| When Audio Denoising Meets SNN (Hao et al.) | 2024 | Audio Denoising | IEEE CAI 2024 |
| SNN-Wave-U-Net | 2025 | Speech Enhancement | 4.63J/inference (3.2x reduction vs ANN Wave-U-Net) |
| SNN-ConvTasNet | 2025 | Speech Enhancement | ~7x energy reduction vs ConvTasNet |
| Three-Stage Hybrid SNN Fine-Tuning for Speech Enhancement | 2025 | Speech Enhancement | Conversion + fine-tuning approach |
| End-to-end Neuromorphic Speech Enhancement with PDM Microphones | 2025 | Speech Enhancement | Direct PDM microphone to enhanced speech |

### Analysis

Speech enhancement is the strongest SNN speech application beyond classification.

**Spiking-FullSubNet** is the standout:
- Uses full-band and sub-band fused approach (inspired by human auditory system)
- Novel spiking neuron model with dynamic input integration and forgetting
- Won the Intel Neuromorphic Deep Noise Suppression (N-DNS) Challenge
- Outperforms SOTA methods with large margins
- Directly applicable to hearing aids, conferencing, edge devices

**DPSNN** addresses real-time requirements:
- Phase 1: Spiking CNNs capture global context
- Phase 2: Spiking RNNs focus on frequency features
- Achieves ~5ms latency (critical for hearing aids)
- Good SNR, perceptual quality, and energy efficiency

**Voice conversion with SNNs:** 0 papers found. Complete gap.

**Speech translation with SNNs:** 0 papers found. Complete gap.

**Speech separation (cocktail party) with SNNs:** Very early stage (1-2 papers from 2007-era using basic spiking models). Modern deep SNN approaches haven't been applied yet.

### Feasibility: HIGH (speech enhancement), VERY HIGH novelty (voice conversion)
- Speech enhancement: well-defined problem, existing SNN baselines, clear metrics (PESQ, STOI, SI-SDR)
- Voice conversion: zero SNN papers -- extremely novel but risky
- Could use snnTorch/SpikingJelly to build SNN-based denoising autoencoders
- Intel N-DNS Challenge dataset is publicly available
- VoiceBank+DEMAND dataset commonly used

### Gaps
1. Voice conversion with SNNs: 0 papers -- wide open
2. Speech separation with modern SNNs: essentially unexplored
3. Speech-to-speech translation with SNNs: 0 papers
4. Real-time SNN speech enhancement on neuromorphic hardware: only Spiking-FullSubNet approaches this
5. SNN for personalized hearing aid processing: limited work despite obvious application

---

## 4. SNN for generative audio

### Paper count: 3-5 papers

| Paper | Year | Task | Key Result |
|-------|------|------|------------|
| Music Neurotechnology for Sound Synthesis (Miranda) | 2009 | Sound synthesis | Neurogranular Sampler using Izhikevich spiking network |
| Spiking Music: Audio Compression with Event-Based Autoencoders | 2024 | Audio Compression | Binary autoencoders perform well on neural audio compression; event-based coding emerges |
| When Audio Denoising Meets SNN | 2024 | Audio Denoising | SNN-based audio denoising framework |
| SNN-based Audio Fidelity Evaluation (SAFE) | 2024 | Audio Quality Assessment | SNN for evaluating audio quality |

**Music generation with SNNs:** Essentially non-existent in the modern deep learning sense. The Neurogranular Sampler (2009) is a creative art/music project using Izhikevich neurons to trigger audio grains, not a trainable generative model. No papers apply modern SNN architectures (Spikformer, surrogate gradient training) to music generation.

**Audio compression with SNNs:** "Spiking Music" (2024) is a notable paper showing that:
- Simple binary autoencoders achieve surprisingly good results on neural audio compression benchmarks
- Event-based coding emerges naturally, synchronized with musical events (piano keystrokes)
- Demonstrates computational advantages of sparsity in audio compression
- Opens a new benchmark where event-based models can leverage advantages

### Feasibility: MEDIUM (audio compression), LOW (music generation)
- Audio compression: "Spiking Music" provides a framework; could extend to different audio types
- Music generation: would be extremely novel but lacks any foundation to build on
- Audio quality assessment (SAFE): interesting but narrow scope for thesis

### Gaps
1. SNN-based music generation: 0 modern papers -- completely open
2. SNN-based sound effect synthesis: 0 papers
3. SNN-based audio super-resolution: 0 papers
4. SNN for environmental sound generation: 0 papers
5. Audio compression on neuromorphic hardware: gap between theory and deployment

---

## 5. How can SNNs even generate continuous signals?

### The core problem
SNNs communicate through discrete binary spike events (0 or 1 at each timestep). Generative tasks like TTS, image generation, and audio synthesis need continuous-valued outputs. So how does this work?

### Solution mechanisms (confirmed by existing papers)

**Mechanism 1: Membrane potential as output**
- The most common approach in modern SNN generative models
- Instead of reading the binary spike output of the final layer, read the membrane potential (a continuous floating-point value)
- The membrane potential is an analog quantity that integrates incoming spikes over time
- Used by SpikeVoice, Spiking Vocos, Spiking VAE, spiking autoencoders
- snnTorch docs explicitly support this: "using membrane potential output from the final layer for image reconstruction"

**Mechanism 2: Rate coding / population coding for output**
- A population of output neurons encodes a continuous value through their collective firing rates
- Higher firing rate = higher output value
- Population coding: each neuron has a different tuning curve; the population superposition encodes vectors
- snnTorch has a tutorial on population coding
- Can encode arbitrary continuous values with sufficient population size

**Mechanism 3: Spike frequency / inter-spike interval decoding**
- Continuous values decoded from the frequency of output spikes
- Time-to-first-spike can also encode continuous values (latency coding)
- Used in some regression tasks and control applications

**Mechanism 4: Two-stage pipeline (SNN predictor + ANN decoder)**
- The SpikeVoice approach: SNN generates spectrograms, conventional vocoder converts to waveform
- Hybrid approach that leverages SNN efficiency for the heavy computation while using a small ANN for final conversion
- Pragmatic and effective -- used in most successful SNN generative systems

**Mechanism 5: Fourier-based spike construction**
- Theoretical framework where each spiking neuron represents a complex exponential (frequency component)
- N spiking neurons assigned integer multiples of fundamental frequency
- Reconstructs arbitrary time-series signals using Fourier Series principles
- More theoretical than practical right now

### The takeaway
SNNs CAN generate continuous signals. The field has moved past the theoretical question. The practical answer is mostly: use membrane potential as output (Mechanism 1) or use a two-stage pipeline (Mechanism 4). These are well-established techniques with multiple successful implementations.

---

## 6. Novel SNN applications beyond classification
