# Spiking Neural Networks for Speech Tasks and Novel Applications
## Comprehensive Research Report - February 2026

---

## EXECUTIVE SUMMARY

This report provides an exhaustive analysis of the current state of Spiking Neural Networks (SNNs) applied to speech tasks and novel application domains beyond classification. The key finding is that **SNN-based speech processing is an active and rapidly evolving field**, but remains significantly behind conventional ANNs in most metrics. Speech-to-Text with SNNs has the most mature research (20+ papers), while Text-to-Speech with SNNs is a genuinely novel frontier with only 2-3 papers in existence. Speech enhancement/denoising is an emerging sweet spot with 5-10 papers and strong practical motivation. Beyond speech, SNNs are breaking into object detection, segmentation, generative models (diffusion, GANs, VAEs), graph neural networks, and reinforcement learning -- many of these represent high-novelty thesis opportunities.

The most critical insight for thesis planning: **SNN generative tasks are no longer purely theoretical**. SpikeVoice (ACL 2024), Spiking Vocos (2025), Spiking-Diffusion, and Spiking-GAN demonstrate that SNNs can generate continuous signals. The mechanism relies on using membrane potential (not discrete spikes) as the output, population coding, and rate-coded spike decoding.

---

## 1. SPEECH-TO-TEXT / AUTOMATIC SPEECH RECOGNITION (ASR) WITH SNNs

### Paper Count: **20+ papers**

### Key Papers

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

### Best Results Achieved

- **Keyword/Command Recognition (GSC v2):** ~95% accuracy (SpikeSCR with knowledge distillation, 2024)
- **Large Vocabulary Continuous ASR (LibriSpeech-960):** WER 3.1%/3.4% (IML-Spikeformer, 2025) -- **this is genuinely competitive with ANN transformers**
- **TIMIT Phone Recognition:** PER ~18.7% (comparable to ANN baseline with same architecture)
- **Energy Efficiency:** 4-5x reduction vs ANN counterparts; as low as 0.68x total synaptic operations per frame

### Comparison with Conventional Models

| Model | LibriSpeech test-clean WER | Type |
|-------|---------------------------|------|
| Whisper Large-v3 | ~2.0% | ANN (Transformer) |
| Wav2Vec 2.0 Large | ~2.3% | ANN (Self-supervised) |
| Conformer (Google) | ~2.1% | ANN (Conformer) |
| **IML-Spikeformer** | **~3.4%** | **SNN (Spiking Transformer)** |
| Earlier SNN models (2020) | ~10-15% | SNN (RNN-based) |

**Gap Assessment:** SNNs are now within 1-1.5% WER of state-of-the-art ANNs on LibriSpeech, which is a remarkable achievement. However, this is with the very latest architectures (2025). Earlier SNN models have a 5-10% gap.

### Datasets Used
- Google Speech Commands v1/v2 (keyword spotting, most common)
- TIMIT (phone recognition)
- LibriSpeech (large vocabulary continuous)
- AISHELL-1 (Mandarin)
- Spiking Heidelberg Digits (SHD) -- neuromorphic audio dataset
- Spiking Speech Commands (SSC) -- neuromorphic version of GSC
- TIDIGITS

### Feasibility for Undergraduate Thesis: **HIGH**
- Keyword spotting on GSC v2 is very doable with snnTorch or SpikingJelly
- SpikingJelly has a built-in `speechcommands.py` example
- snnTorch has extensive tutorials on spike encoding and training
- Could compare multiple SNN architectures (CSNN, RSNN, Spikformer) on GSC
- Could extend to SHD/SSC neuromorphic datasets
- Could explore novel encoding schemes for audio-to-spike conversion

### Specific Gaps
1. No SNN model matches Whisper/wav2vec2 on realistic ASR benchmarks yet (though IML-Spikeformer is closing the gap)
2. Most SNN work focuses on keyword spotting (small vocabulary), not open-vocabulary ASR
3. Limited work on noisy/real-world speech conditions with SNNs
4. Few studies comparing SNN inference latency vs ANN on actual neuromorphic hardware for speech
5. **Gap opportunity:** SNN for low-resource language ASR (no papers found)

---

## 2. TEXT-TO-SPEECH (TTS) WITH SNNs

### Paper Count: **2-3 papers** (genuinely novel frontier)

### Key Papers

| Paper | Year | Venue | Key Result |
|-------|------|-------|------------|
| **SpikeVoice: High-Quality TTS Via Efficient SNN** | 2024 | **ACL 2024** (top NLP venue) | First SNN-based TTS; comparable quality to ANN with only 10.5% energy consumption |
| **Spiking Vocos: An Energy-Efficient Neural Vocoder** | 2025 | arXiv | First SNN-based vocoder; UTMOS 3.74, PESQ 3.45; 14.7% energy of ANN Vocos |

### Detailed Analysis

**SpikeVoice (ACL 2024)** -- This is a landmark paper. It is explicitly stated as "the first TTS work in the SNN field." Key innovations:
- Introduced **Spiking Temporal-Sequential Attention (STSA)** to handle long-term dependencies
- Addressed the "partial-time dependency" problem: spiking neurons' serial nature limits capturing sequence dependencies across timesteps
- Tested on 4 datasets covering Chinese and English, single-speaker and multi-speaker
- Achieved **comparable speech quality to ANN** while using only **10.5% of the energy**
- Published at ACL 2024, demonstrating high-quality peer review

**Spiking Vocos (2025)** -- The first SNN-based frequency-domain vocoder:
- Built on the Vocos framework
- Uses Spiking ConvNeXt module with amplitude shortcut to prevent information loss
- Self-architectural distillation + Temporal Shift Module for temporal modeling
- UTMOS: 3.74, PESQ: 3.45 (comparable to ANN Vocos)
- Only 14.7% energy consumption of ANN counterpart
- **Code available:** https://github.com/pymaster17/Spiking-Vocos

### How SpikeVoice Generates Speech (Theoretical Breakthrough)
The key challenge SpikeVoice solved: SNNs produce discrete spike outputs, but speech requires continuous waveforms. SpikeVoice's approach:
1. Uses membrane potential (not discrete spikes) as the output signal for waveform reconstruction
2. STSA mechanism allows capturing long-range temporal dependencies despite spike-based processing
3. The model generates spectrograms (mel-spectrograms), which are then converted to waveforms by a vocoder
4. This two-stage approach (SNN spectrogram predictor + vocoder) bypasses the need for SNNs to directly generate audio samples

### Feasibility for Undergraduate Thesis: **MEDIUM-HIGH (very high novelty)**
- SpikeVoice code may or may not be publicly available (check ACL proceedings)
- Spiking Vocos has code on GitHub -- could be used as a starting point
- This area has extreme novelty -- even a modest contribution would be publishable
- Risk: complex architecture, may require significant compute
- Approach: could focus on the vocoder component (Spiking Vocos) rather than full TTS pipeline

### Specific Gaps (MASSIVE opportunities)
1. Only 2-3 papers exist in total -- almost any direction is novel
2. No work on SNN-based TTS for specific languages beyond Chinese/English
3. No work on SNN-based emotional/expressive TTS
4. No comparison of different SNN neuron models (LIF vs Izhikevich vs ALIF) for TTS quality
5. No work on SNN-based real-time TTS for edge devices
6. **High-impact gap:** SNN-based vocoder optimized for neuromorphic hardware deployment

---

## 3. SPEECH-TO-SPEECH WITH SNNs (Enhancement, Conversion, Translation)

### Paper Count: **5-10 papers** (growing rapidly, mostly speech enhancement)

### Key Papers

| Paper | Year | Task | Key Result |
|-------|------|------|------------|
| **Spiking-FullSubNet** | 2024 | Speech Enhancement | Won Intel N-DNS Challenge; state-of-the-art with large margins |
| DPSNN: Dual-Path SNN for Streaming Speech Enhancement | 2024 | Speech Enhancement | ~5ms latency; suitable for hearing aids |
| When Audio Denoising Meets SNN (Hao et al.) | 2024 | Audio Denoising | IEEE CAI 2024 |
| SNN-Wave-U-Net | 2025 | Speech Enhancement | 4.63J/inference (3.2x reduction vs ANN Wave-U-Net) |
| SNN-ConvTasNet | 2025 | Speech Enhancement | ~7x energy reduction vs ConvTasNet |
| Three-Stage Hybrid SNN Fine-Tuning for Speech Enhancement | 2025 | Speech Enhancement | Conversion + fine-tuning approach |
| End-to-end Neuromorphic Speech Enhancement with PDM Microphones | 2025 | Speech Enhancement | Direct PDM microphone to enhanced speech |

### Detailed Analysis

**Speech Enhancement is the strongest SNN speech application beyond classification.**

**Spiking-FullSubNet** is the standout:
- Uses full-band and sub-band fused approach (inspired by human auditory system)
- Novel spiking neuron model with dynamic input integration and forgetting
- **Won the Intel Neuromorphic Deep Noise Suppression (N-DNS) Challenge**
- Outperforms state-of-the-art methods with large margins
- Directly applicable to hearing aids, conferencing, edge devices

**DPSNN** addresses real-time requirements:
- Phase 1: Spiking CNNs capture global context
- Phase 2: Spiking RNNs focus on frequency features
- Achieves ~5ms latency (critical for hearing aids)
- Excellent SNR, perceptual quality, and energy efficiency

**Voice Conversion with SNNs:** 0 papers found. This is a complete gap.

**Speech Translation with SNNs:** 0 papers found. Complete gap.

**Speech Separation (Cocktail Party) with SNNs:** Very early stage (1-2 papers from 2007-era using basic spiking models). Modern deep SNN approaches have not been applied yet.

### Feasibility for Undergraduate Thesis: **HIGH (speech enhancement), VERY HIGH novelty (voice conversion)**
- Speech enhancement: well-defined problem, existing SNN baselines, clear metrics (PESQ, STOI, SI-SDR)
- Voice conversion: zero SNN papers exist -- extremely novel but risky
- Could use snnTorch/SpikingJelly to build SNN-based denoising autoencoders
- Intel N-DNS Challenge dataset is publicly available
- VoiceBank+DEMAND dataset commonly used

### Specific Gaps
1. **Voice conversion with SNNs:** 0 papers -- wide open
2. **Speech separation with modern SNNs:** essentially unexplored
3. **Speech-to-speech translation with SNNs:** 0 papers
4. **Real-time SNN speech enhancement on neuromorphic hardware:** only Spiking-FullSubNet approaches this
5. **SNN for personalized hearing aid processing:** limited work despite obvious application

---

## 4. SNN FOR GENERATIVE AUDIO TASKS

### Paper Count: **3-5 papers**

### Key Papers

| Paper | Year | Task | Key Result |
|-------|------|------|------------|
| Music Neurotechnology for Sound Synthesis (Miranda) | 2009 | Sound synthesis | Neurogranular Sampler using Izhikevich spiking network |
| Spiking Music: Audio Compression with Event-Based Autoencoders | 2024 | Audio Compression | Binary autoencoders perform well on neural audio compression; event-based coding emerges |
| When Audio Denoising Meets SNN | 2024 | Audio Denoising | SNN-based audio denoising framework |
| SNN-based Audio Fidelity Evaluation (SAFE) | 2024 | Audio Quality Assessment | SNN for evaluating audio quality |

### Detailed Analysis

**Music Generation with SNNs:** Essentially non-existent in the modern deep learning sense. The Neurogranular Sampler (2009) is a creative art/music project using Izhikevich neurons to trigger audio grains, not a trainable generative model. No papers apply modern SNN architectures (Spikformer, surrogate gradient training) to music generation.

**Audio Compression with SNNs:** "Spiking Music" (2024) is a notable paper showing that:
- Simple binary autoencoders achieve surprisingly good results on neural audio compression benchmarks
- Event-based coding emerges naturally, synchronized with musical events (piano keystrokes)
- Demonstrates computational advantages of sparsity in audio compression
- Opens a new benchmark where event-based models can leverage advantages

**Audio Denoising:** Covered in Section 3 above.

### Feasibility for Undergraduate Thesis: **MEDIUM (audio compression), LOW (music generation)**
- Audio compression: "Spiking Music" provides a framework; could extend to different audio types
- Music generation: would be extremely novel but lacks any foundation to build on
- Audio quality assessment (SAFE): interesting but narrow scope for thesis

### Specific Gaps
1. **SNN-based music generation:** 0 modern papers -- completely open
2. **SNN-based sound effect synthesis:** 0 papers
3. **SNN-based audio super-resolution:** 0 papers
4. **SNN for environmental sound generation:** 0 papers
5. **Audio compression on neuromorphic hardware:** gap between theory and deployment

---

## 5. THE FUNDAMENTAL QUESTION: HOW CAN SNNs GENERATE CONTINUOUS SIGNALS?

### The Core Problem
SNNs communicate through discrete binary spike events (0 or 1 at each timestep). Generative tasks like TTS, image generation, and audio synthesis require producing continuous-valued outputs. How is this bridged?

### Solution Mechanisms (Verified by Existing Papers)

**Mechanism 1: Membrane Potential as Output**
- The most common approach in modern SNN generative models
- Instead of reading the binary spike output of the final layer, read the **membrane potential** (a continuous floating-point value)
- The membrane potential is an analog quantity that integrates incoming spikes over time
- Used by: SpikeVoice, Spiking Vocos, Spiking VAE, spiking autoencoders
- snnTorch documentation explicitly supports this: "using membrane potential output from the final layer for image reconstruction"

**Mechanism 2: Rate Coding / Population Coding for Output**
- A population of output neurons encodes a continuous value through their collective firing rates
- Higher firing rate = higher output value
- Population coding: each neuron has a different tuning curve; the population superposition encodes vectors
- snnTorch has a dedicated tutorial on population coding
- Can encode arbitrary continuous values with sufficient population size

**Mechanism 3: Spike Frequency / Inter-Spike Interval Decoding**
- Continuous values decoded from the frequency of output spikes
- Time-to-first-spike can also encode continuous values (latency coding)
- Used in some regression tasks and control applications

**Mechanism 4: Two-Stage Pipeline (SNN Predictor + ANN Decoder)**
- SpikeVoice approach: SNN generates spectrograms, conventional vocoder converts to waveform
- Hybrid approach that leverages SNN efficiency for the heavy computation while using a small ANN for final conversion
- Pragmatic and effective -- used in most successful SNN generative systems

**Mechanism 5: Fourier-Based Spike Construction**
- A theoretical framework where each spiking neuron represents a complex exponential (frequency component)
- N spiking neurons assigned integer multiples of fundamental frequency
- Reconstructs arbitrary time-series signals using Fourier Series principles
- More theoretical than practical currently

### Key Insight for Thesis
**SNNs CAN generate continuous signals.** The field has moved beyond the theoretical question. The practical answer is primarily: use membrane potential as output (Mechanism 1) or use a two-stage pipeline (Mechanism 4). These are well-established techniques with multiple successful implementations.

---

## 6. NOVEL SNN APPLICATIONS BEYOND CLASSIFICATION

### 6A. Object Detection with SNNs
**Paper Count: 20+ papers | Maturity: HIGH**

| Paper | Year | Venue | Key Result |
|-------|------|-------|------------|
| Spiking-YOLO | 2020 | AAAI | First SNN object detector; 280x energy reduction |
| Trainable Spiking-YOLO | 2023 | Neural Networks | Directly trained; low-latency |
| **SpikeYOLO (Integer-Valued Training)** | **2024** | **ECCV (Best Paper Candidate)** | **66.2% mAP@50 COCO; +2.5% over ANN equivalent; 5.7x energy efficiency** |
| SU-YOLO (Underwater) | 2025 | Neurocomputing | 78.8% mAP@50 underwater; 2.98 mJ energy |
| Deep Directly-Trained SNN for Object Detection | 2023 | arXiv | Direct training without ANN-to-SNN conversion |

**Assessment:** Object detection with SNNs is mature. SpikeYOLO was an ECCV 2024 Best Paper Candidate. The performance gap with ANNs is narrowing rapidly. Good thesis potential but less novel than speech tasks.

### 6B. Semantic Segmentation with SNNs
**Paper Count: 10-20 papers | Maturity: MEDIUM-HIGH**

| Paper | Year | Key Result |
|-------|------|------------|
| SNN for Image Segmentation | 2021 | Early SNN segmentation |
| SNN Fine-Tuning for Brain Image Segmentation | 2023 | Frontiers in Neuroscience |
| NSNPFormer (Transformer-based) | 2024 | mIoU 53.7 (ADE20K), 58.06 (Pascal Context) |
| Spiking U-Net + CBAM + ViT for Medical Segmentation | 2024 | 97.50% accuracy brain tumors |
| Spiking-SSegNet (Low-Latency) | 2025 | Large-scale semantic segmentation |
| Spiking Point Transformer | 2025 | AAAI 2025, point cloud classification |

**Assessment:** Active area with clear applications in medical imaging and autonomous driving. Medical image segmentation with SNNs is particularly promising for thesis work.

### 6C. Anomaly Detection with SNNs
**Paper Count: 5-10 papers | Maturity: MEDIUM**

| Paper | Year | Key Result |
|-------|------|------------|
| Online Evolving SNN for Multivariate Time Series Anomaly | 2022 | Machine Learning journal |
| Recurrent SNN for Time Series Prediction and Anomaly Detection | 2023 | IEEE |
| **Vacuum Spiker: SNN for Time Series Anomaly Detection** | 2025 | Uses STDP; competitive with deep learning; validated on solar inverters |
| SNN Autoencoder for Industrial Process Fault Detection | 2024 | Information Sciences |

**Assessment:** Underexplored but highly promising. SNNs' temporal processing nature makes them naturally suited for time series anomaly detection. The Vacuum Spiker paper shows SNNs can match deep learning with much lower energy. **Excellent thesis opportunity.**

### 6D. Recommendation Systems with SNNs
**Paper Count: 0-1 papers | Maturity: VERY LOW**

No dedicated papers found on SNN-based recommendation systems. Graph Neural Networks dominate this space, and while Spiking GNNs exist (see 6F), they haven't been applied to recommendation.

**Assessment:** Wide open gap. Could combine Spiking GNN + collaborative filtering. Very novel but high risk -- unclear if SNNs provide any advantage here.

### 6E. SNN + Generative Models (Diffusion, GAN, VAE)
**Paper Count: 5-10 papers | Maturity: MEDIUM (rapidly growing)**

| Paper | Year | Model Type | Key Result |
|-------|------|-----------|------------|
| Spiking-GAN | 2021 | GAN | First spike-based GAN; time-to-first-spike coding; 57x lower energy |
| Fully Spiking VAE | 2022 | VAE | First all-SNN VAE; equal or better quality vs ANN VAE |
| Spiking-Diffusion (VQ-SVAE + discrete diffusion) | 2023 | Diffusion | First fully-SNN diffusion model |
| **Spiking DDPM (SDDPM)** | **2024** | **Diffusion** | **WACV 2024; FID 19.20 on CIFAR-10; 37.5% energy of ANN at T=4** |
| **Spiking Diffusion Models** | **2024** | **Diffusion** | Outperforms SNN baselines across multiple datasets |

**Assessment:** Rapidly maturing. Spiking diffusion models are particularly exciting. The FID scores are still much worse than ANN diffusion models (19.20 vs ~2-3 for SOTA ANNs), but the energy savings are dramatic. **Strong thesis opportunity to apply spiking diffusion to new domains (e.g., audio spectrograms).**

### 6F. Spiking Graph Neural Networks
**Paper Count: 10-15 papers | Maturity: MEDIUM-HIGH**

| Paper | Year | Venue | Key Result |
|-------|------|-------|------------|
| A Graph is Worth 1-bit Spikes | 2024 | ICLR | Contrastive learning + SNN for graphs |
| Dynamic Spiking Graph Neural Networks | 2024 | AAAI | Dynamic graph processing |
| Spiking GNN on Riemannian Manifolds | 2024 | NeurIPS | Geometry-aware SGNNs |
| SGNNBench | 2025 | arXiv | Large-scale benchmark for SGNNs |
| Fully Memristive SGNN for Graph Learning | 2025 | Nature Comm. | Hardware implementation |

**Assessment:** Well-established research direction with top venue publications. Multiple architectures exist. Thesis-worthy but competitive.

### 6G. SNN + Reinforcement Learning / Robotics
**Paper Count: 10-20 papers | Maturity: MEDIUM-HIGH**

Key highlights:
- SNNs deployed on Loihi 2 for Astrobee robot control (NASA free-flying robot)
- Spiking RL for Atari games, CartPole, lane-keeping
- 6-DOF manipulator control with fully spiking networks
- **140x less energy** vs DNN approaches for inference
- Autonomous driving with SNNs (NeurIPS 2024)

**Assessment:** Strong practical motivation. Hardware deployment demonstrated. Good thesis potential, especially for robotics labs.

### 6H. SNN + Natural Language Processing
**Paper Count: 5-10 papers | Maturity: MEDIUM**

| Paper | Year | Key Result |
|-------|------|------------|
| SpikeGPT | 2023 | First spiking language model; 46M params; 33.2x energy reduction |
| SpikeBERT | 2023 | Comparable to BERT on text classification |
| SpikingBERT (AAAI) | 2024 | Knowledge distillation from BERT |
| SpikingMiniLM | 2024 | Energy-efficient NLU |
| SNNLP | 2024 | Comprehensive SNN for NLP framework |

**Assessment:** Emerging field. SpikeGPT is notable but very small (46M params vs billions for GPT-3/4). Practical NLP with SNNs remains far behind ANNs. Thesis opportunity in niche tasks (sentiment analysis, intent detection for edge devices).

### 6I. Other Creative Applications

**SNN + Point Cloud / 3D Processing:**
- S3DNet: 92.34% on ModelNet40, first SNN point cloud segmentation (85.0% ShapeNetPart)
- Spiking Point Transformer (AAAI 2025)

**SNN + Image Super-Resolution:**
- SpikeSR for remote sensing super-resolution
- MLFIN for general image super-resolution
- ESDNet for image deraining (IJCAI 2024)

**SNN + Optical Flow / Depth Estimation:**
- Natural pairing with event cameras
- StereoSpike for depth estimation on MVSEC
- SDformerFlow: Spiking transformer for event-based optical flow

**SNN + Continual/Lifelong Learning:**
- CLP-SNN on Loihi 2 for real-time continual learning
- Brain-inspired NACA algorithm for catastrophic forgetting mitigation
