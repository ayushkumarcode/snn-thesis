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

