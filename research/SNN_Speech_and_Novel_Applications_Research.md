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

