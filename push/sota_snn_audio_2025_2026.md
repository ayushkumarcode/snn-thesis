# State-of-the-Art: Spiking Neural Networks for Audio and Environmental Sound Classification (2024--2026)

**Research Report for COMP30040 UoM Thesis**
**Compiled: 5 March 2026**
**Research Agent: Deep Research Investigator (Claude Opus 4.6)**

---

## 1. Executive Summary

This report presents a comprehensive survey of the state-of-the-art in Spiking Neural Networks (SNNs) applied to audio and environmental sound classification, covering the period 2024--2026 with relevant historical context. The investigation spanned multiple search vectors across arXiv, IEEE Xplore, NeurIPS proceedings, OpenReview, Semantic Scholar, Google Scholar, ACM DL, Frontiers, Nature, and university repositories.

**Key finding: No prior work has applied a full SNN to the complete ESC-50 (50-class) dataset.** The closest work is Larroza et al. (2025, arXiv:2503.11206), which applies a 4-layer FC-only SNN to ESC-10 (10-class subset), achieving only 69.0% F1-score with their best encoding (TAE). Our thesis work (47.15% accuracy on full ESC-50 with a convolutional SNN, and 92.50% with PANNs+SNN head) represents a genuine first in the literature.

The field of SNN audio processing is rapidly evolving in 2024--2026, with major advances in:
- Spiking Transformer architectures for speech commands (SpikeSCR: 95.70% SHD; SpikCommander: 96.71% GSC)
- Multimodal audio-visual SNNs (S-CMRL: 98.13% UrbanSound8K-AV)
- Neuromorphic hardware deployment (SpiNNaker2 keyword spotting: 91.12%; Loihi 2 keyword spotting: 200x energy reduction)
- Speech enhancement SNNs (Spiking-FullSubNet: Intel N-DNS Challenge winner)

However, environmental sound classification with SNNs remains severely underexplored, with our work being the most comprehensive study to date.

---

## 2. Papers Applying SNNs to Audio/Sound Classification (2024--2026)

### 2.1 Environmental Sound Classification (Most Relevant to Thesis)

#### Paper 1: Larroza et al. (2025) -- THE Closest Competitor
