# Spiking Neural Networks for Audio Processing: Keyword Spotting & Speech Command Recognition

## Comprehensive Research Report -- February 2025

---

## Executive Summary

Spiking Neural Networks (SNNs) for audio keyword spotting and speech command recognition have matured significantly in 2024-2025, reaching a point where they are a viable and compelling undergraduate thesis topic. The accuracy gap between SNNs and conventional ANNs has narrowed dramatically: state-of-the-art SNNs now achieve 96.9% on Google Speech Commands V2 (35-class), approaching the ANN ceiling of ~97-98%. Multiple open-source frameworks (snnTorch, SpikingJelly, sparch) provide well-documented starting points, and several complete implementations exist on GitHub with 300-600 lines of core Python code. The energy efficiency argument is substantiated by hardware benchmarks showing 10-200x lower energy per inference on neuromorphic hardware (Intel Loihi) versus conventional processors. This is a well-scoped, feasible thesis project with clear benchmarks, available code, and a strong research narrative around energy-efficient edge AI.

---

## 1. SNN vs ANN Accuracy on Google Speech Commands Dataset

### 1.1 Current State of the Art (as of early 2025)

| Model | Type | Dataset (Task) | Accuracy | Parameters | Year | Code Available |
|-------|------|----------------|----------|------------|------|----------------|
| **SpikCommander** | SNN (Spiking Transformer) | GSC V2 (35-class) | **96.92%** | 2.13M | 2025 | Yes |
| **SpikCommander** | SNN (Spiking Transformer) | GSC V2 (35-class) | **97.08%** (T=200) | 2.13M | 2025 | Yes |
| **SpikeSCR** | SNN (Hybrid Attention) | GSC V2 (35-class) | **95.60%** | ~3.3M | 2024 | Pending |
| **SIDC-KWS** | SNN (Conformer) | GSC V2 (12-class) | **96.8%** | -- | 2025 | -- |
| **Spiking LMUFormer** | SNN | GSC V2 (35-class) | **96.12%** | -- | 2024 | -- |
| **RadLIF (sparch)** | SNN (Recurrent) | GSC V2 (35-class) | **96.60%** | ~1M | 2022 | Yes |
| **adLIF (sparch)** | SNN (Non-recurrent) | GSC V2 (35-class) | **95.50%** | ~1M | 2022 | Yes |
| **LSNN** | SNN (Spiking RNN) | GSC V1 (12-class) | **91.2%** | -- | 2020 | Yes |
| **ED-sKWS** | SNN (Early Decision) | GSC V2 (35-class) | **93.04%** | 27.6K | 2024 | No |
| LMUFormer | ANN | GSC V2 (35-class) | 96.53% | -- | 2024 | -- |
| Attention-RNN | ANN | GSC V2 (20-class) | 94.5% | 202K | 2019 | -- |
| LSTM | ANN (Baseline) | GSC V1 (12-class) | 94.4% | -- | 2020 | Yes |
| CNN (Baseline) | ANN | GSC V1 (12-class) | 87.6% | -- | 2020 | Yes |

### 1.2 Key Accuracy Takeaways

- **The gap is nearly closed.** In 2020, the best SNN (LSNN at 91.2%) trailed the best ANN (LSTM at 94.4%) by ~3.2 percentage points on GSC 12-class. By 2025, SpikCommander achieves 96.92% (35-class), which surpasses many ANN baselines.
- **12-class task (simpler):** SNNs now routinely achieve 95-97% accuracy, matching or exceeding ANN baselines.
- **35-class task (harder):** Best SNNs achieve ~96.9%, within 1-2 points of the ANN ceiling.
- **Parameter efficiency:** SpikCommander achieves 96.71% with only 1.12M parameters. ED-sKWS achieves 93% with only 27.6K parameters -- orders of magnitude fewer than typical ANNs.

### 1.3 SHD (Spiking Heidelberg Digits) Benchmark

| Model | Type | SHD Accuracy | Parameters | Year |
|-------|------|-------------|------------|------|
| **SpikCommander** | SNN | **96.41%** | 0.19M | 2025 |
| **SpikeSCR** | SNN | **95.70%** | -- | 2024 |
| **SE-adLIF** | SNN | **95.81%** | 0.45M | 2024 |
| **RadLIF (sparch)** | SNN | **97.60%** | ~1M | 2022 |
| **adLIF (sparch)** | SNN | **97.40%** | ~1M | 2022 |
| Hardware deployment | SNN | **93.4%** | -- | 2024 |

### 1.4 SSC (Spiking Speech Commands) Benchmark

| Model | Type | SSC Accuracy | Parameters | Year |
|-------|------|-------------|------------|------|
| **SpikCommander** | SNN | **83.49%** | 2.13M | 2025 |
| **SpikeSCR** | SNN | **82.79%** | -- | 2024 |
| **RadLIF (sparch)** | SNN | **93.40%** | ~1M | 2022 |
| CNN (Cramer et al.) | ANN | 77.7% | -- | 2020 |
| GRU | ANN | 79.05% | -- | 2020 |

---

## 2. Frameworks and Tools Available

### 2.1 Framework Comparison

| Framework | Maintainer | Language | Backend | Audio Support | Tutorials | Difficulty | PyPI |
|-----------|-----------|----------|---------|--------------|-----------|------------|------|
| **snnTorch** | UCSC (Eshraghian) | Python | PyTorch | SHD loader built-in | 18 tutorials | Beginner-friendly | Yes |
| **SpikingJelly** | Peking Univ. | Python | PyTorch | Speech Commands example (594 LOC) | Extensive docs | Intermediate | Yes |
| **sparch** | Idiap Research | Python | PyTorch | SHD, SSC, GSC, HD | Minimal (research code) | Intermediate | No |
| **Norse** | Community | Python | PyTorch | No dedicated audio | Intro notebooks | Intermediate | Yes |
| **Lava** | Intel | Python | Custom | Loihi deployment | Good docs | Advanced | Yes |
| **BindsNET** | UMass | Python | PyTorch | No dedicated audio | Examples | Intermediate | Yes |
| **Tonic** | Community | Python | PyTorch | SHD, SSC loaders | Data loading tutorials | Beginner-friendly | Yes |
| **Rockpool** | SynSense | Python | PyTorch | WaveSense tutorial | Good docs | Intermediate | Yes |

### 2.2 snnTorch (Recommended for Beginners)

- **Website:** https://snntorch.readthedocs.io/
- **GitHub:** https://github.com/jeshraghian/snntorch
- **Key features:**
  - 18 tutorials covering neuron models, feedforward SNNs, training, surrogate gradients, neuromorphic datasets
  - Built-in SHD dataset loader via `snntorch.spikevision.spikedata.SHD`
  - Google Colab notebook support (no local GPU needed)
  - Active maintenance, good community
- **Audio-specific:** Has SHD dataset example, but no dedicated audio classification tutorial. The general tutorials are directly applicable.
- **Install:** `pip install snntorch`

### 2.3 SpikingJelly

- **GitHub:** https://github.com/fangwei123456/spikingjelly
- **Published in Science Advances** (high-quality, peer-reviewed framework)
- **Key features:**
  - Includes a complete 594-line Speech Commands audio recognition example
  - Supports both activation-based and timestep-based training
  - CuPy acceleration for faster training
  - Internal MelScale implementation
- **Audio-specific:** `spikingjelly/activation_based/examples/speechcommands.py` -- a complete convolutional SNN for 12-class GSC
- **Install:** `pip install spikingjelly` or build from source

### 2.4 sparch (Purpose-Built for Audio)

- **GitHub:** https://github.com/idiap/sparch
- **Paper:** "A Surrogate Gradient Spiking Baseline for Speech Command Recognition" (Frontiers in Neuroscience, 2022)
- **Key features:**
  - Purpose-built for SNN speech command recognition
  - Supports 4 datasets: SHD, SSC, HD, GSC
  - Implements 4 neuron types: LIF, RLIF, adLIF, RadLIF
  - Clean PyTorch module design
  - Command-line experiment runner
- **Best for:** Reproducing published results and running comparative experiments
- **Install:** Clone from GitHub

### 2.5 Tonic (Data Loading)

- **Website:** https://tonic.readthedocs.io/
- **Purpose:** PyTorch-compatible loader for neuromorphic datasets (analogous to torchvision)
- **Key features:**
  - SHD and SSC dataset support built-in
  - Transform pipeline for event-based data
  - Works seamlessly with snnTorch and SpikingJelly
- **Install:** `pip install tonic`

---

## 3. Undergraduate-Level Implementations

### 3.1 Available Implementations Ranked by Accessibility

| Repository | Accessibility | Framework | Accuracy | LOC (core) | Dataset |
|-----------|-------------|-----------|----------|------------|---------|
| **SpikingJelly speechcommands.py** | Good | SpikingJelly/PyTorch | Competitive | ~494 | GSC V1 (12-class) |
| **sparch** | Good | PyTorch | SOTA | ~500-800 | SHD, SSC, GSC |
| **GoogleSpeechCommandsRNN** | Moderate | TensorFlow 2 | 91.2% (SNN) | ~1000+ | GSC V1 (12-class) |
| **SCommander** | Moderate | SpikingJelly | 96.9% | ~800+ | SHD, SSC, GSC |
| **RSNN** | Difficult | TensorFlow 1.2 | -- | ~500 | Custom |

### 3.2 Recommended Path for an Undergraduate

**Phase 1: Learning (Weeks 1-4)**
1. Complete snnTorch tutorials 1-5 (neuron models, feedforward SNNs, training)
2. Complete snnTorch tutorial 7 (neuromorphic datasets with Tonic)
3. Load and explore the SHD dataset using Tonic

**Phase 2: Baseline Implementation (Weeks 5-8)**
4. Implement a basic LIF-based SNN on SHD using snnTorch (~200-300 lines)
5. Implement the same architecture using SpikingJelly for comparison
6. Train and evaluate, achieving ~90% on SHD as a baseline

**Phase 3: Speech Commands (Weeks 9-14)**
7. Adapt to Google Speech Commands V2 (12-class first, then 35-class)
8. Implement Mel-spectrogram preprocessing pipeline
9. Build convolutional SNN architecture
10. Compare with an equivalent ANN baseline

**Phase 4: Analysis & Writing (Weeks 15-20)**
11. Energy consumption estimation (synaptic operations counting)
12. Accuracy vs. energy tradeoff analysis
13. Parameter sensitivity study
14. Thesis writing

### 3.3 Estimated Code Complexity

| Component | Estimated Lines | Difficulty |
|-----------|----------------|------------|
| Data loading + preprocessing | 50-100 | Easy |
| SNN model definition | 50-100 | Moderate |
| Training loop | 80-150 | Moderate |
| Evaluation + metrics | 50-80 | Easy |
| Visualization + analysis | 50-100 | Easy |
| **Total core implementation** | **280-530** | -- |
| ANN baseline for comparison | 100-200 | Easy |
| Full project with utilities | 500-1000 | -- |

A minimal working SNN for SHD classification can be achieved in approximately **200-300 lines** of Python using snnTorch. A full thesis-quality implementation with preprocessing, training, evaluation, comparison baselines, and visualization would typically be **500-1000 lines**.

---

## 4. Dataset Comparison and Recommendations

### 4.1 Dataset Overview

| Dataset | Classes | Samples | Format | Pre-spiked | Size | Task | Availability |
|---------|---------|---------|--------|------------|------|------|-------------|
| **SHD** | 20 (digits 0-9, EN+DE) | ~10,420 | Spike trains | Yes | ~700 MB | Digit recognition | Free (Zenke Lab) |
| **SSC** | 35 (speech commands) | ~105,829 | Spike trains | Yes | ~6 GB | Command recognition | Free (Zenke Lab) |
| **GSC V2** | 35 (or 12 subset) | ~105,829 | Raw audio (16kHz) | No | ~2.3 GB | Command recognition | Free (TensorFlow) |
| **TIDIGITS** | 11 (digits 0-9 + "oh") | ~25,104 | Raw audio | No | ~500 MB | Digit recognition | Licensed (LDC) |

### 4.2 Recommendation for Thesis

**Primary dataset: SHD (Spiking Heidelberg Digits)**
- Already in spike format (no encoding pipeline needed)
- Small enough for rapid iteration (~10K samples)
- Well-established benchmarks for comparison
- 20 classes -- enough complexity for a thesis
- Built-in loader in snnTorch and Tonic
- Published state-of-the-art: 96.41% (SpikCommander) to 97.60% (RadLIF)

**Secondary dataset: GSC V2 (Google Speech Commands, 12-class subset)**
- Industry-standard benchmark
- Requires audio-to-spike encoding (adds thesis content)
- Large community with many baselines
- 12-class subset is manageable; 35-class is stretch goal
- Published SNN SOTA: ~96.9%

**Why NOT TIDIGITS:**
- Requires LDC license (may cost money or institutional access)
- Fewer published SNN benchmarks
- Less active research community

**Why NOT SSC alone:**
- Very large dataset (6GB, long training times)
- 35 classes is challenging for a first SNN project
- Better as a stretch goal after SHD

### 4.3 Recommended Dataset Strategy

1. **Start with SHD** -- validate your approach quickly
2. **Move to GSC V2 12-class** -- show generalization to raw audio
3. **Optional stretch: GSC V2 35-class or SSC** -- if time permits

---

## 5. Implementation Complexity Analysis

### 5.1 Minimal Working Example (SHD with snnTorch)

A minimal classification pipeline requires approximately:

```
Data loading (Tonic/snnTorch):     ~30 lines
Model definition (LIF network):   ~40 lines
Training loop:                     ~60 lines
Evaluation:                        ~30 lines
Main script / config:              ~40 lines
-----------------------------------------
TOTAL:                             ~200 lines
```

### 5.2 Thesis-Quality Implementation

A full implementation with proper structure requires approximately:

```
Data loading + preprocessing:      ~100 lines
Model definition (multiple archs): ~150 lines
Training with logging:             ~150 lines
Evaluation + confusion matrix:     ~80 lines
Energy estimation:                 ~60 lines
Visualization:                     ~100 lines
ANN baseline:                      ~150 lines
Config / argument parsing:         ~60 lines
Utils (checkpointing, logging):    ~100 lines
-----------------------------------------
TOTAL:                             ~950 lines
```

### 5.3 Complexity Comparison with SpikingJelly Example

The SpikingJelly `speechcommands.py` example is 594 lines total (~494 source lines excluding comments/blanks). It includes:
- Mel-spectrogram feature extraction with delta features
- 3-layer convolutional SNN with dilated convolutions
- LIF neurons with tau=10/7
- Sigmoid surrogate gradient function
- Full training pipeline with 50 epochs
- Confusion matrix evaluation

This represents a complete, working, publishable implementation.

### 5.4 Key Technical Challenges

| Challenge | Difficulty | Mitigation |
|-----------|-----------|------------|
| Understanding LIF neuron dynamics | Medium | snnTorch tutorials 1-3 |
| Surrogate gradient training | Medium | snnTorch tutorial 5 |
| Audio-to-spike encoding | Medium | Use SHD (pre-encoded) or Mel-spectrograms |
| Hyperparameter tuning | Medium | Start from published configs (sparch) |
| GPU memory management | Low-Medium | Use small batch sizes, SHD is small |
| Reproducing published results | Medium | Use sparch or SpikingJelly examples |
| Energy estimation | Low | Count synaptic operations (MAC vs AC) |

---

## 6. Energy Efficiency Argument for SNNs in Audio

### 6.1 Theoretical Basis

SNNs achieve energy efficiency through three mechanisms:
1. **Event-driven computation:** Neurons only compute when they receive or emit a spike (sparse activity)
2. **Addition-only operations:** SNN inference uses accumulate (AC) operations instead of multiply-accumulate (MAC) operations. AC operations cost ~0.9 pJ vs ~4.6 pJ for MAC in 45nm CMOS.
3. **Temporal sparsity:** Audio signals are naturally sparse -- silence and low-activity periods require no computation

### 6.2 Concrete Energy Numbers

| Platform | Task | Energy/Inference | Power | Relative |
|----------|------|-----------------|-------|----------|
| **Intel Loihi** | KWS | ~110 mJ | 23 mW dynamic | **1x (baseline)** |
| **Intel Loihi 2** | Audio | -- | -- | **200x less than Jetson Orin Nano** |
| NVIDIA Jetson TX1 | KWS | ~1.7 J (est.) | -- | ~15x more |
| CPU (Cortex-M7) | KWS | ~1.65 J (est.) | -- | ~15x more |
| GPU (general) | KWS | -- | -- | ~109x more |
| NVIDIA Jetson Orin Nano | Audio | -- | -- | 200x more (vs Loihi 2) |

### 6.3 SNN-Specific Energy Measurements (from SpikCommander, 2025)

| Model | Energy (mJ) | SOPs (G) | Accuracy (GSC) |
|-------|-------------|---------|----------------|
| SpikCommander 1L | **0.028** | 0.008 | 96.71% |
| SpikCommander 2L | **0.042** | 0.020 | 96.92% |
| Spiking LMUFormer | 0.059 | 0.031 | 96.12% |
| SpikeSCR 2L | ~0.093 | -- | 95.60% |

Note: These energy numbers are estimated from synaptic operation counts (SOPs), not measured on neuromorphic hardware.

### 6.4 Energy Efficiency Caveats

- **Hardware dependency:** True energy gains require neuromorphic hardware (Loihi, SpiNNaker). On standard GPUs, SNNs may actually be SLOWER and less efficient than ANNs due to the timestep loop.
- **Sparsity requirement:** For SNNs to be more efficient than ANNs, the spike count per synapse must be below ~0.42-0.44 (verified empirically for VGG16/AlexNet topologies).
- **Estimation vs measurement:** Most papers estimate energy using synaptic operation counts and assumed hardware energy costs. True measurements require actual neuromorphic hardware deployment.
- **Thesis angle:** An undergraduate thesis can credibly argue the energy efficiency case using SOP counting methodology, which is standard practice in the field.

### 6.5 Energy Argument for the Thesis

The strongest thesis narrative is:

> "Keyword spotting must run continuously on edge devices with strict power budgets (smartwatches, hearing aids, IoT sensors). SNNs offer a path to sub-milliwatt keyword spotting by exploiting temporal sparsity in audio and event-driven computation. This thesis demonstrates that SNNs can achieve competitive accuracy (>95%) while requiring 5-55x fewer synaptic operations than equivalent ANNs, projecting significant energy savings on neuromorphic hardware."

---

## 7. Recent Papers (2024-2025) with Code

### 7.1 Papers with Available Code

| Paper | Venue | Year | Key Result | Code |
|-------|-------|------|-----------|------|
| **SpikCommander** | AAAI 2026 | 2025 | 96.92% GSC, 96.41% SHD, 83.49% SSC | [GitHub](https://github.com/JackieWang9811/SCommander) |
| **SpikeSCR (KDCL)** | arXiv | 2024 | 95.60% GSC, 95.70% SHD | Code promised (pending) |
| **A Surrogate Gradient Spiking Baseline** (sparch) | Frontiers Neurosci. | 2022 | 96.60% GSC, 97.60% SHD | [GitHub](https://github.com/idiap/sparch) |
| **GoogleSpeechCommandsRNN** (LSNN) | NeurIPS workshop | 2020 | 91.2% GSC (spiking) | [GitHub](https://github.com/dsalaj/GoogleSpeechCommandsRNN) |
| **SpikingJelly speechcommands** | -- | 2023 | Competitive on GSC 12-class | [GitHub (example)](https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/activation_based/examples/speechcommands.py) |
| **RSNN** | IEEE ICSICT | 2022 | -- (speech KWS) | [GitHub](https://github.com/edwardzcl/RSNN) |

### 7.2 Papers without Code (but Highly Relevant)

| Paper | Venue | Year | Key Result |
|-------|-------|------|-----------|
| **ED-sKWS** | Interspeech 2024 | 2024 | 93% GSC with 52% energy reduction, 27.6K params |
| **SIDC-KWS** | Interspeech 2025 | 2025 | 96.8% GSC 12-class, 75.6% less energy than ANN |
| **Speech2Spikes** | NICE 2023 | 2023 | Efficient audio encoding for neuromorphic systems |
| **WaveSense** | arXiv | 2021 | Efficient temporal convolutions for KWS |

### 7.3 Key Survey/Review Papers

- **"Snn and sound: a comprehensive review of spiking neural networks in sound"** (PMC, 2024) -- broad survey of SNNs in all audio tasks
- **"Advances in Small-Footprint Keyword Spotting"** (arXiv, 2025) -- comprehensive review including SNN approaches
- **"Overview of Spiking Neural Network Learning Approaches and Their Computational Complexities"** (MDPI Sensors, 2023)
- **"A Practical Tutorial on Spiking Neural Networks"** (Preprints.org, 2025) -- benchmarks multiple frameworks

---

## 8. Thesis Project Feasibility Assessment

### 8.1 Strengths as an Undergraduate Thesis Topic

| Factor | Assessment | Notes |
|--------|-----------|-------|
| **Novelty** | Strong | SNN audio is an active research area; many angles unexplored |
| **Feasibility** | Strong | Frameworks, code, and tutorials available |
| **Scope** | Well-defined | Clear benchmarks (SHD, GSC) with published baselines |
| **Literature** | Abundant | 50+ relevant papers, several surveys |
| **Code availability** | Good | Multiple complete implementations exist |
| **Compute requirements** | Moderate | SHD trainable on a single GPU in hours; GSC in a day |
| **Learning curve** | Moderate | Requires PyTorch proficiency + SNN concepts |
| **Originality opportunity** | Good | Many possible contributions (see below) |
| **Industry relevance** | Strong | Edge AI, smart devices, always-on audio |
| **Academic interest** | High | Neuromorphic computing is a growing field |

### 8.2 Potential Thesis Contributions

An undergraduate thesis does not need to beat SOTA. Credible contributions include:

1. **Systematic comparison:** Compare LIF, RLIF, adLIF, RadLIF on SHD and GSC using identical experimental setup (using sparch)
2. **Framework comparison:** Same model architecture, different frameworks (snnTorch vs SpikingJelly vs Norse) -- compare training speed, accuracy, ease of use
3. **Encoding study:** Compare Mel-spectrogram vs. cochlea model vs. rate coding for audio-to-spike conversion on GSC
4. **Efficiency-accuracy tradeoff:** Characterize how reducing timesteps, network size, or spike rates affects accuracy and estimated energy
5. **Robustness study:** Test SNN keyword spotting under noise conditions vs ANN baselines
6. **Small-footprint models:** Demonstrate competitive accuracy with very few parameters (following ED-sKWS direction)

### 8.3 Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Cannot reproduce published results | Medium | Medium | Use sparch (designed for reproducibility) |
| Training takes too long | Low | Medium | Start with SHD (10K samples); use Colab GPU |
| Difficulty understanding SNN math | Medium | Low | snnTorch tutorials are excellent |
| Framework bugs / compatibility | Medium | Low | snnTorch and SpikingJelly are mature |
| Cannot access neuromorphic hardware | High | Low | Energy estimation via SOP counting is standard |
| Scope creep | Medium | Medium | Fix scope to SHD + GSC 12-class + 1 contribution |

### 8.4 Minimum Viable Thesis

At minimum, a passing thesis could:
1. Implement a LIF-based SNN for SHD classification using snnTorch (~90%+ accuracy)
2. Implement an equivalent ANN baseline for comparison
3. Extend to GSC V2 12-class with Mel-spectrogram preprocessing
4. Compare accuracy, parameter count, and estimated energy (SOP counting)
5. Write up with proper literature review and analysis

This is achievable in one semester of focused work.

---

## 9. Recommended Tool Stack

```
Primary framework:     snnTorch (beginner-friendly, great tutorials)
Secondary framework:   SpikingJelly (for code comparison, has audio example)
Data loading:          Tonic (PyTorch-compatible neuromorphic data loaders)
Audio preprocessing:   torchaudio (Mel-spectrograms, MFCC)
Deep learning:         PyTorch 2.x
Experiment tracking:   Weights & Biases (free for students)
Visualization:         matplotlib, seaborn
Notebook environment:  Google Colab (free GPU) or local Jupyter
Version control:       Git + GitHub
```

### Installation commands:
```bash
pip install snntorch
pip install spikingjelly
pip install tonic
pip install torchaudio
pip install wandb
```

---

## 10. Key References and Resources

### Frameworks
- snnTorch: https://github.com/jeshraghian/snntorch
- SpikingJelly: https://github.com/fangwei123456/spikingjelly
- sparch: https://github.com/idiap/sparch
- Tonic: https://tonic.readthedocs.io/
- Norse: https://github.com/norse/norse
- Lava (Intel): https://github.com/lava-nc/lava
- Open Neuromorphic community: https://open-neuromorphic.org/

### Datasets
- SHD/SSC (Zenke Lab): https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/
- Google Speech Commands: via torchaudio or TensorFlow datasets
- Heidelberg Spiking Datasets (IEEE): https://ieee-dataport.org/open-access/heidelberg-spiking-datasets

### Code Repositories
- SpikCommander: https://github.com/JackieWang9811/SCommander
- sparch: https://github.com/idiap/sparch
- GoogleSpeechCommandsRNN: https://github.com/dsalaj/GoogleSpeechCommandsRNN
- RSNN: https://github.com/edwardzcl/RSNN
- SpikingJelly speech example: https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/activation_based/examples/speechcommands.py

### Key Papers
- Bittar & Garner (2022). "A Surrogate Gradient Spiking Baseline for Speech Command Recognition." Frontiers in Neuroscience.
- Wang et al. (2025). "SpikCommander: A High-performance Spiking Transformer." AAAI 2026. arXiv:2511.07883
- Song et al. (2024). "ED-sKWS: Early-Decision Spiking Neural Networks for Keyword Spotting." Interspeech 2024. arXiv:2406.12726
- SpikeSCR (2024). "Efficient Speech Command Recognition Leveraging SNN and Curriculum Learning-based Knowledge Distillation." arXiv:2412.12858
- Blouw et al. (2019). "Benchmarking Keyword Spotting Efficiency on Neuromorphic Hardware." arXiv:1812.01739
- Cramer et al. (2020). "The Heidelberg Spiking Data Sets." IEEE TNNLS.
- Weidel & Sheik (2021). "WaveSense: Efficient Temporal Convolutions with SNNs for KWS." arXiv:2111.01456
- Lim et al. (2025). "SIDC-KWS: Efficient Spiking Inception-Dilated Conformer." Interspeech 2025.

### Tutorials
- snnTorch Tutorial Series: https://snntorch.readthedocs.io/en/latest/tutorials/
- snnTorch SHD Example: https://snntorch.readthedocs.io/en/latest/examples/examples_svision/example_sv_shd.html
- Rockpool WaveSense Tutorial: https://rockpool.ai/tutorials/wavesense_training.html
- Open Neuromorphic Software Guide: https://open-neuromorphic.org/neuromorphic-computing/software/

---

## 11. Research Gaps and Confidence Assessment

### What I could not find:
- Exact SIDC-KWS parameter counts and full results table (behind Interspeech paywall)
- SpikeSCR released code (authors stated "upon camera-ready" as of Dec 2024)
- NEAT framework details (referenced on Open Neuromorphic but limited documentation found)
- Specific undergraduate theses on SNN audio processing (likely exist but not indexed well)

### Confidence levels:
- **High confidence:** Accuracy numbers from published papers, framework availability, dataset details
- **Medium confidence:** Energy efficiency projections (based on SOP counting, not hardware measurement), code line estimates
- **Lower confidence:** Whether SpikeSCR code has been released since late 2024, exact training times on consumer hardware

---

*Report generated February 2025. Research based on publicly available papers, repositories, and documentation.*
