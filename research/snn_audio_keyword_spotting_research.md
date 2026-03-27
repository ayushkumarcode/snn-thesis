# SNNs for Audio Processing: Keyword Spotting & Speech Command Recognition

Looking into whether SNN-based keyword spotting / speech command recognition could work as a thesis topic. Short answer: yes, this has gotten pretty mature in 2024-2025. The accuracy gap between SNNs and ANNs has narrowed a lot -- state-of-the-art SNNs now hit 96.9% on Google Speech Commands V2 (35-class), which is close to the ANN ceiling of ~97-98%. There are multiple open-source frameworks (snnTorch, SpikingJelly, sparch) with good documentation, and several complete implementations on GitHub in 300-600 lines of core Python. The energy efficiency argument holds up with hardware benchmarks showing 10-200x lower energy per inference on neuromorphic hardware (Intel Loihi) vs conventional processors.

---

## SNN vs ANN Accuracy on Google Speech Commands

### Current state of the art (early 2025)

| Model | Type | Dataset (Task) | Accuracy | Parameters | Year | Code? |
|-------|------|----------------|----------|------------|------|-------|
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

The gap is basically closed. In 2020, best SNN (LSNN at 91.2%) trailed best ANN (LSTM at 94.4%) by ~3.2 points on GSC 12-class. By 2025, SpikCommander gets 96.92% (35-class), beating many ANN baselines. On the 12-class task, SNNs routinely hit 95-97%, matching or exceeding ANNs. On the harder 35-class task, best SNNs get ~96.9%, within 1-2 points of ANN ceiling.

Parameter efficiency is interesting too: SpikCommander gets 96.71% with only 1.12M params. ED-sKWS gets 93% with just 27.6K parameters -- orders of magnitude fewer than typical ANNs.

### SHD (Spiking Heidelberg Digits) Benchmark

| Model | Type | SHD Accuracy | Parameters | Year |
|-------|------|-------------|------------|------|
| **SpikCommander** | SNN | **96.41%** | 0.19M | 2025 |
| **SpikeSCR** | SNN | **95.70%** | -- | 2024 |
| **SE-adLIF** | SNN | **95.81%** | 0.45M | 2024 |
| **RadLIF (sparch)** | SNN | **97.60%** | ~1M | 2022 |
| **adLIF (sparch)** | SNN | **97.40%** | ~1M | 2022 |
| Hardware deployment | SNN | **93.4%** | -- | 2024 |

### SSC (Spiking Speech Commands) Benchmark

| Model | Type | SSC Accuracy | Parameters | Year |
|-------|------|-------------|------------|------|
| **SpikCommander** | SNN | **83.49%** | 2.13M | 2025 |
| **SpikeSCR** | SNN | **82.79%** | -- | 2024 |
| **RadLIF (sparch)** | SNN | **93.40%** | ~1M | 2022 |
| CNN (Cramer et al.) | ANN | 77.7% | -- | 2020 |
| GRU | ANN | 79.05% | -- | 2020 |

---

## Frameworks and Tools

### Framework Comparison

| Framework | Maintainer | Audio Support | Tutorials | Difficulty | Install |
|-----------|-----------|--------------|-----------|------------|---------|
| **snnTorch** | UCSC (Eshraghian) | SHD loader built-in | 18 tutorials | Beginner-friendly | `pip install snntorch` |
| **SpikingJelly** | Peking Univ. | Speech Commands example (594 LOC) | Extensive docs | Intermediate | `pip install spikingjelly` |
| **sparch** | Idiap Research | SHD, SSC, GSC, HD | Minimal (research code) | Intermediate | Clone from GitHub |
| **Norse** | Community | No dedicated audio | Intro notebooks | Intermediate | `pip install norse` |
| **Lava** | Intel | Loihi deployment | Good docs | Advanced | `pip install lava` |
| **BindsNET** | UMass | No dedicated audio | Examples | Intermediate | `pip install bindsnet` |
| **Tonic** | Community | SHD, SSC loaders | Data loading tutorials | Beginner-friendly | `pip install tonic` |
| **Rockpool** | SynSense | WaveSense tutorial | Good docs | Intermediate | `pip install rockpool` |

### snnTorch (probably the best starting point)
- https://github.com/jeshraghian/snntorch
- 18 tutorials covering neuron models, feedforward SNNs, training, surrogate gradients, neuromorphic datasets
- Built-in SHD dataset loader via `snntorch.spikevision.spikedata.SHD`
- Google Colab support (no local GPU needed)
- Has SHD example but no dedicated audio classification tutorial -- the general tutorials apply directly though.

### SpikingJelly
- https://github.com/fangwei123456/spikingjelly
- Published in Science Advances
- Has a complete 594-line Speech Commands audio recognition example
- Supports activation-based and timestep-based training, CuPy acceleration
- Internal MelScale implementation
- The audio example is at `spikingjelly/activation_based/examples/speechcommands.py` -- complete convolutional SNN for 12-class GSC

### sparch (purpose-built for audio)
- https://github.com/idiap/sparch
- Paper: "A Surrogate Gradient Spiking Baseline for Speech Command Recognition" (Frontiers in Neuroscience, 2022)
- Purpose-built for SNN speech command recognition
- Supports 4 datasets: SHD, SSC, HD, GSC
- Implements 4 neuron types: LIF, RLIF, adLIF, RadLIF
- Clean PyTorch module design, command-line experiment runner
- Best for reproducing published results and running comparative experiments

### Tonic (data loading)
- https://tonic.readthedocs.io/
- PyTorch-compatible loader for neuromorphic datasets (like torchvision but for spikes)
- SHD and SSC support built-in, transform pipeline for event-based data
- Works with snnTorch and SpikingJelly

---

## Implementations and How to Get Started

### Available implementations ranked by accessibility

| Repository | Accessibility | Framework | Accuracy | LOC (core) | Dataset |
|-----------|-------------|-----------|----------|------------|---------|
| **SpikingJelly speechcommands.py** | Good | SpikingJelly/PyTorch | Competitive | ~494 | GSC V1 (12-class) |
| **sparch** | Good | PyTorch | SOTA | ~500-800 | SHD, SSC, GSC |
| **GoogleSpeechCommandsRNN** | Moderate | TensorFlow 2 | 91.2% (SNN) | ~1000+ | GSC V1 (12-class) |
| **SCommander** | Moderate | SpikingJelly | 96.9% | ~800+ | SHD, SSC, GSC |
| **RSNN** | Difficult | TensorFlow 1.2 | -- | ~500 | Custom |

### How i'd approach this as an undergrad

**Phase 1: Learning (Weeks 1-4)**
1. Do snnTorch tutorials 1-5 (neuron models, feedforward SNNs, training)
2. Do tutorial 7 (neuromorphic datasets with Tonic)
3. Load and play with SHD dataset using Tonic

**Phase 2: Baseline (Weeks 5-8)**
4. Implement basic LIF-based SNN on SHD with snnTorch (~200-300 lines)
5. Implement same architecture in SpikingJelly for comparison
6. Train and evaluate, shoot for ~90% on SHD as baseline

**Phase 3: Speech Commands (Weeks 9-14)**
7. Move to Google Speech Commands V2 (12-class first, then 35-class)
8. Implement Mel-spectrogram preprocessing
9. Build convolutional SNN architecture
10. Compare with equivalent ANN baseline

**Phase 4: Analysis & Writing (Weeks 15-20)**
11. Energy estimation (synaptic operations counting)
12. Accuracy vs energy tradeoff analysis
13. Parameter sensitivity study
14. Write thesis

### Estimated code complexity

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

Minimal working SNN for SHD can be done in about 200-300 lines of Python with snnTorch. Full thesis-quality implementation with preprocessing, training, evaluation, comparison, and visualization would be 500-1000 lines.

### Key technical challenges

| Challenge | Difficulty | How to deal with it |
|-----------|-----------|---------------------|
| Understanding LIF neuron dynamics | Medium | snnTorch tutorials 1-3 |
| Surrogate gradient training | Medium | snnTorch tutorial 5 |
| Audio-to-spike encoding | Medium | Use SHD (pre-encoded) or Mel-spectrograms |
| Hyperparameter tuning | Medium | Start from published configs (sparch) |
| GPU memory management | Low-Medium | Small batch sizes, SHD is small |
| Reproducing published results | Medium | Use sparch or SpikingJelly examples |
| Energy estimation | Low | Count synaptic operations (MAC vs AC) |

---

## Datasets

| Dataset | Classes | Samples | Format | Pre-spiked | Size | Availability |
|---------|---------|---------|--------|------------|------|-------------|
| **SHD** | 20 (digits 0-9, EN+DE) | ~10,420 | Spike trains | Yes | ~700 MB | Free (Zenke Lab) |
| **SSC** | 35 (speech commands) | ~105,829 | Spike trains | Yes | ~6 GB | Free (Zenke Lab) |
| **GSC V2** | 35 (or 12 subset) | ~105,829 | Raw audio (16kHz) | No | ~2.3 GB | Free (TensorFlow) |
| **TIDIGITS** | 11 (digits 0-9 + "oh") | ~25,104 | Raw audio | No | ~500 MB | Licensed (LDC) |

**Recommendation: start with SHD.** Already in spike format (no encoding pipeline needed), small enough for rapid iteration (~10K samples), well-established benchmarks, 20 classes is enough complexity, built-in loaders in snnTorch and Tonic.

**Then move to GSC V2 12-class** -- industry standard benchmark, requires audio-to-spike encoding (adds thesis content), large community with many baselines.

TIDIGITS requires a license (may cost money or need institutional access), and SSC alone is very large (6GB, long training times) -- better as a stretch goal after SHD.

Strategy:
1. Start with SHD -- validate approach quickly
2. Move to GSC V2 12-class -- show generalization to raw audio
3. Optional stretch: GSC V2 35-class or SSC if time permits

---

## Energy Efficiency Argument

### How SNNs save energy

Three mechanisms:
1. **Event-driven computation:** neurons only compute when they receive or emit a spike (sparse activity)
2. **Addition-only operations:** SNN inference uses accumulate (AC) operations instead of multiply-accumulate (MAC). AC costs ~0.9 pJ vs ~4.6 pJ for MAC in 45nm CMOS.
3. **Temporal sparsity:** audio signals are naturally sparse -- silence and low-activity periods need no computation

### Concrete energy numbers

| Platform | Task | Energy/Inference | Relative |
|----------|------|-----------------|----------|
| **Intel Loihi** | KWS | ~110 mJ | 1x (baseline) |
| **Intel Loihi 2** | Audio | -- | 200x less than Jetson Orin Nano |
| NVIDIA Jetson TX1 | KWS | ~1.7 J (est.) | ~15x more |
| CPU (Cortex-M7) | KWS | ~1.65 J (est.) | ~15x more |
| GPU (general) | KWS | -- | ~109x more |

### SNN-specific energy from SpikCommander (2025)

| Model | Energy (mJ) | SOPs (G) | Accuracy (GSC) |
|-------|-------------|---------|----------------|
| SpikCommander 1L | **0.028** | 0.008 | 96.71% |
| SpikCommander 2L | **0.042** | 0.020 | 96.92% |
| Spiking LMUFormer | 0.059 | 0.031 | 96.12% |
| SpikeSCR 2L | ~0.093 | -- | 95.60% |

Note: these energy numbers are estimated from synaptic operation counts, not measured on actual hardware.

### Caveats to keep in mind

- True energy gains need neuromorphic hardware (Loihi, SpiNNaker). On standard GPUs, SNNs may actually be slower and less efficient than ANNs because of the timestep loop.
- For SNNs to be more efficient than ANNs, the spike count per synapse needs to be below ~0.42-0.44 (verified empirically for VGG16/AlexNet topologies).
- Most papers estimate energy using synaptic operation counts and assumed hardware energy costs. True measurements require actual neuromorphic hardware.
- An undergrad thesis can credibly argue energy efficiency using SOP counting methodology -- that's standard practice in the field.

The thesis narrative would be something like: keyword spotting must run continuously on edge devices with strict power budgets (smartwatches, hearing aids, IoT sensors). SNNs can exploit temporal sparsity in audio and event-driven computation. If you can show competitive accuracy (>95%) with 5-55x fewer synaptic operations than equivalent ANNs, that projects significant energy savings on neuromorphic hardware.

---

## Would This Work as an Undergrad Thesis?

i think so. Here's the breakdown:

| Factor | Assessment | Notes |
|--------|-----------|-------|
| Novelty | Strong | SNN audio is active but many angles unexplored |
| Feasibility | Strong | Frameworks, code, tutorials all available |
| Scope | Well-defined | Clear benchmarks (SHD, GSC) with published baselines |
| Literature | Abundant | 50+ relevant papers, several surveys |
| Code availability | Good | Multiple complete implementations |
| Compute needs | Moderate | SHD trainable on single GPU in hours; GSC in a day |
| Learning curve | Moderate | Need PyTorch proficiency + SNN concepts |
| Originality opportunity | Good | See below |
| Industry relevance | Strong | Edge AI, smart devices, always-on audio |

### Possible contributions (don't need to beat SOTA)

1. **Neuron type comparison:** Compare LIF, RLIF, adLIF, RadLIF on SHD and GSC using identical experimental setup (using sparch)
2. **Framework comparison:** Same model, different frameworks (snnTorch vs SpikingJelly vs Norse) -- compare training speed, accuracy, ease of use
3. **Encoding study:** Compare Mel-spectrogram vs cochlea model vs rate coding for audio-to-spike conversion on GSC
4. **Efficiency-accuracy tradeoff:** How does reducing timesteps, network size, or spike rates affect accuracy and estimated energy
5. **Robustness study:** Test SNN keyword spotting under noise vs ANN baselines
6. **Small-footprint models:** Competitive accuracy with very few parameters (following ED-sKWS direction)
