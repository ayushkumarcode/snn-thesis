# SNNs for Audio Classification: SHD & SSC Benchmarks

so i've been looking into whether SNNs are actually competitive for audio classification, specifically on the Spiking Heidelberg Digits (SHD) and Spiking Speech Commands (SSC) benchmarks. turns out they are -- like genuinely beating ANNs, which is kind of surprising honestly.

the best SNN methods on SHD are hitting around 96.4% accuracy, which blows past the best ANN baselines (92.4% CNN, 90.4% GRU). on the larger SSC benchmark, best SNNs get 83.5-86% vs the GRU baseline of 79%. the field moved fast since 2022 -- key innovations include learnable synaptic delays (DCLS-Delays, ICLR 2024), adaptive neuron models (adLIF, RadLIF, SE-adLIF), parameter-free attention (Pfa-SNN), and spiking transformers (SpikCommander). most of these have open-source code and train on a single GPU in minutes to hours. this seems really feasible for a 3rd year thesis.

---

## 1. The Datasets

### 1.1 Spiking Heidelberg Digits (SHD)

| Property | Value |
|----------|-------|
| Task | Spoken digit classification (0-9 in English and German) |
| Classes | 20 |
| Training samples | 8,156 |
| Test samples | 2,264 |
| Input channels | 700 (artificial cochlea model) |
| Data format | Spike trains with timestamps |
| Created by | Cramer et al. (2020), Zenke Lab, Heidelberg |
| Reference | [arXiv:1910.07407](https://arxiv.org/abs/1910.07407) |

SHD encodes spoken digit recordings into spike trains using "Lauscher," an artificial cochlea model that mimics the human inner ear. each sample is spike events across 700 frequency channels with precise temporal information.

### 1.2 Spiking Speech Commands (SSC)

| Property | Value |
|----------|-------|
| Task | Speech command classification |
| Classes | 35 |
| Total samples | ~100,000 |
| Base dataset | Google Speech Commands v0.2 |
| Input channels | 700 (same cochlea model) |
| Data format | Spike trains with timestamps |
| Created by | Same group (Cramer et al.) |

SSC is the bigger, harder sibling of SHD. 35 speech command classes, lots of speakers, same cochlea encoding.

### 1.3 Dataset Access

both datasets are available through multiple loaders:
- **Tonic** library: `pip install tonic` then `tonic.datasets.SHD('./data', train=True)` or `tonic.datasets.SSC('./data', split='train')`
- **snnTorch**: built-in `snntorch.spikevision.spikedata.SHD()`
- **Norse**: `norse.dataset.spiking_heidelberg`
- **sparch toolkit**: auto download via config
- **SNN-delays repo**: auto download and preprocessing
- **Direct download**: [Zenke Lab resources page](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/)

---

## 2. State-of-the-Art Results

### 2.1 SHD Leaderboard (as of February 2026)

| Rank | Model | Accuracy | Params | Time Steps | Recurrent | Year | Code Available |
|------|-------|----------|--------|------------|-----------|------|----------------|
| 1 | MCRE (Multi-Scale Chunked Residual Encoding) | **96.44%** | -- | -- | -- | 2025 | -- |
| 2 | SpikCommander (1L-8-128) | **96.41%** | 0.19M | 100 | No | 2025 | -- |
| 3 | Pfa-SNN (Parameter-free Attention) | **96.26%** | 0.20M | 100 | -- | 2025 | -- |
| 4 | Event-SSMA (ANN model) | 95.90% | 0.40M | -- | -- | -- | -- |
| 5 | SE-adLIF (2L) | 95.81% | 0.45M | 250 | Yes | 2024 | [GitHub](https://github.com/IGITUGraz/SE-adlif) |
| 6 | SpikeSCR (1L) | 95.60% | 0.26M | 100 | -- | 2025 | -- |
| 7 | DCLS-Delays (2L) | **95.07%** | 0.20M | 100 | **No** | 2024 | [GitHub](https://github.com/Thvnvtos/SNN-delays) |
| 8 | d-cAdLIF (2L) | 94.85% | 0.08M | 100 | -- | 2024 | -- |
| 9 | RadLIF (3x1024) | 94.62% | -- | -- | Yes | 2022 | [sparch](https://github.com/idiap/sparch) |
| 10 | adLIF (3x128) | 93.06% | -- | -- | No | 2022 | [sparch](https://github.com/idiap/sparch) |
| 11 | DH-SNN (2L) | 92.10% | 0.05M | 1000 | -- | -- | -- |
| 12 | Spikformer (1L) | 90.10% | 1.77M | 100 | No | -- | -- |
| 13 | SDT (1L) | 89.61% | 1.77M | 100 | No | -- | -- |

### 2.2 SSC Leaderboard (as of February 2026)

| Rank | Model | Accuracy | Params | Time Steps | Year | Code Available |
|------|-------|----------|--------|------------|------|----------------|
| 1 | SpikCommander (2L, T=250) | **85.98%** | 2.13M | 250 | 2025 | -- |
| 2 | SpikCommander (2L, T=200) | 85.52% | 2.13M | 200 | 2025 | -- |
| 3 | SpikCommander (2L, T=100) | 83.49% | 2.13M | 100 | 2025 | -- |
| 4 | SpikCommander (1L, T=100) | 83.26% | 1.12M | 100 | 2025 | -- |
| 5 | SpikeSCR (2L) | 82.79% | 3.30M | 100 | 2025 | -- |
| 6 | SpikeSCR (1L) | 82.54% | 1.71M | 100 | 2025 | -- |
| 7 | DH-SNN (3L) | 82.46% | 0.35M | 1000 | -- | -- |
| 8 | MCRE | 80.92% | -- | -- | 2025 | -- |
| 9 | DCLS-Delays (3L) | 80.69% | 2.50M | 100 | 2024 | [GitHub](https://github.com/Thvnvtos/SNN-delays) |
| 10 | SE-adLIF (2L) | 80.44% | 1.60M | 250 | 2024 | [GitHub](https://github.com/IGITUGraz/SE-adlif) |
| 11 | d-cAdLIF (2L) | 80.23% | 0.70M | 100 | 2024 | -- |
| 12 | Pfa-SNN | 80.18% | 0.71M | 100 | 2025 | -- |
| 13 | DCLS-Delays (2L) | 80.16% | 1.40M | 100 | 2024 | [GitHub](https://github.com/Thvnvtos/SNN-delays) |
| 14 | Spikformer (2L) | 80.18% | 2.57M | 100 | -- | -- |
| 15 | SDT (2L) | 79.82% | 2.57M | 100 | -- | -- |
| 16 | RadLIF (3x1024) | 77.40% | -- | -- | 2022 | [sparch](https://github.com/idiap/sparch) |

### 2.3 Key Methods

**DCLS-Delays (ICLR 2024)** -- this was the breakthrough paper. they use Dilated Convolutions with Learnable Spacings to learn synaptic delays in feedforward SNNs. each synapse gets a 1D Gaussian kernel whose position (representing the delay) is learned during training. hits 95.07% on SHD with just 2 feedforward layers of 256 LIF neurons each, no recurrent connections at all. the Gaussians narrow during training to produce discrete delays that'd work on neuromorphic hardware. open source, clean implementation, pretty easy to reproduce.

**SE-adLIF (2024)** -- uses an improved discretization scheme (Symplectic Euler) for adaptive LIF neurons. the standard Euler-forward discretization introduces systematic errors; SE corrects this. gets 95.81% on SHD with recurrent connections.

**SpikCommander (2025)** -- a fully spike-driven transformer using Multi-view Spiking Temporal-Aware Self-Attention (MSTASA) and Spiking Contextual Refinement Channel MLP (SCR-MLP). 96.41% on SHD with only 0.19M params and 83.49-85.98% on SSC.

**Pfa-SNN (2025)** -- adds parameter-free attention directly into the spiking neuron. no additional parameters needed. 96.26% on SHD. pretty elegant approach actually.

**MCRE (2025)** -- Multi-Scale Chunked Residual Encoding inspired by hippocampus-cortex information reorganization. 96.44% on SHD (current best) and 80.92% on SSC, while reducing energy consumption by up to 55%.

**RadLIF / adLIF (2022)** -- the surrogate gradient baseline from Bittar and Garner. RadLIF = Recurrent Adaptive LIF, adLIF = Adaptive LIF (non-recurrent). these established the competitive baselines. open source via the sparch toolkit.

---

## 3. SNN vs Traditional ANN Comparison

### 3.1 Accuracy Comparison on SHD

| Architecture | Type | Best SHD Accuracy | Notes |
|-------------|------|-------------------|-------|
| SpikCommander | SNN (Transformer) | 96.41% | 0.19M params |
| DCLS-Delays | SNN (Feedforward) | 95.07% | 0.20M params, no recurrence |
| RadLIF | SNN (Recurrent) | 94.62% | Surrogate gradient |
| **CNN (Cramer 2020)** | **ANN** | **92.4%** | **Best ANN baseline** |
| GRU (3x128) | ANN | 90.40% | Gated recurrent unit |
| liBRU (3x128) | ANN | 89.61% | Lightweight bistable RNN |
| LSTM | ANN | ~89% | Standard LSTM |

the big takeaway: SNNs actually beat ANNs on SHD by a pretty wide margin (96.4% vs 92.4%). this is one of the few benchmarks where SNNs clearly outperform traditional deep learning.

### 3.2 Accuracy Comparison on SSC

| Architecture | Type | Best SSC Accuracy | Notes |
|-------------|------|-------------------|-------|
| SpikCommander | SNN (Transformer) | 85.98% (T=250) | Current SOTA |
| SpikeSCR | SNN | 82.54% | Curriculum distillation |
| DCLS-Delays | SNN (Feedforward) | 80.69% | 3-layer, no recurrence |
| **GRU (3x512)** | **ANN** | **79.05%** | **Best ANN baseline** |
| liBRU (3x512) | ANN | 78.70% | Lightweight bistable RNN |
| CNN | ANN | 77.7% | Convolutional |
| RadLIF (3x1024) | SNN (Recurrent) | 77.40% | 2022 baseline |
| LSTM | ANN | ~73% | Standard LSTM |

on SSC, SNNs also beat ANNs (85.98% vs 79.05%), but the gap took longer to emerge. the 2022 SNN baseline (77.4%) was actually below the GRU (79.05%), but by 2025 SNNs lead convincingly.

### 3.3 Energy Efficiency

| Metric | SNN | ANN |
|--------|-----|-----|
| Inference energy (general) | 5-15 mJ | ~200 mJ |
| SynOps ratio vs ANN | 0.68x | 1.0x (baseline) |
| Energy-delay product | Up to 8.2x better | Baseline |
| Neuromorphic hardware (SpiNNaker) | ~0.3W | N/A |

SNNs get their energy efficiency through sparse, event-driven computation. the advantage is biggest on neuromorphic hardware (Loihi, SpiNNaker, BrainScaleS), where SNNs can be 10-100x more efficient than equivalent ANNs on GPU. even when both run on GPU, SNNs use fewer multiply-accumulate operations thanks to spike sparsity.

### 3.4 Why SNNs Win on These Audio Benchmarks

been thinking about why SNNs do so well here specifically:

1. **Natural temporal affinity**: audio is inherently temporal. spike trains naturally encode timing info, unlike frame-based ANNs that have to learn temporal structure from scratch.
2. **Data is already spike-encoded**: SHD/SSC comes from an artificial cochlea that outputs spike trains. SNNs process this natively; ANNs have to convert it to dense tensors which loses temporal precision.
3. **Learnable delays**: methods like DCLS-Delays exploit the temporal structure of spike trains by learning optimal signal propagation delays -- something ANNs can't really do naturally.
4. **Biological plausibility**: the cochlea-to-spike encoding mirrors biological auditory processing, and SNNs process these in a biologically plausible way.

---

## 4. Student and Undergraduate Projects

### 4.1 Confirmed Student SNN Audio Projects

| Project | Level | University | Tools | Results |
|---------|-------|-----------|-------|---------|
| Musical Pattern Recognition in SNNs | BEng | Unknown (~2016) | Brian 2, STDP | First layer of multi-layer SNN for music patterns. Author notes "only a small portion of what was originally intended was achieved" -- realistic and honest. [GitHub](https://github.com/mrahtz/musical-pattern-recognition-in-spiking-neural-networks) |
| Audio Classification with SNN (SpiNNaker) | Research | Seville | PyNN, SpiNNaker | 3-layer SNN for 8 pure tones. [GitHub](https://github.com/jpdominguez/Multilayer-SNN-for-audio-samples-classification-using-SpiNNaker) |
| Biologically Inspired Audio Classification | MSc | U. Calgary (2021) | Custom | Compared spike encoding schemes for audio. Developed bio-plausible learning rule. [Thesis](https://ucalgary.scholaris.ca/items/d4288028-91ef-436a-ae12-62f07e5ac43b) |
| Hardware Implementation of SNN for Speech | MSc | U. Padova | FPGA | Hardware SNN for SHD. [Thesis](https://thesis.unipd.it/retrieve/b7626846-af0e-4988-8610-90698072a72d/Toffano_Marco.pdf) |
| Energy-efficient SNNs at the Edge | PhD | U. Grenoble Alpes (2023) | Custom | Models and algorithms for energy-efficient SNNs on neuromorphic hardware. [Thesis](https://theses.hal.science/tel-04331152/file/DAMPFHOFFER_2023_archivage.pdf) |
| Randomised Time-Stepping for SNN | BSc | Imperial College (2021) | Python | Mathematical investigation. [GitHub](https://github.com/Fabio752/Randomised-time-stepping-methods-for-SNN-simulations) |

### 4.2 what this means

i couldn't find any undergrad projects specifically doing SNN audio classification on SHD/SSC. that's interesting because:
- it would be a **novel undergraduate contribution** (good for thesis originality)
- the MSc thesis from Calgary (2021) and the BEng music project are the closest things out there
- SHD didn't even exist before 2020, and surrogate gradient training only became accessible to students around 2022-2023

---

## 5. Frameworks and Tools

### 5.1 Framework Comparison for SHD/SSC Work

| Framework | Language | SHD Support | Difficulty | Speed | Best For |
|-----------|----------|-------------|------------|-------|----------|
| **snnTorch** | PyTorch | Built-in loader + tutorial | Beginner-friendly | Moderate | Learning SNNs, modular design, great tutorials |
| **SpikingJelly** | PyTorch | Via Tonic | Intermediate | Fast (CuPy backend) | Performance, built-in models, Science Advances paper |
| **sparch** | PyTorch | Built-in, automatic | Beginner-friendly | Moderate | Specifically designed for SHD/SSC speech tasks |
| **Spyx** | JAX | Built-in tutorial | Intermediate | **Fastest** (JIT) | Speed, GPU/TPU optimization, research iteration |
| **Norse** | PyTorch | Built-in dataset class | Intermediate | Moderate | Bio-inspired components |
| **Rockpool** | PyTorch/JAX | Built-in tutorial | Intermediate | Moderate | Neuromorphic hardware deployment (Xylo) |
| **SNN-delays** | PyTorch | Built-in, automatic | Intermediate | Moderate | Reproducing ICLR 2024 delay learning results |
| **Tonic** | Python | Dedicated dataset class | Easy (data only) | N/A | Dataset loading and transforms only |

### 5.2 What i'd recommend for an undergrad thesis

**Primary: snnTorch + Tonic**
- best tutorial series (8 interactive Jupyter notebooks)
- built-in SHD dataset loader
- modular neuron models (LIF, Leaky, Synaptic, Alpha)
- easy integration with PyTorch
- active community and documentation
- `pip install snntorch tonic`

**If you need speed: Spyx (JAX)**
- train SHD models in under 60 seconds on a single GPU
- JIT compilation for maximum GPU utilization
- steeper learning curve but dramatically faster iteration

**For reproducing SOTA: SNN-delays or sparch**
- SNN-delays: clean ICLR 2024 code, easy to set up, auto-downloads SHD/SSC
- sparch: purpose-built for speech command recognition with SNNs, supports LIF/RLIF/adLIF/RadLIF

### 5.3 Training Time Benchmarks (100 epochs, SHD, single GPU)

| Framework | Batch 64 | Batch 128 | Batch 256 |
|-----------|----------|-----------|-----------|
| snnTorch (uncompiled) | 4543s (~76 min) | 2313s (~39 min) | 1212s (~20 min) |
| snnTorch (compiled) | 196s (~3.3 min) | 103s (~1.7 min) | 59s (~1 min) |
| mlGeNN | 216s (~3.6 min) | 161s (~2.7 min) | 124s (~2 min) |
| **Spyx** | **69s (~1.2 min)** | **46s (~0.8 min)** | **36s (~0.6 min)** |

these are for a 128-neuron feedforward SNN getting ~70-75% accuracy. SOTA models (256-1024 neurons, multiple layers) take longer but still stay in the minutes-to-low-hours range on a single GPU. SHD is a small dataset so training is fast no matter what.

### 5.4 Energy Analysis Tools

- **syops** library: computes Synaptic Operations (SynOps) and estimates energy based on 45nm technology
- **Manual SynOps counting**: count spike-triggered MAC operations
- **Comparison approach**: run equivalent ANN, count FLOPs with `torchprofile` or `thop`, compare with SNN SynOps

---

## 6. Feasibility for a 3rd-Year Thesis

### 6.1 Why This is Doable

| Factor | Assessment |
|--------|------------|
| Dataset size | Small (8K training, 2K test). Fits in RAM. Fast to iterate. |
| Training time | Minutes per experiment on a single GPU (even a laptop GPU) |
| Available code | Multiple open-source baselines: sparch, SNN-delays, snnTorch tutorials |
| Framework maturity | snnTorch has 8 tutorials, extensive docs, active community |
| Prior undergraduate work | Musical pattern recognition BEng project provides scope calibration |
| Novelty potential | No known undergraduate SHD/SSC project. Novel at this level. |
| Apple M-series support | PyTorch MPS backend works for snnTorch/SpikingJelly. No CUDA needed. |
| Results achievable | A 2-layer SNN should reach 90%+ on SHD with surrogate gradients |
| ANN comparison | Easy to implement LSTM/GRU baseline in PyTorch for comparison |
| Energy analysis | Feasible via SynOps counting, no neuromorphic hardware needed |

### 6.2 Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Can't reproduce SOTA results | Low-Medium | Start with sparch baseline (RadLIF, 94.62%). Use provided code and configs. |
| Training instability | Low | SHD is well-behaved. Surrogate gradients are stable. Use proven hyperparams from papers. |
| Dataset download issues | Low | Multiple mirrors. Tonic handles caching. |
| Understanding surrogate gradients | Medium | snnTorch tutorials 5-6 cover this well. Start there. |
| Running out of time | Low | Minimum viable project (1 SNN + 1 ANN baseline on SHD) can be done in 2-3 weeks of focused work. |
| Scope creep | Medium | Stick to SHD initially. SSC is a stretch goal. |

### 6.3 Rough Timeline (semester-length)

| Phase | Duration | Activities |
|-------|----------|------------|
| 1. Foundation | 2-3 weeks | Complete snnTorch tutorials 1-7. Understand LIF neurons, surrogate gradients, spike encoding. Load SHD with Tonic. |
| 2. Baseline Implementation | 2-3 weeks | Implement ANN baselines (LSTM, GRU) on SHD. Implement basic LIF SNN using sparch or snnTorch. Aim for ~85-90% on SHD. |
| 3. Core Experiments | 3-4 weeks | Compare neuron models (LIF vs adLIF vs RadLIF). Add learnable delays (DCLS approach). Hyperparameter search. Target: 92-95% on SHD. |
| 4. Analysis | 2-3 weeks | Energy efficiency analysis (SynOps vs FLOPs). Spike activity visualization. Ablation studies. Error analysis. |
| 5. Writing & Polish | 2-3 weeks | Write thesis. Create figures. Run final experiments. |

### 6.4 Minimum Viable (guaranteed to pass)

- implement 1 SNN (LIF or adLIF) on SHD using snnTorch or sparch
- implement 1 ANN baseline (GRU or LSTM) on SHD
- compare accuracy, training time, and estimated energy
- write up with clear methodology and analysis
- expected accuracy: SNN 85-92%, ANN 88-92%
- time required: 4-6 weeks of focused work

### 6.5 Ambitious Version (first-class potential)

- compare 3-4 neuron models (LIF, RLIF, adLIF, RadLIF) on both SHD and SSC
- implement delay learning (reproduce DCLS-Delays ICLR 2024 paper)
- compare against GRU, LSTM, and small Transformer
- detailed energy analysis with SynOps
- spike train visualization and interpretability
- ablation study on key hyperparameters (layers, neurons, time steps)
- expected accuracy: 93-95% on SHD
- time required: full semester of consistent work

---

## 7. Research Question Ideas

### 7.1 Options

**Option A (i think this is the strongest):**
> "To what extent can feedforward spiking neural networks with learnable synaptic delays match or exceed recurrent networks for spoken digit classification on the SHD benchmark, and what are the energy efficiency implications?"

i like this one because:
- clear independent variable (delays vs. recurrence)
- measurable outcomes (accuracy, energy)
- builds on DCLS-Delays (ICLR 2024) which showed feedforward SNNs with delays can match recurrent ones
- achievable -- the SNN-delays codebase can be directly used
- contributes to an open question about whether temporal processing needs recurrence or can be done through delays

**Option B (comparative focus):**
> "How do spiking neural networks compare to traditional recurrent architectures (LSTM, GRU) in accuracy, computational cost, and energy efficiency for audio classification on neuromorphic speech benchmarks?"

broader and more survey-like. good for showing breadth but risks becoming just a benchmark-running exercise without deep insight.

**Option C (neuron model focus):**
> "What is the effect of adaptive neuron dynamics on temporal audio classification in spiking neural networks, and how does neuron model complexity trade off against accuracy and energy consumption?"

focuses on comparing LIF, adLIF, RadLIF, and potentially SE-adLIF. sparch toolkit is designed for exactly this.

**Option D (encoding focus):**
> "How do different spike encoding strategies affect the accuracy and efficiency of spiking neural networks for audio classification?"

explores rate coding vs temporal coding vs the cochlea-based encoding of SHD. more neuroscience-flavored. could include custom encoding of raw audio.

### 7.2 What Makes a Good Research Question Here

for a 3rd year thesis, the question should be:
- **Answerable** with available tools and data
- **Bounded** (1-2 datasets, not 5)
- **Novel at the undergraduate level** (not just reproducing a paper)
- **Measurable** (quantitative metrics: accuracy, SynOps, training time)
- **Interesting** to a thesis examiner who might not be an SNN expert

option A seems strongest because the delay-vs-recurrence question is currently debated in the literature, the tools exist to answer it, and the results have clear implications for neuromorphic hardware deployment (feedforward networks are way easier to put on hardware than recurrent ones).

---

## 8. Key Papers to Read

### Essential (read these first)

1. **Cramer et al. (2020)** -- "The Heidelberg Spiking Data Sets for the Systematic Evaluation of Spiking Neural Networks" -- [arXiv:1910.07407](https://arxiv.org/abs/1910.07407)
   the dataset paper. defines SHD and SSC.

2. **Bittar & Garner (2022)** -- "A Surrogate Gradient Spiking Baseline for Speech Command Recognition" -- [Frontiers in Neuroscience](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.865897/full)
   establishes the baseline SNN results on SHD/SSC. introduces adLIF and RadLIF. sparch toolkit comes from this.

3. **Hammouamri, Khalfaoui-Hassani & Masquelier (2024)** -- "Learning Delays in Spiking Neural Networks using Dilated Convolutions with Learnable Spacings" -- [ICLR 2024](https://openreview.net/forum?id=4r2ybzJnmN)
   the DCLS-Delays breakthrough. shows feedforward SNNs with delays beat recurrent SNNs. clean code available.

### Important (read for depth)

4. **Ding et al. (2024)** -- "Advancing Spatio-Temporal Processing in Spiking Neural Networks through Adaptation" -- [arXiv:2408.07517](https://arxiv.org/html/2408.07517)
   the SE-adLIF paper. improved neuron discretization.

5. **Sun & Wu (2025)** -- "Towards Parameter-free Attentional Spiking Neural Networks" -- [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0893608025000334)
   Pfa-SNN. attention in SNNs without extra parameters.

6. **SpikCommander (2025)** -- "A High-performance Spiking Transformer with Multi-view Learning for Efficient Speech Command Recognition" -- [arXiv:2511.07883](https://arxiv.org/abs/2511.07883)
   current SOTA on SSC. spiking transformer architecture.

### Review Papers

7. **Baek & Lee (2024)** -- "SNN and Sound: A Comprehensive Review of Spiking Neural Networks in Sound" -- [Biomedical Engineering Letters](https://link.springer.com/article/10.1007/s13534-024-00406-y)
   review of all SNN audio applications.

8. **Li et al. (2025)** -- "Spiking Neural Networks for Temporal Processing: Status Quo and Future Prospects" -- [arXiv:2502.09449](https://arxiv.org/html/2502.09449v1)
   critical assessment. argues current benchmarks may not adequately test temporal processing. worth keeping in mind.

---

## 9. Open Research Gaps

some things that haven't been done yet that could be interesting:

1. **Benchmark adequacy**: Li et al. (2025) argue that SHD/SSC may not truly test temporal processing because "temporal credit assignment during the backward pass is not necessary for these datasets." could test this by comparing SNNs trained with and without temporal backpropagation.

2. **Feedforward vs recurrent with delays**: DCLS-Delays showed feedforward SNNs with delays can match recurrent ones, but this hasn't been tested across all neuron models.

3. **Energy analysis standardization**: most papers report accuracy but few report standardized energy metrics. a project that measures SynOps and estimates energy across multiple methods would actually be a genuine contribution.

4. **Undergraduate accessibility**: there's no tutorial or reproducible notebook walking through a complete SHD experiment from data loading to analysis with modern methods. making one would help the community.

5. **Encoding comparison**: comparing the cochlea-based SHD encoding against simpler encodings (rate coding of mel spectrograms, delta encoding) hasn't been done for SNN audio classification.

---

## 10. How Confident Am I In All This

| Finding | Confidence | Source Quality |
|---------|------------|---------------|
| SHD SOTA is ~96.4% | HIGH | Multiple recent papers agree |
| SSC SOTA is ~85.98% | HIGH | SpikCommander paper with detailed tables |
| ANN baselines (GRU 90.4%, CNN 92.4% on SHD) | HIGH | Original dataset paper + sparch paper |
| SNNs beat ANNs on SHD/SSC | HIGH | Consistent across multiple papers since 2024 |
| SHD trainable in minutes on single GPU | HIGH | Spyx benchmarks + multiple framework docs |
| snnTorch is most beginner-friendly | HIGH | Tutorial quality, documentation, community |
| No undergrad SHD project exists publicly | MEDIUM-HIGH | Searched GitHub pretty thoroughly, but private projects might exist |
| Energy efficiency claims (5-100x) | MEDIUM | Varies hugely by hardware platform and methodology |
| Feasibility for 3rd year thesis | HIGH | Small dataset, available tools, multiple baselines |

---

## 11. Sources

### Datasets
- [Zenke Lab SHD/SSC page](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/)
- [IEEE DataPort](https://ieee-dataport.org/open-access/heidelberg-spiking-datasets)
- [Papers with Code SHD Benchmark](https://paperswithcode.com/sota/audio-classification-on-shd)

### Key Papers
- [Cramer et al. 2020 (SHD/SSC dataset)](https://arxiv.org/abs/1910.07407)
- [Bittar & Garner 2022 (sparch baseline)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.865897/full)
- [Hammouamri et al. 2024 ICLR (DCLS-Delays)](https://openreview.net/forum?id=4r2ybzJnmN)
- [SpikCommander 2025](https://arxiv.org/abs/2511.07883)
- [SE-adLIF 2024](https://arxiv.org/html/2408.07517)
- [Pfa-SNN 2025](https://www.sciencedirect.com/science/article/abs/pii/S0893608025000334)
- [MCRE 2025](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5534158)

