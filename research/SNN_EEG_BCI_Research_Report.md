# SNN-based EEG classification for brain-computer interfaces

i looked into whether SNNs could be used for EEG-based BCIs, and honestly it's a pretty active area. the field has picked up a lot since 2023, with new architectures, benchmarks, and open-source code coming out regularly. the basic pitch is that SNNs are biologically plausible (the brain itself uses spikes), energy efficient (up to 95% less energy than DNNs on neuromorphic hardware), and naturally suited to temporal signals like EEG.

that said, SNNs currently trail state-of-the-art CNNs and transformers by about 3-10 percentage points on most EEG benchmarks, though the gap is closing. the main tasks people are tackling with SNN-EEG include motor imagery classification, emotion recognition, seizure detection, stress detection, and SSVEP classification. there's plenty of public datasets (BCI Competition IV-2a/2b, PhysioNet EEGMMIDB, DEAP, SEED) and frameworks (snnTorch, SpikingJelly, Norse, combra-lab/snn-eeg) to make this doable for an undergrad who knows Python/PyTorch but not neuroscience.

---

## 1. what people are doing with SNN-based EEG classification

### motor imagery (MI) classification

this is the most studied task by far. subjects imagine moving body parts (left hand, right hand, feet, tongue) and you classify the EEG patterns over the motor cortex.

key papers:
- **SCNet** (2023): CNN feature extraction + SNN biological interpretability with adaptive coding and surrogate gradient learning. Tested on PhysioNet, BCI IV-2a, and BCI IV-2b. Beats prior SNN methods.
- **HR-SNN** (2024): End-to-end SNN getting 77.58% average on BCI IV-2a (4-class), beating all compared SNN models. On PhysioNet: 67.24% (global) and 74.95% (transfer learning).
- **NiSNN-A** (2024): Non-iterative SNN with attention for motor imagery. Combines accuracy gains with energy reduction.
- **LENet/RDSNN** (2024): Lightweight SNN getting 73.65% on PhysioNet, 81.75% on BCI IV-2a, 84.56% on BCI IV-2b.
- **Lightweight SNN** (2025, ScienceDirect): Within-subject and cross-subject experiments on three public datasets show the SNN beats classical CNN-based models.
- **combra-lab SNN-EEG** (2022, TMLR): Deployed on Intel Loihi, consuming 95% less energy than DNNs on NVIDIA Jetson TX2 with similar accuracy.

### emotion recognition

second-most studied task. subjects watch emotional stimuli while EEG is recorded, and you classify valence, arousal, dominance.

key papers:
- **EESCN** (2024, Computer Methods and Programs in Biomedicine): Gets 94.56% (valence), 94.81% (arousal), 94.73% (dominance) on DEAP and 79.65% on SEED-IV. Faster and uses less memory than prior SNN methods.
- **Fractal-SNN** (2023): Uses multi-scale temporal-spectral-spatial information. Tested on DREAMER, DEAP, SEED-IV, and MPED.
- **NeuroSense** (Tan et al., 2021): 78.97%/67.76% (arousal/valence) on DEAP.
- **Bidirectional SNN** (Alzhrani et al., 2021): 94.83% accuracy on DREAMER.
- **BISNN** (2025): Bio-information-fused SNN for emotion recognition.

### epilepsy/seizure detection

growing application area with obvious clinical motivation.

key papers:
- **EESNN** (2024): Recurrent spiking convolution structure, energy reduction by several orders of magnitude vs ANNs.
- **Spiking Conformer** (2024): Trained on raw EEG, no preprocessing needed. 10x fewer operations than non-spiking equivalent.
- **SyNSense Xylo deployment** (2024): Real-time sub-milliwatt epilepsy detection on a neuromorphic edge processor.
- **Cross-patient SNN** (2024, Frontiers in Neuroscience): Efficient cross-patient seizure detection.

### other tasks

- **Stress detection:** CSNN (2025, Scientific Reports) gets 98.75% with 10-fold CV and F1 of 98.60%.
- **SSVEP:** Event-driven SNN (2024, IEEE) using empirical mode decomposition and CCA with excitation-inhibition balanced SNN.
- **P300:** SNNs have been used for P300 signal reconstruction and data augmentation (2021, Frontiers), but dedicated P300 SNN classifiers are rare. this is a potential gap.
- **Sleep staging:** Hybrid SNN (HSNN) demonstrated for automatic sleep staging.
- **Situational awareness:** SNN with SCTN neurons (2024, Applied Sciences).

### how architectures have evolved

| Generation | Approach | Example |
|---|---|---|
| Early (pre-2020) | Reservoir/NeuCube + STDP | NeuCube framework |
| Mid (2020-2022) | Surrogate gradient SNNs | combra-lab SNN-EEG |
| Recent (2023-2024) | Hybrid CNN-SNN, attention | SCNet, HR-SNN, NiSNN-A |
| Emerging (2024-2025) | Spiking Transformers | Spikeformer, Spiking Conformer |
| Frontier (2025+) | Lightweight + edge deployment | LENet, Xylo-based systems |

---

## 2. accuracy comparison: SNNs vs conventional approaches

### motor imagery (BCI Competition IV Dataset 2a, 4-class)

| Method | Type | Accuracy (%) | Year |
|---|---|---|---|
| EEGEncoder (Transformer) | ANN | 86.46 | 2024 |
| CIACNet (Attention CNN) | ANN | 85.15 | 2024 |
| SNA-MHC (custom SNN+Attn) | SNN | 92.80* | 2024 |
| RDSNN (Lightweight SNN) | SNN | 81.75 | 2024 |
| HR-SNN (End-to-End SNN) | SNN | 77.58 | 2024 |
| CNN1D_MF | ANN | 69.20 | 2023 |
| DFBRTS | ANN | 78.16 | 2024 |

*SNA-MHC's 92.80% is an outlier -- might use different evaluation protocols. most SNN results cluster around 75-82% on this benchmark.

### motor imagery (PhysioNet EEGMMIDB)

| Method | Type | Accuracy (%) | Year |
|---|---|---|---|
| Novel DL approach | ANN | 95.70 | 2025 |
| Optimized DL + DWT | ANN | 97.05 | 2024 |
| HR-SNN (transfer) | SNN | 74.95 | 2024 |
| HR-SNN (global) | SNN | 67.24 | 2024 |
| RDSNN | SNN | 73.65 | 2024 |
| combra-lab SNN-EEG | SNN | ~similar to DNN* | 2022 |

*combra-lab reports "similar classification performance" to DNNs with 95% less energy.

### emotion recognition (DEAP dataset)

| Method | Type | Valence (%) | Arousal (%) | Year |
|---|---|---|---|---|
| Graph CNN + Dual Attention | ANN | ~90+ | ~90+ | 2024 |
| EESCN | SNN | 94.56 | 94.81 | 2024 |
| NeuroSense | SNN | 67.76 | 78.97 | 2021 |
| Bidirectional SNN (DREAMER) | SNN | 94.83 (overall) | -- | 2021 |

### stress detection

| Method | Type | Accuracy (%) | Year |
|---|---|---|---|
| CSNN | SNN | 98.75 | 2025 |
| Hybrid SNN | SNN | 94.00 | 2024 |

### the takeaway on accuracy

the gap between SNNs and ANNs depends a lot on the task:
- **emotion recognition and stress detection:** SNNs can match or beat ANNs (EESCN at 94.81% is competitive with SOTA on DEAP)
- **motor imagery (4-class):** SNNs still trail top ANNs by ~5-10pp (77-82% vs 85-87% for best ANNs), though the gap is narrowing
- **seizure detection:** SNNs get comparable accuracy with orders-of-magnitude lower energy

the real argument for SNNs isn't just accuracy -- it's the accuracy-energy tradeoff. an SNN at 80% accuracy that uses 1/20th the energy of an 85% CNN might be the better choice for a wearable BCI.

---

## 3. available datasets

### motor imagery

| Dataset | Subjects | Channels | Classes | Sampling Rate | Access | Notes |
|---|---|---|---|---|---|---|
| **BCI Competition IV-2a** | 9 | 22 EEG + 3 EOG | 4 (left hand, right hand, feet, tongue) | 250 Hz | [BBCI website](https://www.bbci.de/competition/iv/) / [Kaggle](https://www.kaggle.com/datasets/thngdngvn/bci-competition-iv-data-sets-2a) | Most popular MI benchmark (31+ studies). 288 trials/subject across 2 sessions. |
| **BCI Competition IV-2b** | 9 | 3 EEG + 3 EOG | 2 (left hand, right hand) | 250 Hz | [BBCI website](https://www.bbci.de/competition/iv/) | Second most popular (14+ studies). |
| **PhysioNet EEGMMIDB** | 109 | 64 | 4 (open/close fists, imagine fists/feet) | 160 Hz | [PhysioNet](https://www.physionet.org/content/eegmmidb/1.0.0/) | Freely downloadable, no application needed. Largest free MI dataset. |
| **BNCI Horizon 2020** | Various | Various | Various | Various | [BNCI database](https://bnci-horizon-2020.eu/database/data-sets) | Collection of multiple BCI datasets. |

### emotion recognition

| Dataset | Subjects | Channels | Classes | Stimuli | Access | Notes |
|---|---|---|---|---|---|---|
| **DEAP** | 32 | 32 EEG + 8 peripheral | Valence/Arousal/Dominance (continuous) | 40 music videos (1 min each) | [DEAP official](http://eecs.qmul.ac.uk/mmv/datasets/deap/) | Needs university email application (~1 month wait). 512 Hz, preprocessed files at 128 Hz available. |
| **SEED** | 15 | 62 EEG | 3 (negative, neutral, positive) | Film clips | [BCMI Lab](https://bcmi.sjtu.edu.cn/home/seed/seed.html) | 200 Hz preprocessed. 3 sessions/subject (~1 week apart). |
| **SEED-IV** | 15 | 62 EEG | 4 (happy, sad, fear, neutral) | Film clips | [BCMI Lab](https://bcmi.sjtu.edu.cn/home/seed/index.html) | Extension of SEED with 4 emotions. |
| **SEED-VII** | -- | -- | 6 basic emotions + continuous | Multimodal | [BCMI Lab](https://bcmi.sjtu.edu.cn/home/seed/) | Newest, continuous labels. |
| **DREAMER** | 23 | 14 EEG | Valence/Arousal/Dominance | Film clips | Public | Lower channel count, good for lightweight models. |

### seizure detection

| Dataset | Notes |
|---|---|
| **CHB-MIT** | Scalp EEG from pediatric subjects with intractable seizures. PhysioNet. |
| **Bonn University** | 5 classes (healthy, epileptic zone, seizure). Classic benchmark. |
| **TUH EEG Corpus** | Large-scale clinical EEG from Temple University Hospital. |

### what i'd recommend for an undergrad project

**best starting point:** PhysioNet EEGMMIDB -- freely available, no application process, large (109 subjects), well-documented, lots of published baselines, and directly supported by the combra-lab/snn-eeg codebase.

**best for emotion recognition:** DEAP -- industry standard, supported by TorchEEG, but you need to apply with a university email (plan ahead, takes ~1 month).

---

## 4. why SNNs actually make sense for BCI

### biological plausibility

SNNs communicate through discrete spikes, which is what real neurons do. this creates a natural alignment between the computation model and the biological signals being decoded:

- EEG reflects aggregate neural spiking activity
- SNNs process through spike timing and rates -- same coding schemes biological neurons use
- STDP in SNNs mirrors actual synaptic learning rules
- membrane potential and refractory periods naturally capture temporal structure in EEG

this isn't just philosophical -- SNN architectures can leverage neuroscience knowledge about EEG generation to inform network design.

### energy efficiency

this is probably the strongest practical argument:

- combra-lab SNN on Intel Loihi: 95% less energy per inference than DNNs on NVIDIA Jetson TX2
- Spiking Conformer for seizure detection: 10x fewer operations than non-spiking equivalent
- SyNSense Xylo: sub-milliwatt real-time epilepsy detection
- SNNs use additions instead of multiplications (no MACs), and event-driven processing means energy only gets used during spike transmission

this matters for BCI because wearable/implantable devices need to run on batteries for days or weeks. that's the kind of energy budget only neuromorphic computing can deliver. the brain runs on about 20W total -- SNNs on neuromorphic hardware approach this kind of efficiency.

### real-time processing

- SNNs process temporal information natively -- don't need to accumulate a window before processing
- event-driven computation means asynchronous processing as spikes arrive
- neuromorphic hardware (Loihi, SpiNNaker, Xylo, TrueNorth) enables parallel event-driven operations with minimal latency
- critical for BCI where milliseconds matter (e.g., motor intent decoding for prosthetic control)

### temporal dynamics

EEG is inherently temporal, and SNNs handle this naturally:
- spiking neurons maintain internal state (membrane potential) that integrates temporal info
- no need for explicit temporal feature engineering or sliding windows
- recurrent connections can capture long-range temporal dependencies
- contrast with CNNs that treat EEG as quasi-static images and might miss temporal dynamics

### compact models

- SNNs tend to have fewer parameters than equivalent ANNs
- EESCN showed faster speed and less memory than prior methods
- lightweight architectures like LENet replace FC layers with classification convolution blocks
- smaller models = easier edge deployment + less training data needed

---

## 5. open-source code on GitHub

### SNN-EEG specific repos

| Repository | What it does | Stars | Paper |
|---|---|---|---|
| [combra-lab/snn-eeg](https://github.com/combra-lab/snn-eeg) | PyTorch + Intel Loihi for decoding EEG on neuromorphic hardware. Spatial conv, temporal conv, recurrent layers. Trained on EEGMMIDB. | Active | TMLR 2022 |
| [SuperBruceJia/EEG-DL](https://github.com/SuperBruceJia/EEG-DL) | Deep Learning library for EEG tasks including SNN-related approaches. TensorFlow. | Popular | -- |
| [TheBrainLab/Awesome-Spiking-Neural-Networks](https://github.com/TheBrainLab/Awesome-Spiking-Neural-Networks) | Curated paper list with codes for SNN research including EEG stuff. | Large | -- |
| [SpikingChen/SNN-Daily-Arxiv](https://github.com/SpikingChen/SNN-Daily-Arxiv) | Daily updated arXiv SNN papers. Handy for tracking new stuff. | Active | -- |

### general SNN frameworks that work for EEG

| Framework | Repo | Key features | Docs |
|---|---|---|---|
| **snnTorch** | [jeshraghian/snntorch](https://github.com/jeshraghian/snntorch) | PyTorch-based, great tutorials (7 parts), surrogate gradients, GPU-accelerated. Best for beginners. | [snntorch.readthedocs.io](https://snntorch.readthedocs.io/) |
| **SpikingJelly** | [fangwei123456/spikingjelly](https://github.com/fangwei123456/spikingjelly) | Full-stack SNN toolkit: preprocessing, building, training, neuromorphic deployment. Published in Science Advances. | [spikingjelly.readthedocs.io](https://spikingjelly.readthedocs.io/) |
| **Norse** | [norse/norse](https://github.com/norse/norse) | PyTorch primitives for bio-inspired neural components. More research-oriented. | [norse.github.io/norse](https://norse.github.io/norse/) |

### EEG processing libraries

| Library | Repo | What it does |
|---|---|---|
| **TorchEEG** | [torcheeg/torcheeg](https://github.com/torcheeg/torcheeg) | PyTorch-based EEG analysis. Plug-and-play API, built-in DEAP/SEED loaders, multiple DL models. |
| **MNE-Python** | [mne-tools/mne-python](https://github.com/mne-tools/mne-python) | Industry standard for EEG preprocessing, filtering, artifact removal. Essential. |
| **MOABB** | [NeuroTechX/moabb](https://github.com/NeuroTechX/moabb) | Mother of All BCI Benchmarks. Standardized evaluation framework for BCI algorithms. |

### what i'd use for a project

```
EEG Data Loading:    TorchEEG (for DEAP/SEED) or MNE-Python (for PhysioNet/BCI IV)
EEG Preprocessing:   MNE-Python (filtering, artifact removal, epoching)
SNN Framework:       snnTorch (best tutorials) or SpikingJelly (most features)
Base Framework:      PyTorch
Spike Encoding:      snnTorch built-in encoders (rate coding, latency coding)
Evaluation:          scikit-learn metrics, MOABB benchmarks
```

---

## 6. could an undergrad without neuroscience background do this?

### short answer: yes, with the right scope

**why it's feasible:**

1. **no wet-lab work** -- all datasets are public and pre-recorded. no recruiting subjects, no ethics approval, no EEG equipment.

2. **good tooling** -- TorchEEG, MNE-Python, and snnTorch abstract away a lot of complexity. TorchEEG loads DEAP/SEED with a few lines of code.

3. **PyTorch foundation** -- if you know PyTorch, snnTorch is a manageable jump. the tutorials assume DL knowledge, not neuroscience.

4. **tutorials exist:**
   - snnTorch has a 7-part tutorial series from spike encoding fundamentals
   - combra-lab/snn-eeg has a complete working pipeline
   - TorchEEG has workflow examples for DEAP and SEED

5. **published reference implementations** -- there are multiple working codebases you can study, reproduce, and extend.

6. **the neuroscience you actually need is learnable in 1-2 weeks:**
   - what EEG measures (aggregate electrical activity from scalp)
   - what motor imagery is (imagining movement)
   - basic frequency bands (alpha, beta, mu rhythms)
   - what ERD/ERS means (event-related desynchronization/synchronization)

### challenges and how to deal with them

| Challenge | How hard | What to do |
|---|---|---|
| EEG preprocessing (filtering, artifact removal) | Medium | MNE-Python tutorials; many datasets come preprocessed |
| Spike encoding (EEG to spikes) | Medium | snnTorch built-in encoders; follow tutorial 1 |
| Surrogate gradient training | Medium-Hard | snnTorch/SpikingJelly handle this automatically |
| Cross-subject variability | Hard | Start with within-subject, attempt cross-subject later |
| Hyperparameter tuning | Medium | Use published papers' configs as starting points |
| Neuroscience terminology | Medium | Keep a glossary; you only need ~20 terms |
| Debugging temporal SNN dynamics | Medium | snnTorch visualization tools for membrane potentials |

### scoping it

**minimum viable project (safe scope):**
- reproduce combra-lab/snn-eeg results on PhysioNet EEGMMIDB
- compare SNN vs EEGNet (standard CNN baseline) on same dataset
- report accuracy, parameter count, estimated energy consumption
- timeline: ~3-4 months

**ambitious but achievable:**
- implement SNN for motor imagery using snnTorch on BCI Competition IV-2a
- compare multiple spike encodings (rate vs temporal)
- benchmark against 2-3 CNN baselines (EEGNet, ShallowConvNet, DeepConvNet)
- energy analysis using theoretical MAC operation counts
- timeline: ~4-6 months

**stretch goals (if time permits):**
- cross-subject transfer learning with SNNs
- hybrid CNN-SNN architecture
- deployment on neuromorphic hardware simulator
- real-time inference demo

### crash course reading list (1-2 weeks)

1. "An Introduction to EEG" -- any intro neuroscience textbook chapter
2. BCI Competition IV website documentation
3. snnTorch Tutorial Series (Tutorials 1-5)
4. "Spiking neural networks for EEG signal analysis: From theory to practice" (ScienceDirect, 2025)
