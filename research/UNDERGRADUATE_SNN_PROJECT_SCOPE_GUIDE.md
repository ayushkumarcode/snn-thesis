# Undergraduate SNN/Neuromorphic Project Scope Guide

> Compiled from 4 parallel research investigations across Manchester eScholar, GitHub, UK university repositories, and the SpiNNaker/APT group.
> **Date:** 2026-02-24

---

## TL;DR

- **UK universities do NOT publicly archive undergraduate dissertations.** Only PhD/MPhil theses appear in institutional repositories.
- **GitHub is the primary way to find undergraduate SNN work.** 3 confirmed BSc/BEng projects found, plus 15+ masters theses.
- **Heidelberg University (BrainScaleS group)** publicly archives 60+ bachelor's theses on neuromorphic hardware -- the best resource for scope calibration.
- **Manchester's SpiNNaker group** has 16+ PhD theses (all freely downloadable) but no public undergrad work. They almost certainly supervise 3rd-year projects but these are behind authentication.
- **A realistic 3rd-year SNN project:** 1-2 SNN architectures, 1-2 datasets, comparison with ANN baseline, energy analysis. Deliverable: Jupyter notebooks + 40-80 page report.

---

## Part 1: Confirmed Undergraduate (BSc/BEng) Projects

Only 3 confirmed undergraduate SNN projects were found publicly:

### 1. Shape Detector SNN -- Filippo Ferrari (Manchester, ~2018, BSc AI)
- **Supervisor:** Prof. Steve Furber
- **What:** Single-layer SNN for shape detection
- **Tools:** Python, pyDVS library
- **Code:** https://github.com/filippoferrari/shape_detector_snn
- **Thesis source:** https://github.com/filippoferrari/bsc_dissertation (LaTeX)
- **Scope:** MODERATE -- well-structured with tests, CI, configs. 107 commits.
- **Takeaway:** Good example of a BSc project at a top university. Clean code, proper testing. Supervised by the creator of SpiNNaker.

### 2. Musical Pattern Recognition in SNNs -- mrahtz (~2016, BEng)
- **What:** First layer of a multi-layer SNN for recognising musical patterns in audio
- **Tools:** Brian 2 simulator, STDP learning, custom .wav audio
- **Code:** https://github.com/mrahtz/musical-pattern-recognition-in-spiking-neural-networks
- **Thesis PDF:** http://amid.fish/beng_project_report.pdf
- **Scope:** MODERATE -- 49 stars, 17 forks. Author candidly notes "only a small portion of what was originally intended was actually achieved"
- **Takeaway:** Novel application domain (music). High community interest. Honest about limitations -- this is realistic and expected for a final-year project.

### 3. Randomised Time-Stepping Methods for SNN Simulations -- Fabio Deo (Imperial, 2021)
- **What:** Mathematical investigation of randomness in SNN time-stepping methods
- **Tools:** Python (98.9% of repo)
- **Code:** https://github.com/Fabio752/Randomised-time-stepping-methods-for-SNN-simulations
- **Thesis PDF:** Included in repository
- **Scope:** MODERATE -- theoretical/computational, appropriate for Imperial

### Other Noteworthy Undergraduate Projects (not SNN-specific but adjacent):

| Project | University | Tools | What |
|---------|-----------|-------|------|
| SNN for Digit Recognition (C++) | King's College London, 2018 | C++, OpenCV | From-scratch SNN, no framework |
| SNN for Autonomous Locomotion | Unknown, bachelor thesis | V-REP, ROS, R-STDP | Robot following red object (did not converge -- honest about failure) |
| Spiking Stereo Matching | Unknown, bachelor | SpiNNaker, sPyNNaker | Event-based stereo matching, 2ms latency, published paper |
| QuadBot Neuromorphic | Cambridge (summer project) | MATLAB, VEX robots | CPG-based quadrupedal locomotion |
| SNN NoC Architecture | Sri Lanka (4th year) | Verilog, FPGA, RISC-V | Team of 3, custom hardware |

---

## Part 2: Masters Thesis Projects (Closer Scope Reference)

Masters projects are more ambitious than undergrad but give a useful upper bound.

### Most Relevant Masters Projects:

| Project | Tools | Results | Thesis PDF? |
|---------|-------|---------|-------------|
| Spiking Grid Cells on SpiNNaker (Manchester MSc, Buttigieg 2019) | SpiNNaker, sPyNNaker, Python 2.7 | Grid cell spatial navigation model | Yes (Google Drive) |
| Event-Based Robot Control (TUM, 59 stars) | TensorFlow, V-REP, ROS, NEST, DVS | Lane-following with DRL+SNN | Yes |
| Deep Spiking Q-Networks (TUM, 11 stars) | SpyTorch, NEST, OpenAI Gym | CartPole + RL with SNNs | Yes |
| Brain-Machine Interface (SpiNNaker) | SpiNNaker 4 chips, STDP | 73.4% EEG classification | Yes |
| SNN-RL Actor-Critic (21 stars) | NEST 3, Docker, MongoDB | R-STDP for line-following | Yes |
| SNN for Hand Kinematics from sEMG | Brian2, C++/Cython | EMG-to-hand-movement decoding | No |
| SNN Formation Control | Norse, PyTorch | Multi-agent formation control | No |

---

## Part 3: Course/Research Projects (Best Scope Calibration)

These are the most realistic comparisons for what a 3rd-year project looks like:

### Tier 1 -- Achievable (Good Grade)

| Project | Framework | Datasets | Key Result |
|---------|-----------|----------|------------|
| SNN vs CNN Comparison (sofi12321) | snnTorch | SOCOFing, EMNIST, Fashion-MNIST | SNN 98% vs CNN 83% on fingerprints |
| SNN Image Classification (HaoyiZhu) | snnTorch | Static + spike data | 99.12% static, 97.05% spike |
| Simple SNN with STDP (4 students, Osnabruck) | Python from scratch | MNIST | Plateaued quickly, honest about limits |
| ANN vs SNN Comparison (NicolaCST) | Python | Various | Power consumption analysis |

### Tier 2 -- Ambitious (Very Good Grade)

| Project | Framework | Datasets | Key Result |
|---------|-----------|----------|------------|
| Deep Learning with Biologically Plausible NNs (chiralevy) | snnTorch | MNIST, CIFAR-10, Speech Commands | MNIST 98.06%, CIFAR-10 70.60%, Speech 91.20% |
| SNN Gesture Classification (DerrickL25) | snnTorch | DVS128 Gesture | Neuromorphic event data classification |
| Convolutional SNN for Speech Recognition | PyTorch, scikit-learn | TIDIGITS | 92% accuracy |

### Tier 3 -- Outstanding (but risky)

| Project | Framework | What Makes It Special |
|---------|-----------|----------------------|
| Shape Detector SNN (Ferrari, Manchester) | pyDVS, Python | Supervised by Furber, clean engineering |
| Musical Pattern Recognition (mrahtz) | Brian 2, STDP | Novel domain, published thesis PDF |
| RL-SNN-Quadrupeds (UC Berkeley) | MuJoCo, PPO | Real hardware deployment (partially failed) |

---

## Part 4: Heidelberg BrainScaleS -- The Gold Standard for Bachelor's Scope

The Kirchhoff Institute for Physics at Heidelberg publicly archives ALL bachelor's and master's theses on neuromorphic hardware. **This is the single best resource for understanding what a bachelor's thesis in this field looks like.**

**Full listing:** http://www.kip.uni-heidelberg.de/vision/publications/mscbsc/

### Selected Bachelor's Theses (most relevant):

| Year | Title | What They Did |
|------|-------|---------------|
| 2021 | Real-time Image Classification on Analog Neuromorphic Hardware | Ran image classifier on BrainScaleS-2 chip |
| 2023 | Multi-Single-Chip Training of SNNs with BrainScaleS-2 | Multi-chip SNN training pipeline |
| 2017 | Accelerated Classification in Hierarchical Neural Networks | Classification on neuromorphic hardware |
| 2018 | Solving Map Coloring Problems on Analog Neuromorphic Hardware | Constraint satisfaction on hardware |
| 2019 | Structural Plasticity for Feature Selection in Auditory Stimuli | Plasticity algorithms for audio |
| 2015 | Boltzmann Sampling with Neuromorphic Hardware | Probabilistic computing |
| 2014 | Binaural Sound Localization on Neuromorphic Hardware | Audio processing |
| 2011 | Analysis of the Liquid Computing Paradigm on a Neuromorphic System | Reservoir computing |

**Key observation:** Heidelberg bachelor's theses typically involve ONE focused task: characterising a component, implementing a specific algorithm, building a software tool, or running a specific experiment. They are 40-60 pages.

---

## Part 5: Manchester PhD Theses (For Context Only)

These are NOT representative of undergraduate scope but show what the SpiNNaker group works on. All PDFs are freely downloadable.

| Author | Year | Title | Key Achievement |
|--------|------|-------|-----------------|
| Mollie Ward | 2024 | Modelling Neural Dynamics on Neuromorphic Hardware | HH models on SpiNNaker/SpiNNaker2 |
| Luca Peres | 2022 | Parallelisation of Neural Processing | World's first real-time Cortical Microcircuit |
| Qian Liu | 2018 | Deep Spiking Neural Networks | 99.07% MNIST, Noisy Softplus activation |
| Petrut Bogdan | 2019 | Structural Plasticity on SpiNNaker | STDP + structural plasticity for digit classification |
| Andrew Mundy | 2016 | Real Time Spaun on SpiNNaker | 9000x speedup of 2.5M neuron brain model |
| Gabriel Fonseca Guerra | 2020 | Stochastic Processes for Neuromorphic HW | Cross-platform: SpiNNaker + Loihi |
| Sergio Davies | 2012 | Learning in Spiking Neural Networks | Novel STDP learning rule |
| James Knight | 2016 | Plasticity in Large-Scale Models | Largest plastic SNN on neuromorphic HW |

**All 16 PhD theses:** https://research.manchester.ac.uk/en/studentTheses/ (search "SpiNNaker" or "spiking")

### Other UK PhD Theses Found:

| Author | Year | University | Topic |
|--------|------|-----------|-------|
| N. Perez | 2023 | Imperial College | Sparse backward pass (150x speedup) for deep SNNs |
| Florian Bacho | 2024 | Kent | Exact gradients for temporally-coded SNNs |
| Yin Bi | 2020 | UCL | Graph neural networks for event cameras, created ASL-DVS dataset |
| William Peer Berg | 2022 | Edinburgh | Modular PyTorch framework for SNN optimization |
| Jinqi Huang | 2022 | Southampton | Memristor-based SNNs, NeuroPack simulator, 168 pages |
| Yannan Xing | 2020 | Strathclyde | Deep SNNs for gesture recognition |

---

## Part 6: What Distinguishes a Good Undergraduate SNN Project

Based on analysing 40+ projects:

### Do:
1. Pick a **clear research question** (not just "implement an SNN")
2. Include a **meaningful comparison** (SNN vs ANN on the same task)
3. Use **multiple evaluation metrics** (accuracy, training time, energy estimates, spike count)
4. Be **honest about limitations** (convergence issues, accuracy gaps are expected and normal)
5. Use a **well-documented framework** (snnTorch has the best tutorials for beginners)

### Don't:
1. Try to build from scratch in C++ (unless that IS the project)
2. Attempt multiple complex novel contributions (that's PhD territory)
3. Expect to match ANN accuracy (the gap is a known open problem)
4. Use obscure datasets without good reason

### Typical Deliverables:
- 1-3 Jupyter notebooks or 1-5 Python files
- 40-80 page report including literature review
- Clear experimental results with tables/graphs
- Energy efficiency analysis (even estimated)

---

## Part 7: Framework and Dataset Recommendations

### Frameworks (ranked by student-friendliness):

| Framework | Best For | Tutorials? | Recommendation |
|-----------|----------|-----------|----------------|
| **snnTorch** | PyTorch users, gradient-based training | Excellent (18 Colab notebooks) | **Start here** |
| **Brian2** | Neuroscience-oriented simulation | Good documentation | Good for biological realism |
| **Norse** | Deep learning + SNNs in PyTorch | Moderate | Good alternative to snnTorch |
| **BindsNET** | STDP learning, PyTorch integration | Good examples | Good for unsupervised learning |
| **SpikingJelly** | Full-stack SNN, high performance | Moderate (some Chinese docs) | Best pre-built DVS models |

### Datasets (ranked by suitability):

| Dataset | Type | Difficulty | Notes |
|---------|------|-----------|-------|
| MNIST / Fashion-MNIST | Static images | Easy | Standard baseline, everyone uses it |
| N-MNIST | Neuromorphic events | Medium | Natural fit for SNNs |
| DVS128 Gesture | Neuromorphic events | Medium | 11 gestures, recommended for your project |
| SHD (Heidelberg Spiking Dataset) | Audio spikes | Medium | Spoken digits, native spike format |
| CIFAR-10 | Static images | Medium | Good for ANN comparison |
| Google Speech Commands | Audio | Medium | Good for temporal SNN capabilities |

---

## Part 8: Repositories Where This Research Was Compiled From

### Individual Report Files:
1. `manchester_escholar_thesis_research.md` -- 27 Manchester theses (all PhD+)
2. `snn_thesis_projects_research.md` -- 40+ GitHub projects with detailed analysis
3. `uk_thesis_research_neuromorphic_snn.md` -- Multi-university repository search
4. `spinnaker_apt_student_projects_research.md` -- SpiNNaker/APT group deep dive
5. `figshare_manchester_research_report.md` -- Figshare investigation (no theses found)

### Key External Resources:
- Heidelberg Bachelor's Theses: http://www.kip.uni-heidelberg.de/vision/publications/mscbsc/
- Manchester Research Explorer: https://research.manchester.ac.uk/en/studentTheses/
- Open Neuromorphic Community: https://open-neuromorphic.org/
- snnTorch Tutorials: https://snntorch.readthedocs.io/en/latest/tutorials/index.html
- awesome-snn: https://github.com/coderonion/awesome-snn

---

*Last updated: 2026-02-24*
