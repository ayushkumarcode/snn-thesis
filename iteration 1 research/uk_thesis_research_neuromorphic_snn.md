# UK University Thesis Repository Research: Neuromorphic Computing & Spiking Neural Networks

**Research conducted: 24 February 2026**
**Focus: Undergraduate (BSc/BEng) and Masters (MSc) theses; PhD theses included for scope calibration**

---

## EXECUTIVE SUMMARY

This investigation systematically searched 8+ university repositories, the British Library EThOS database, GitHub, and Google Scholar for undergraduate and postgraduate theses related to neuromorphic computing, spiking neural networks (SNNs), and adjacent topics.

**Key finding:** Genuine undergraduate (BSc/BEng) final-year projects on spiking neural networks are rare in public repositories. UK universities generally do not publicly archive undergraduate dissertations -- only PhD theses are routinely deposited. However, several confirmed undergraduate projects were found via GitHub, and the University of Heidelberg (BrainScaleS group) maintains an extraordinary public archive of 100+ bachelor's and master's theses on neuromorphic hardware. The University of Manchester (SpiNNaker group) is the most prolific UK source for PhD-level SNN theses.

**Repositories searched:**
- Imperial College London (Spiral)
- University of Edinburgh (ERA)
- UCL Discovery
- University of Southampton (ePrints Soton)
- White Rose eTheses (Leeds/Sheffield/York)
- University of Bristol
- University of Manchester (Research Explorer)
- ETH Zurich (Research Collection)
- University of Heidelberg (KIP)
- University of Kent (KAR)
- University of Strathclyde (STAX)
- University of Glasgow (Enlighten)
- British Library EThOS
- GitHub (multiple searches)
- Google Scholar / general web

---

## CONFIRMED UNDERGRADUATE (BSc/BEng) FINAL YEAR PROJECTS

### 1. Shape Detector Spiking Neural Network
- **Author:** Filippo Ferrari
- **University:** University of Manchester
- **Year:** ~2018 (estimated from repository dates)
- **Degree:** BSc Artificial Intelligence (Final Year Project)
- **Supervisor:** Prof. Steve Furber
- **Description:** Developed a spiking neural network for shape detection, likely running on or related to the SpiNNaker platform.
- **Tools:** Python, SNN simulation (likely sPyNNaker/PyNN)
- **Repository:** https://github.com/filippoferrari/shape_detector_snn and https://github.com/filippoferrari/bsc_dissertation (LaTeX thesis source)
- **Full text:** Thesis source available in LaTeX; compiled PDF likely available
- **Accessibility:** FULL TEXT AVAILABLE (GitHub)
- **Scope:** Single-layer SNN for shape classification -- appropriate BSc scope
- **Confidence:** HIGH -- explicitly labeled as BSc dissertation

### 2. Musical Pattern Recognition in Spiking Neural Networks
- **Author:** mrahtz (GitHub username)
- **University:** Not specified (likely UK based on "BEng" designation)
- **Year:** ~2016 (repository created April 2016)
- **Degree:** BEng (Final Year Project)
- **Supervisor:** Not specified
- **Description:** Implemented the first layer of a spiking neural network model capable of recognizing musical patterns in audio input, based on Peter Diehl's unsupervised learning model.
- **Tools:** Brian 2 (neural simulation), Brian 1 with Hears bridge, Python, NumPy, matplotlib, FFmpeg, Mingus music library, Fluid R3 SoundFont
- **Datasets:** Custom .wav test sequences including monophonic audio and a sequence from Yann Tiersen
- **Repository:** https://github.com/mrahtz/musical-pattern-recognition-in-spiking-neural-networks
- **Full text:** Code available; author notes "only a small portion of what was originally intended was actually achieved"
- **Accessibility:** CODE AVAILABLE (GitHub), report not found online
- **Scope:** Partial implementation -- first network layer only. Honest about limitations.
- **Confidence:** HIGH -- explicitly labeled as BEng final year project

### 3. Randomised Time-Stepping Methods for SNN Simulations
- **Author:** Fabio Deo (Fabio752 on GitHub)
- **University:** Imperial College London
- **Year:** 2021
- **Degree:** Final Year Project (undergraduate -- likely MEng/BEng)
- **Supervisor:** Not specified
- **Description:** Investigated the effect of introducing various degrees of randomness in spiking neural network time-stepping methods. Explored how randomization impacts computational methods used in SNN simulations.
- **Tools:** Python (98.9% of repository)
- **Repository:** https://github.com/Fabio752/Randomised-time-stepping-methods-for-SNN-simulations
- **Full text:** Complete thesis PDF included in repository
- **Accessibility:** FULL TEXT AVAILABLE (GitHub)
- **Scope:** Mathematical/computational investigation -- appropriate for Imperial undergraduate
- **Confidence:** HIGH -- profile indicates final year project

---

## HEIDELBERG UNIVERSITY (BrainScaleS) -- BACHELOR'S THESES ON NEUROMORPHIC HARDWARE

This is the richest single source of undergraduate-level neuromorphic theses found anywhere. The KIP (Kirchhoff Institute for Physics) at Heidelberg publicly archives ALL bachelor's and master's theses. These are NOT UK universities but provide exceptional scope calibration.

**Full listing:** http://www.kip.uni-heidelberg.de/vision/publications/mscbsc/

### Selected Bachelor's Theses (most relevant to SNN/neuromorphic):

| Year | Title | Author | Key Topic |
|------|-------|--------|-----------|
| 2025 | Model-Guided Parametrisation for Flexible Operating Points on BrainScaleS-2 | A.K. Stsepankova | Hardware calibration |
| 2025 | An Intermediate Data Format for In-the-loop Training on Neuromorphic Hardware | B.L. Kroehs | Training pipeline |
| 2025 | Event-based Learning of Synaptic Delays and Arbitrary Topologies | F. Fischer | Learning algorithms |
| 2025 | Gradient-Based Training of Multi-Compartmental Neuron Models on BrainScaleS-2 | T. Janz | Neuron model training |
| 2024 | Placement strategies for multi-compartment neurons on BrainScaleS-2 | D. Baumeister | Hardware mapping |
| 2024 | Enabling Delay Learning in a Scalable ML Framework for Neuromorphic Hardware | L. Tabel | ML framework |
| 2024 | Runtime-dynamic reconfiguration for a time-continuous neuromorphic system | T. Auberer | System reconfiguration |
| 2023 | Multi-Single-Chip Training of SNNs with BrainScaleS-2 | J.V. Straub | Multi-chip SNN training |
| 2023 | Implementation of an FPGA-based memory mapped buffer for real-time communication | R. Heinemann | FPGA interface |
| 2022 | Validation and Automation of the Calibration Routine of the ANANAS board | J. Sawatzki | Hardware calibration |
| 2022 | Hardware Design and Reference Current Generation for Neuromorphic Multi-Chip Systems | M. Stucke | Circuit design |
| 2021 | Migration and Enhancement of the Advanced Lab Course on Neuromorphic Computing | A. Nock | Education/lab |
| 2021 | Real-time Image Classification on Analog Neuromorphic Hardware | F.L. Ebert | Image classification |
| 2020 | Reconstruction of Synaptic Weight on the Neuromorphic BrainScaleS-1 System | M. Wehrheim | Synapse characterization |
| 2020 | PyNN for BrainScaleS-2 | M. Czierlinski | Software interface |
| 2020 | Characterization of silicon neurons on HICANN-X v2 | P. Dauer | Hardware characterization |
| 2019 | Structural Plasticity for Feature Selection in Auditory Stimuli on Neuromorphic Hardware | M. Kreft | Plasticity algorithms |
| 2019 | Towards Balanced Random Networks on the BrainScaleS I System | Q. Schwarzenboeck | Network dynamics |
| 2018 | Solving Map Coloring Problems on Analog Neuromorphic Hardware | J. Steidel | Constraint satisfaction |
| 2018 | Characterization and Calibration of Synaptic Plasticity on Neuromorphic Hardware | J. Weis | STDP characterization |
| 2018 | Towards Spike-based Expectation Maximization on Neuromorphic Chip | F. Schneider | Probabilistic computing |
| 2017 | Accelerated Classification in Hierarchical Neural Networks on Neuromorphic Hardware | C. Fischer | Classification |
| 2016 | Towards Fast Iterative Learning on the BrainScaleS System | L. Pilz | Learning algorithms |
| 2015 | Boltzmann Sampling with Neuromorphic Hardware | D. Stockel | Sampling |
