# UK University Thesis Repository Research: Neuromorphic Computing & SNNs

searched through 8+ university repositories, the British Library EThOS database, GitHub, and Google Scholar on 24 February 2026, looking for undergraduate and postgraduate theses related to neuromorphic computing and spiking neural networks.

the main takeaway: genuine undergraduate (BSc/BEng) final-year projects on spiking neural networks are really rare in public repositories. UK universities generally don't publicly archive undergrad dissertations -- only PhD theses get routinely deposited. but i did find several confirmed undergraduate projects via GitHub, and the University of Heidelberg (BrainScaleS group) maintains an amazing public archive of 100+ bachelor's and master's theses on neuromorphic hardware. Manchester's SpiNNaker group is the most prolific UK source for PhD-level SNN theses.

repositories i searched:
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

## confirmed undergraduate (BSc/BEng) final year projects

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
- **Scope:** Single-layer SNN for shape classification -- appropriate BSc scope
- confidence that this is legit: HIGH -- explicitly labeled as BSc dissertation

### 2. Musical Pattern Recognition in Spiking Neural Networks
- **Author:** mrahtz (GitHub username)
- **University:** Not specified (likely UK based on "BEng" designation)
- **Year:** ~2016 (repository created April 2016)
- **Degree:** BEng (Final Year Project)
- **Description:** Implemented the first layer of a spiking neural network model capable of recognizing musical patterns in audio input, based on Peter Diehl's unsupervised learning model.
- **Tools:** Brian 2 (neural simulation), Brian 1 with Hears bridge, Python, NumPy, matplotlib, FFmpeg, Mingus music library, Fluid R3 SoundFont
- **Datasets:** Custom .wav test sequences including monophonic audio and a sequence from Yann Tiersen
- **Repository:** https://github.com/mrahtz/musical-pattern-recognition-in-spiking-neural-networks
- **Full text:** Code available; author notes "only a small portion of what was originally intended was actually achieved"
- **Scope:** Partial implementation -- first network layer only. Honest about limitations, which i appreciate.
- confidence: HIGH -- explicitly labeled as BEng final year project

### 3. Randomised Time-Stepping Methods for SNN Simulations
- **Author:** Fabio Deo (Fabio752 on GitHub)
- **University:** Imperial College London
- **Year:** 2021
- **Degree:** Final Year Project (undergraduate -- likely MEng/BEng)
- **Description:** Investigated the effect of introducing various degrees of randomness in spiking neural network time-stepping methods.
- **Tools:** Python (98.9% of repository)
- **Repository:** https://github.com/Fabio752/Randomised-time-stepping-methods-for-SNN-simulations
- **Full text:** Complete thesis PDF included in repository
- **Scope:** Mathematical/computational investigation -- appropriate for Imperial undergraduate
- confidence: HIGH -- profile indicates final year project

---

## Heidelberg University (BrainScaleS) -- bachelor's theses on neuromorphic hardware

this is honestly the richest single source of undergrad-level neuromorphic theses i found anywhere. the KIP (Kirchhoff Institute for Physics) at Heidelberg publicly archives ALL bachelor's and master's theses. these are not UK universities but they're incredibly useful for scope calibration.

full listing: http://www.kip.uni-heidelberg.de/vision/publications/mscbsc/

### selected bachelor's theses (most relevant to SNN/neuromorphic):

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
| 2015 | Investigating Competitive Dynamics in a Recurrent Neural Network on Neuromorphic Hardware | D. Alevi | Network dynamics |
| 2014 | Binaural Sound Localization on Neuromorphic Hardware | L. Kriener | Audio processing |
| 2011 | Analysis of the Liquid Computing Paradigm on a Neuromorphic Hardware System | D. Probst | Reservoir computing |

total bachelor's theses at Heidelberg (neuromorphic): ~60+ spanning 2003-2025
total master's theses: ~50+ spanning 2003-2025

the key observation for scope: Heidelberg bachelor's theses typically involve ONE focused task on BrainScaleS hardware -- characterizing a component, implementing a specific algorithm, building a software tool, or running a specific experiment. they're well-scoped, hardware-oriented, and typically 40-60 pages.

---

## UK PhD theses (for scope calibration)

### University of Manchester (SpiNNaker Group) -- most prolific UK source

| Title | Author | Year | Supervisors | Key Contribution |
|-------|--------|------|-------------|-----------------|
| Deep Spiking Neural Networks | Qian Liu | 2018 | Furber, Lester | Noisy Softplus activation, 99.07% MNIST, spike-based autoencoders |
| Ensemble Learning for Spiking Neural Networks | Alina Neculae | 2020 | Brown, Furber | Ensemble methods for SNNs, class probability interpretation |
| Stochastic Processes For Neuromorphic Hardware | Gabriel Fonseca Guerra | 2020 | Furber, Lester | Ion channel models on SpiNNaker, constraint satisfaction |
| Spikes from Sound: Human Auditory Periphery on SpiNNaker | Robert James | 2019 | Garside, Koch | Cochlear model on SpiNNaker, auditory processing |
| Learning in Spiking Neural Networks | Sergio Davies | 2012 | Furber | STDP learning rule, SpiNNaker routing |
| Parallel Simulation of Neural Networks on SpiNNaker | Andrew Rowley | ~2012 | Furber | STDP with plasticity, SpiNNaker feasibility |
| Parallelisation of Neural Processing on Neuromorphic Hardware | Luca Peres | 2022 | Furber, Rhodes | 20x performance improvement, cortical microcircuit |
| Modelling Neural Dynamics On Neuromorphic Hardware | Mollie Ward | 2024 | Rhodes, Garside | Hodgkin-Huxley on SpiNNaker/SpiNNaker2 |

URL pattern: https://research.manchester.ac.uk/en/studentTheses/[thesis-name]

### Imperial College London

| Title | Author | Year | Key Contribution |
|-------|--------|------|-----------------|
| Robust and efficient training on deep spiking neural networks | N. Perez | 2023 | Sparse backward pass (150x speedup), weight initialization for SNNs |

URL: https://spiral.imperial.ac.uk/entities/publication/0bfc1732-2dac-4589-b50e-7eec99f8efdf

### University of Kent

| Title | Author | Year | Supervisor | Key Contribution |
|-------|--------|------|------------|-----------------|
| Towards Neuromorphic Gradient Descent | Florian Bacho | 2024 | Dominique Chu | Exact gradients for temporally-coded SNNs, local online learning |
| Towards Modelling of Autonomous Neuromorphic Learning | Jakub Fil | 2022 | -- | Autonomous neuromorphic learning systems |

URLs:
- https://kar.kent.ac.uk/104801/
- https://kar.kent.ac.uk/95778/

### UCL

| Title | Author | Year | Key Contribution |
|-------|--------|------|-----------------|
| Bio-mimetic SNNs for unsupervised clustering | Katarzyna Kozdon | 2022 | STDP, structural plasticity, evolutionary learning |
| Graph-based Feature Learning for Neuromorphic Vision | Yin Bi | 2020 | Graph neural networks for event cameras, ASL-DVS dataset |
| Classification of Finger Movements from ECoG using SNN | -- | -- | First SNN-based ECoG decoder |

URLs:
- https://discovery.ucl.ac.uk/id/eprint/10142370/
- https://discovery.ucl.ac.uk/id/eprint/10109453/
- https://discovery.ucl.ac.uk/id/eprint/10213947/

### University of Edinburgh

| Title | Author | Year | Key Contribution |
|-------|--------|------|-----------------|
| Spiking neural network model construction, inference, analysis and applications | William Peer Berg | 2022 | Modular PyTorch framework for SNN optimization |

URL: https://era.ed.ac.uk/handle/1842/43087
