# UK University Thesis Search: Neuromorphic Computing & SNNs

Searched 8+ university repositories, the British Library EThOS database, GitHub, and Google Scholar for undergraduate and postgraduate theses on neuromorphic computing, SNNs, and adjacent topics.

The main takeaway: genuine undergraduate (BSc/BEng) final-year projects on spiking neural networks are really rare in public repositories. UK universities generally don't publicly archive undergraduate dissertations -- only PhD theses get routinely deposited. But i found several confirmed undergrad projects via GitHub, and the University of Heidelberg (BrainScaleS group) has an amazing public archive of 100+ bachelor's and master's theses on neuromorphic hardware. Manchester's SpiNNaker group is the most prolific UK source for PhD-level SNN theses.

Repositories searched:
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

## Confirmed Undergraduate (BSc/BEng) Final Year Projects

### 1. Shape Detector Spiking Neural Network
- **Filippo Ferrari**, University of Manchester, ~2018
- BSc Artificial Intelligence Final Year Project, supervised by Steve Furber
- SNN for shape detection, likely related to SpiNNaker
- Python, SNN simulation (likely sPyNNaker/PyNN)
- Code: https://github.com/filippoferrari/shape_detector_snn
- Thesis LaTeX: https://github.com/filippoferrari/bsc_dissertation
- Single-layer SNN for shape classification -- appropriate BSc scope
- Explicitly labeled as BSc dissertation

### 2. Musical Pattern Recognition in SNNs
- **mrahtz** (GitHub username), likely UK based on "BEng" designation, ~2016
- BEng Final Year Project
- First layer of an SNN model for recognizing musical patterns in audio input, based on Peter Diehl's unsupervised learning model
- Brian 2, Brian 1 Hears bridge, Python, NumPy, matplotlib, FFmpeg, Mingus music library
- Custom .wav test sequences including monophonic audio and Yann Tiersen
- Code: https://github.com/mrahtz/musical-pattern-recognition-in-spiking-neural-networks
- Author notes "only a small portion of what was originally intended was actually achieved" -- honest about limitations, which is good

### 3. Randomised Time-Stepping Methods for SNN Simulations
- **Fabio Deo** (Fabio752 on GitHub), Imperial College London, 2021
- Final Year Project (likely MEng/BEng)
- Investigated introducing various degrees of randomness in SNN time-stepping methods
- Python (98.9% of repo)
- Code + **full thesis PDF** included: https://github.com/Fabio752/Randomised-time-stepping-methods-for-SNN-simulations
- Mathematical/computational investigation -- appropriate for Imperial

---

## Heidelberg University (BrainScaleS) -- Bachelor's Theses

This is the richest single source of undergrad-level neuromorphic theses i found anywhere. The KIP (Kirchhoff Institute for Physics) publicly archives ALL bachelor's and master's theses. Not UK, but amazing for scope calibration.

Full listing: http://www.kip.uni-heidelberg.de/vision/publications/mscbsc/

Selected bachelor's theses (most relevant to SNN/neuromorphic):

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

Total bachelor's theses at Heidelberg (neuromorphic): ~60+ spanning 2003-2025. Master's theses: ~50+.

The pattern for Heidelberg bachelor's theses: typically ONE focused task on BrainScaleS hardware -- characterizing a component, implementing a specific algorithm, building a software tool, or running one specific experiment. Well-scoped, hardware-oriented, typically 40-60 pages.

---

## UK PhD Theses (for scope calibration)

### University of Manchester (SpiNNaker Group) -- most prolific UK source

| Title | Author | Year | Supervisors | Key Contribution |
|-------|--------|------|-------------|-----------------|
| Deep Spiking Neural Networks | Qian Liu | 2018 | Furber, Lester | Noisy Softplus activation, 99.07% MNIST |
| Ensemble Learning for SNNs | Alina Neculae | 2020 | Brown, Furber | Ensemble methods for SNNs |
| Stochastic Processes For Neuromorphic HW | Gabriel Fonseca Guerra | 2020 | Furber, Lester | Ion channel models on SpiNNaker |
| Spikes from Sound: Auditory Periphery on SpiNNaker | Robert James | 2019 | Garside, Koch | Cochlear model on SpiNNaker |
| Learning in SNNs | Sergio Davies | 2012 | Furber | STDP learning rule, SpiNNaker routing |
| Parallel Simulation of NNs on SpiNNaker | Andrew Rowley | ~2012 | Furber | STDP with plasticity |
| Parallelisation on Neuromorphic HW | Luca Peres | 2022 | Furber, Rhodes | 20x performance improvement |
| Modelling Neural Dynamics on Neuromorphic HW | Mollie Ward | 2024 | Rhodes, Garside | HH on SpiNNaker/SpiNNaker2 |

URL pattern: https://research.manchester.ac.uk/en/studentTheses/[thesis-name]

### Imperial College London
- N. Perez, 2023 -- "Robust and efficient training on deep spiking neural networks" -- sparse backward pass (150x speedup), weight initialization for SNNs
- https://spiral.imperial.ac.uk/entities/publication/0bfc1732-2dac-4589-b50e-7eec99f8efdf

### University of Kent
- Florian Bacho, 2024 -- "Towards Neuromorphic Gradient Descent" -- exact gradients for temporally-coded SNNs, local online learning
- Jakub Fil, 2022 -- "Towards Modelling of Autonomous Neuromorphic Learning"
- https://kar.kent.ac.uk/104801/ and https://kar.kent.ac.uk/95778/

### UCL
- Katarzyna Kozdon, 2022 -- "Bio-mimetic SNNs for unsupervised clustering" -- STDP, structural plasticity, evolutionary learning
- Yin Bi, 2020 -- "Graph-based Feature Learning for Neuromorphic Vision" -- graph NNs for event cameras, ASL-DVS dataset
- "Classification of Finger Movements from ECoG using SNN" -- first SNN-based ECoG decoder
- https://discovery.ucl.ac.uk/id/eprint/10142370/ and https://discovery.ucl.ac.uk/id/eprint/10109453/

### University of Edinburgh
- William Peer Berg, 2022 -- "Spiking neural network model construction, inference, analysis and applications" -- modular PyTorch framework for SNN optimization
- https://era.ed.ac.uk/handle/1842/43087

### University of Southampton
- Jinqi Huang, 2022 (168 pages) -- "Memristor-based Spiking Neural Networks" -- FPGA interface, NeuroPack simulator, ANN-to-SNN conversion
- Kier Dugan, 2016 -- "Non-neural computing on the SpiNNaker" -- non-neural apps on neuromorphic hardware
- Patrick Foster, 2023 -- "Neural spike classification acceleration with RRAM"
- https://eprints.soton.ac.uk/471765/ and https://eprints.soton.ac.uk/400083/

### University of Strathclyde
- Yannan Xing, 2020 -- "Deep spiking neural networks with applications to human gesture recognition" -- SCRNN for gesture recognition, speech emotion recognition
- https://stax.strath.ac.uk/concern/theses/x059c7885

### University of Glasgow
- Jude Haris, 2025 -- "Hardware-software co-design of FPGA-based neural network accelerators for edge inference" -- SECDA methodology, 84x speedup on TConv
- https://theses.gla.ac.uk/85185/

### University of Sheffield
- Dorian Florescu, 2016 -- "Reconstruction, identification and implementation methods for spiking neural circuits"
- Carlos Fernandez Musoles, 2020 -- "Improving scalability of large-scale distributed SNN simulations on HPC" -- HyperPRAW partitioning algorithm
- https://etheses.whiterose.ac.uk/12625/ and https://etheses.whiterose.ac.uk/id/eprint/29007/

---

## White Rose eTheses (Leeds/Sheffield/York)

13 theses found matching "neuromorphic spiking neural network":

| Title | Author | Year | Degree | University |
|-------|--------|------|--------|------------|
| Bio-inspired reinforcement learning | Zhile Yang | 2025 | PhD | Leeds |
| Energy-efficient Tracking of Mobile Audio Sources | Dimitrios Pappas | 2025 | PhD | Sheffield |
| Qubit chains' emergent behaviour from biologically-inspired dynamics | Xavier Laurent | 2025 | MSc by research | York |
| System Analysis and Design of Physical Delay-Feedback Reservoir Computing | Alexander McDonnell | 2024 | PhD | York |
| Scaling up ESNs with Heterogeneous Reservoirs | Chester Wringe | 2024 | PhD | York |
