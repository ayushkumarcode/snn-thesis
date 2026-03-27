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
