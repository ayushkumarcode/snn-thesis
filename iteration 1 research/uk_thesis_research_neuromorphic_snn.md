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
