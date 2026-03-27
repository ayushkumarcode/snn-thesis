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

