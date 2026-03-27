# SNN Thesis & Student Projects on GitHub

looked through github on 2026-02-24 to find undergraduate and masters thesis/dissertation projects related to SNNs, neuromorphic computing, etc. found 40+ student and research projects.

the overall pattern for undergrad projects is pretty clear: most BSc/3rd-year projects focus on a single well-defined task (usually MNIST or gesture classification), use one established framework (snnTorch, Brian2, SpikingJelly, or BindsNET), compare SNN performance against a conventional ANN baseline, and work with 1-2 datasets. masters-level stuff tends to be more ambitious -- hardware deployment (SpiNNaker, Loihi, FPGA), multi-domain applications (robotics, RL), or novel architectures. worth noting that few projects include actual thesis PDFs in the repo, though several reference external documents.

---

## confirmed undergraduate / BSc / final year projects

these are repos explicitly identified as undergraduate or final-year projects.

### 1.1 Shape Detector SNN (University of Manchester)
- **URL**: https://github.com/filippoferrari/shape_detector_snn
- **Dissertation repo**: https://github.com/filippoferrari/bsc_dissertation
- **Description**: Shape detection using spiking neural networks, BSc AI dissertation supervised by Prof. Steve Furber (creator of SpiNNaker)
- **Framework**: Python, pyDVS library
- **Dataset**: Custom shape images processed through DVS simulation
- **Results**: Not quantified in README
- **Thesis PDF**: LaTeX source in dissertation repo (may need compilation)
- **Complexity**: MODERATE - well-structured with tests, CI, configs
- **Stars**: 2 | **Last updated**: Archived 2024 | **Commits**: 107
- good example of a BSc project at a top university. clean code structure, proper testing, supervised by a world expert. feels like a solid reference for what a strong undergrad project looks like.

### 1.2 Musical Pattern Recognition in SNNs (BEng Final Year)
- **URL**: https://github.com/mrahtz/musical-pattern-recognition-in-spiking-neural-networks
- **Description**: Spiking neural network that differentiates musical notes from audio sequences
- **Framework**: Brian 2 (neural simulator), Brian 1 Hears bridge, NumPy, matplotlib, Mingus music library
