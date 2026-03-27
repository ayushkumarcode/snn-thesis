# SNN Thesis & Student Projects on GitHub

Went through GitHub looking for undergraduate and masters thesis/dissertation projects related to SNNs and neuromorphic computing. Found 40+ projects.

The pattern that emerges is pretty clear: most BSc/3rd-year projects focus on a single well-defined task (usually MNIST or gesture classification), use one established framework (snnTorch, Brian2, SpikingJelly, or BindsNET), compare SNN performance against a conventional ANN baseline, and work with 1-2 datasets. Masters-level projects tend to be more ambitious -- hardware deployment (SpiNNaker, Loihi, FPGA), multi-domain applications (robotics, RL), or novel architectural stuff. Not many projects include formal thesis PDFs in the repo, but several reference external documents.

---

## Confirmed Undergraduate / BSc / Final Year Projects

### Shape Detector SNN (University of Manchester)
- https://github.com/filippoferrari/shape_detector_snn
- Dissertation repo: https://github.com/filippoferrari/bsc_dissertation
- Shape detection using SNNs, BSc AI dissertation supervised by **Steve Furber** (creator of SpiNNaker)
- Uses Python, pyDVS library, custom shape images through DVS simulation
- Results not quantified in README. LaTeX source in dissertation repo.
- Well-structured with tests, CI, configs. 2 stars, 107 commits, archived 2024.
- Good example of a BSc project at a top university. Clean code, proper testing, supervised by a world expert.

### Musical Pattern Recognition in SNNs (BEng Final Year)
- https://github.com/mrahtz/musical-pattern-recognition-in-spiking-neural-networks
- SNN that differentiates musical notes from audio sequences
- Brian 2, Brian 1 Hears bridge, NumPy, matplotlib, Mingus music library
- Custom monophonic audio (.wav). Successfully differentiated notes with visualizations of spike patterns, membrane potentials, synaptic weight evolution.
- **Thesis PDF available:** http://amid.fish/beng_project_report.pdf
- 49 stars, 17 forks. That's a lot for a student project -- clearly struck a chord (pun intended). Novel application domain, strong documentation. Good model for an ambitious but achievable BEng project.

### SNN for Digit Recognition (King's College London BSc)
- https://github.com/LucaMozzo/SpikingNeuralNetwork
- Efficient C++ implementation of a stochastic SNN for handwritten digit recognition
- C/C++ (custom, no framework), OpenCV3, SQLite, Visual Studio 2017. MNIST.
- Multiple decoding methods explored (rate, first-to-spike). Original implementation needed 2h37m per epoch, then heavily optimized.
- Impressively low-level for an undergrad. Building from scratch in C++ is unusual -- most students use existing frameworks. The optimization challenge shows real engineering effort.
- 11 stars, 3 forks, 45 commits

### SNN for Autonomous Locomotion Control (Bachelor Thesis)
- https://github.com/romenr/bachelorthesis
- Using SNNs to control a mobile robot following a red object
- V-REP (robot sim), ROS, Python, R-STDP learning
- **Honest about failure:** "network weights do not seem to converge"
- TeX source and presentation slides in repo. 2 stars, 191 commits.
- Realistic bachelor thesis outcome. The project was ambitious (robotics + SNN + learning), and partial results are normal for this scope. Not every thesis succeeds fully and that's ok.

### Spiking Stereo Matching (BSc Thesis)
- https://github.com/gdikov/SpikingStereoMatching
- Real-time event-based stereo matching using DVS, deployed on SpiNNaker
- sPyNNaker toolchain, PyNN, SpiNNaker hardware
- 2ms latency with neuromorphic hardware (published in Biomimetic and Biohybrid Systems, 2017)
- Very advanced for a BSc thesis. Published paper from the work. Probably had SpiNNaker access through Manchester's facilities.
- 2 stars, 2 forks, 182 commits

### QuadBot-NeuroMorphic (Cambridge Undergraduate Research)
- https://github.com/Cambridge-Control-Lab/QuadBot-NeuroMorphic
- Neuromorphic control of quadrupedal robot locomotion using spiking neural circuits (CPGs)
- MATLAB/Simulink R2022b+, SOLIDWORKS, VEX V5 Robot Brain, Python
- Successfully implemented biologically-inspired gaits on physical robots (NeuroPup and Synapider)
- 10-week summer research project (not a thesis per se). Funded by MathWorks. Very well-documented. Shows what's possible with good supervision and resources at a top university.
- 5 stars, 2 forks, 93 commits

### SNN Accelerator Hardware (EE552 Class Project)
- https://github.com/zwhexplorer/Spiking-Neural-Network-Accelerator-EE552-project
- Hardware accelerator for SNNs inspired by TrueNorth and Loihi, 3x3 mesh network
- 100% SystemVerilog, asynchronous hardware design. Working 9-router mesh topology with XY routing.
- Final report PDF included.
- Graduate-level course project. Very hardware-focused, not typical for a software-oriented thesis.
- 15 stars, 4 forks, 64 commits

### Neuromorphic NoC Architecture for SNNs (4th Year Project)
- https://github.com/cepdnaclk/e18-4yp-Neuromorphic-NoC-Architecture-for-SNNs
- Scalable Network-on-Chip architecture based on RISC-V ISA for SNN processing on FPGA
- 99.7% Verilog, working FPGA implementation with custom accelerators
- Team of 3 students. Very ambitious hardware project.
- 7 stars, 5 forks, 207 commits

---

## Confirmed Masters Thesis Projects

### Volr DSL - Modelling Learning Systems
- https://github.com/Jegp/thesis
- Domain-specific language (Volr) enabling unified modelling of ANNs and SNNs
- Haskell (DSL), Futhark+OpenCL (ANN backend), NEST+PyNN (SNN backend), BrainScaleS
- NAND, XOR, MNIST. Demonstrated topology-preserving translation between paradigms.
- **Thesis PDF: report/report.pdf in repo**
