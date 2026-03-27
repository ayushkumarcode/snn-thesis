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
- **Dataset**: Custom monophonic audio sequences (.wav)
- **Results**: Successfully differentiated notes; visualizations of spike patterns, membrane potentials, synaptic weight evolution
- **Thesis PDF**: YES - available at http://amid.fish/beng_project_report.pdf
- **Complexity**: MODERATE - integrates neuromorphic simulation, audio processing, STDP
- **Stars**: 49 | **Forks**: 17 | **Last updated**: Archived
- this one's really good. novel application domain (music), strong documentation, thesis PDF available. the high star count shows people are interested. good model for an ambitious but achievable BEng project.

### 1.3 Spiking Neural Network for Digit Recognition (King's College London BSc)
- **URL**: https://github.com/LucaMozzo/SpikingNeuralNetwork
- **Description**: Efficient C++ implementation of a stochastic SNN for handwritten digit recognition
- **Framework**: C/C++ (custom implementation), OpenCV3, SQLite, Visual Studio 2017
- **Dataset**: MNIST
- **Results**: Multiple decoding methods explored (rate decoding, first-to-spike). Original implementation required 2h37m per epoch, heavily optimized.
- **Thesis PDF**: Not in repo but referenced as 2018 BSc thesis
- **Complexity**: HIGH for an undergrad - built from scratch in C++, no framework used
- **Stars**: 11 | **Forks**: 3 | **Commits**: 45
- pretty impressive that they built everything from scratch in C++. most students use existing frameworks. the optimization challenge (2h37m -> much faster) shows real engineering effort.

### 1.4 SNN for Autonomous Locomotion Control (Bachelor Thesis)
- **URL**: https://github.com/romenr/bachelorthesis
- **Description**: Using SNNs to control autonomous locomotion of a mobile robot following a red object
- **Framework**: V-REP (robot simulation), ROS, Python, R-STDP learning
- **Dataset**: N/A (simulation-based)
- **Results**: INCOMPLETE - "network weights do not seem to converge"
- **Thesis PDF**: TeX source in repo (thesis folder), presentation slides included
- **Complexity**: MODERATE-HIGH - robotics simulation + SNN + RL
- **Stars**: 2 | **Forks**: 1 | **Commits**: 191
- honest about failing to converge, which is actually quite realistic for a bachelor thesis. the project was ambitious (robotics + SNN + learning), and partial results are normal for this scope. good reminder that not all thesis projects succeed fully.

### 1.5 Spiking Stereo Matching (BSc Thesis)
- **URL**: https://github.com/gdikov/SpikingStereoMatching
- **Description**: SNN for real-time event-based stereo matching using dynamic vision sensors, deployed on SpiNNaker
- **Framework**: sPyNNaker toolchain, PyNN, SpiNNaker hardware
- **Dataset**: Custom event-based vision sensor data
- **Results**: 2ms latency with neuromorphic hardware (published in Biomimetic and Biohybrid Systems, 2017)
- **Thesis PDF**: Not in repo, but published paper exists
- **Complexity**: ADVANCED - neuromorphic hardware, event-based vision, SpiNNaker
- **Stars**: 2 | **Forks**: 2 | **Commits**: 182
- very advanced for a BSc thesis. they actually got a published paper out of it. access to SpiNNaker hardware was likely through Manchester's facilities.

### 1.6 QuadBot-NeuroMorphic (Cambridge Undergraduate Research)
- **URL**: https://github.com/Cambridge-Control-Lab/QuadBot-NeuroMorphic
- **Description**: Neuromorphic control of quadrupedal robot locomotion using spiking neural circuits (CPGs)
- **Framework**: MATLAB/Simulink R2022b+, SOLIDWORKS, VEX V5 Robot Brain, Python
- **Dataset**: N/A (physical robot experiments)
- **Results**: Successfully implemented biologically-inspired gaits on physical robots (NeuroPup and Synapider)
- **Thesis PDF**: No formal thesis, but good documentation
- **Complexity**: ADVANCED - neuroscience + control systems + embedded hardware
- **Stars**: 5 | **Forks**: 2 | **Commits**: 93 | **Last updated**: Sept 2023
- this was actually a 10-week summer research project (not a thesis per se). funded by MathWorks. well-documented. shows what's possible with good supervision and resources at a top university.

### 1.7 SNN Accelerator Hardware (EE552 Class Project)
- **URL**: https://github.com/zwhexplorer/Spiking-Neural-Network-Accelerator-EE552-project
- **Description**: Hardware accelerator for SNNs inspired by TrueNorth and Loihi, 3x3 mesh network
- **Framework**: SystemVerilog (100%), asynchronous hardware design
- **Dataset**: N/A (hardware design)
- **Results**: Working 9-router mesh topology with XY routing
- **Report**: Final report PDF included
- **Complexity**: HIGH - hardware design, NoC architecture, asynchronous circuits
- **Stars**: 15 | **Forks**: 4 | **Commits**: 64
- graduate-level course project (EE552). very hardware-focused. not typical for a software-oriented thesis but interesting to see the hardware side of neuromorphic computing.

### 1.8 Neuromorphic NoC Architecture for SNNs (4th Year Project)
- **URL**: https://github.com/cepdnaclk/e18-4yp-Neuromorphic-NoC-Architecture-for-SNNs
- **Description**: Scalable Network-on-Chip architecture based on RISC-V ISA for SNN processing on FPGA
- **Framework**: Verilog (99.7%), FPGA implementation
- **Dataset**: N/A (hardware design)
- **Results**: Working FPGA implementation with custom accelerators
- **Complexity**: VERY HIGH - custom hardware, ISA extensions, FPGA
- **Stars**: 7 | **Forks**: 5 | **Commits**: 207 | **Last updated**: June 2024
- 4th-year engineering project (likely undergrad final year in a 4-year program). team of 3 students. very ambitious hardware project.

---

## confirmed masters thesis projects

### 2.1 Volr DSL - Modelling Learning Systems (Masters)
- **URL**: https://github.com/Jegp/thesis
- **Description**: Domain-specific language (Volr) enabling unified modelling of ANNs and SNNs
- **Framework**: Haskell (DSL), Futhark+OpenCL (ANN backend), NEST+PyNN (SNN backend), BrainScaleS
- **Dataset**: NAND, XOR, MNIST
- **Results**: Demonstrated topology-preserving translation between ANN and SNN paradigms
- **Thesis PDF**: YES - report/report.pdf included
- **Complexity**: VERY HIGH - multiple frameworks, Haskell DSL, neuromorphic hardware
- **Stars**: 4 | **Commits**: 99

### 2.2 Recurrent SNNs for POMDPs (Masters)
- **URL**: https://github.com/Quickblink/rsnn
- **Description**: Recurrent Spiking Neural Networks for Partially Observable Markov Decision Processes
- **Framework**: PyTorch, Docker, multiple neuron architectures (LIF, Adaptive, etc.)
- **Dataset**: Sequential MNIST, encoded MNIST variants
- **Results**: Not explicitly documented
- **Thesis PDF**: Not in repo
- **Complexity**: ADVANCED
- **Stars**: 3 | **Commits**: 185+

### 2.3 SNNs for Reinforcement Learning Tasks (Masters by Research, UTS)
- **URL**: https://github.com/andrewrafeUTS/SNNTechnicalAppendix
- **Description**: Evolutionary experiments with SNNs for CartPole and LunarLander RL tasks
- **Framework**: Python 3.8, matplotlib, numpy, gym (custom SNN implementation)
- **Dataset**: CartPole, LunarLander environments
- **Results**: Tested multiple decoding methods (f2f, rate, etc.)
- **Thesis PDF**: Not in repo
- **Complexity**: MODERATE-ADVANCED
- **Stars**: 1 | **Commits**: 5

### 2.4 Deep Spiking Q-Networks (TUM Masters)
- **URL**: https://github.com/vhris/Deep-Spiking-Q-Networks
- **Description**: Spiking DQN training using conversion and surrogate gradients for RL tasks
- **Framework**: SpyTorch, NEST 2.16.0, PyNN, OpenAI Gym
- **Dataset**: CartPole, MountainCar, Breakout (OpenAI Gym)
- **Results**: Both conversion and direct training methods succeeded on CartPole
- **Thesis PDF**: YES - included in repository
- **Complexity**: ADVANCED
- **Stars**: 11 | **Forks**: 3 | **Last updated**: Feb 2021

### 2.5 SNN for Hand Kinematics from sEMG (Masters)
- **URL**: https://github.com/davidkubanek/SNN-hand-kinematics-estimation-from-sEMG-signals
- **Description**: Neuromorphic reservoir network for estimating hand movements from muscle signals
- **Framework**: Brian2, Python, C++/Cython
- **Dataset**: NinaPro public EMG database
- **Results**: Not explicitly documented
- **Thesis PDF**: Not in repo
- **Complexity**: HIGH
- **Stars**: 4 | **Commits**: 30

### 2.6 Spiking Grid Cell Models on Neuromorphic Hardware (MSc, Manchester)
- **URL**: https://github.com/nickybu/spiking_grid_cell_model
- **Description**: Spiking grid cell models on SpiNNaker, supervised by Prof. Furber
- **Framework**: SpiNNaker, sPyNNaker, Python 2.7, Brian2
- **Dataset**: N/A (computational neuroscience simulation)
- **Results**: Successfully implemented grid cell models on neuromorphic hardware
- **Thesis PDF**: YES - linked in repository
- **Complexity**: ADVANCED
- **Stars**: 0 | **Commits**: 9 | **Last updated**: 2019

### 2.7 Brain-Machine Interface using SpiNNaker (Masters)
- **URL**: https://github.com/solversa/Master-Thesis-Brain-Machine-Interface
- **Description**: Decoding 3D imaginary reach/grasp movements from EEG using SNNs on SpiNNaker
- **Framework**: SpiNNaker (4 chips, 64 cores), STDP with reward-based training, Python
- **Dataset**: Motor imagery EEG data
- **Results**: 73.4% mean classification accuracy (only 4.12% below state-of-art ML)
- **Thesis PDF**: YES - included in repository
- **Complexity**: VERY HIGH
- **Stars**: 2 | **Forks**: 5 | **Commits**: 102

### 2.8 Spiking Deep Belief Network (Masters)
- **URL**: https://github.com/MazdakFatahi/Spiking-Deep-Belief-Network
- **Description**: Spike-Based Deep Belief Network with LIF neurons using contrastive divergence
- **Framework**: Custom implementation, LIF neurons, rate-based CD
- **Dataset**: MNIST
- **Results**: 94.9% accuracy on MNIST
- **Thesis PDF**: YES - "MazdakFatahi(Ms Thesis).pdf"
- **Complexity**: HIGH
- **Stars**: 1 | **Commits**: 3

### 2.9 SNN-RL: Training SNNs with Reinforcement Learning (Masters)
- **URL**: https://github.com/BSVogler/SNN-RL
- **Description**: Actor-critic RL framework with spiking neural network actors using R-STDP
- **Framework**: NEST 3, Python 3.7/3.8, Docker, MongoDB
- **Dataset**: Line-following task environments
- **Results**: Successful line-following behavior
- **Thesis PDF**: YES - Thesis.pdf in repository
- **Complexity**: HIGH
- **Stars**: 21 | **Forks**: 3

### 2.10 Event-Based End-to-End Robot Control (TUM Masters)
- **URL**: https://github.com/clamesc/Training-Neural-Networks-for-Event-Based-End-to-End-Robot-Control
- **Description**: Robot steering with DVS event camera using DRL and SNNs for lane-keeping
- **Framework**: TensorFlow, V-REP, ROS, NEST 2.10.0, Python 2.7
- **Dataset**: Simulated lane-following task with DVS
- **Results**: Both DQN-SNN and R-STDP methods succeeded at lane following
- **Thesis PDF**: YES - full thesis PDF included
- **Complexity**: VERY HIGH
- **Stars**: 59 | **Forks**: 23
- most popular thesis project i found. excellent documentation. combines DVS, robotics, and SNNs.

### 2.11 CartPole with SNNs inspired by Theory of Mind (Masters)
- **URL**: https://github.com/atenagm1375/cartpole
- **Description**: SNN-based CartPole control inspired by Theory of Mind concepts
- **Framework**: PyTorch, BindsNet, OpenAI Gym
- **Dataset**: CartPole, River Raid environments
- **Complexity**: MODERATE-ADVANCED
- **Stars**: 1 | **Commits**: 77

### 2.12 GANs for Spiking Time Series (Masters, UvA)
- **URL**: https://github.com/HitLuca/GANs_for_spiking_time_series
- **Description**: Generating spiking time series patterns using GANs (Master in AI, University of Amsterdam)
- **Framework**: Not detailed
- **Dataset**: Spiking time series data
- **Complexity**: HIGH
- interesting intersection of GANs and spiking data

### 2.13 Use of Spiking Neural Networks (Thesis)
- **URL**: https://github.com/honzikv/use-of-snn
- **Description**: Three experiments - EEG classification, P300 detection, surrogate gradient MNIST/Fashion-MNIST
- **Framework**: PyTorch, TensorFlow, Jupyter
- **Dataset**: BNCI Horizon 2020 EEG, Harvard P300, MNIST, Fashion MNIST
- **Results**: Successfully converted CNN to SNN for EEG; surrogate gradient training on image datasets
- **Thesis PDF**: Not in repo (title: "Use of Spiking Neural Networks")
- **Complexity**: MODERATE-ADVANCED (three separate experiments)
- **Stars**: 2 | **Commits**: 157

### 2.14 SNN Formation Control for Multi-Agent Systems
- **URL**: https://github.com/ViktorNfa/SpikingNeuralNet_FormationControl
- **Description**: SNN using Norse framework to learn formation control with collision avoidance
- **Framework**: Norse, PyTorch
- **Results**: SNN learned formation control comparable to classical controllers
- **Complexity**: MODERATE-HIGH
- **Stars**: 4 | **Commits**: 18 | **Last updated**: July 2024

### 2.15 Design Space Exploration of Associative Memories (Masters, Bielefeld)
- **URL**: https://github.com/astoeckel/master-thesis-astoeckel-2015
- **Description**: Willshaw model for associative memories using spiking neurons targeting neuromorphic hardware
- **Framework**: LaTeX, Python, MATLAB, C++
- **Thesis PDF**: YES - downloadable v1.2 from GitHub releases (CC BY-ND 4.0)
- **Complexity**: HIGH
- **Stars**: 2

---

## student course projects / research group projects

### 3.1 Simple SNN with STDP (University of Osnabruck Course Project)
