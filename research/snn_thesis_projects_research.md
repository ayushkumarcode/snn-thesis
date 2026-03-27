# Spiking Neural Network Thesis & Student Projects: GitHub Research Report

**Date**: 2026-02-24
**Researcher**: Deep Research Investigation
**Scope**: GitHub repositories of undergraduate and masters thesis/dissertation projects related to SNNs, neuromorphic computing, and adjacent topics

---

## Executive Summary

This report catalogs **40+ student and research projects** found on GitHub related to spiking neural networks (SNNs) and neuromorphic computing. The investigation covered direct thesis repositories, course projects, research implementations, curated awesome-lists, and framework-specific project ecosystems. The findings reveal a clear pattern for typical undergraduate project scope: most BSc/3rd-year projects focus on a single well-defined task (usually MNIST or gesture classification), use one established framework (snnTorch, Brian2, SpikingJelly, or BindsNET), compare SNN performance against a conventional ANN baseline, and produce results within 1-2 datasets. Masters-level projects tend to be more ambitious, often involving hardware deployment (SpiNNaker, Loihi, FPGA), multi-domain applications (robotics, RL), or novel architectural contributions. Notably, few projects include formal thesis PDFs in the repository, though several reference external documents.

---

## Section 1: Confirmed Undergraduate / BSc / Final Year Projects

These are repositories explicitly identified as undergraduate or final-year projects.

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
- **Assessment**: Good example of a BSc project at a top university. Clean code structure, proper testing, supervised by a world expert. Representative of what a strong undergraduate project looks like.

### 1.2 Musical Pattern Recognition in SNNs (BEng Final Year)
- **URL**: https://github.com/mrahtz/musical-pattern-recognition-in-spiking-neural-networks
- **Description**: Spiking neural network that differentiates musical notes from audio sequences
- **Framework**: Brian 2 (neural simulator), Brian 1 Hears bridge, NumPy, matplotlib, Mingus music library
- **Dataset**: Custom monophonic audio sequences (.wav)
- **Results**: Successfully differentiated notes; visualizations of spike patterns, membrane potentials, synaptic weight evolution
- **Thesis PDF**: YES - available at http://amid.fish/beng_project_report.pdf
- **Complexity**: MODERATE - integrates neuromorphic simulation, audio processing, STDP
- **Stars**: 49 | **Forks**: 17 | **Last updated**: Archived
- **Assessment**: Excellent undergraduate project. Novel application domain (music), strong documentation, thesis PDF available. High star count indicates community interest. Good model for an ambitious but achievable BEng project.

### 1.3 Spiking Neural Network for Digit Recognition (King's College London BSc)
- **URL**: https://github.com/LucaMozzo/SpikingNeuralNetwork
- **Description**: Efficient C++ implementation of a stochastic SNN for handwritten digit recognition
- **Framework**: C/C++ (custom implementation), OpenCV3, SQLite, Visual Studio 2017
- **Dataset**: MNIST
- **Results**: Multiple decoding methods explored (rate decoding, first-to-spike). Original implementation required 2h37m per epoch, heavily optimized.
- **Thesis PDF**: Not in repo but referenced as 2018 BSc thesis
- **Complexity**: HIGH for an undergraduate - built from scratch in C++, no framework used
- **Stars**: 11 | **Forks**: 3 | **Commits**: 45
- **Assessment**: Impressive low-level implementation. Building from scratch in C++ is unusually ambitious for a BSc project. Most students use existing frameworks. The optimization challenge (2h37m -> much faster) shows real engineering effort.

### 1.4 SNN for Autonomous Locomotion Control (Bachelor Thesis)
- **URL**: https://github.com/romenr/bachelorthesis
- **Description**: Using SNNs to control autonomous locomotion of a mobile robot following a red object
- **Framework**: V-REP (robot simulation), ROS, Python, R-STDP learning
- **Dataset**: N/A (simulation-based)
- **Results**: INCOMPLETE - "network weights do not seem to converge"
- **Thesis PDF**: TeX source in repo (thesis folder), presentation slides included
- **Complexity**: MODERATE-HIGH - robotics simulation + SNN + RL
- **Stars**: 2 | **Forks**: 1 | **Commits**: 191
- **Assessment**: Honest about failure to converge - this is realistic for a bachelor thesis. The project was ambitious (robotics + SNN + learning), and partial results are normal for this scope. Good example that not all thesis projects succeed fully.

### 1.5 Spiking Stereo Matching (BSc Thesis)
- **URL**: https://github.com/gdikov/SpikingStereoMatching
- **Description**: SNN for real-time event-based stereo matching using dynamic vision sensors, deployed on SpiNNaker
- **Framework**: sPyNNaker toolchain, PyNN, SpiNNaker hardware
- **Dataset**: Custom event-based vision sensor data
- **Results**: 2ms latency with neuromorphic hardware (published in Biomimetic and Biohybrid Systems, 2017)
- **Thesis PDF**: Not in repo, but published paper exists
- **Complexity**: ADVANCED - neuromorphic hardware, event-based vision, SpiNNaker
- **Stars**: 2 | **Forks**: 2 | **Commits**: 182
- **Assessment**: Very advanced for a BSc thesis. Published academic paper from the work. Access to SpiNNaker hardware was likely through Manchester's facilities.

### 1.6 QuadBot-NeuroMorphic (Cambridge Undergraduate Research)
- **URL**: https://github.com/Cambridge-Control-Lab/QuadBot-NeuroMorphic
- **Description**: Neuromorphic control of quadrupedal robot locomotion using spiking neural circuits (CPGs)
- **Framework**: MATLAB/Simulink R2022b+, SOLIDWORKS, VEX V5 Robot Brain, Python
- **Dataset**: N/A (physical robot experiments)
- **Results**: Successfully implemented biologically-inspired gaits on physical robots (NeuroPup and Synapider)
- **Thesis PDF**: No formal thesis, but comprehensive documentation
- **Complexity**: ADVANCED - neuroscience + control systems + embedded hardware
- **Stars**: 5 | **Forks**: 2 | **Commits**: 93 | **Last updated**: Sept 2023
- **Assessment**: 10-week summer research project (not a thesis per se). Funded by MathWorks. Very well-documented. Shows what is possible with good supervision and resources at a top university.

### 1.7 SNN Accelerator Hardware (EE552 Class Project)
- **URL**: https://github.com/zwhexplorer/Spiking-Neural-Network-Accelerator-EE552-project
- **Description**: Hardware accelerator for SNNs inspired by TrueNorth and Loihi, 3x3 mesh network
- **Framework**: SystemVerilog (100%), asynchronous hardware design
- **Dataset**: N/A (hardware design)
- **Results**: Working 9-router mesh topology with XY routing
- **Report**: Final report PDF included
- **Complexity**: HIGH - hardware design, NoC architecture, asynchronous circuits
- **Stars**: 15 | **Forks**: 4 | **Commits**: 64
- **Assessment**: Graduate-level course project (EE552). Very hardware-focused. Not typical for a software-oriented thesis but shows the hardware side of neuromorphic computing.

### 1.8 Neuromorphic NoC Architecture for SNNs (4th Year Project)
- **URL**: https://github.com/cepdnaclk/e18-4yp-Neuromorphic-NoC-Architecture-for-SNNs
- **Description**: Scalable Network-on-Chip architecture based on RISC-V ISA for SNN processing on FPGA
- **Framework**: Verilog (99.7%), FPGA implementation
- **Dataset**: N/A (hardware design)
- **Results**: Working FPGA implementation with custom accelerators
- **Complexity**: VERY HIGH - custom hardware, ISA extensions, FPGA
- **Stars**: 7 | **Forks**: 5 | **Commits**: 207 | **Last updated**: June 2024
- **Assessment**: 4th-year engineering project (likely undergraduate final year in a 4-year program). Team of 3 students. Very ambitious hardware project.

---

## Section 2: Confirmed Masters Thesis Projects

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
