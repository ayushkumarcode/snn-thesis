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
- **Assessment**: Most popular thesis project found. Excellent documentation. Combines DVS, robotics, and SNNs.

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
- **Assessment**: Novel intersection of GANs and spiking data

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

## Section 3: Student Course Projects / Research Group Projects

### 3.1 Simple SNN with STDP (University of Osnabruck Course Project)
- **URL**: https://github.com/cowolff/Simple-Spiking-Neural-Network-STDP
- **Description**: From-scratch SNN with STDP trained on MNIST
- **Framework**: Python, TensorFlow/Keras (for MNIST loading only)
- **Dataset**: MNIST (partial)
- **Results**: Reasonable accuracy after 1 epoch, plateaued quickly. Dense NNs outperformed significantly.
- **Report**: YES - Paper.pdf included
- **Complexity**: MODERATE (no hidden layers, single-layer architecture)
- **Stars**: 47 | **Forks**: 9 | **Commits**: 73
- **Assessment**: EXCELLENT model for a course project. Four students, from-scratch implementation, honest about limitations. Very achievable scope for a group undergraduate project.

### 3.2 SNN Image Classification - SNN vs CNN Comparison
- **URL**: https://github.com/sofi12321/SNN_image_classification
- **Description**: Comparing SNN and CNN for image classification across multiple datasets
- **Framework**: snnTorch, PyTorch
- **Dataset**: SOCOFing (fingerprints), EMNIST, Fashion-MNIST
- **Results**: SOCOFing: SNN 98% vs CNN 83%; EMNIST: both 99%; Fashion-MNIST: both 86%. SNN training ~1.5x slower.
- **Report**: No separate report; comprehensive README
- **Complexity**: MODERATE
- **Stars**: 9 | **Forks**: 2 | **Commits**: 21
- **Assessment**: GREAT model for an undergraduate project. Three datasets, clear comparison, achievable scope. Uses snnTorch which has good documentation. Single Jupyter notebook format.

### 3.3 SNN Image Classification (AI3610 Homework)
- **URL**: https://github.com/HaoyiZhu/SNN_Image_Classification
- **Description**: Convolutional SNN with 12C5-MP2-64C5-MP2-1024FC10 architecture
- **Framework**: snnTorch, PyTorch, Hydra config
- **Dataset**: Static images and spike-based neuromorphic inputs
- **Results**: Static: 99.12% accuracy; Spike data: 97.05% accuracy (20 epochs, RTX 3090)
- **Complexity**: MODERATE
- **Stars**: 7 | **Commits**: 7
- **Assessment**: Clean implementation. Shows what a course assignment looks like - focused, achievable, good results.

### 3.4 Deep Learning with Biologically Plausible Neural Networks
- **URL**: https://github.com/chiralevy/Deep-Learning-with-Biologically-Plausible-Neural-Networks
- **Description**: Performance comparison between SNNs and conventional NNs on three tasks
- **Framework**: snnTorch
- **Dataset**: MNIST, CIFAR-10, Google Speech Commands
- **Results**:
  - MNIST: CSNN 98.06% vs CNN 98.39%
  - CIFAR-10: CSNN 70.60% vs CNN 68.00%
  - Speech Commands: LSNN 91.20% vs LSTM 94.40% vs CNN 87.60%
- **Report**: No PDF; comprehensive README with results tables
- **Complexity**: MODERATE-HIGH (three different domains: vision, vision, audio)
- **Stars**: 4
- **Assessment**: EXCELLENT scope for a thesis project. Three tasks, clear comparisons, multiple architectures. The CIFAR-10 and Speech Commands results show meaningful contribution beyond MNIST.

### 3.5 SNN Gesture Classification with DVS128
- **URL**: https://github.com/DerrickL25/SNN_Gesture_Classification
- **Description**: Neuromorphic gesture classification from DVS128 event camera data
- **Framework**: snnTorch, PyTorch
- **Dataset**: DVSGesture from IBM (1,077 samples, 11 gesture classes, 29 subjects)
- **Results**: Not explicitly stated
- **Complexity**: MODERATE
- **Stars**: 5
- **Assessment**: Good focused project using real neuromorphic data. Single Jupyter notebook. Research group project.

### 3.6 ANN vs SNN Comparison (Course Project)
- **URL**: https://github.com/NicolaCST/ANN-vs-SNN
- **Description**: Comparing performances and power consumption between ANNs and SNNs
- **Framework**: Python, Jupyter Notebook
- **Dataset**: Not specified
- **Report**: YES - VCS_doc.pdf included
- **Complexity**: MODERATE
- **Stars**: 0 | **Commits**: 3
- **Assessment**: Simple course project with PDF report. Good starting point for understanding SNN vs ANN tradeoffs.

### 3.7 RL-SNN-Quadrupeds (UC Berkeley EECS206B Final Project)
- **URL**: https://github.com/tganamur/RL-SNN-Quadrupeds
- **Description**: Teaching quadruped robots to walk using SNNs and RL
- **Framework**: MuJoCo, Stable-baselines3, PPO
- **Results**: MLP learned ape-like gait; SNN achieved standing but not walking. First SNN-based RL on physical quadruped.
- **Complexity**: HIGH
- **Stars**: 13
- **Assessment**: Ambitious but partially successful. Real hardware deployment (PuppyPi robot). Shows sim-to-real challenges.

### 3.8 Convolutional SNN for Speech Recognition
- **URL**: https://github.com/verrannt/snn_speechrec
- **Description**: Unsupervised convolutional SNN for speech recognition using STDP
- **Framework**: Python, PyTorch, scikit-learn (SVM classifier)
- **Dataset**: TIDIGITS
- **Results**: Achieved 92% accuracy (vs 97.5% in reference paper)
- **Report**: YES - Report.pdf with analysis of implementation differences
- **Complexity**: MODERATE-HIGH
- **Stars**: 9 | **Commits**: 140

### 3.9 Backpropagation for Amplitude Classification using SNNs
- **URL**: https://github.com/aravsi77/spiking_neural_network_thesis
- **Description**: SNN for 4QAM modulation classification
- **Framework**: BindsNet
- **Dataset**: 4QAM signal data at 18dB SNR
- **Complexity**: MODERATE
- **Stars**: 1 | **Commits**: 34

---

## Section 4: Notable Non-Student Projects (Relevant for Scope Comparison)

### 4.1 Pure Python SNN (IIT Guwahati)
- **URL**: https://github.com/Shikhargupta/Spiking-Neural-Network
- **Description**: Hardware-efficient SNN with STDP and WTA lateral inhibition
- **Framework**: Pure Python
- **Dataset**: MNIST
- **Results**: Successful binary and multi-class classification; clear neuron specialization
- **Stars**: 1,200+ | **Forks**: 294
- **Assessment**: Most popular SNN educational implementation on GitHub. Shows learned digit patterns via weight reconstruction. Great reference for understanding SNNs from scratch.

### 4.2 SNN++ (C++ High-Performance)
- **URL**: https://github.com/ianmkim/snnpp
- **Description**: C++ SNN implementation with SIMD optimization, 2000% faster than reference Python
- **Framework**: C++, OpenCV, CMake, Intel SSE
- **Dataset**: MNIST
- **Results**: ~50 seconds vs ~18 minutes for reference implementation
- **Stars**: 13 | **Commits**: 30

### 4.3 Python SNN with STDP and RL
- **URL**: https://github.com/maael/SpikingNeuralNetwork
- **Description**: SNN with basic STDP, homeostatic STDP, and reward-based RL STDP variants
- **Framework**: Python 3
- **Stars**: 133 | **Forks**: 37

### 4.4 FPGA SNN STDP Acceleration
- **URL**: https://github.com/rafamedina97/FPGA_SNN_STDP
- **Description**: FPGA hardware acceleration of STDP learning for SNNs
- **Framework**: VHDL, SystemVerilog, Vivado, MATLAB
- **Dataset**: MNIST (784-20-10 network)
- **Stars**: 40 | **Forks**: 7

### 4.5 SNN4Space (ESA)
- **URL**: https://github.com/AndrzejKucik/SNN4Space
- **Description**: ANN-to-SNN conversion for satellite land cover classification
- **Framework**: KerasSpiking, TensorFlow
- **Dataset**: EuroSAT RGB (27,000 examples), UC Merced (2,100 examples)
- **Results**: UC Merced 91.43%, EuroSAT 95.07%
- **Stars**: 14 | **Commits**: 128

### 4.6 Bayesian Optimization 1D-CSNN for Fraud Detection
- **URL**: https://github.com/dylanperdigao/Bayesian-Optimization-1D-CSNN
- **Description**: 1D-Convolutional SNN optimized with Bayesian methods for fraud detection
- **Framework**: snnTorch, Python
- **Dataset**: Bank Account Fraud (BAF) Dataset from NeurIPS 2022
- **Results**: Published at EPIA 2024 conference
- **Stars**: 3

---

## Section 5: Curated Lists and Resource Collections

### 5.1 awesome-snn
- **URL**: https://github.com/coderonion/awesome-snn
- **Description**: Collection of public SNN projects including frameworks, tools, and applications

### 5.2 Awesome-Spiking-Neural-Networks (yfguo91)
- **URL**: https://github.com/yfguo91/Awesome-Spiking-Neural-Networks
- **Description**: Comprehensive survey including papers organized by topic

### 5.3 Awesome-SNN-Conference-Paper
- **URL**: https://github.com/AXYZdong/awesome-snn-conference-paper
- **Description**: Papers from top-tier conferences with code links

### 5.4 Awesome-SNN-Paper-Collection
- **URL**: https://github.com/Ruichen0424/Awesome-SNN-Paper-Collection
- **Description**: Papers organized by topic including spike cameras

### 5.5 Open Neuromorphic
- **URL**: https://github.com/open-neuromorphic/open-neuromorphic
- **Website**: https://open-neuromorphic.org/
- **Description**: Global community with workshops, student talks, and project peer review

### 5.6 Event-Based Vision Resources
- **URL**: https://github.com/uzh-rpg/event-based_vision_resources
- **Description**: Comprehensive resource list for event-based vision technology

---

## Section 6: Framework Ecosystem Summary

| Framework | URL | Best For | Student-Friendly? |
|-----------|-----|----------|-------------------|
| **snnTorch** | https://github.com/jeshraghian/snntorch | PyTorch-based training, gradient descent, tutorials | YES - excellent tutorials |
| **SpikingJelly** | https://github.com/fangwei123456/spikingjelly | Full-stack SNN development, PyTorch-based | MODERATE - Chinese docs |
| **Brian2** | https://github.com/brian-team/brian2 | Biological neuroscience simulation | YES - great documentation |
| **BindsNET** | https://github.com/BindsNET/bindsnet | STDP learning, PyTorch integration | YES - good examples |
| **Norse** | https://github.com/norse/norse | Deep learning + SNNs in PyTorch | MODERATE |
| **SpyTorch** | https://github.com/fzenke/spytorch | Surrogate gradient learning tutorials | YES - tutorial focused |
| **Lava** | https://github.com/lava-nc/lava-dl | Intel Loihi deployment | NO - hardware specific |
| **PySNN** | https://github.com/BasBuller/PySNN | Simple PyTorch SNN | YES - beginner friendly |

---

## Section 7: snnTorch-Tagged Projects (from GitHub Topics)

Additional smaller projects using snnTorch found via the snntorch GitHub topic:

| Project | Description | Stars |
|---------|-------------|-------|
| neuromorphic_classifier | MNIST classification with SNN | 1 |
| snn-tre | SNN model to classify MNIST | 0 |
| IA_in_complex_game_snn | Evolving SNNs for trash collection game | 2 |
| Spiking-ResNet | Blood pressure prediction from PPG | 2 |
| search-and-rescue | Drone computer vision with SNN | 1 |
| snn-image-classification | Computer vision basics | 0 |
| Spiking-Classifier | Image classification with Gradio UI | 0 |
| PredictiveSNNModels | MNIST sequence prediction | 0 |
| snn-glacier-segmentation | Glacier image segmentation | 0 |
| SNN-CL-AutonomousDriving | Autonomous driving + continual learning | 0 |
| PulsePod | Arrhythmia detection framework | 0 |

---

## Section 8: Analysis - Typical Scope for a 3rd Year Undergraduate Project

Based on analyzing 40+ projects, here is what a realistic 3rd-year undergraduate SNN project looks like:

### Typical Characteristics
1. **Single focused task**: Classification on one primary dataset, possibly tested on a second
2. **One framework**: snnTorch (most accessible) or Brian2 (more neuroscience-oriented)
3. **Standard datasets**: MNIST, Fashion-MNIST, CIFAR-10, DVS Gesture, N-MNIST
4. **Comparison angle**: SNN vs ANN/CNN performance comparison is the most common approach
5. **Duration**: Semester-long or year-long
6. **Deliverable**: Jupyter notebooks + report/dissertation
7. **Code volume**: 1-5 Python files or 1-3 Jupyter notebooks

### Scope Tiers for Undergraduate Projects

**Tier 1 - Achievable (Good Grade)**
- SNN classification on MNIST/Fashion-MNIST using snnTorch
- Compare accuracy and training time with equivalent CNN
- Example: sofi12321/SNN_image_classification

**Tier 2 - Ambitious (Very Good Grade)**
- SNN on multiple datasets (MNIST + CIFAR-10 + one more)
- OR neuromorphic dataset (DVS Gesture, N-MNIST)
- OR novel application domain (audio, time series, medical)
- Example: chiralevy/Deep-Learning-with-Biologically-Plausible-Neural-Networks

**Tier 3 - Very Ambitious (Outstanding)**
- Novel architecture or training method
- Hardware deployment (FPGA, SpiNNaker if available)
- Robotics integration
- Published results
- Example: filippoferrari/shape_detector_snn, mrahtz/musical-pattern-recognition-in-spiking-neural-networks

### What Distinguishes Good Projects
1. **Clear research question** (not just "implement an SNN")
2. **Meaningful comparison** (SNN vs ANN on same task)
3. **Multiple evaluation metrics** (accuracy, training time, energy estimates)
4. **Good documentation** (README, code comments, report)
5. **Honest about limitations** (convergence issues, accuracy gaps)

---

## Section 9: Projects with Thesis PDFs Available

| Project | Level | PDF Location |
|---------|-------|--------------|
| Musical Pattern Recognition | BEng | http://amid.fish/beng_project_report.pdf |
| Jegp/thesis (Volr DSL) | Masters | report/report.pdf in repo |
| Deep Spiking Q-Networks | Masters | In repository |
| Spiking Grid Cell Models | MSc | Linked in repo |
| Brain-Machine Interface | Masters | In repository |
| Spiking Deep Belief Network | Masters | MazdakFatahi(Ms Thesis).pdf |
| SNN-RL | Masters | Thesis.pdf in repo |
| Event-Based Robot Control | Masters | In repository |
| Master thesis Astoeckel 2015 | Masters | GitHub releases v1.2 |
| Simple SNN STDP | Course | Paper.pdf in repo |
| SNN Speech Recognition | Research | Report.pdf in repo |
| ANN vs SNN | Course | VCS_doc.pdf in repo |
| SNN Accelerator EE552 | Course | Final report PDF in repo |

---

## Section 10: Datasets Commonly Used Across Projects

| Dataset | Usage Frequency | Type | Difficulty |
|---------|----------------|------|------------|
| MNIST | Very High | Static images | Easy |
| Fashion-MNIST | High | Static images | Easy-Medium |
| CIFAR-10 | Medium | Static images | Medium |
| N-MNIST | Medium | Neuromorphic events | Medium |
| DVS Gesture (IBM) | Medium | Neuromorphic events | Medium-Hard |
| CIFAR10-DVS | Low-Medium | Neuromorphic events | Hard |
| Google Speech Commands | Low | Audio | Medium |
| TIDIGITS | Low | Audio | Medium |
| EEG datasets (various) | Low | Time series | Hard |
| Custom datasets | Variable | Various | Variable |

---

## Section 11: Research Gaps and Limitations

### What I Could Not Find
1. **Very few explicitly labeled "3rd year" projects** - Most undergraduate projects are labeled "BSc" or "final year" without specifying year
2. **Limited thesis PDFs for undergraduate work** - Masters students are more likely to include PDFs
3. **Energy consumption benchmarks** - Few student projects actually measure or estimate energy savings
4. **Deployment on edge devices** - Very few projects deploy SNNs on actual neuromorphic hardware (those that do are typically at Manchester with SpiNNaker access)
5. **NLP/text applications with SNNs** - Almost no student projects found in this domain

### Why Some Searches Were Unproductive
- "Third year project" is not a universal term (varies by country)
- Many student repos lack proper README documentation
- Some projects are in private repos or institutional GitLab instances
- Neuromorphic hardware projects require lab access not available to most students

---

## Section 12: Recommended Follow-ups

1. **Search institutional repositories** (e.g., ETH Zurich Research Collection, Manchester e-scholar) for thesis PDFs not hosted on GitHub
2. **Check Open Neuromorphic student talks** (https://open-neuromorphic.org/) for recent student work
3. **Browse snnTorch discussion forum** (GitHub Discussions) for student project questions
4. **Review NICE (Neuro-Inspired Computational Elements) workshop proceedings** for student presentations
5. **Search arXiv** for papers from student authors at universities (often link to GitHub repos)

---

## Confidence Assessment

| Finding | Confidence |
|---------|------------|
| Project URLs and basic metadata | HIGH - verified from GitHub |
| Accuracy numbers and results | HIGH - from README/repos |
| Student vs research classification | MODERATE - inferred from context |
| Thesis PDF availability | HIGH - verified in repos |
| Complexity assessments | MODERATE - based on code inspection |
| Scope tier recommendations | HIGH - based on pattern analysis across 40+ projects |
| Framework recommendations | HIGH - based on ecosystem analysis |
