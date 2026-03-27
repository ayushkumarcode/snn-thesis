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
- **URL**: https://github.com/cowolff/Simple-Spiking-Neural-Network-STDP
- **Description**: From-scratch SNN with STDP trained on MNIST
- **Framework**: Python, TensorFlow/Keras (for MNIST loading only)
- **Dataset**: MNIST (partial)
- **Results**: Reasonable accuracy after 1 epoch, plateaued quickly. Dense NNs outperformed significantly.
- **Report**: YES - Paper.pdf included
- **Complexity**: MODERATE (no hidden layers, single-layer architecture)
- **Stars**: 47 | **Forks**: 9 | **Commits**: 73
- great model for a course project. four students, from-scratch implementation, honest about limitations. very achievable scope for a group undergrad project.

### 3.2 SNN Image Classification - SNN vs CNN Comparison
- **URL**: https://github.com/sofi12321/SNN_image_classification
- **Description**: Comparing SNN and CNN for image classification across multiple datasets
- **Framework**: snnTorch, PyTorch
- **Dataset**: SOCOFing (fingerprints), EMNIST, Fashion-MNIST
- **Results**: SOCOFing: SNN 98% vs CNN 83%; EMNIST: both 99%; Fashion-MNIST: both 86%. SNN training ~1.5x slower.
- **Report**: No separate report; comprehensive README
- **Complexity**: MODERATE
- **Stars**: 9 | **Forks**: 2 | **Commits**: 21
- this is a great model for an undergrad project. three datasets, clear comparison, achievable scope. uses snnTorch which has good documentation. single Jupyter notebook format.

### 3.3 SNN Image Classification (AI3610 Homework)
- **URL**: https://github.com/HaoyiZhu/SNN_Image_Classification
- **Description**: Convolutional SNN with 12C5-MP2-64C5-MP2-1024FC10 architecture
- **Framework**: snnTorch, PyTorch, Hydra config
- **Dataset**: Static images and spike-based neuromorphic inputs
- **Results**: Static: 99.12% accuracy; Spike data: 97.05% accuracy (20 epochs, RTX 3090)
- **Complexity**: MODERATE
- **Stars**: 7 | **Commits**: 7
- clean implementation. shows what a course assignment looks like -- focused, achievable, good results.

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
- excellent scope for a thesis project i think. three tasks, clear comparisons, multiple architectures. the CIFAR-10 and Speech Commands results show meaningful contribution beyond just MNIST.

### 3.5 SNN Gesture Classification with DVS128
- **URL**: https://github.com/DerrickL25/SNN_Gesture_Classification
- **Description**: Neuromorphic gesture classification from DVS128 event camera data
- **Framework**: snnTorch, PyTorch
- **Dataset**: DVSGesture from IBM (1,077 samples, 11 gesture classes, 29 subjects)
- **Results**: Not explicitly stated
- **Complexity**: MODERATE
- **Stars**: 5
- good focused project using real neuromorphic data. single Jupyter notebook. research group project.

### 3.6 ANN vs SNN Comparison (Course Project)
- **URL**: https://github.com/NicolaCST/ANN-vs-SNN
- **Description**: Comparing performances and power consumption between ANNs and SNNs
- **Framework**: Python, Jupyter Notebook
- **Dataset**: Not specified
- **Report**: YES - VCS_doc.pdf included
- **Complexity**: MODERATE
- **Stars**: 0 | **Commits**: 3
- simple course project with PDF report. good starting point for understanding SNN vs ANN tradeoffs.

### 3.7 RL-SNN-Quadrupeds (UC Berkeley EECS206B Final Project)
- **URL**: https://github.com/tganamur/RL-SNN-Quadrupeds
- **Description**: Teaching quadruped robots to walk using SNNs and RL
- **Framework**: MuJoCo, Stable-baselines3, PPO
- **Results**: MLP learned ape-like gait; SNN achieved standing but not walking. First SNN-based RL on physical quadruped.
- **Complexity**: HIGH
- **Stars**: 13
- ambitious but only partially successful. real hardware deployment (PuppyPi robot). shows the sim-to-real challenges well.

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

## notable non-student projects (useful for scope comparison)

### 4.1 Pure Python SNN (IIT Guwahati)
- **URL**: https://github.com/Shikhargupta/Spiking-Neural-Network
- **Description**: Hardware-efficient SNN with STDP and WTA lateral inhibition
- **Framework**: Pure Python
- **Dataset**: MNIST
- **Results**: Successful binary and multi-class classification; clear neuron specialization
- **Stars**: 1,200+ | **Forks**: 294
- most popular SNN educational implementation on github. shows learned digit patterns via weight reconstruction. great reference for understanding SNNs from scratch.

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
