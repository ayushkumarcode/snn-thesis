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
- Very high complexity -- multiple frameworks, Haskell DSL, neuromorphic hardware
- 4 stars, 99 commits

### Recurrent SNNs for POMDPs
- https://github.com/Quickblink/rsnn
- Recurrent SNNs for Partially Observable Markov Decision Processes
- PyTorch, Docker, multiple neuron architectures (LIF, Adaptive, etc.)
- Sequential MNIST, encoded MNIST variants. Results not explicitly documented.
- 3 stars, 185+ commits

### SNNs for RL Tasks (Masters by Research, UTS)
- https://github.com/andrewrafeUTS/SNNTechnicalAppendix
- Evolutionary experiments with SNNs for CartPole and LunarLander
- Python 3.8, custom SNN implementation. Tested multiple decoding methods.
- 1 star, 5 commits

### Deep Spiking Q-Networks (TUM Masters)
- https://github.com/vhris/Deep-Spiking-Q-Networks
- Spiking DQN training using conversion and surrogate gradients for RL
- SpyTorch, NEST 2.16.0, PyNN, OpenAI Gym
- CartPole, MountainCar, Breakout. Both conversion and direct training worked on CartPole.
- **Thesis PDF included in repository**
- 11 stars, 3 forks

### SNN for Hand Kinematics from sEMG
- https://github.com/davidkubanek/SNN-hand-kinematics-estimation-from-sEMG-signals
- Neuromorphic reservoir network for estimating hand movements from muscle signals
- Brian2, Python, C++/Cython. NinaPro public EMG database.
- 4 stars, 30 commits

### Spiking Grid Cell Models on SpiNNaker (MSc, Manchester)
- https://github.com/nickybu/spiking_grid_cell_model
- Spiking grid cell models on SpiNNaker, supervised by Prof. Furber
- SpiNNaker, sPyNNaker, Python 2.7, Brian2
- **Thesis PDF linked in repository**
- 0 stars, 9 commits

### Brain-Machine Interface using SpiNNaker (Masters)
- https://github.com/solversa/Master-Thesis-Brain-Machine-Interface
- Decoding 3D imaginary reach/grasp movements from EEG using SNNs on SpiNNaker
- SpiNNaker (4 chips, 64 cores), STDP with reward-based training
- 73.4% mean accuracy (only 4.12% below state-of-the-art ML)
- **Thesis PDF included**
- 2 stars, 5 forks, 102 commits

### Spiking Deep Belief Network (Masters)
- https://github.com/MazdakFatahi/Spiking-Deep-Belief-Network
- Spike-Based Deep Belief Network with LIF neurons using contrastive divergence
- MNIST, 94.9% accuracy
- **Thesis PDF: "MazdakFatahi(Ms Thesis).pdf"**
- 1 star, 3 commits

### SNN-RL: Training SNNs with RL (Masters)
- https://github.com/BSVogler/SNN-RL
- Actor-critic RL framework with spiking network actors using R-STDP
- NEST 3, Python 3.7/3.8, Docker, MongoDB. Line-following task.
- **Thesis PDF: Thesis.pdf in repo**
- 21 stars, 3 forks

### Event-Based End-to-End Robot Control (TUM Masters)
- https://github.com/clamesc/Training-Neural-Networks-for-Event-Based-End-to-End-Robot-Control
- Robot steering with DVS event camera using DRL and SNNs for lane-keeping
- TensorFlow, V-REP, ROS, NEST 2.10.0, Python 2.7
- Both DQN-SNN and R-STDP methods worked for lane following.
- **Thesis PDF included**
- 59 stars, 23 forks -- most popular thesis project i found. Great documentation. Combines DVS, robotics, and SNNs.

### CartPole with SNNs inspired by Theory of Mind
- https://github.com/atenagm1375/cartpole
- SNN-based CartPole control inspired by Theory of Mind concepts
- PyTorch, BindsNet, OpenAI Gym. CartPole, River Raid.
- 1 star, 77 commits

### GANs for Spiking Time Series (UvA Masters)
- https://github.com/HitLuca/GANs_for_spiking_time_series
- Generating spiking time series patterns using GANs (University of Amsterdam)
- Interesting intersection of GANs and spiking data

### Use of Spiking Neural Networks (Thesis)
- https://github.com/honzikv/use-of-snn
- Three experiments: EEG classification, P300 detection, surrogate gradient MNIST/Fashion-MNIST
- PyTorch, TensorFlow, Jupyter
- BNCI Horizon 2020 EEG, Harvard P300, MNIST, Fashion MNIST
- Successfully converted CNN to SNN for EEG; surrogate gradient training on image datasets
- 2 stars, 157 commits

### SNN Formation Control for Multi-Agent Systems
- https://github.com/ViktorNfa/SpikingNeuralNet_FormationControl
- SNN using Norse framework to learn formation control with collision avoidance
- SNN learned formation control comparable to classical controllers
- 4 stars, 18 commits

### Design Space Exploration of Associative Memories (Masters, Bielefeld)
- https://github.com/astoeckel/master-thesis-astoeckel-2015
- Willshaw model for associative memories using spiking neurons targeting neuromorphic hardware
- LaTeX, Python, MATLAB, C++
- **Thesis PDF: downloadable v1.2 from GitHub releases (CC BY-ND 4.0)**
- 2 stars

---

## Student Course Projects / Research Group Projects

### Simple SNN with STDP (University of Osnabruck Course Project)
- https://github.com/cowolff/Simple-Spiking-Neural-Network-STDP
- From-scratch SNN with STDP trained on MNIST
- Python, TF/Keras (just for MNIST loading), partial MNIST
- Got reasonable accuracy after 1 epoch then plateaued quickly. Dense NNs outperformed significantly.
- **Report: Paper.pdf in repo**
- Four students, from-scratch, honest about limitations. Very achievable scope for a group undergrad project.
- 47 stars, 9 forks, 73 commits

### SNN Image Classification - SNN vs CNN Comparison
- https://github.com/sofi12321/SNN_image_classification
- Comparing SNN and CNN on multiple datasets
- snnTorch, PyTorch
- SOCOFing (fingerprints): SNN 98% vs CNN 83%; EMNIST: both 99%; Fashion-MNIST: both 86%. SNN training ~1.5x slower.
- Great model for an undergrad project. Three datasets, clear comparison, achievable scope. Single Jupyter notebook format.
- 9 stars, 2 forks, 21 commits

### SNN Image Classification (AI3610 Homework)
- https://github.com/HaoyiZhu/SNN_Image_Classification
- Convolutional SNN with 12C5-MP2-64C5-MP2-1024FC10 architecture
- snnTorch, PyTorch, Hydra config
- Static: 99.12% accuracy; Spike data: 97.05% (20 epochs, RTX 3090)
- Clean implementation. Shows what a course assignment looks like -- focused, achievable, good results.
- 7 stars, 7 commits

### Deep Learning with Biologically Plausible NNs
- https://github.com/chiralevy/Deep-Learning-with-Biologically-Plausible-Neural-Networks
- Performance comparison between SNNs and conventional NNs on three tasks
- snnTorch. MNIST, CIFAR-10, Google Speech Commands.
- MNIST: CSNN 98.06% vs CNN 98.39%. CIFAR-10: CSNN 70.60% vs CNN 68.00%. Speech Commands: LSNN 91.20% vs LSTM 94.40% vs CNN 87.60%.
- Really good scope for a thesis project. Three tasks, clear comparisons, multiple architectures. The CIFAR-10 and Speech Commands results go meaningfully beyond MNIST.
- 4 stars

### SNN Gesture Classification with DVS128
- https://github.com/DerrickL25/SNN_Gesture_Classification
- Neuromorphic gesture classification from DVS128 event camera data
- snnTorch, PyTorch. DVSGesture from IBM (1,077 samples, 11 gesture classes, 29 subjects).
