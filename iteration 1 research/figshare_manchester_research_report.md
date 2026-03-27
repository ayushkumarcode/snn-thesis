# Figshare Manchester Repository: Neuromorphic Computing and Adjacent Topics
# Comprehensive Research Investigation Report
# Date: 2026-02-24

---

## EXECUTIVE SUMMARY

After an exhaustive investigation of the University of Manchester's Figshare institutional repository (https://figshare.manchester.ac.uk/), searching across **all 17 Manchester group IDs**, reviewing **833 unique articles**, running **100+ search terms** spanning neuromorphic computing, spiking neural networks, brain-inspired computing, FPGA implementations, edge AI, BCI/EEG, machine learning, deep learning, computer vision, robotics, and dozens more adjacent topics, the core finding is:

**There are NO undergraduate third-year (final year) thesis projects on the Figshare Manchester repository related to neuromorphic computing or closely adjacent topics.** In fact, the repository does not appear to host undergraduate theses at all. The Manchester Figshare portal is used primarily for:
- Research datasets supporting published academic papers
- Software/code supplements for journal publications
- Conference presentations and posters (mostly about teaching/learning)
- PhD thesis supplementary data
- Architectural models from the B.15 Model Archive

The repository contains **zero items** for the following core neuromorphic search terms:
- "spiking neural network" (exact phrase)
- SpiNNaker
- FPGA (any context)
- memristor / memristive
- STDP / spike timing dependent plasticity
- leaky integrate and fire / LIF
- reservoir computing
- TinyML
- Intel Loihi / IBM TrueNorth
- event camera / dynamic vision sensor / DVS
- neuromorphic engineering
- model compression / quantization / pruning
- brain-computer interface (exact phrase)
- gesture recognition
- image classification
- object detection
- edge AI / edge computing
- embedded machine learning

The single match for "neuromorphic" is a **research-level publication** (not an undergraduate project) by members of Steve Furber's SpiNNaker group.

---

## DETAILED FINDINGS

### 1. DIRECTLY NEUROMORPHIC-RELEVANT ITEMS (1 found)

#### 1.1 BitBrain and Sparse Binary Coincidence (SBC) Memories
- **Title:** Implementation of BitBrain algorithm in C
- **Authors:** Jakub Fil, Michael Hopkins, Steve Furber, Edward Jones
- **Year:** 2022
- **URL:** https://figshare.manchester.ac.uk/articles/software/Implementation_of_BitBrain_algorithm_in_C/21679610
- **DOI:** https://doi.org/10.48420/21679610
- **Type:** Software (C code implementation)
- **Categories:** Machine learning not elsewhere classified, Artificial intelligence not elsewhere classified
- **Tags:** neuromorphic, single-pass learning, efficient inference, classification, machine learning, robust, event-based, IoT
- **Description:** Code for an innovative working mechanism (SBC memory) and surrounding infrastructure (BitBrain) based upon a novel synthesis of ideas from sparse coding, computational neuroscience and information theory. Supports single-pass and single-shot learning, accurate and robust inference, and the potential for continuous adaptive learning. Designed for implementation on current and future neuromorphic devices as well as conventional CPU architectures.
- **Funding:** Human Brain Project Specific Grant Agreement 3 (H2020 945539)
- **Assessment:** This is a **research-level publication supplement** by members of the Advanced Processor Technologies (APT) group at Manchester -- Steve Furber is the creator of SpiNNaker. This is NOT an undergraduate project. It is a supplement for "Frontiers in Neuroinformatics: Physical Neuromorphic Computing and its Industrial Applications."
- **Tools/Frameworks:** C programming, MNIST dataset
- **Relevance to your project:** HIGH -- directly neuromorphic. Shows what Manchester researchers are doing in neuromorphic computing, and the BitBrain algorithm could be a reference point for your work.

---

### 2. ADJACENT ML/AI/NEURAL ENGINEERING ITEMS (Closest to relevant)

#### 2.1 Deep Autoencoder for Real-Time EEG Artifact Removal
- **Title:** Deep autoencoder for real-time EEG artifact removal and its Android smartphone implementation
- **Authors:** Le Xing, Alex Casson
- **Year:** 2023
- **URL:** https://figshare.manchester.ac.uk/articles/software/Deep_autoencoder_for_real-time_EEG_artifact_removal_and_its_Android_smartphone_implementation/23093759
- **DOI:** https://doi.org/10.48420/23093759
- **Type:** Software (GitHub linked)
- **Categories:** Neural engineering, Deep learning, Neural networks
- **Tags:** Autoencoder Neural Network, EEG artifact removal, Smartphone App, Deep learning application on EEG, brain computer interface research, Hardware Acceleration
- **Description:** A novel Deep Convolutional Autoencoder neural network for single-channel EEG artifact removal in real-time. Implemented as an Android Smartphone App for mobile EEG and potential portable Brain-Computer Interfaces applications. Contains pre-processed EEG data, Python code and Android Studio project.
- **Assessment:** This is a **research publication** by Alex Casson's Non-Invasive Bioelectronics Lab. NOT an undergraduate project. However, very relevant to BCI/neural engineering topics.
- **Tools:** Python, TensorFlow Lite, Android Studio, EEG data
- **Relevance:** HIGH for BCI/neural engineering scope calibration

#### 2.2 LSTM Robot Navigation in Warehouse Environments
- **Title:** Optimizing Warehouse Robot Navigation with LSTM Networks and Adaptive Greedy Weight Techniques
- **Authors:** Jieru Zhou
- **Year:** 2024
- **URL:** https://figshare.manchester.ac.uk/articles/software/Optimizing_Warehouse_Robot_Navigation_with_LSTM_Networks_and_Adaptive_Greedy_Weight_Techniques/26870128
- **DOI:** https://doi.org/10.48420/26870128
- **Type:** Software (Python code, 9 kB)
- **Categories:** Control engineering, Intelligent robotics
- **Tags:** LSTM prediction model, Robot path planning, A* Algorithm
- **Description:** Combines Long Short-Term Memory (LSTM) networks with adaptive greedy weight strategies for optimized path planning in dynamic warehouse environments. LSTM networks predict future robot paths based on historical data, while the adaptive greedy weight strategy balances exploration and exploitation. Includes LSTM model implementation, training code, and two pathfinding algorithms.
- **Assessment:** Single author (Jieru Zhou), School of Engineering. **Possibly a student project** based on single authorship and scope, but cannot confirm if undergraduate. The code is only 9 kB, suggesting a focused implementation. The scope (LSTM + A* for robot path planning) is consistent with what might be a final-year project.
- **Tools:** Python, LSTM neural network
- **Accompanying media:** Robot Path Planning Using LSTM+Adaptive Greedy Weight and A*+Adaptive Greedy Weight Strategies (video demonstration, ID: 26869981)
- **Relevance:** MEDIUM -- shows ML-based robotics project scope

#### 2.3 ECG, EEG and IMU Data for Motion Artefact Removal
- **Title:** ECG, EEG and IMU data for local motion artefact removal
- **Authors:** Christopher Beach, Mingjie Li, Ertan Balaban, Alex Casson
- **Year:** 2021
- **URL:** https://figshare.manchester.ac.uk/articles/dataset/ECG_EEG_and_IMU_data_for_local_motion_artefact_removal/13626395
- **Type:** Dataset
- **Categories:** Biomedical engineering, Signal processing
- **Tags:** EEG data, electrocardiograms, wearable sensors, IMUs, adaptive filtering
- **Description:** ECG and EEG data with inertial measurement units (IMUs) placed on each electrode to enable recording of local motion activity during electrophysiological recordings for improved motion artefact removal.
- **Assessment:** Research dataset from Alex Casson's lab. NOT undergraduate.
- **Relevance:** MEDIUM -- biomedical signal processing / wearable computing

#### 2.4 Low Power Computer Vision for Landmine Detection
- **Title:** Evaluation of a Low Power Computer Vision Based Positioning System for a Handheld Landmine Detector using AprilTag Markers - Supporting Data
