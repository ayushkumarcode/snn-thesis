# Searching Manchester's Figshare for Neuromorphic / SNN Projects

Went through the University of Manchester's Figshare institutional repository (https://figshare.manchester.ac.uk/) pretty thoroughly. Searched across all 17 Manchester group IDs, reviewed 833 unique articles, tried 100+ search terms covering neuromorphic computing, SNNs, brain-inspired computing, FPGA, edge AI, BCI/EEG, ML, deep learning, computer vision, robotics, and a bunch of adjacent topics.

The bottom line: **there are no undergraduate third-year thesis projects on the Figshare Manchester repository related to neuromorphic computing or anything close to it.** The repository doesn't seem to host undergrad theses at all. It's mainly used for:
- Research datasets supporting published papers
- Software/code supplements for journal publications
- Conference presentations and posters (mostly teaching/learning related)
- PhD thesis supplementary data
- Architectural models from the B.15 Model Archive

Zero results for all these core neuromorphic terms:
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

The only match for "neuromorphic" was a research-level publication (not undergrad) by members of Steve Furber's SpiNNaker group.

---

## What i actually found

### Directly Neuromorphic-Relevant (just 1)

#### BitBrain and Sparse Binary Coincidence (SBC) Memories
- **Title:** Implementation of BitBrain algorithm in C
- **Authors:** Jakub Fil, Michael Hopkins, Steve Furber, Edward Jones
- **Year:** 2022
- **URL:** https://figshare.manchester.ac.uk/articles/software/Implementation_of_BitBrain_algorithm_in_C/21679610
- **DOI:** https://doi.org/10.48420/21679610
- Software (C code implementation)
- Tags: neuromorphic, single-pass learning, efficient inference, classification, machine learning, robust, event-based, IoT
- It's about an innovative working mechanism (SBC memory) and surrounding infrastructure (BitBrain) based on a synthesis of ideas from sparse coding, computational neuroscience and information theory. Supports single-pass and single-shot learning, efficient inference, and continuous adaptive learning. Designed for neuromorphic devices and conventional CPUs.
- Funded by Human Brain Project Specific Grant Agreement 3 (H2020 945539)
- This is a **research-level publication supplement** by members of the APT group -- Steve Furber is the creator of SpiNNaker. Definitely not an undergrad project. It's a supplement for "Frontiers in Neuroinformatics: Physical Neuromorphic Computing and its Industrial Applications."
- Uses C programming, MNIST dataset
- Still relevant as reference for what Manchester researchers are doing in neuromorphic computing

---

### Adjacent ML/AI/Neural Engineering Stuff (closest things to relevant)

#### Deep Autoencoder for Real-Time EEG Artifact Removal
- Le Xing, Alex Casson, 2023
- https://figshare.manchester.ac.uk/articles/software/Deep_autoencoder_for_real-time_EEG_artifact_removal_and_its_Android_smartphone_implementation/23093759
- A Deep Convolutional Autoencoder for single-channel EEG artifact removal in real-time, implemented as an Android App for mobile EEG and portable BCI applications.
- Research publication from Alex Casson's Non-Invasive Bioelectronics Lab, not undergrad.
- Uses Python, TensorFlow Lite, Android Studio, EEG data

#### LSTM Robot Navigation in Warehouse Environments
- Jieru Zhou, 2024
- https://figshare.manchester.ac.uk/articles/software/Optimizing_Warehouse_Robot_Navigation_with_LSTM_Networks_and_Adaptive_Greedy_Weight_Techniques/26870128
- Combines LSTM networks with adaptive greedy weight strategies for path planning in dynamic warehouses. LSTM predicts future robot paths from historical data, then adaptive greedy weight balances exploration/exploitation. Includes LSTM model, training code, and two pathfinding algorithms.
- Single author, School of Engineering. Only 9 kB of code. **Possibly a student project** based on single authorship and scope but can't confirm if undergrad.
- Has a video demo: Robot Path Planning Using LSTM+Adaptive Greedy Weight and A*+Adaptive Greedy Weight Strategies (ID: 26869981)

#### ECG, EEG and IMU Data for Motion Artefact Removal
- Christopher Beach, Mingjie Li, Ertan Balaban, Alex Casson, 2021
- https://figshare.manchester.ac.uk/articles/dataset/ECG_EEG_and_IMU_data_for_local_motion_artefact_removal/13626395
- ECG and EEG data with IMUs on each electrode for motion artefact removal
- Research dataset from Alex Casson's lab

#### Low Power Computer Vision for Landmine Detection
- Adam Fletcher, John Davidson, Edward Cheadle, Daniel Conniffe, Anthony Peyton, Frank Podd, 2025
- https://figshare.manchester.ac.uk/articles/dataset/Evaluation_of_a_Low_Power_Computer_Vision_Based_Positioning_System_for_a_Handheld_Landmine_Detector_using_AprilTag_Markers_-_Supporting_Data/30030226
- Positioning system using AprilTags, designed for low-power battery-powered real-time applications
- Research publication, tangentially related (low power + computer vision)

#### Autonomous Robotics for Nuclear Environments
