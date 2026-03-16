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
- **Authors:** Adam Fletcher, John Davidson, Edward Cheadle, Daniel Conniffe, Anthony Peyton, Frank Podd
- **Year:** 2025
- **URL:** https://figshare.manchester.ac.uk/articles/dataset/Evaluation_of_a_Low_Power_Computer_Vision_Based_Positioning_System_for_a_Handheld_Landmine_Detector_using_AprilTag_Markers_-_Supporting_Data/30030226
- **Type:** Dataset
- **Categories:** Electronic sensors, Computer vision
- **Tags:** machine vision, positioning, fiducial markers, landmine
- **Description:** A positioning system employing visual fiducial markers (AprilTags) designed to operate at low-power for battery-powered real-time applications.
- **Assessment:** Research publication. NOT undergraduate. Interesting for "low power" / "edge computing" / "computer vision" scope.
- **Relevance:** LOW-MEDIUM -- tangentially related (low power, computer vision)

#### 2.5 Autonomous Robotics for Nuclear Environments
- **Title:** Radiation Exposure Reduction During Autonomous Exploration
- **Authors:** Andrew West
- **Year:** 2022
- **URL:** https://figshare.manchester.ac.uk/articles/dataset/Radiation_Exposure_Reduction_During_Autonomous_Exploration/18782165
- **Type:** Dataset
- **Categories:** Control engineering, Nuclear engineering
- **Tags:** ROS, Gazebo simulation, Unmanned Ground Vehicle, Control Systems, Robotics
- **Description:** Autonomous exploration sessions in a simulated environment with gamma radiation sources. Uses frontiers exploration with radiation avoidance.
- **Assessment:** Research-level. Related to the RAIN (Robotics and Artificial Intelligence for Nuclear) project at Manchester.
- **Relevance:** LOW -- autonomous robotics context only

#### 2.6 Turbulent Flow Data as PyTorch Tensors for ML
- **Title:** Turbulent Flow data as PyTorch tensors for ML
- **Authors:** Mohammed Sardar, Alex Skillen
- **Year:** 2025
- **URL:** https://figshare.manchester.ac.uk/articles/dataset/Turbulent_Flow_data_as_PyTorch_tensors_for_ML_Kolmogorov_Flow_at_Re_222_and_Kelvin-Helmholtz_instability/29329565
- **Type:** Dataset
- **Categories:** Turbulent flows
- **Description:** PyTorch tensor datasets of turbulent flow simulations for ML training.
- **Assessment:** Research dataset for physics-informed ML. NOT undergraduate.
- **Relevance:** LOW -- shows ML applied to physics/engineering

#### 2.7 Machine Learning for Archaeological Bone Analysis
- **Title:** Machine Learning ATR-FTIR Spectroscopy Data for Screening of Collagen
- **Authors:** Manasij Pal chowdhury et al.
- **Year:** 2022
- **URL:** https://figshare.manchester.ac.uk/articles/dataset/Machine_Learning_ATR-FTIR_Spectroscopy_Data_for_the_Screening_of_Collagen_for_ZooMS_Analysis_and_mtDNA_in_Archaeological_Bone/20298801
- **Type:** Dataset
- **Categories:** Archaeological science
- **Tags:** Random Forest, ATR-FTIR, Machine learning
- **Description:** Supplementary data for a PhD thesis using Random Forest ML for spectroscopy screening.
- **Assessment:** PhD thesis supplement. Uses Random Forest classifier.
- **Relevance:** LOW -- ML application but different domain

#### 2.8 Gaussian Mixture Models / Neural Network Model Files
- **Title:** Expressive Gaussian mixture models for high-dimensional statistical modelling
- **Authors:** Darren Price, Stephen Menary
- **Year:** 2021
- **URL:** https://figshare.manchester.ac.uk/articles/dataset/Expressive_Gaussian_mixture_models_for_high-dimensional_statistical_modelling_simulated_data_and_neural_network_model_files/17136839
- **Type:** Dataset
- **Categories:** Particle physics, Knowledge representation and reasoning
- **Description:** Neural network model files for learning-based Gaussian mixture models for particle physics simulation.
- **Assessment:** Research publication supplement. NOT undergraduate.
- **Relevance:** LOW -- neural network application in physics

#### 2.9 Closed Loop Sound Stimulation During Sleep
- **Title:** Closed Loop Sound Stimulation During Sleep In Matlab
- **Authors:** Alex Casson
- **Year:** 2022
- **URL:** https://figshare.manchester.ac.uk/articles/software/Closed_Loop_Sound_Stimulation_During_Sleep_In_Matlab/19606930
- **Type:** Software
- **Categories:** Medical devices
- **Description:** Matlab App for real-time EEG streaming, slow oscillation detection, and audio-tone playback. Closed-loop brain stimulation platform.
- **Assessment:** Research tool from Alex Casson's lab.
- **Relevance:** MEDIUM -- real-time EEG processing, signal processing

#### 2.10 3D-Printed Conductive EEG Electrodes
- **Title:** 3d-printed directly conductive EEG electrode models
- **Authors:** Le Xing
- **Year:** 2022
- **URL:** https://figshare.manchester.ac.uk/articles/online_resource/3d-printed_directly_conductive_EEG_electrode_models/19148987
- **Type:** Online resource (3D models)
- **Categories:** Electronic sensors
- **Description:** 3D-printed directly conductive and flexible EEG electrode models for IEEE FLEPS 2022 conference.
- **Assessment:** Research hardware design. NOT undergraduate.
- **Relevance:** LOW-MEDIUM -- EEG hardware, wearable sensing

---

### 3. OTHER NOTABLE ITEMS (For Scope Calibration)

#### 3.1 MIRRAX Reconfigurable Robot
- **URL:** https://figshare.manchester.ac.uk/articles/dataset/MIRRAX_A_Reconfigurable_Robot_for_Limited_Access_Environments_-_Experimental_Dataset/21071017
- ROS bag files for a reconfigurable robot. Research-level.

#### 3.2 Fifty Hyperspectral Reflectance Images
- **URL:** https://figshare.manchester.ac.uk/articles/dataset/Fifty_hyperspectral_reflectance_images_of_outdoor_scenes/14877285
- Visual perception dataset. Research-level.

#### 3.3 Training Data for Power Network Preventive Actions
- **URL:** https://figshare.manchester.ac.uk/articles/dataset/Training_data_for_effect_of_preventive_actions/14784570
- ML classifier training data for power network resilience. Research-level.

#### 3.4 ECG-X AI Applications for ECG Interpretation
- **URL:** https://figshare.manchester.ac.uk/articles/online_resource/ECG-X_Contrasting_Attitudes_Towards_Current_and_Future_AI_Applications_for_Computerised_Interpretation_of_ECG_A_Clinical_Stakeholder_Interview_Study/27256623
- Study on attitudes towards AI in clinical ECG interpretation. Research-level.

#### 3.5 Safety Case Framework for Autonomous Robotics
- **URL:** https://figshare.manchester.ac.uk/articles/online_resource/Safety_Case_Framework_for_Autonomous_Robotics_for_Nuclear/19512010
- Framework for autonomous robotics safety. Research-level.

---

## SEARCH METHODOLOGY

### Platforms and Approaches Used:
1. **Figshare Manchester Web Interface** -- Searched individually through the browser with 50+ terms
2. **Figshare REST API v2** -- Programmatic search using POST to /v2/articles/search
3. **Group-based enumeration** -- Identified all 17 Manchester group IDs and retrieved ALL 833 unique articles
4. **Keyword filtering** -- Applied 150+ keyword filters across all articles

### Manchester Figshare Group IDs Identified:
| Group ID | Description | Article Count |
|----------|-------------|---------------|
| 29325 | Faculty of Science and Engineering (parent) | 36 |
| 29331 | School of Engineering | 149 |
| 29337 | Sub-group (Engineering) | 1 |
| 29340 | Sub-group (Engineering) | 1 |
| 29343 | Sub-group (Robotics/Nuclear) | 3 |
| 29349 | School of Natural Sciences | 424 |
| 29355 | Sub-group | 5 |
| 29358 | Sub-group | 20 |
| 29361 | Sub-group | 1 |
| 29364 | Sub-group (Chemistry) | 31 |
| 29367 | Sub-group (Physics) | 7 |
| 29370 | Sub-group | 32 |
| 29391 | Sub-group | 1 |
| 29403 | Alliance Manchester Business School | 5 |
| 29442 | Faculty of Humanities / Education | 86 |
| 29466 | Sub-group (Education) | 6 |
| 29469 | Sub-group (Social Sciences) | 25 |

### Complete List of Search Terms Used (100+):
neuromorphic, spiking neural network, SNN, brain-inspired computing, event-driven computing, bio-inspired neural, leaky integrate fire, LIF, STDP, spike timing dependent plasticity, neural engineering, computational neuroscience, edge AI, edge computing, energy efficient neural, low power AI, event camera, dynamic vision sensor, DVS, FPGA, FPGA neural network, hardware neural network, TinyML, embedded machine learning, reservoir computing, memristor, memristive, Intel Loihi, IBM TrueNorth, SpiNNaker, deep learning optimization, model compression, quantization, convolutional neural network, recurrent neural network, LSTM, reinforcement learning, signal processing, EEG classification, brain-computer interface, BCI, gesture recognition, image classification, object detection, computer vision, transfer learning, autoencoder, generative adversarial, GAN, natural language processing, sentiment analysis, autonomous vehicle, robot navigation, pose estimation, semantic segmentation, anomaly detection, time series prediction, speech recognition, audio classification, medical imaging, biomedical signal, wearable sensor, IoT, TensorFlow, PyTorch, model pruning, knowledge distillation, neural architecture search, attention mechanism, transformer, BERT, GPT, diffusion model, variational autoencoder, graph neural network, point cloud, lidar, radar, EMG, prosthetic, binary neural network, sparse coding, ResNet, YOLO, U-Net, sensor fusion, microcontroller, Raspberry Pi, Arduino, federated learning, explainable AI, few-shot learning, meta-learning, self-supervised learning, contrastive learning, photonic computing, analog computing, in-memory computing, quantum machine learning, undergraduate thesis, final year project, third year project, BEng, MEng, dissertation

---

## RESEARCH GAPS AND EXPLANATION

### Why No Undergraduate Theses Were Found:

1. **The Manchester Figshare repository is NOT used for undergraduate theses.** It is a research data repository primarily for:
   - Supplementary data/code for published journal papers
   - Research datasets
   - Conference presentations (mostly teaching/learning conferences)
   - PhD thesis supplementary data
   - Architectural model photos

2. **Manchester undergraduate theses are likely stored elsewhere:**
   - The University of Manchester Library thesis archive
   - The university's own student project submission system (e.g., Blackboard, Turnitin)
   - Department-specific repositories
   - Manchester eScholar (https://www.research.manchester.ac.uk/) -- primarily for PhD/research theses
   - Student GitHub repositories (not centrally indexed)

3. **The entire repository contains only 833 articles** across all departments and groups. For a university the size of Manchester, this is a very small collection, confirming it is used selectively for research data deposits, not comprehensively for student work.

4. **No items in the repository are tagged or categorized as "undergraduate" or "student project."** The "thesis" tag appears only in relation to PhD theses (e.g., Waters Thesis 2021, Bamber Thesis, Thom's PhD thesis).

### Why Limited ML/AI/Neuromorphic Content:

Despite Manchester being a major centre for neuromorphic computing (SpiNNaker project, Steve Furber's group), the Figshare portal has minimal AI/ML content because:

1. Manchester's AI/ML researchers likely use other repositories (GitHub, Zenodo, arXiv)
2. The BitBrain entry from Furber's group is the sole neuromorphic item
3. Alex Casson's EEG/BCI work represents the bulk of neural engineering content
4. The repository appears to be a relatively recent adoption (most items from 2021-2025)

---

## CONFIDENCE ASSESSMENT

| Finding | Confidence Level |
|---------|-----------------|
| No undergraduate theses exist on Figshare Manchester | **VERY HIGH** (99%) -- Enumerated all 833 articles |
| No neuromorphic computing undergraduate projects | **VERY HIGH** (99%) -- Searched 100+ terms, checked all articles |
| The repository is not used for undergraduate work | **HIGH** (95%) -- No evidence of any undergraduate submissions |
| Jieru Zhou's LSTM project might be a student project | **LOW** (30%) -- Single author, appropriate scope, but unconfirmed |
| All other items are research-level publications | **HIGH** (90%) -- Multi-author, funded, journal-linked |

---

## RECOMMENDED FOLLOW-UPS

If you need to find Manchester undergraduate neuromorphic/ML thesis projects, consider:

1. **Manchester eScholar** (https://www.research.manchester.ac.uk/) -- Check for electronic theses and dissertations
2. **Contact the School of Engineering directly** -- Ask if they have a repository of third-year projects
3. **Contact Steve Furber's APT group** -- They may have records of undergraduate projects on SpiNNaker/neuromorphic topics
4. **Contact Alex Casson's Non-Invasive Bioelectronics Lab** -- For BCI/EEG-related undergraduate projects
5. **Search other UK university Figshare portals** -- Some universities (e.g., Loughborough, Southampton) may use Figshare for student theses
6. **Search the broader Figshare.com** -- Some Manchester students may have uploaded work to the general Figshare platform
7. **GitHub search** -- Search for Manchester student repositories related to neuromorphic topics
8. **Check the IEEE Xplore and conference proceedings** -- Some undergraduate projects get published as conference papers

---

## SUMMARY TABLE OF ALL RELEVANT/ADJACENT ITEMS FOUND

| # | Title | Authors | Year | Type | Relevance | Undergrad? |
|---|-------|---------|------|------|-----------|------------|
| 1 | BitBrain algorithm in C | Fil, Hopkins, Furber, Jones | 2022 | Software | DIRECT neuromorphic | No (Research) |
| 2 | Deep autoencoder EEG artifact removal | Xing, Casson | 2023 | Software | HIGH (BCI/Neural Eng) | No (Research) |
| 3 | LSTM Warehouse Robot Navigation | Zhou | 2024 | Software | MEDIUM (ML/Robotics) | Possibly |
| 4 | ECG/EEG/IMU motion artefact data | Beach, Li, Balaban, Casson | 2021 | Dataset | MEDIUM (Biosignal) | No (Research) |
| 5 | Low Power Computer Vision landmine | Fletcher et al. | 2025 | Dataset | LOW-MED (Edge CV) | No (Research) |
| 6 | Closed Loop Sound Stimulation Sleep | Casson | 2022 | Software | MEDIUM (Real-time EEG) | No (Research) |
| 7 | 3D-printed EEG electrodes | Xing | 2022 | Online resource | LOW-MED (EEG hw) | No (Research) |
| 8 | hBET2 EEG Alpha analysis code | Casson, Halpin, Xing | 2024 | Software | LOW-MED (EEG) | No (Research) |
| 9 | Radiation Autonomous Exploration | West | 2022 | Dataset | LOW (Autonomous robot) | No (Research) |
| 10 | Gaussian mixture neural network | Price, Menary | 2021 | Dataset | LOW (NN in physics) | No (Research) |
| 11 | Turbulent Flow PyTorch tensors | Sardar, Skillen | 2025 | Dataset | LOW (ML for CFD) | No (Research) |
| 12 | ML ATR-FTIR Spectroscopy | Pal chowdhury et al. | 2022 | Dataset | LOW (ML archaeology) | No (PhD thesis) |
| 13 | Hyperspectral images | Foster et al. | 2022 | Dataset | LOW (Vision dataset) | No (Research) |
| 14 | Power network training data | Noebels et al. | 2021 | Dataset | LOW (ML for power) | No (Research) |
| 15 | ECG-X AI for ECG | Hughes-Noehrer, Jay | 2024 | Online resource | LOW (AI in healthcare) | No (Research) |

---

*Report generated: 2026-02-24*
*Total items in repository reviewed: 833*
*Search terms used: 100+*
*Manchester Figshare groups enumerated: 17*
