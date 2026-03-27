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
- Andrew West, 2022
- https://figshare.manchester.ac.uk/articles/dataset/Radiation_Exposure_Reduction_During_Autonomous_Exploration/18782165
- Autonomous exploration sessions in simulated environment with gamma radiation sources. Uses frontier exploration with radiation avoidance. ROS, Gazebo, UGV.
- Part of the RAIN (Robotics and AI for Nuclear) project at Manchester

#### Turbulent Flow Data as PyTorch Tensors for ML
- Mohammed Sardar, Alex Skillen, 2025
- https://figshare.manchester.ac.uk/articles/dataset/Turbulent_Flow_data_as_PyTorch_tensors_for_ML_Kolmogorov_Flow_at_Re_222_and_Kelvin-Helmholtz_instability/29329565
- PyTorch tensor datasets of turbulent flow simulations for ML training

#### ML for Archaeological Bone Analysis
- Manasij Pal chowdhury et al., 2022
- https://figshare.manchester.ac.uk/articles/dataset/Machine_Learning_ATR-FTIR_Spectroscopy_Data_for_the_Screening_of_Collagen_for_ZooMS_Analysis_and_mtDNA_in_Archaeological_Bone/20298801
- PhD thesis supplement using Random Forest classifier for spectroscopy screening

#### Gaussian Mixture Models / NN Model Files
- Darren Price, Stephen Menary, 2021
- https://figshare.manchester.ac.uk/articles/dataset/Expressive_Gaussian_mixture_models_for_high-dimensional_statistical_modelling_simulated_data_and_neural_network_model_files/17136839
- Neural network model files for learning-based GMMs for particle physics

#### Closed Loop Sound Stimulation During Sleep
- Alex Casson, 2022
- https://figshare.manchester.ac.uk/articles/software/Closed_Loop_Sound_Stimulation_During_Sleep_In_Matlab/19606930
- Matlab App for real-time EEG streaming, slow oscillation detection, and audio-tone playback

#### 3D-Printed Conductive EEG Electrodes
- Le Xing, 2022
- https://figshare.manchester.ac.uk/articles/online_resource/3d-printed_directly_conductive_EEG_electrode_models/19148987
- 3D-printed directly conductive and flexible EEG electrode models for IEEE FLEPS 2022

---

### Other Stuff (for scope calibration)

- MIRRAX Reconfigurable Robot -- ROS bag files, research-level
- Fifty Hyperspectral Reflectance Images -- visual perception dataset, research-level
- Training Data for Power Network Preventive Actions -- ML classifier training data, research-level
- ECG-X AI Applications for ECG Interpretation -- attitudes towards AI in clinical ECG, research-level
- Safety Case Framework for Autonomous Robotics -- framework doc, research-level

---

## How i searched

1. **Figshare Manchester Web Interface** -- searched through the browser with 50+ terms
2. **Figshare REST API v2** -- programmatic search using POST to /v2/articles/search
3. **Group-based enumeration** -- identified all 17 Manchester group IDs and pulled ALL 833 unique articles
4. **Keyword filtering** -- applied 150+ keyword filters across everything

### Manchester Figshare Group IDs

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

### All the search terms i used (100+):
neuromorphic, spiking neural network, SNN, brain-inspired computing, event-driven computing, bio-inspired neural, leaky integrate fire, LIF, STDP, spike timing dependent plasticity, neural engineering, computational neuroscience, edge AI, edge computing, energy efficient neural, low power AI, event camera, dynamic vision sensor, DVS, FPGA, FPGA neural network, hardware neural network, TinyML, embedded machine learning, reservoir computing, memristor, memristive, Intel Loihi, IBM TrueNorth, SpiNNaker, deep learning optimization, model compression, quantization, convolutional neural network, recurrent neural network, LSTM, reinforcement learning, signal processing, EEG classification, brain-computer interface, BCI, gesture recognition, image classification, object detection, computer vision, transfer learning, autoencoder, generative adversarial, GAN, natural language processing, sentiment analysis, autonomous vehicle, robot navigation, pose estimation, semantic segmentation, anomaly detection, time series prediction, speech recognition, audio classification, medical imaging, biomedical signal, wearable sensor, IoT, TensorFlow, PyTorch, model pruning, knowledge distillation, neural architecture search, attention mechanism, transformer, BERT, GPT, diffusion model, variational autoencoder, graph neural network, point cloud, lidar, radar, EMG, prosthetic, binary neural network, sparse coding, ResNet, YOLO, U-Net, sensor fusion, microcontroller, Raspberry Pi, Arduino, federated learning, explainable AI, few-shot learning, meta-learning, self-supervised learning, contrastive learning, photonic computing, analog computing, in-memory computing, quantum machine learning, undergraduate thesis, final year project, third year project, BEng, MEng, dissertation

---

## Why there's nothing here

### Why no undergrad theses?

1. **Manchester Figshare just isn't used for undergrad theses.** It's a research data repository for supplementary data/code, research datasets, conference presentations, PhD supplements, and architectural model photos.

2. **Manchester undergrad theses are probably stored elsewhere:**
   - University of Manchester Library thesis archive
   - University's student submission system (Blackboard, Turnitin)
   - Department-specific repositories
   - Manchester eScholar (https://www.research.manchester.ac.uk/) -- mostly PhD/research theses
   - Student GitHub repos (not centrally indexed)

3. **The entire repository only has 833 articles** across ALL departments. For a university Manchester's size, that's tiny -- confirms it's used selectively for research deposits, not broadly for student work.

4. **Nothing is tagged as "undergraduate" or "student project."** The "thesis" tag only appears for PhD theses.

### Why so little ML/AI/neuromorphic content?

Despite Manchester being a major centre for neuromorphic computing (SpiNNaker project, Steve Furber's group), the Figshare portal has minimal AI/ML content because:

1. Manchester's AI/ML researchers probably use other repositories (GitHub, Zenodo, arXiv)
2. The BitBrain entry is the only neuromorphic item
3. Alex Casson's EEG/BCI work is the bulk of neural engineering content
4. The repository seems to be a relatively recent adoption (most items from 2021-2025)

---

## How confident am i

| Finding | Confidence |
|---------|-----------|
| No undergrad theses exist on Figshare Manchester | Very high (99%) -- went through all 833 articles |
| No neuromorphic undergrad projects | Very high (99%) -- searched 100+ terms, checked everything |
| The repository isn't used for undergrad work | High (95%) -- no evidence of any undergrad submissions |
| Jieru Zhou's LSTM project might be a student project | Low (30%) -- single author, right scope, but unconfirmed |
| Everything else is research-level | High (90%) -- multi-author, funded, journal-linked |

---

## Where else to look

If i need to find Manchester undergrad neuromorphic/ML thesis projects:

1. **Manchester eScholar** (https://www.research.manchester.ac.uk/) -- electronic theses and dissertations
2. **Contact School of Engineering** -- ask about a repository of third-year projects
3. **Contact Steve Furber's APT group** -- might have records of undergrad projects on SpiNNaker/neuromorphic
4. **Contact Alex Casson's lab** -- for BCI/EEG undergrad projects
5. **Search other UK university Figshare portals** -- some unis (Loughborough, Southampton) may use Figshare for student theses
6. **Search broader Figshare.com** -- students might have uploaded to the general platform
7. **GitHub search** -- look for Manchester student repos on neuromorphic topics
8. **IEEE Xplore / conference proceedings** -- some undergrad projects get published as conference papers

---

## Summary table of everything found

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
