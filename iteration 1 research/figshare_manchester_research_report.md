# Figshare Manchester Repository: Neuromorphic Computing and Adjacent Topics

searched through the University of Manchester's Figshare institutional repository (https://figshare.manchester.ac.uk/) on 2026-02-24, looking through all 17 Manchester group IDs, reviewing 833 unique articles, running 100+ search terms spanning neuromorphic computing, spiking neural networks, brain-inspired computing, FPGA implementations, edge AI, BCI/EEG, machine learning, deep learning, computer vision, robotics, etc.

**the short version: there's basically nothing here for what i need.** there are no undergraduate third-year thesis projects on Figshare Manchester related to neuromorphic computing or anything close. actually, the repository doesn't seem to host undergrad theses at all. it's mostly used for research datasets supporting papers, software/code supplements, conference presentations (mostly about teaching/learning), PhD thesis supplementary data, and architectural models.

zero results for all of these core terms:
- "spiking neural network", SpiNNaker, FPGA, memristor, STDP, LIF, reservoir computing, TinyML, Intel Loihi, IBM TrueNorth, event camera, DVS, neuromorphic engineering, model compression, brain-computer interface, gesture recognition, image classification, object detection, edge AI, embedded ML

the one match for "neuromorphic" is a research-level publication from Steve Furber's SpiNNaker group, not a student project.

---

## the one directly neuromorphic item

### BitBrain and Sparse Binary Coincidence (SBC) Memories
- **Title:** Implementation of BitBrain algorithm in C
- **Authors:** Jakub Fil, Michael Hopkins, Steve Furber, Edward Jones
- **Year:** 2022
- **URL:** https://figshare.manchester.ac.uk/articles/software/Implementation_of_BitBrain_algorithm_in_C/21679610
- **DOI:** https://doi.org/10.48420/21679610
- **Type:** Software (C code implementation)
- **Tags:** neuromorphic, single-pass learning, efficient inference, classification, machine learning, robust, event-based, IoT
- **Description:** Code for a working mechanism (SBC memory) and surrounding infrastructure (BitBrain) based on ideas from sparse coding, computational neuroscience and information theory. Supports single-pass and single-shot learning, accurate and robust inference, and the potential for continuous adaptive learning. Designed for implementation on neuromorphic devices and conventional CPUs.
- **Funding:** Human Brain Project Specific Grant Agreement 3 (H2020 945539)
- this is a research-level publication supplement by members of the APT group -- Steve Furber's team. not an undergrad project. it's a supplement for "Frontiers in Neuroinformatics: Physical Neuromorphic Computing and its Industrial Applications." but it does show what Manchester researchers are doing in neuromorphic computing.

---

## closest adjacent ML/AI/neural engineering items

### Deep Autoencoder for Real-Time EEG Artifact Removal
- **Authors:** Le Xing, Alex Casson (2023)
- **URL:** https://figshare.manchester.ac.uk/articles/software/Deep_autoencoder_for_real-time_EEG_artifact_removal_and_its_Android_smartphone_implementation/23093759
- **Type:** Software (GitHub linked)
- **Description:** A Deep Convolutional Autoencoder for single-channel EEG artifact removal in real-time. Implemented as Android app for mobile EEG and portable BCI applications.
- research publication by Alex Casson's Non-Invasive Bioelectronics Lab. not an undergrad project.

### LSTM Robot Navigation in Warehouse Environments
- **Author:** Jieru Zhou (2024)
- **URL:** https://figshare.manchester.ac.uk/articles/software/Optimizing_Warehouse_Robot_Navigation_with_LSTM_Networks_and_Adaptive_Greedy_Weight_Techniques/26870128
- **Type:** Software (Python code, 9 kB)
- **Description:** Combines LSTM with adaptive greedy weight strategies for path planning in dynamic warehouse environments.
- single author, School of Engineering. **possibly a student project** based on single authorship and scope, but can't confirm if undergraduate. the code is only 9 kB, so it's pretty focused.

### ECG, EEG and IMU Data for Motion Artefact Removal
- **Authors:** Christopher Beach, Mingjie Li, Ertan Balaban, Alex Casson (2021)
- **URL:** https://figshare.manchester.ac.uk/articles/dataset/ECG_EEG_and_IMU_data_for_local_motion_artefact_removal/13626395
- **Type:** Dataset
- research dataset from Casson's lab.

### Low Power Computer Vision for Landmine Detection
- **Authors:** Adam Fletcher et al. (2025)
- **URL:** https://figshare.manchester.ac.uk/articles/dataset/Evaluation_of_a_Low_Power_Computer_Vision_Based_Positioning_System_for_a_Handheld_Landmine_Detector_using_AprilTag_Markers_-_Supporting_Data/30030226
- **Type:** Dataset
- research publication, tangentially related through low power + computer vision.

### Autonomous Robotics for Nuclear Environments
- **Author:** Andrew West (2022)
- **URL:** https://figshare.manchester.ac.uk/articles/dataset/Radiation_Exposure_Reduction_During_Autonomous_Exploration/18782165
- **Type:** Dataset
- related to the RAIN (Robotics and Artificial Intelligence for Nuclear) project at Manchester.

### Turbulent Flow Data as PyTorch Tensors for ML
- **Authors:** Mohammed Sardar, Alex Skillen (2025)
- **URL:** https://figshare.manchester.ac.uk/articles/dataset/Turbulent_Flow_data_as_PyTorch_tensors_for_ML_Kolmogorov_Flow_at_Re_222_and_Kelvin-Helmholtz_instability/29329565
- PyTorch tensor datasets of turbulent flow simulations. ML applied to physics, not really relevant.

### Other items found
- Machine Learning ATR-FTIR Spectroscopy Data (2022) -- Random Forest for archaeology, PhD thesis supplement
- Gaussian Mixture Models / Neural Network Model Files (2021) -- particle physics, research-level
- Closed Loop Sound Stimulation During Sleep (2022) -- real-time EEG processing, Casson's lab
- 3D-Printed Conductive EEG Electrodes (2022) -- EEG hardware design, research-level

---

## why i found nothing useful

### why no undergrad theses exist here

1. **Manchester Figshare is not used for undergrad theses.** it's a research data repository for supplementary data/code for published papers, research datasets, conference presentations, PhD thesis supplementary data, and architectural model photos.

2. **Manchester undergrad theses are stored elsewhere** -- probably the university library thesis archive, Blackboard/Turnitin submission system, department-specific repos, Manchester eScholar (mostly PhD/research theses), or student GitHub repos (not centrally indexed).

3. **the entire repository only has 833 articles** across all departments and groups. for a university Manchester's size, that's tiny, confirming it's used selectively for research data, not broadly for student work.

4. **no items are tagged as "undergraduate" or "student project."** the "thesis" tag only appears in relation to PhD theses.

### why limited ML/AI/neuromorphic content

despite Manchester being a major centre for neuromorphic computing (SpiNNaker, Furber's group), Figshare has minimal AI/ML content because:
- Manchester's AI/ML researchers probably use GitHub, Zenodo, arXiv instead
- the BitBrain entry from Furber's group is the only neuromorphic item
- Alex Casson's EEG/BCI work is most of the neural engineering content
- the repository seems to be a relatively recent adoption (most items from 2021-2025)

---

## Manchester Figshare group IDs

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

---

## summary of everything found

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
