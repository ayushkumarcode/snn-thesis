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
