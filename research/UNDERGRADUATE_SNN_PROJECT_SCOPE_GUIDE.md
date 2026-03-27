# Undergraduate SNN/Neuromorphic Project Scope Guide

> Compiled from 4 parallel research investigations across Manchester eScholar, GitHub, UK university repositories, and the SpiNNaker/APT group.
> **Date:** 2026-02-24

---

## TL;DR

- **UK universities do NOT publicly archive undergraduate dissertations.** Only PhD/MPhil theses appear in institutional repositories.
- **GitHub is the primary way to find undergraduate SNN work.** 3 confirmed BSc/BEng projects found, plus 15+ masters theses.
- **Heidelberg University (BrainScaleS group)** publicly archives 60+ bachelor's theses on neuromorphic hardware -- the best resource for scope calibration.
- **Manchester's SpiNNaker group** has 16+ PhD theses (all freely downloadable) but no public undergrad work. They almost certainly supervise 3rd-year projects but these are behind authentication.
- **A realistic 3rd-year SNN project:** 1-2 SNN architectures, 1-2 datasets, comparison with ANN baseline, energy analysis. Deliverable: Jupyter notebooks + 40-80 page report.

---

## Part 1: Confirmed Undergraduate (BSc/BEng) Projects

Only 3 confirmed undergraduate SNN projects were found publicly:

### 1. Shape Detector SNN -- Filippo Ferrari (Manchester, ~2018, BSc AI)
- **Supervisor:** Prof. Steve Furber
- **What:** Single-layer SNN for shape detection
- **Tools:** Python, pyDVS library
- **Code:** https://github.com/filippoferrari/shape_detector_snn
- **Thesis source:** https://github.com/filippoferrari/bsc_dissertation (LaTeX)
- **Scope:** MODERATE -- well-structured with tests, CI, configs. 107 commits.
- **Takeaway:** Good example of a BSc project at a top university. Clean code, proper testing. Supervised by the creator of SpiNNaker.

### 2. Musical Pattern Recognition in SNNs -- mrahtz (~2016, BEng)
- **What:** First layer of a multi-layer SNN for recognising musical patterns in audio
- **Tools:** Brian 2 simulator, STDP learning, custom .wav audio
- **Code:** https://github.com/mrahtz/musical-pattern-recognition-in-spiking-neural-networks
- **Thesis PDF:** http://amid.fish/beng_project_report.pdf
- **Scope:** MODERATE -- 49 stars, 17 forks. Author candidly notes "only a small portion of what was originally intended was actually achieved"
- **Takeaway:** Novel application domain (music). High community interest. Honest about limitations -- this is realistic and expected for a final-year project.

### 3. Randomised Time-Stepping Methods for SNN Simulations -- Fabio Deo (Imperial, 2021)
- **What:** Mathematical investigation of randomness in SNN time-stepping methods
- **Tools:** Python (98.9% of repo)
- **Code:** https://github.com/Fabio752/Randomised-time-stepping-methods-for-SNN-simulations
- **Thesis PDF:** Included in repository
- **Scope:** MODERATE -- theoretical/computational, appropriate for Imperial

### Other Noteworthy Undergraduate Projects (not SNN-specific but adjacent):

| Project | University | Tools | What |
|---------|-----------|-------|------|
| SNN for Digit Recognition (C++) | King's College London, 2018 | C++, OpenCV | From-scratch SNN, no framework |
| SNN for Autonomous Locomotion | Unknown, bachelor thesis | V-REP, ROS, R-STDP | Robot following red object (did not converge -- honest about failure) |
| Spiking Stereo Matching | Unknown, bachelor | SpiNNaker, sPyNNaker | Event-based stereo matching, 2ms latency, published paper |
| QuadBot Neuromorphic | Cambridge (summer project) | MATLAB, VEX robots | CPG-based quadrupedal locomotion |
| SNN NoC Architecture | Sri Lanka (4th year) | Verilog, FPGA, RISC-V | Team of 3, custom hardware |

---

## Part 2: Masters Thesis Projects (Closer Scope Reference)

Masters projects are more ambitious than undergrad but give a useful upper bound.
