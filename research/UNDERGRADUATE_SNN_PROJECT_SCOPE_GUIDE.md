# Undergraduate SNN/Neuromorphic Project Scope Guide

Compiled from 4 parallel research dives across Manchester eScholar, GitHub, UK university repositories, and the SpiNNaker/APT group.

Quick summary of what i found:

- **UK universities do NOT publicly archive undergrad dissertations.** Only PhD/MPhil theses show up in institutional repositories.
- **GitHub is the main way to find undergrad SNN work.** 3 confirmed BSc/BEng projects found, plus 15+ masters theses.
- **Heidelberg University (BrainScaleS group)** publicly archives 60+ bachelor's theses on neuromorphic hardware -- best resource for scope calibration.
- **Manchester's SpiNNaker group** has 16+ PhD theses (all freely downloadable) but no public undergrad work. They almost certainly supervise 3rd-year projects but those are behind authentication.
- **A realistic 3rd-year SNN project:** 1-2 SNN architectures, 1-2 datasets, comparison with ANN baseline, energy analysis. Deliverable: Jupyter notebooks + 40-80 page report.

---

## Confirmed Undergraduate (BSc/BEng) Projects

Only 3 confirmed undergrad SNN projects found publicly:

### 1. Shape Detector SNN -- Filippo Ferrari (Manchester, ~2018, BSc AI)
- Supervised by Steve Furber
- Single-layer SNN for shape detection
- Python, pyDVS library
- Code: https://github.com/filippoferrari/shape_detector_snn
- Thesis source: https://github.com/filippoferrari/bsc_dissertation (LaTeX)
- Well-structured with tests, CI, configs. 107 commits.
- Good example of a BSc project at a top university. Clean code, proper testing, supervised by the creator of SpiNNaker.

### 2. Musical Pattern Recognition in SNNs -- mrahtz (~2016, BEng)
- First layer of a multi-layer SNN for recognising musical patterns in audio
- Brian 2 simulator, STDP learning, custom .wav audio
- Code: https://github.com/mrahtz/musical-pattern-recognition-in-spiking-neural-networks
- Thesis PDF: http://amid.fish/beng_project_report.pdf
- 49 stars, 17 forks. Author says "only a small portion of what was originally intended was actually achieved" -- honest about limitations, which is realistic and expected.

### 3. Randomised Time-Stepping Methods for SNN Simulations -- Fabio Deo (Imperial, 2021)
- Mathematical investigation of randomness in SNN time-stepping methods
- Python (98.9% of repo)
- Code + full thesis PDF: https://github.com/Fabio752/Randomised-time-stepping-methods-for-SNN-simulations
- Theoretical/computational focus, appropriate for Imperial

### Other noteworthy undergrad projects (not SNN-specific but adjacent):

| Project | University | Tools | What |
|---------|-----------|-------|------|
| SNN for Digit Recognition (C++) | King's College London, 2018 | C++, OpenCV | From-scratch SNN, no framework |
| SNN for Autonomous Locomotion | Unknown, bachelor thesis | V-REP, ROS, R-STDP | Robot following red object (didn't converge -- honest about failure) |
| Spiking Stereo Matching | Unknown, bachelor | SpiNNaker, sPyNNaker | Event-based stereo matching, 2ms latency, published paper |
| QuadBot Neuromorphic | Cambridge (summer project) | MATLAB, VEX robots | CPG-based quadrupedal locomotion |
| SNN NoC Architecture | Sri Lanka (4th year) | Verilog, FPGA, RISC-V | Team of 3, custom hardware |

---

## Masters Thesis Projects (useful as an upper bound on scope)

| Project | Tools | Results | Thesis PDF? |
|---------|-------|---------|-------------|
