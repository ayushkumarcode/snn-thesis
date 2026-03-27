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
