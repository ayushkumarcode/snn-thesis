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
