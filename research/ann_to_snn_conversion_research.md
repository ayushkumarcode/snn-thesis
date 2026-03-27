# ANN-to-SNN Conversion: Comprehensive Research Report for Undergraduate Thesis Direction

**Research Date:** 2026-02-25
**Scope:** Evaluating ANN-to-SNN conversion as a practical and contributory undergraduate thesis direction

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [State of the Art (2024-2026)](#2-state-of-the-art-2024-2026)
3. [Available Tools and Frameworks](#3-available-tools-and-frameworks)
4. [Accuracy Loss During Conversion](#4-accuracy-loss-during-conversion)
5. [Which Architectures Convert Best](#5-which-architectures-convert-best)
6. [Timestep Requirements](#6-timestep-requirements)
7. [Undergraduate Contribution Opportunities](#7-undergraduate-contribution-opportunities)
8. [Recent Papers with Reproducible Code](#8-recent-papers-with-reproducible-code)
9. [Time to Get a Working Pipeline](#9-time-to-get-a-working-pipeline)
10. [Thesis Framing Recommendations](#10-thesis-framing-recommendations)
11. [Consolidated Accuracy Tables](#11-consolidated-accuracy-tables)
12. [Research Gaps and Open Problems](#12-research-gaps-and-open-problems)
13. [Risk Assessment](#13-risk-assessment)
14. [Sources](#14-sources)

---

## 1. Executive Summary

ANN-to-SNN conversion is one of the two dominant methods for building deep spiking neural networks (the other being direct training with surrogate gradients). The core idea is straightforward: take a pre-trained artificial neural network, replace ReLU activations with integrate-and-fire spiking neurons, normalize thresholds, and run inference where spike rates encode activation values. This is the **most cost-effective** method for obtaining high-accuracy SNNs because it leverages the mature ANN training ecosystem.
