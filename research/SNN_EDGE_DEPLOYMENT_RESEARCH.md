# SNN Edge Deployment Research: Deploying Spiking Neural Networks on Real Hardware

**Research Date:** 2026-02-25
**Scope:** Comprehensive investigation into deploying SNNs on edge devices, FPGAs, microcontrollers, and accessible hardware for an undergraduate thesis project.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Can SNNs Be Deployed on Real Hardware?](#2-can-snns-be-deployed-on-real-hardware)
3. [Frameworks Supporting Edge Deployment](#3-frameworks-supporting-edge-deployment)
4. [Power Consumption: SNN vs ANN on Edge Devices](#4-power-consumption-snn-vs-ann-on-edge-devices)
5. [Student Projects That Have Done This](#5-student-projects-that-have-done-this)
6. [FPGA-Based SNN Deployment for Undergraduates](#6-fpga-based-snn-deployment-for-undergraduates)
7. [Raspberry Pi and Accessible Hardware](#7-raspberry-pi-and-accessible-hardware)
8. [Recent Papers and Projects (2023-2026)](#8-recent-papers-and-projects-2023-2026)
9. [Concrete Hardware Cost Breakdown](#9-concrete-hardware-cost-breakdown)
10. [Feasibility Assessment and Recommended Project Paths](#10-feasibility-assessment-and-recommended-project-paths)
11. [Sources](#11-sources)

---

## 1. Executive Summary

Deploying spiking neural networks on real edge hardware is not only feasible but is an active and growing area of research with multiple proven pathways accessible to undergraduate students. The research reveals three tiers of deployment difficulty, all of which are achievable within a thesis project timeline:

**Tier 1 -- Software SNN on CPU (Easiest, 2-4 weeks to deploy):** Train an SNN using snnTorch or SpikingJelly in Python, then convert the trained model to optimized C code and run inference on a microcontroller (STM32, Arduino Portenta H7) or Raspberry Pi. This path has been demonstrated with concrete benchmarks: a C runtime achieves 21x speedup over Python snnTorch, and compiled models fit within 250 KB of SRAM.

**Tier 2 -- FPGA Accelerated SNN (Moderate, 6-10 weeks):** Train an SNN in snnTorch, apply quantization-aware training, and deploy to an FPGA using HLS (High-Level Synthesis). The open-neuromorphic/fpga-snntorch workshop from ISFPGA 2024 provides a complete pipeline for deploying on the AMD Kria KV260 ($199). The Spiker+ framework can auto-generate VHDL from a Python description, consuming only 180 mW per inference on small FPGAs.

**Tier 3 -- Neuromorphic Hardware (Advanced but possible):** Deploy via NIR (Neuromorphic Intermediate Representation) to SpiNNaker2, SynSense Xylo/Speck, or Intel Loihi 2. The NIR standard now connects 7 simulators to 4+ hardware platforms, and SynSense claims interns can deploy their first application within 1-2 months.

The strongest undergraduate thesis angle would combine Tier 1 and Tier 2: train an SNN for DVS gesture recognition using snnTorch, deploy it on both a microcontroller (C runtime) and an FPGA (HLS pipeline), then measure and compare power, latency, and accuracy across platforms. This produces a highly demonstrable project with concrete, measurable results.

---

## 2. Can SNNs Be Deployed on Real Hardware?

### 2.1 Yes -- Verified Across Multiple Platforms

SNNs have been successfully deployed on the following hardware categories, with published results:

**Microcontrollers (ARM Cortex-M):**
- STM32F407VG6 (Cortex-M4, 168 MHz, 192 KB RAM): eLSNN implementation achieving 54% lower execution time vs naive implementation
