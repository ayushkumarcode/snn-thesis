# SNN for Robot Reflexes in Simulation: Component Verification

checked on 2026-02-27. verdict: feasible on macOS with caveats -- see details below.

---

## first question: can this whole project run on macOS without an NVIDIA GPU?

**yes, with the right stack choices. but NOT with the originally-proposed Isaac Gym.**

| Component | macOS Compatible? | Notes |
|-----------|------------------|-------|
| MuJoCo (physics sim) | YES | Native Apple Silicon support, fast on M-series chips |
| Gymnasium (RL environments) | YES | Officially tested on macOS, Python 3.10-3.13 |
| snnTorch (SNN library) | YES | Pure PyTorch, works wherever PyTorch works |
| PyTorch + MPS | YES | GPU acceleration via Metal Performance Shaders on Apple Silicon |
| Stable-Baselines3 (RL algorithms) | YES | CPU-based PPO works fine on macOS |
| Isaac Gym | **NO** | Linux + NVIDIA GPU only. HARD BLOCKER on macOS |
| SpikeGym | **PARTIAL** | Depends on Isaac Gym for full GPU pipeline; MuJoCo mode may work |

**Recommended macOS-native stack:** MuJoCo + Gymnasium + snnTorch + Stable-Baselines3 (or CleanRL)

**Cloud fallback for faster training:** Google Colab (free T4 GPU, 30 hrs/week) or Kaggle (free P100, 30 hrs/week)

---

## Component-by-Component Verification

---

### 1. SpikeGym

| Field | Value |
|-------|-------|
| **EXISTS** | YES -- but barely |
| **URL** | https://gitlab.com/ecs-lab/spikegym |
| **Stars** | Not displayed (GitLab); only 3 commits, 1 branch, 0 tags |
| **Last commit** | Created May 2024; only 3 commits total |
| **Maintained?** | BARELY -- 3 commits is extremely low activity |
| **What it is** | Modifications to the `skrl` RL library to support SNN policy networks with PPO |
| **Environments** | Cartpole and Ant (via Isaac Gym and MuJoCo) |
| **Paper** | "Exploring spiking neural networks for deep reinforcement learning in robotic tasks" -- Scientific Reports, Dec 2024 |
| **POTENTIAL BLOCKER** | **YES (MEDIUM-HIGH)** |

**Why it is a blocker:** SpikeGym has only 3 commits. It is a research artifact from a paper, not a maintained library. It was built on top of Isaac Gym (Linux/NVIDIA only) primarily, with MuJoCo as secondary. Using it on macOS would require significant adaptation. You should NOT depend on this as a core tool -- instead, treat it as reference code and build your own integration.

**Source:** [Scientific Reports paper](https://www.nature.com/articles/s41598-024-77779-8) | [GitLab repo](https://gitlab.com/ecs-lab/spikegym)

---

### 2. Isaac Gym

| Field | Value |
|-------|-------|
| **EXISTS** | YES -- but DEPRECATED |
| **URL** | https://developer.nvidia.com/isaac-gym |
