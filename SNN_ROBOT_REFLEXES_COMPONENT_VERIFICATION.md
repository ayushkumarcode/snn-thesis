# SNN for Robot Reflexes in Simulation: Component Verification Report

**Date:** 2026-02-27
**Purpose:** Exhaustive verification of every technical component for the thesis project
**Verdict:** FEASIBLE ON macOS WITH CAVEATS -- see details below

---

## CRITICAL ANSWER FIRST: Can This Entire Project Run on macOS Without an NVIDIA GPU?

**YES, with the right stack choices. But NOT with the originally-proposed Isaac Gym.**

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
| **Free?** | Yes, after accepting NVIDIA license |
| **NVIDIA GPU Required?** | **YES -- ABSOLUTELY REQUIRED** |
| **CUDA Version** | Requires NVIDIA driver 470+ (implies CUDA 11.x+) |
| **macOS Support** | **NO. Linux only (Ubuntu 18.04/20.04).** |
| **Current Status** | DEPRECATED. NVIDIA recommends Isaac Lab as replacement |
| **POTENTIAL BLOCKER** | **YES -- HARD BLOCKER FOR macOS** |

**Why it is a blocker:** Isaac Gym requires (1) Linux, (2) an NVIDIA GPU with at least 8GB VRAM, (3) NVIDIA driver 470+. It is completely impossible to run on macOS with Apple Silicon. It is also deprecated and no longer supported by NVIDIA. Its replacement, Isaac Lab, also requires Linux + NVIDIA GPU.

**Verdict:** Do NOT use Isaac Gym for this project. Use MuJoCo instead.

**Source:** [NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym) | [NVIDIA Forum on deprecation](https://forums.developer.nvidia.com/t/isaac-gym-deprecation-transition-to-isaac-lab/322978) | [Forum on macOS](https://forums.developer.nvidia.com/t/is-it-possible-to-develop-using-macos/164267)

---

### 3. MuJoCo

| Field | Value |
|-------|-------|
| **EXISTS** | YES |
| **URL** | https://github.com/google-deepmind/mujoco |
| **Stars** | 12,100+ |
| **Latest Version** | 3.5.0 (released February 13, 2026) |
| **Free?** | YES -- Apache 2.0 license, fully open source |
| **macOS Support** | **YES -- native universal binary (Intel + Apple Silicon)** |
| **Apple Silicon** | YES -- "MuJoCo on the M1 Max is lightning fast" (Google DeepMind) |
| **GPU Acceleration** | MJX (MuJoCo XLA) supports Apple Silicon via JAX |
| **POTENTIAL BLOCKER** | **NO** |

**This is the correct simulation engine for this project.** MuJoCo is:
- Free and open source
- Actively maintained by Google DeepMind (monthly releases)
- Has native macOS Apple Silicon support
- Has 10 built-in Gymnasium locomotion environments (Ant, Humanoid, HalfCheetah, Hopper, Walker2d, etc.)
- Can be installed with a single command: `pip install "gymnasium[mujoco]"`

**Installation test (should work on your Mac):**
```python
import gymnasium as gym
env = gym.make("Ant-v5", render_mode="human")
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

**Source:** [MuJoCo GitHub](https://github.com/google-deepmind/mujoco) | [Gymnasium MuJoCo docs](https://gymnasium.farama.org/environments/mujoco/) | [Google DeepMind tweet on Apple Silicon](https://x.com/GoogleDeepMind/status/1471535887867592708)

---

### 4. SNN + RL Integration

| Field | Value |
|-------|-------|
| **EXISTS** | YES -- but NO off-the-shelf library |
| **Pre-built library?** | **NO.** No library combines snnTorch + standard RL algorithms out of the box |
| **Must build from scratch?** | **PARTIALLY YES** |
