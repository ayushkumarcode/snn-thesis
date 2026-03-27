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

