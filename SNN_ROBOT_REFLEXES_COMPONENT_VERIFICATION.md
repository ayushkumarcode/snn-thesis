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
| **Complexity** | HIGH -- this is the core research contribution |
| **POTENTIAL BLOCKER** | **YES (MEDIUM)** |

**The integration does NOT exist as a pip-installable package.** Here is what actually exists:

1. **SpikeGym** (3 commits on GitLab) -- modifies `skrl` for SNN support, research-quality code only
2. **PopSAN** (https://github.com/combra-lab/pop-spiking-deep-rl) -- 64 stars, last commit Oct 2020, custom SNN implementation (NOT snnTorch), supports PPO/DDPG/TD3/SAC with MuJoCo. Outdated dependencies (Python 3.5, MuJoCo 2.0)
3. **RL-SNN-Quadrupeds** (https://github.com/tganamur/RL-SNN-Quadrupeds) -- 13 stars, 4 commits, UC Berkeley course project, uses Stable-Baselines3 + MuJoCo. SNN approach only achieved "steady standing" (did not walk successfully)

**What you actually need to do:**
- Use Stable-Baselines3 with its custom policy network API
- Replace the MLP policy with an snnTorch-based SNN policy
- SB3 allows custom `features_extractor` and custom network architectures via `policy_kwargs`
- This is a non-trivial integration requiring ~200-500 lines of custom code

**Source:** [SB3 custom policy docs](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html) | [PopSAN repo](https://github.com/combra-lab/pop-spiking-deep-rl) | [RL-SNN-Quadrupeds](https://github.com/tganamur/RL-SNN-Quadrupeds)

---

### 5. SNN Policy Network (Continuous Action Output)

| Field | Value |
|-------|-------|
| **EXISTS** | YES -- proven in research |
| **Off-the-shelf?** | NO |
| **Mechanism verified?** | YES -- three known approaches |
| **POTENTIAL BLOCKER** | **YES (MEDIUM)** -- requires careful implementation |

**The core problem:** SNNs output binary spikes (0 or 1), but RL needs continuous actions (e.g., joint torques from -1.0 to 1.0).

**Three verified mechanisms to bridge this gap:**

1. **Membrane potential readout (RECOMMENDED):** Disable spiking in the output layer by setting the firing threshold to infinity. Read the raw membrane potential as the continuous action value. This is the simplest and most commonly used approach.
   ```python
   # Conceptual snnTorch implementation
   import snntorch as snn
   # Output layer: LIF neuron with threshold=infinity (never spikes)
   lif_out = snn.Leaky(beta=0.9, threshold=float('inf'))
   # membrane potential IS the continuous action
   ```

2. **Rate coding:** Run the SNN for T timesteps, count spikes per output neuron, normalize to get a rate in [0, 1], then scale to action range. Slower but more biologically plausible.

3. **Population coding (PopSAN):** Use a population of neurons per action dimension. Each sub-population encodes a Gaussian-tuned response. Decode by weighted average of population activity. Most complex but highest capacity.

**Source:** [SpikeGym paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11680704/) | [PopSAN paper](https://proceedings.mlr.press/v155/tang21a.html) | [Fully Spiking Neural Network for Legged Robots](https://arxiv.org/abs/2310.05022)

---

### 6. Reward Function

| Field | Value |
|-------|-------|
| **EXISTS** | YES -- pre-built in Gymnasium |
| **Pre-built?** | **YES -- fully implemented** |
| **Customization needed?** | NO (for standard locomotion) |
| **POTENTIAL BLOCKER** | **NO** |

**Standard reward functions are already implemented in Gymnasium MuJoCo environments:**

**Ant-v5 (recommended starting environment):**
```
reward = healthy_reward + forward_reward - ctrl_cost - contact_cost
```
- `healthy_reward`: +1.0 per timestep the ant is upright (z in [0.2, 1.0])
- `forward_reward`: proportional to forward velocity (x-direction)
- `ctrl_cost`: penalty for large joint torques
- `contact_cost`: penalty for external contact forces

**Humanoid-v5:**
```
reward = healthy_reward + forward_reward - ctrl_cost - contact_cost
```
- `healthy_reward`: +5.0 per timestep (higher than Ant because harder to balance)

**Termination (implicit balance enforcement):**
- Episode ends if the torso height leaves the healthy range (e.g., the robot falls over)
- This naturally forces the agent to learn balance as a prerequisite for locomotion

**No custom reward engineering needed for an undergraduate thesis.** The built-in rewards are the standard benchmark used by all RL papers.

**Source:** [Gymnasium Ant docs](https://gymnasium.farama.org/environments/mujoco/ant/) | [Gymnasium Humanoid docs](https://gymnasium.farama.org/environments/mujoco/humanoid/)

---

### 7. Training Stability

| Field | Value |
|-------|-------|
| **EXISTS** | YES -- documented extensively in literature |
| **Stable?** | **NO -- SNN+RL training is known to be unstable** |
| **Known issues** | Multiple critical issues documented |
| **Typical runs needed** | 5+ seeds minimum, report mean +/- std |
| **POTENTIAL BLOCKER** | **YES (HIGH)** -- this is the hardest part of the project |

**Documented stability issues with SNN + RL:**

1. **Non-differentiable spike function:** The derivative does not exist at the spike threshold. Surrogate gradients approximate this but introduce bias. Training can oscillate or fail to converge.

2. **Membrane potential oscillation:** The last membrane voltage can cause nonperiodic oscillatory behavior, making learning unstable and hard to converge.

3. **Discrete-continuous mismatch:** Discrete SNN outputs conflict with continuous target network soft updates in off-policy RL, causing "abrupt output shifts," oscillatory updates, and high sensitivity to random seed initialization.

4. **Temporal encoding challenges:** If input signals arrive after the first output neuron spikes, they cannot effectively participate in training.

5. **Gradient vanishing/explosion:** Due to non-differentiable spiking signals; restricts effective SNN depth to shallow architectures in some cases.

**Mitigation strategies:**
- Use PPO (on-policy) rather than SAC/TD3 (off-policy) -- avoids the discrete-continuous mismatch
- Use membrane potential readout instead of rate coding -- smoother gradients
- Use surrogate gradient with appropriate steepness parameter
- Train with 5+ random seeds and report statistics
- Start with simpler environments (CartPole, InvertedPendulum) before Ant

**Source:** [Spiking Q-learning paper](https://arxiv.org/html/2201.09754v2) | [Temporal coding paper](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.877701/full) | [Proxy Target paper](https://arxiv.org/html/2505.24161)

---

### 8. Evaluation Metrics

| Field | Value |
|-------|-------|
| **EXISTS** | YES -- well-established standards |
| **Pre-built?** | Partially (reward is automatic; others need manual logging) |
| **POTENTIAL BLOCKER** | **NO** |

**Standard metrics for robot locomotion RL (all verifiable from literature):**

| Metric | How to Measure | Standard? |
|--------|---------------|-----------|
| **Cumulative reward** | Sum of rewards per episode | YES -- primary metric |
| **Episode length** | Timesteps before termination/truncation | YES -- measures stability |
| **Forward velocity** | x-velocity from `info` dict | YES |
| **Distance traveled** | Cumulative x-displacement | YES |
| **Cost of Transport (CoT)** | Energy / (weight * distance) | YES -- standard efficiency metric |
| **Torque magnitude** | L2 norm of actions | YES -- energy proxy |
| **Balance duration** | Timesteps before falling | YES -- for reflex tasks |
| **Disturbance recovery** | Time to recover after perturbation push | NOVEL -- good thesis contribution |
| **SNN-specific: Spike count** | Total spikes per inference | YES -- for energy efficiency argument |
| **SNN-specific: Inference latency** | Wall-clock time per action | YES -- for real-time argument |

**The reflex-specific contribution for the thesis could be:**
- Apply random perturbation forces during evaluation
- Measure recovery time and maximum perturbation the agent can withstand
- Compare SNN policy vs. ANN policy on disturbance rejection speed

**Source:** [Locomotion benchmarking](https://arxiv.org/html/2501.16590) | [Cost of Transport metric](https://www.mdpi.com/2075-1702/10/3/185)

---

### 9. GPU Requirements

| Field | Value |
|-------|-------|
| **Simulation requires GPU?** | NO -- MuJoCo runs on CPU |
| **SNN training requires GPU?** | Recommended but NOT required |
| **Can run on laptop?** | **YES** |
| **POTENTIAL BLOCKER** | **NO (but training will be slow)** |

**Estimated training times:**

| Setup | Environment | Estimated Wall-Clock Time |
|-------|------------|--------------------------|
| MacBook (CPU only) | InvertedPendulum | ~10-30 minutes |
| MacBook (CPU only) | Ant-v5 (20M steps PPO) | ~4-12 hours |
| MacBook (MPS GPU) | Ant-v5 (20M steps PPO) | ~2-6 hours (estimated) |
| Google Colab (T4 GPU) | Ant-v5 (20M steps PPO) | ~1-3 hours |
| Isaac Gym (full GPU) | Ant-v5 | ~7 minutes |
