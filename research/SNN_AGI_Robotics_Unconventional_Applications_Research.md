# SNN applications in AGI, robotics, and unconventional domains

i investigated 25+ SNN application domains across three categories: AGI-related concepts, robotics applications, and unconventional uses. Searched across arxiv, IEEE, Nature, Frontiers, GitHub, and multiple academic databases.

**AGI-Related:** The Kyambadde thesis turned out to be a Philosophy PhD with zero SNN content. SNN + continual learning is the most viable AGI-adjacent direction, with good literature and clear undergrad-feasible scope (Split-MNIST with EWC adaptation). SNN + meta-learning exists but is too technically demanding. Cognitive architectures (SPAUN, BrainCog) are fascinating but way too much scope for a few weeks.

**Robotics:** The most promising directions are (1) SNN for obstacle avoidance in simulation (CartPole/LunarLander), (2) SNN for robotic arm target-reaching in MuJoCo, and (3) SNN for optical flow estimation from event camera data. All can be done simulation-only on macOS. Tactile sensing, SLAM, and swarm robotics are interesting but need either specialized hardware data or way too much implementation.

**Unconventional:** SNN for cybersecurity/intrusion detection is the strongest unconventional option -- tabular data, public datasets (NSL-KDD, CICIDS2017), clear SNN efficiency narrative, and published results showing 99%+ accuracy. SNN for ECG classification is another strong contender. Game AI and autonomous driving are established fields with limited novelty. Drug discovery and weather prediction have minimal SNN presence.

**Top 3 new candidates from this research (not previously considered):**
1. SNN for continual learning (Split-MNIST, catastrophic forgetting mitigation)
2. SNN for robotic arm control in MuJoCo (RL-based, simulation-only)
3. SNN for image denoising (novel, few papers, competitive results)

---

## Part 1: AGI-related SNN research

---

### 1.1 The Kyambadde thesis: what it actually is

**Full title:** "Replicating cognitive self-sufficiency to realize computational self-sufficiency in Artificial General Intelligence models"
**Author:** Ivan Serwano Kyambadde
**Institution:** University of York, Department of **Philosophy** (NOT Computer Science)
**Supervisor:** Tom Stoneham (philosopher)
**Completion:** November 2023
