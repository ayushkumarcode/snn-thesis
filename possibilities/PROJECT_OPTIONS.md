# SNN Thesis Project Options -- Honest Breakdown

**Context:** COMP30040, University of Manchester. ~28 days to code submission. Working heavily with Claude Code. Short iteration cycles preferred. Novelty is nice but not required -- report quality is 55% of the mark.

---

## Option 1: SNN on ESC-50 (Environmental Sound Classification)

**The pitch:** You'd be the first person to ever publish SNN results on ESC-50. That's not hype -- a March 2025 peer-reviewed paper explicitly confirms zero SNN papers exist on this dataset.

**What you'd actually build:**
- Load ESC-50 (2,000 audio clips, 50 classes like "dog bark", "chainsaw", "rain")
- Convert audio to mel-spectrograms (librosa, ~10 lines of code)
- Encode spectrograms as spikes (rate coding or direct coding)
- Train a convolutional SNN in snnTorch
- Train the same architecture as a regular ANN (swap snn.Leaky for nn.ReLU)
- Compare accuracy, energy (SynOps counting), and at least one more axis

**Iteration cycle:** Train loop runs in minutes on a laptop GPU. Each experiment = tweak config, run, see results. Fully Claude Code friendly -- it's standard PyTorch with snnTorch on top.

**What "good" looks like:** Even 70-80% accuracy is a genuine result because nobody has done this before. ANN baseline on ESC-50 is ~97% (with pre-trained models) or ~75-85% (from scratch). Matching or getting close to a from-scratch ANN = strong thesis.

**Novelty:** VERY HIGH -- automatic, because you're first.

**Risk:** LOW. The worst case is "SNNs don't work well on ESC-50" and that's still a publishable negative result with good analysis.

**Why pick this:** Best novelty-to-effort ratio. The literature review practically writes itself ("no SNN papers exist, here's why this matters"). Your supervisor will see you identified a genuine gap. The report framing is strong.

**Why not:** It's an application paper, not technically groundbreaking. You're applying existing methods to a new dataset. That's fine for undergrad, but it won't feel "cool" in the way robot reflexes would.

---

## Option 2: SNN ECG Classification on PTB-XL

**The pitch:** SNNs for cardiac monitoring. Wearable heart monitors need to run on tiny batteries for days. SNNs use 30-1000x less energy than standard deep learning. PTB-XL (21,799 ECGs, 12-lead) has no comprehensive SNN benchmark.

**What you'd actually build:**
- Load PTB-XL dataset (free, well-documented)
- Use snnTorch's delta encoding (designed for ECG-like signals)
- Train SNN classifier for cardiac conditions (5 superclasses or 23 subclasses)
- Compare against ANN baseline
- Report energy efficiency

**Iteration cycle:** Similar to ESC-50 -- standard train loops. PTB-XL is bigger (21K samples) so training takes a bit longer but still minutes, not hours. The 12-lead aspect adds preprocessing complexity compared to single-lead MIT-BIH.

**What "good" looks like:** MIT-BIH SNN SOTA is 98.3% (SparrowSNN). PTB-XL is harder -- current DNN benchmarks hit ~75-85% depending on the task. Getting competitive SNN results here would be strong.

**Novelty:** HIGH. Some SNN-ECG papers exist (~20-30 total) but none do a proper benchmark on PTB-XL with modern frameworks.

**Risk:** LOW-MEDIUM. 12-lead ECG has more preprocessing complexity than audio spectrograms. If it gets annoying, you can fall back to MIT-BIH (single-lead, simpler, but less novel).

**Why pick this:** Killer real-world narrative. "SNNs for energy-efficient cardiac monitoring" is a sentence that makes sense to everyone. The clinical angle makes for a compelling motivation section.

**Why not:** Slightly more preprocessing than ESC-50. The 12-lead format can be fiddly. Less "automatic novelty" than ESC-50 since some SNN-ECG work exists.

---

## Option 3: Wildlife Camera Trap Classification

**The pitch:** Zero SNN papers on wildlife camera traps. Camera traps are battery-powered in remote locations -- exactly where SNN energy efficiency matters most.

**What you'd actually build:**
- Load Snapshot Serengeti (3.2M images, 48 species) or Caltech Camera Traps (smaller)
- Standard image classification pipeline with convolutional SNN
- Rate-encode RGB images as spike trains
- Compare SNN vs CNN on same architecture
- Energy comparison

**Iteration cycle:** Standard image classification -- fast iterations. Large dataset though (3.2M images), so you'd probably subsample.

**Novelty:** VERY HIGH (zero papers).

**Risk:** LOW-MEDIUM. The problem: these are RGB images. SNNs have no natural advantage over CNNs on static images. Your SNN will almost certainly be worse than the CNN, and the energy argument is weaker for images than for temporal data (audio, ECG). You'd need to frame this carefully.

**Why pick this:** Zero papers = automatic novelty. Great real-world narrative. Large, clean datasets.

**Why not:** Weakest natural SNN fit of the top options. Static RGB images don't play to SNN strengths. You'd be fighting uphill to explain why SNNs matter here beyond energy efficiency.

---

## Option 4: Framework Shootout (snnTorch vs SpikingJelly vs Norse)

**The pitch:** Nobody has ever done a proper three-way comparison of SNN frameworks on the same tasks with the same hyperparameters. The community doesn't know if framework choice affects accuracy, training time, or energy estimates. You'd answer that.

**What you'd actually build:**
- Pick 2-3 datasets (SHD, DVS128, and one static like CIFAR-10)
- Implement the SAME architecture in all three frameworks
- Train with the SAME hyperparameters
- Compare: accuracy, training time, memory usage, lines of code, API ergonomics
- Maybe add energy estimation comparison

**Iteration cycle:** Very short per experiment, but lots of experiments (3 frameworks x 2-3 datasets x multiple runs). Each individual run is fast. The work is breadth, not depth.

**What "good" looks like:** A clear comparison table showing where frameworks diverge. Even if they all get similar accuracy, documenting the API differences, gotchas, and performance characteristics is valuable.

**Novelty:** GENUINE confirmed gap. No three-way comparison exists. A 2025 benchmark deliberately excluded snnTorch and Norse.

**Risk:** LOW. Even if all frameworks perform identically, that's a result. The worst case is still a useful contribution.

**Why pick this:** Lowest risk option. Immediately useful to anyone starting SNN research. Multiple small deliverables rather than one big bet. Good for "systematic, thorough evaluation" framing that markers love.

**Why not:** Less exciting narrative than "first SNN on X domain." It's a meta-study, not an application. Some might see it as "just running benchmarks."

---

## Option 5: SNN Robot Reflexes in Simulation

**The pitch:** Biological reflexes use spiking neurons. You'd implement SNN-based reflex controllers for simulated robots using SpikeGym + Isaac Gym.

**What you'd actually build:**
- Set up Isaac Gym or MuJoCo simulation environment
- Use SpikeGym framework (SNN + RL)
- Train SNN policy for balance/locomotion tasks
- Compare SNN vs ANN policy on latency, energy, and task performance

**Iteration cycle:** THIS IS THE PROBLEM. Environment setup can eat 1-2 weeks. RL training is unstable -- you can spend days tuning reward functions and hyperparameters with no progress. SpikeGym is <1 year old, likely has bugs and sparse documentation. When it works, training is ~7 minutes. When it doesn't, you're debugging physics simulations.

**Novelty:** HIGH (~5-10 papers in the area).

**Risk:** MEDIUM-HIGH. RL + SNN is double the debugging complexity. The bottleneck here isn't development speed -- it's training stability, which Claude Code can't fix.

**Why pick this:** Most impressive if it works. Strongest biological motivation. Best Manchester/SpiNNaker alignment. Would genuinely impress a supervisor.

**Why not:** You said you want short iteration cycles and to get it done. This is the opposite. RL training instability is exactly the kind of "fixed iteration time" problem that doesn't compress with better tooling.

---

## Option 6: SNN Satellite Image Classification

**The pitch:** Edge AI in space. Satellites can't afford GPU power budgets. Neuromorphic chips are already in orbit.

**Novelty:** MODERATE (~15-25 papers already exist). ESA-funded SNN4Space codebase achieves 95.07% on EuroSAT.

**Why not (short version):** ESC-50 beats it on novelty (0 vs 15-25 papers). ECG beats it on narrative. Robot reflexes beats it on excitement. It sits in an awkward middle ground -- not novel enough to be exciting, not easy enough to be pragmatic. The SNN fit is also weak (standard RGB images).

---

## Option 7: SNN Food Recognition (Food-101)

**The pitch:** Zero SNN papers on food image classification.

**Novelty:** VERY HIGH (zero papers).

**Why not (short version):** Weakest real-world narrative. "Why would you use an SNN to classify food?" doesn't have a compelling answer beyond "nobody's tried it." Static images, no natural SNN advantage. It works but it's boring.

---

## The Honest Recommendation

Given what you've told me -- Claude Code heavy, short iteration cycles, ~28 days, pragmatic but wouldn't mind some novelty:

**Go with ESC-50.** It's the sweet spot:
- Zero papers = automatic novelty without trying
- Audio = natural SNN fit (temporal data)
- Standard PyTorch train loop = Claude Code friendly
- Each experiment runs in minutes
- Even mediocre accuracy is publishable (you're first)
- The report writes itself (clear gap, clear method, clear results)
- Fall back to SHD/SSC if things go sideways (same pipeline, known good results)

**Second choice: Framework Shootout** if you'd rather do breadth over depth. Lower novelty but also lower risk. Many small experiments instead of one big bet.

**Everything else is either harder (robot reflexes, ECG 12-lead), less novel (satellite), or weaker narrative (food, wildlife).**

---

*Generated from 35+ research agents across 3 waves of analysis. See `research/WAVE1_MASTER_SYNTHESIS.md` for full technical details and source citations.*
