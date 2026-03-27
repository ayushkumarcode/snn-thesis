# SNN Thesis Project Options -- Honest Breakdown

COMP30040, University of Manchester. about 28 days to code submission. working heavily with Claude Code, so short iteration cycles are ideal. novelty is nice but not required -- report quality is 55% of the mark.

---

## Option 1: SNN on ESC-50 (Environmental Sound Classification)

the pitch: i'd be the first person to ever publish SNN results on ESC-50. a March 2025 peer-reviewed paper explicitly confirms zero SNN papers exist on this dataset. that's not hype, it's just a gap nobody's filled yet.

what i'd actually build:
- load ESC-50 (2,000 audio clips, 50 classes like "dog bark", "chainsaw", "rain")
- convert audio to mel-spectrograms (librosa, ~10 lines of code)
- encode spectrograms as spikes (rate coding or direct coding)
- train a convolutional SNN in snnTorch
- train the same architecture as a regular ANN (swap snn.Leaky for nn.ReLU)
- compare accuracy, energy (SynOps counting), and at least one more axis

iteration cycle: train loop runs in minutes on a laptop GPU. each experiment = tweak config, run, see results. fully Claude Code friendly -- it's standard PyTorch with snnTorch on top.

what "good" looks like: even 70-80% accuracy is a genuine result because nobody has done this before. ANN baseline on ESC-50 is ~97% (with pre-trained models) or ~75-85% (from scratch). matching or getting close to a from-scratch ANN = strong thesis.

novelty: VERY HIGH -- automatic, because i'm first.

risk: LOW. the worst case is "SNNs don't work well on ESC-50" and that's still a publishable negative result with good analysis.

why pick this: best novelty-to-effort ratio. the literature review practically writes itself ("no SNN papers exist, here's why this matters"). supervisor will see i identified a genuine gap. the report framing is strong.

why not: it's an application paper, not technically groundbreaking. i'm applying existing methods to a new dataset. that's fine for undergrad, but it won't feel "cool" in the way robot reflexes would.

---

## Option 2: SNN ECG Classification on PTB-XL

the pitch: SNNs for cardiac monitoring. wearable heart monitors need to run on tiny batteries for days. SNNs use 30-1000x less energy than standard deep learning. PTB-XL (21,799 ECGs, 12-lead) has no proper SNN benchmark.

what i'd actually build:
- load PTB-XL dataset (free, well-documented)
- use snnTorch's delta encoding (designed for ECG-like signals)
- train SNN classifier for cardiac conditions (5 superclasses or 23 subclasses)
- compare against ANN baseline
- report energy efficiency

iteration cycle: similar to ESC-50 -- standard train loops. PTB-XL is bigger (21K samples) so training takes a bit longer but still minutes, not hours. the 12-lead aspect adds preprocessing complexity compared to single-lead MIT-BIH.

what "good" looks like: MIT-BIH SNN SOTA is 98.3% (SparrowSNN). PTB-XL is harder -- current DNN benchmarks hit ~75-85% depending on the task. getting competitive SNN results here would be strong.

novelty: HIGH. some SNN-ECG papers exist (~20-30 total) but none do a proper benchmark on PTB-XL with modern frameworks.

risk: LOW-MEDIUM. 12-lead ECG has more preprocessing complexity than audio spectrograms. if it gets annoying, i can fall back to MIT-BIH (single-lead, simpler, but less novel).

why pick this: killer real-world narrative. "SNNs for energy-efficient cardiac monitoring" makes sense to everyone. the clinical angle makes for a compelling motivation section.

why not: slightly more preprocessing than ESC-50. the 12-lead format can be fiddly. less "automatic novelty" than ESC-50 since some SNN-ECG work exists.

---

## Option 3: Wildlife Camera Trap Classification

the pitch: zero SNN papers on wildlife camera traps. camera traps are battery-powered in remote locations -- exactly where SNN energy efficiency matters most.

what i'd actually build:
- load Snapshot Serengeti (3.2M images, 48 species) or Caltech Camera Traps (smaller)
- standard image classification pipeline with convolutional SNN
- rate-encode RGB images as spike trains
- compare SNN vs CNN on same architecture
- energy comparison

iteration cycle: standard image classification -- fast iterations. large dataset though (3.2M images), so i'd probably subsample.

novelty: VERY HIGH (zero papers).

risk: LOW-MEDIUM. the problem: these are RGB images. SNNs have no natural advantage over CNNs on static images. the SNN will almost certainly be worse than the CNN, and the energy argument is weaker for images than for temporal data (audio, ECG). would need to frame this carefully.

why pick this: zero papers = automatic novelty. great real-world narrative. large, clean datasets.

why not: weakest natural SNN fit of the top options. static RGB images don't play to SNN strengths. i'd be fighting uphill to explain why SNNs matter here beyond energy efficiency.

---

## Option 4: Framework Shootout (snnTorch vs SpikingJelly vs Norse)

the pitch: nobody has ever done a proper three-way comparison of SNN frameworks on the same tasks with the same hyperparameters. the community doesn't know if framework choice affects accuracy, training time, or energy estimates. i'd answer that.

what i'd actually build:
- pick 2-3 datasets (SHD, DVS128, and one static like CIFAR-10)
- implement the SAME architecture in all three frameworks
- train with the SAME hyperparameters
- compare: accuracy, training time, memory usage, lines of code, API ergonomics
- maybe add energy estimation comparison

iteration cycle: very short per experiment, but lots of experiments (3 frameworks x 2-3 datasets x multiple runs). each individual run is fast. the work is breadth, not depth.

what "good" looks like: a clear comparison table showing where frameworks diverge. even if they all get similar accuracy, documenting the API differences, gotchas, and performance characteristics is valuable.

novelty: genuine confirmed gap. no three-way comparison exists. a 2025 benchmark deliberately excluded snnTorch and Norse.

risk: LOW. even if all frameworks perform identically, that's a result. worst case is still a useful contribution.

why pick this: lowest risk option. immediately useful to anyone starting SNN research. multiple small deliverables rather than one big bet.

why not: less exciting narrative than "first SNN on X domain." it's a meta-study, not an application. some might see it as "just running benchmarks."

---

## Option 5: SNN Robot Reflexes in Simulation

the pitch: biological reflexes use spiking neurons. i'd implement SNN-based reflex controllers for simulated robots using SpikeGym + Isaac Gym.

what i'd actually build:
- set up Isaac Gym or MuJoCo simulation environment
- use SpikeGym framework (SNN + RL)
- train SNN policy for balance/locomotion tasks
- compare SNN vs ANN policy on latency, energy, and task performance

iteration cycle: THIS IS THE PROBLEM. environment setup can eat 1-2 weeks. RL training is unstable -- you can spend days tuning reward functions and hyperparameters with no progress. SpikeGym is <1 year old, likely has bugs and sparse documentation. when it works, training is ~7 minutes. when it doesn't, you're debugging physics simulations.

novelty: HIGH (~5-10 papers in the area).

risk: MEDIUM-HIGH. RL + SNN is double the debugging complexity. the bottleneck isn't development speed -- it's training stability, which Claude Code can't fix.

why pick this: most impressive if it works. strongest biological motivation. best Manchester/SpiNNaker alignment. would genuinely impress a supervisor.

why not: i want short iteration cycles and to get it done. this is the opposite. RL training instability is exactly the kind of problem that doesn't compress with better tooling.

---

## Option 6: SNN Satellite Image Classification

the pitch: edge AI in space. satellites can't afford GPU power budgets. neuromorphic chips are already in orbit.

novelty: MODERATE (~15-25 papers already exist). ESA-funded SNN4Space codebase achieves 95.07% on EuroSAT.

why not (short version): ESC-50 beats it on novelty (0 vs 15-25 papers). ECG beats it on narrative. robot reflexes beats it on excitement. it sits in an awkward middle ground -- not novel enough to be exciting, not easy enough to be pragmatic. the SNN fit is also weak (standard RGB images).

---

## Option 7: SNN Food Recognition (Food-101)

the pitch: zero SNN papers on food image classification.

novelty: VERY HIGH (zero papers).

why not (short version): weakest real-world narrative. "why would you use an SNN to classify food?" doesn't have a compelling answer beyond "nobody's tried it." static images, no natural SNN advantage. it works but it's boring.

---

## my recommendation to myself

given what i know -- Claude Code heavy, short iteration cycles, ~28 days, pragmatic but wouldn't mind some novelty:

**go with ESC-50.** it's the sweet spot:
- zero papers = automatic novelty without trying
- audio = natural SNN fit (temporal data)
- standard PyTorch train loop = Claude Code friendly
- each experiment runs in minutes
- even mediocre accuracy is publishable (i'm first)
- the report writes itself (clear gap, clear method, clear results)
- can fall back to SHD/SSC if things go sideways (same pipeline, known good results)

second choice: framework shootout if i'd rather do breadth over depth. lower novelty but also lower risk. many small experiments instead of one big bet.

everything else is either harder (robot reflexes, ECG 12-lead), less novel (satellite), or weaker narrative (food, wildlife).
