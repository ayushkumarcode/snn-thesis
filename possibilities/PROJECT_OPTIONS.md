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

