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
