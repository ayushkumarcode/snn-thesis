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

